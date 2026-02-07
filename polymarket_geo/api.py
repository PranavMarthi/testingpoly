"""
FastAPI service exposing market location data.

Endpoints:
  GET /nearby       - Markets within radius km of a lat/lon point
  GET /search       - Search by city/country string
  GET /market/{id}  - Single market with all inferred locations
  GET /health       - Pipeline health metrics
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from polymarket_geo.config import get_settings
from polymarket_geo.db import (
    close_pool,
    get_market_by_condition_id,
    get_market_by_id,
    get_pipeline_metrics,
    get_pool,
    query_markets_by_text,
    query_nearby_markets,
    run_migrations,
)
from polymarket_geo.geocode import geocode_location
from polymarket_geo.models import (
    HealthResponse,
    MarketLocationResponse,
    MarketResponse,
    NearbyResponse,
    SearchResponse,
)

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB pool + run migrations. Shutdown: close pool."""
    logger.info("Starting up API server...")
    await get_pool()
    try:
        await run_migrations()
    except Exception as e:
        logger.warning("Migration failed (may already exist): %s", e)
    yield
    await close_pool()
    logger.info("API server shut down.")


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Polymarket Geo API",
    description="Query Polymarket prediction markets by geographic location",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────

def _format_market(row: dict) -> MarketResponse:
    """Convert a DB row dict into a MarketResponse."""
    locations = []
    if "locations" in row:
        for loc in row["locations"]:
            locations.append(MarketLocationResponse(
                location_name=loc["location_name"],
                location_type=loc["location_type"],
                confidence=loc["confidence"],
                reason=loc.get("reason"),
                latitude=loc.get("latitude"),
                longitude=loc.get("longitude"),
                inference_method=loc.get("inference_method", "unknown"),
            ))

    return MarketResponse(
        id=row["id"],
        condition_id=row["condition_id"],
        question=row["question"],
        description=row.get("description"),
        category=row.get("category"),
        active=row.get("active", True),
        volume=row.get("volume"),
        locations=locations,
    )


def _group_rows_to_markets(rows: list[dict]) -> list[MarketResponse]:
    """
    Group joined market+location rows into MarketResponse objects.
    Multiple rows per market (one per location) get merged.
    """
    markets_map: dict[int, dict] = {}

    for row in rows:
        mid = row["id"]
        if mid not in markets_map:
            markets_map[mid] = {
                "id": mid,
                "condition_id": row["condition_id"],
                "question": row["question"],
                "description": row.get("description"),
                "category": row.get("category"),
                "active": row.get("active", True),
                "volume": row.get("volume"),
                "locations": [],
            }

        if row.get("location_id"):
            markets_map[mid]["locations"].append({
                "location_name": row["location_name"],
                "location_type": row["location_type"],
                "confidence": row["confidence"],
                "reason": row.get("reason"),
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "inference_method": row.get("inference_method", "unknown"),
            })

    return [_format_market(m) for m in markets_map.values()]


# ══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.get("/nearby", response_model=NearbyResponse)
async def nearby_markets(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    radius_km: float = Query(None, ge=0.1, description="Search radius in km"),
    min_confidence: float = Query(0.0, ge=0, le=1, description="Minimum confidence threshold"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """
    Find markets within a radius of a geographic point.

    Example PostGIS query used internally:
    ```sql
    SELECT m.*, ml.*,
           ST_Distance(ml.geog, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography) AS distance_m
    FROM markets m
    JOIN market_locations ml ON ml.market_id = m.id
    WHERE ml.geocoded = TRUE
      AND ST_DWithin(ml.geog, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :radius_m)
    ORDER BY distance_m ASC, ml.confidence DESC
    LIMIT :limit OFFSET :offset;
    ```
    """
    settings = get_settings().api
    if radius_km is None:
        radius_km = settings.default_radius_km
    if radius_km > settings.max_radius_km:
        raise HTTPException(400, f"radius_km must be <= {settings.max_radius_km}")

    rows, total = await query_nearby_markets(
        lat=lat, lon=lon, radius_km=radius_km,
        min_confidence=min_confidence, limit=limit, offset=offset,
    )

    markets = _group_rows_to_markets(rows)

    return NearbyResponse(
        markets=markets,
        total=total,
        center_lat=lat,
        center_lon=lon,
        radius_km=radius_km,
    )


@app.get("/search", response_model=SearchResponse)
async def search_markets(
    q: str = Query(..., min_length=1, max_length=200, description="City, country, or market text"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    geocode: bool = Query(True, description="Also geocode the search query and search nearby"),
):
    """
    Search markets by city/country/text.

    Strategy:
      1. Text search: ILIKE on question and location_name (trigram-indexed)
      2. If geocode=True, also geocode the query and do a spatial search
      3. Merge and deduplicate results

    This allows searching "Atlanta" to find both:
      - Markets mentioning "Atlanta" in text
      - Markets geocoded near Atlanta's coordinates
    """
    # Text search
    rows, total = await query_markets_by_text(q, limit=limit, offset=offset)
    markets = _group_rows_to_markets(rows)

    resolved_location = None
    resolved_lat = None
    resolved_lon = None

    # Optionally geocode the search term and do a spatial search too
    if geocode and len(markets) < limit:
        try:
            geo_result = await geocode_location(q)
            if geo_result.latitude and geo_result.longitude:
                resolved_location = geo_result.display_name
                resolved_lat = geo_result.latitude
                resolved_lon = geo_result.longitude

                # Spatial search around the geocoded point
                spatial_rows, spatial_total = await query_nearby_markets(
                    lat=geo_result.latitude,
                    lon=geo_result.longitude,
                    radius_km=50.0,
                    limit=limit - len(markets),
                )
                spatial_markets = _group_rows_to_markets(spatial_rows)

                # Merge, deduplicating by market id
                existing_ids = {m.id for m in markets}
                for sm in spatial_markets:
                    if sm.id not in existing_ids:
                        markets.append(sm)
                        existing_ids.add(sm.id)

                total = max(total, len(markets))
        except Exception as e:
            logger.warning("Search geocoding failed for '%s': %s", q, e)

    return SearchResponse(
        markets=markets,
        total=total,
        query=q,
        resolved_location=resolved_location,
        resolved_lat=resolved_lat,
        resolved_lon=resolved_lon,
    )


@app.get("/market/{market_id}", response_model=MarketResponse)
async def get_market(market_id: int | str):
    """
    Get a single market with all its inferred locations.
    Accepts numeric DB id or Polymarket condition_id string.
    """
    try:
        # Try as numeric ID first
        numeric_id = int(market_id)
        result = await get_market_by_id(numeric_id)
    except (ValueError, TypeError):
        # Try as condition_id
        result = await get_market_by_condition_id(str(market_id))

    if result is None:
        raise HTTPException(404, "Market not found")

    return _format_market(result)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Pipeline and data health metrics."""
    try:
        metrics = await get_pipeline_metrics()
        total = metrics.get("total_markets", 0)
        with_locs = metrics.get("markets_with_locations", 0)
        pct = round(100.0 * with_locs / total, 1) if total > 0 else 0.0

        return HealthResponse(
            status="ok",
            total_markets=total,
            processed_markets=metrics.get("processed_markets", 0),
            pct_with_location=pct,
            avg_confidence=float(metrics["avg_confidence"]) if metrics.get("avg_confidence") else None,
        )
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return HealthResponse(status="error")


@app.get("/metrics")
async def detailed_metrics():
    """Detailed pipeline metrics for monitoring dashboards."""
    metrics = await get_pipeline_metrics()
    return metrics
