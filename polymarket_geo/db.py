"""
Database connection management and query functions.
Uses asyncpg for async Postgres access with connection pooling.
All spatial queries use PostGIS functions.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import asyncpg

from polymarket_geo.config import get_settings
from polymarket_geo.models import LocationCandidate, MarketInferenceResult, RawMarket

logger = logging.getLogger(__name__)

# ── Connection Pool ────────────────────────────────────────────────────

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await asyncpg.create_pool(
            dsn=settings.db.dsn,
            min_size=settings.db.min_pool_size,
            max_size=settings.db.max_pool_size,
        )
        logger.info("Database connection pool created (min=%d, max=%d)",
                     settings.db.min_pool_size, settings.db.max_pool_size)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")


@asynccontextmanager
async def get_connection() -> AsyncIterator[asyncpg.Connection]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


# ── Schema Initialization ─────────────────────────────────────────────

async def run_migrations() -> None:
    """Execute SQL migrations in order (idempotent)."""
    from pathlib import Path

    migrations_dir = Path(__file__).parent / "migrations"
    migration_paths = sorted(migrations_dir.glob("*.sql"))

    async with get_connection() as conn:
        for path in migration_paths:
            sql = path.read_text()
            await conn.execute(sql)
            logger.info("Applied migration: %s", path.name)
    logger.info("Migrations applied successfully (%d files)", len(migration_paths))


# ── Market Upserts ────────────────────────────────────────────────────

async def upsert_market(conn: asyncpg.Connection, market: RawMarket, raw_payload: dict) -> int:
    """
    Insert or update a market by condition_id.
    Returns the database id of the market row.

    Upsert strategy: ON CONFLICT (condition_id) DO UPDATE
    - Always refresh price/volume/active/closed (these change frequently)
    - Never overwrite question/description (stable after creation)
    """
    row = await conn.fetchrow(
        """
        INSERT INTO markets (
            condition_id, question, description, market_slug, category,
            end_date_iso, active, closed, volume, liquidity,
            outcomes, outcome_prices, tags, raw_payload
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        ON CONFLICT (condition_id) DO UPDATE SET
            active = EXCLUDED.active,
            closed = EXCLUDED.closed,
            volume = EXCLUDED.volume,
            liquidity = EXCLUDED.liquidity,
            outcome_prices = EXCLUDED.outcome_prices,
            raw_payload = EXCLUDED.raw_payload,
            updated_at = NOW()
        RETURNING id
        """,
        market.condition_id,
        market.question,
        market.description,
        market.market_slug,
        market.category,
        market.end_date_iso,
        market.active,
        market.closed,
        market.volume,
        market.liquidity,
        json.dumps(market.outcomes) if market.outcomes else None,
        json.dumps(market.outcome_prices) if market.outcome_prices else None,
        json.dumps(market.tags) if market.tags else None,
        json.dumps(raw_payload),
    )
    return row["id"]


async def upsert_markets_batch(markets: list[tuple[RawMarket, dict]]) -> dict:
    """
    Batch upsert markets. Returns stats dict.
    """
    stats = {"fetched": len(markets), "new": 0, "updated": 0}
    async with get_connection() as conn:
        for market, raw in markets:
            # Check if exists before upsert to count new vs updated
            existing = await conn.fetchval(
                "SELECT id FROM markets WHERE condition_id = $1",
                market.condition_id,
            )
            market_id = await upsert_market(conn, market, raw)
            if existing is None:
                stats["new"] += 1
            else:
                stats["updated"] += 1
    return stats


# ── Location Upserts ─────────────────────────────────────────────────

async def upsert_location(
    conn: asyncpg.Connection,
    market_id: int,
    loc: LocationCandidate,
    geo_version: int,
) -> int:
    """
    Insert or update a location candidate for a market.
    UNIQUE constraint on (market_id, location_name, geo_version) prevents duplicates.
    When the same location is inferred again in the same version, we update confidence/reason.
    """
    row = await conn.fetchrow(
        """
        INSERT INTO market_locations (
            market_id, location_name, location_type, confidence, reason,
            inference_method, latitude, longitude, geog, geocoded,
            geocode_source, geo_version
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8,
            CASE WHEN $7 IS NOT NULL AND $8 IS NOT NULL
                 THEN ST_SetSRID(ST_MakePoint($8, $7), 4326)::geography
                 ELSE NULL END,
            CASE WHEN $7 IS NOT NULL AND $8 IS NOT NULL THEN TRUE ELSE FALSE END,
            $9, $10
        )
        ON CONFLICT (market_id, location_name, geo_version) DO UPDATE SET
            confidence = EXCLUDED.confidence,
            reason = EXCLUDED.reason,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            geog = EXCLUDED.geog,
            geocoded = EXCLUDED.geocoded,
            geocode_source = EXCLUDED.geocode_source,
            updated_at = NOW()
        RETURNING id
        """,
        market_id,
        loc.location_name,
        loc.location_type.value,
        loc.confidence,
        loc.reason,
        loc.inference_method.value,
        loc.latitude,
        loc.longitude,
        "cache" if loc.latitude else None,  # will be updated after geocoding
        geo_version,
    )
    return row["id"]


async def save_inference_result(
    result: MarketInferenceResult,
    geo_version: int,
) -> None:
    """
    Save all location candidates for a market and mark it as geo_processed.
    Runs in a transaction so either all locations save or none.
    """
    async with get_connection() as conn:
        async with conn.transaction():
            # Get market_id from condition_id
            market_id = await conn.fetchval(
                "SELECT id FROM markets WHERE condition_id = $1",
                result.condition_id,
            )
            if market_id is None:
                logger.warning("Market %s not found in DB, skipping location save",
                               result.condition_id)
                return

            # Delete old locations for this version (clean re-inference)
            await conn.execute(
                "DELETE FROM market_locations WHERE market_id = $1 AND geo_version = $2",
                market_id, geo_version,
            )

            # Insert new locations
            for loc in result.locations:
                await upsert_location(conn, market_id, loc, geo_version)

            # Mark market as processed
            await conn.execute(
                """
                UPDATE markets
                SET geo_processed = TRUE,
                    geo_processed_at = NOW(),
                    geo_version = $2
                WHERE id = $1
                """,
                market_id, geo_version,
            )


async def update_location_geocode(
    location_id: int,
    latitude: float,
    longitude: float,
    source: str,
    raw_response: Optional[dict] = None,
) -> None:
    """Update a location row with geocoded coordinates."""
    async with get_connection() as conn:
        await conn.execute(
            """
            UPDATE market_locations
            SET latitude = $2,
                longitude = $3,
                geog = ST_SetSRID(ST_MakePoint($3, $2), 4326)::geography,
                geocoded = TRUE,
                geocode_source = $4,
                geocode_raw = $5,
                updated_at = NOW()
            WHERE id = $1
            """,
            location_id, latitude, longitude, source,
            json.dumps(raw_response) if raw_response else None,
        )


# ── Fetch Unprocessed Markets ────────────────────────────────────────

async def get_unprocessed_markets(
    geo_version: int,
    limit: int = 500,
) -> list[dict]:
    """
    Fetch markets that haven't been processed at the current geo_version.
    This enables re-processing when inference logic changes.
    """
    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT id, condition_id, question, description, category, tags
            FROM markets
            WHERE geo_processed = FALSE OR geo_version < $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            geo_version, limit,
        )
        return [dict(r) for r in rows]


async def get_ungeooded_locations(limit: int = 500) -> list[dict]:
    """Fetch location candidates that need geocoding."""
    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT id, location_name, location_type
            FROM market_locations
            WHERE geocoded = FALSE AND location_type != 'global'
            ORDER BY confidence DESC
            LIMIT $1
            """,
            limit,
        )
        return [dict(r) for r in rows]


# ── Spatial Queries (used by API) ─────────────────────────────────────

async def query_nearby_markets(
    lat: float,
    lon: float,
    radius_km: float = 50.0,
    min_confidence: float = 0.0,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """
    Find markets within radius_km of a lat/lon point.
    Uses ST_DWithin for indexed spatial filtering.
    Returns (rows, total_count).
    """
    radius_meters = radius_km * 1000

    async with get_connection() as conn:
        # Count total matches
        total = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT m.id)
            FROM markets m
            JOIN market_locations ml ON ml.market_id = m.id
            WHERE ml.geocoded = TRUE
              AND ml.confidence >= $3
              AND ST_DWithin(
                  ml.geog,
                  ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography,
                  $4
              )
            """,
            lat, lon, min_confidence, radius_meters,
        )

        # Fetch paginated results, ordered by distance then confidence
        rows = await conn.fetch(
            """
            SELECT
                m.id,
                m.condition_id,
                m.question,
                m.description,
                m.category,
                m.active,
                m.volume,
                ml.id AS location_id,
                ml.location_name,
                ml.location_type,
                ml.confidence,
                ml.reason,
                ml.latitude,
                ml.longitude,
                ml.inference_method,
                ST_Distance(
                    ml.geog,
                    ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography
                ) AS distance_meters
            FROM markets m
            JOIN market_locations ml ON ml.market_id = m.id
            WHERE ml.geocoded = TRUE
              AND ml.confidence >= $3
              AND ST_DWithin(
                  ml.geog,
                  ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography,
                  $4
              )
            ORDER BY distance_meters ASC, ml.confidence DESC
            LIMIT $5 OFFSET $6
            """,
            lat, lon, min_confidence, radius_meters, limit, offset,
        )

        return [dict(r) for r in rows], total


async def query_markets_by_text(
    query: str,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """
    Search markets by location name or question text using trigram similarity.
    """
    pattern = f"%{query}%"

    async with get_connection() as conn:
        total = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT m.id)
            FROM markets m
            LEFT JOIN market_locations ml ON ml.market_id = m.id
            WHERE m.question ILIKE $1
               OR ml.location_name ILIKE $1
            """,
            pattern,
        )

        rows = await conn.fetch(
            """
            SELECT
                m.id,
                m.condition_id,
                m.question,
                m.description,
                m.category,
                m.active,
                m.volume,
                ml.id AS location_id,
                ml.location_name,
                ml.location_type,
                ml.confidence,
                ml.reason,
                ml.latitude,
                ml.longitude,
                ml.inference_method
            FROM markets m
            LEFT JOIN market_locations ml ON ml.market_id = m.id
            WHERE m.question ILIKE $1
               OR ml.location_name ILIKE $1
            ORDER BY ml.confidence DESC NULLS LAST
            LIMIT $2 OFFSET $3
            """,
            pattern, limit, offset,
        )

        return [dict(r) for r in rows], total


async def get_market_by_id(market_id: int) -> Optional[dict]:
    """Fetch a single market with all its location candidates."""
    async with get_connection() as conn:
        market = await conn.fetchrow(
            """
            SELECT id, condition_id, question, description, category,
                   active, closed, volume, liquidity, outcomes, outcome_prices
            FROM markets WHERE id = $1
            """,
            market_id,
        )
        if market is None:
            return None

        locations = await conn.fetch(
            """
            SELECT location_name, location_type, confidence, reason,
                   latitude, longitude, inference_method
            FROM market_locations
            WHERE market_id = $1
            ORDER BY confidence DESC
            """,
            market_id,
        )

        result = dict(market)
        result["locations"] = [dict(r) for r in locations]
        return result


async def get_market_by_condition_id(condition_id: str) -> Optional[dict]:
    """Fetch a single market by its Polymarket condition_id."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM markets WHERE condition_id = $1", condition_id
        )
        if row is None:
            return None
        return await get_market_by_id(row["id"])


# ── Pipeline Metrics ──────────────────────────────────────────────────

async def get_pipeline_metrics() -> dict:
    """Fetch aggregate metrics for the pipeline health dashboard."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                (SELECT COUNT(*) FROM markets) AS total_markets,
                (SELECT COUNT(*) FROM markets WHERE geo_processed) AS processed_markets,
                (SELECT COUNT(DISTINCT market_id) FROM market_locations) AS markets_with_locations,
                (SELECT ROUND(AVG(confidence)::numeric, 4) FROM market_locations) AS avg_confidence,
                (SELECT COUNT(*) FROM market_locations WHERE confidence < 0.5) AS low_confidence,
                (SELECT COUNT(*) FROM geocode_cache) AS cache_entries,
                (SELECT SUM(hit_count) FROM geocode_cache) AS total_cache_hits
            """
        )
        return dict(row) if row else {}


# ── Geocode Cache (DB-backed) ────────────────────────────────────────

async def get_cached_geocode(query_normalized: str) -> Optional[dict]:
    """Look up a geocode result from the persistent cache."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT latitude, longitude, display_name, source, raw_response
            FROM geocode_cache
            WHERE query_normalized = $1 AND expires_at > NOW()
            """,
            query_normalized,
        )
        if row is not None:
            # Increment hit counter
            await conn.execute(
                "UPDATE geocode_cache SET hit_count = hit_count + 1 WHERE query_normalized = $1",
                query_normalized,
            )
            return dict(row)
        return None


async def set_cached_geocode(
    query_normalized: str,
    latitude: Optional[float],
    longitude: Optional[float],
    display_name: Optional[str],
    source: str,
    raw_response: Optional[dict],
    ttl_days: int = 30,
) -> None:
    """Store a geocode result in the persistent cache."""
    async with get_connection() as conn:
        await conn.execute(
            """
            INSERT INTO geocode_cache (
                query_normalized, latitude, longitude, display_name,
                source, raw_response, expires_at
            ) VALUES ($1, $2, $3, $4, $5, $6, NOW() + make_interval(days => $7))
            ON CONFLICT (query_normalized) DO UPDATE SET
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                display_name = EXCLUDED.display_name,
                raw_response = EXCLUDED.raw_response,
                fetched_at = NOW(),
                expires_at = NOW() + make_interval(days => $7)
            """,
            query_normalized, latitude, longitude, display_name,
            source, json.dumps(raw_response) if raw_response else None,
            ttl_days,
        )


# ── Event Venue Cache (DB-backed) ─────────────────────────────────────

async def get_cached_event_venue(event_key: str, event_year: Optional[int]) -> Optional[dict]:
    """Look up cached event venue resolution from Postgres cache."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT status, venue_name, city, country, latitude, longitude,
                   source_url, source_type, confidence, reason, raw_payload
            FROM event_venue_cache
            WHERE event_key = $1
              AND event_year IS NOT DISTINCT FROM $2
              AND expires_at > NOW()
            """,
            event_key,
            event_year,
        )
        return dict(row) if row else None


async def set_cached_event_venue(
    event_key: str,
    event_year: Optional[int],
    status: str,
    venue_name: Optional[str],
    city: Optional[str],
    country: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    source_url: Optional[str],
    source_type: str,
    confidence: float,
    reason: str,
    raw_payload: Optional[dict],
    ttl_days: int = 7,
) -> None:
    """Upsert event venue cache row in Postgres."""
    async with get_connection() as conn:
        await conn.execute(
            """
            INSERT INTO event_venue_cache (
                event_key, event_year, status, venue_name, city, country,
                latitude, longitude, geog, source_url, source_type,
                confidence, reason, raw_payload, fetched_at, expires_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8,
                CASE WHEN $7 IS NOT NULL AND $8 IS NOT NULL
                     THEN ST_SetSRID(ST_MakePoint($8, $7), 4326)::geography
                     ELSE NULL END,
                $9, $10,
                $11, $12, $13,
                NOW(), NOW() + make_interval(days => $14)
            )
            ON CONFLICT (event_key, event_year) DO UPDATE SET
                status = EXCLUDED.status,
                venue_name = EXCLUDED.venue_name,
                city = EXCLUDED.city,
                country = EXCLUDED.country,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                geog = EXCLUDED.geog,
                source_url = EXCLUDED.source_url,
                source_type = EXCLUDED.source_type,
                confidence = EXCLUDED.confidence,
                reason = EXCLUDED.reason,
                raw_payload = EXCLUDED.raw_payload,
                fetched_at = NOW(),
                expires_at = NOW() + make_interval(days => $14),
                updated_at = NOW()
            """,
            event_key,
            event_year,
            status,
            venue_name,
            city,
            country,
            latitude,
            longitude,
            source_url,
            source_type,
            confidence,
            reason,
            json.dumps(raw_payload) if raw_payload else None,
            ttl_days,
        )


# ── Pipeline Run Tracking ─────────────────────────────────────────────

async def start_pipeline_run() -> int:
    """Record the start of a pipeline run. Returns run_id."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "INSERT INTO pipeline_runs DEFAULT VALUES RETURNING id"
        )
        return row["id"]


async def finish_pipeline_run(run_id: int, stats: dict, error: Optional[str] = None) -> None:
    """Record the completion of a pipeline run."""
    async with get_connection() as conn:
        await conn.execute(
            """
            UPDATE pipeline_runs SET
                finished_at = NOW(),
                status = $2,
                markets_fetched = $3,
                markets_new = $4,
                markets_updated = $5,
                locations_inferred = $6,
                locations_geocoded = $7,
                geocode_cache_hits = $8,
                geocode_cache_misses = $9,
                avg_confidence = $10,
                error_message = $11,
                metadata = $12
            WHERE id = $1
            """,
            run_id,
            "failed" if error else "completed",
            stats.get("markets_fetched", 0),
            stats.get("markets_new", 0),
            stats.get("markets_updated", 0),
            stats.get("locations_inferred", 0),
            stats.get("locations_geocoded", 0),
            stats.get("geocode_cache_hits", 0),
            stats.get("geocode_cache_misses", 0),
            stats.get("avg_confidence"),
            error,
            json.dumps(stats),
        )
