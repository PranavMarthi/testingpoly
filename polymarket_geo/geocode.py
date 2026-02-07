"""
Geocoding module with aggressive caching, rate limiting, and normalization.

Strategy:
  1. Normalize the location string (lowercase, trim, expand abbreviations)
  2. Check Postgres cache (geocode_cache table)
  3. If miss, call geocoder API with exponential backoff
  4. Store result in cache with TTL
  5. Update the market_locations row with lat/lon

Supports Nominatim (free, rate-limited) and Google Geocoding API.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import httpx

from polymarket_geo.config import get_settings
from polymarket_geo.db import (
    get_cached_geocode,
    get_ungeooded_locations,
    set_cached_geocode,
    update_location_geocode,
)
from polymarket_geo.models import GeocodeResult

logger = logging.getLogger(__name__)

# ── Location Name Normalization ────────────────────────────────────────

# US state abbreviations -> full names (for geocoder disambiguation)
US_STATE_ABBREVS = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

# City -> preferred full form for geocoding
CITY_NORMALIZATIONS: dict[str, str] = {
    "atlanta": "Atlanta, GA, USA",
    "atlanta, ga": "Atlanta, GA, USA",
    "new york": "New York, NY, USA",
    "new york, ny": "New York, NY, USA",
    "nyc": "New York, NY, USA",
    "los angeles": "Los Angeles, CA, USA",
    "los angeles, ca": "Los Angeles, CA, USA",
    "la": "Los Angeles, CA, USA",
    "chicago": "Chicago, IL, USA",
    "chicago, il": "Chicago, IL, USA",
    "houston": "Houston, TX, USA",
    "houston, tx": "Houston, TX, USA",
    "phoenix": "Phoenix, AZ, USA",
    "philadelphia": "Philadelphia, PA, USA",
    "san antonio": "San Antonio, TX, USA",
    "san diego": "San Diego, CA, USA",
    "dallas": "Dallas, TX, USA",
    "san francisco": "San Francisco, CA, USA",
    "sf": "San Francisco, CA, USA",
    "seattle": "Seattle, WA, USA",
    "denver": "Denver, CO, USA",
    "boston": "Boston, MA, USA",
    "miami": "Miami, FL, USA",
    "washington": "Washington, DC, USA",
    "washington, dc": "Washington, DC, USA",
    "dc": "Washington, DC, USA",
    "london": "London, United Kingdom",
    "london, uk": "London, United Kingdom",
    "paris": "Paris, France",
    "berlin": "Berlin, Germany",
    "tokyo": "Tokyo, Japan",
    "beijing": "Beijing, China",
    "moscow": "Moscow, Russia",
    "mumbai": "Mumbai, India",
    "new delhi": "New Delhi, India",
    "toronto": "Toronto, ON, Canada",
    "toronto, on, canada": "Toronto, ON, Canada",
    "sydney": "Sydney, NSW, Australia",
    "palm beach": "Palm Beach, FL, USA",
    "palm beach, fl": "Palm Beach, FL, USA",
}


def normalize_location_name(name: str) -> str:
    """
    Normalize a location string for consistent caching and geocoding.
    Rules:
      1. Lowercase and strip whitespace
      2. Apply known city normalizations
      3. Expand US state abbreviations (e.g., "Atlanta, GA" -> "Atlanta, Georgia, USA")
      4. Remove extra punctuation
    """
    normalized = name.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)  # collapse whitespace

    # Check known normalizations first
    if normalized in CITY_NORMALIZATIONS:
        return CITY_NORMALIZATIONS[normalized]

    # Try expanding state abbreviations for "City, ST" patterns
    match = re.match(r"^(.+?),\s*([A-Z]{2})$", name.strip())
    if match:
        city = match.group(1).strip()
        state_abbr = match.group(2).upper()
        if state_abbr in US_STATE_ABBREVS:
            return f"{city}, {US_STATE_ABBREVS[state_abbr]}, USA"

    # Return cleaned version
    return normalized


# ── Rate Limiter ───────────────────────────────────────────────────────

class RateLimiter:
    """Token bucket rate limiter for geocoding API calls."""

    def __init__(self, rate_per_second: float = 1.0):
        self._rate = rate_per_second
        self._interval = 1.0 / rate_per_second
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_call
            if elapsed < self._interval:
                await asyncio.sleep(self._interval - elapsed)
            self._last_call = asyncio.get_event_loop().time()


# ── Geocoder Implementations ──────────────────────────────────────────

class NominatimGeocoder:
    """Geocode using OpenStreetMap Nominatim (free, 1 req/sec limit)."""

    def __init__(self):
        self.settings = get_settings().geocoding
        self.rate_limiter = RateLimiter(self.settings.rate_limit_rps)

    async def geocode(self, query: str) -> Optional[GeocodeResult]:
        """
        Geocode a location string. Returns None on failure.
        Implements exponential backoff on errors.
        """
        await self.rate_limiter.acquire()

        for attempt in range(self.settings.max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{self.settings.nominatim_url}/search",
                        params={
                            "q": query,
                            "format": "json",
                            "limit": 1,
                            "addressdetails": 1,
                        },
                        headers={
                            "User-Agent": self.settings.nominatim_user_agent,
                        },
                        timeout=10,
                    )
                    resp.raise_for_status()
                    results = resp.json()

                    if not results:
                        logger.debug("Nominatim: no results for '%s'", query)
                        return GeocodeResult(
                            query=query,
                            source="nominatim",
                            from_cache=False,
                        )

                    top = results[0]
                    return GeocodeResult(
                        query=query,
                        latitude=float(top["lat"]),
                        longitude=float(top["lon"]),
                        display_name=top.get("display_name"),
                        source="nominatim",
                        from_cache=False,
                        raw=top,
                    )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = self.settings.backoff_base ** (attempt + 1)
                    logger.warning("Nominatim rate limited, backing off %.1fs", wait)
                    await asyncio.sleep(wait)
                    continue
                logger.error("Nominatim HTTP error: %s", e)
                return None

            except httpx.RequestError as e:
                wait = self.settings.backoff_base ** (attempt + 1)
                logger.warning("Nominatim request error (attempt %d/%d): %s, backing off %.1fs",
                               attempt + 1, self.settings.max_retries, e, wait)
                await asyncio.sleep(wait)
                continue

        logger.error("Nominatim: all %d retries exhausted for '%s'",
                     self.settings.max_retries, query)
        return None


class GoogleGeocoder:
    """Geocode using Google Maps Geocoding API (paid, high rate limits)."""

    def __init__(self):
        self.settings = get_settings().geocoding
        self.rate_limiter = RateLimiter(min(self.settings.rate_limit_rps, 50.0))

    async def geocode(self, query: str) -> Optional[GeocodeResult]:
        if not self.settings.google_api_key:
            logger.error("Google Geocoding API key not configured")
            return None

        await self.rate_limiter.acquire()

        for attempt in range(self.settings.max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://maps.googleapis.com/maps/api/geocode/json",
                        params={
                            "address": query,
                            "key": self.settings.google_api_key,
                        },
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    if data.get("status") != "OK" or not data.get("results"):
                        logger.debug("Google Geocoding: no results for '%s' (status=%s)",
                                     query, data.get("status"))
                        return GeocodeResult(query=query, source="google", from_cache=False)

                    top = data["results"][0]
                    loc = top["geometry"]["location"]
                    return GeocodeResult(
                        query=query,
                        latitude=float(loc["lat"]),
                        longitude=float(loc["lng"]),
                        display_name=top.get("formatted_address"),
                        source="google",
                        from_cache=False,
                        raw=top,
                    )

            except Exception as e:
                wait = self.settings.backoff_base ** (attempt + 1)
                logger.warning("Google Geocoding error (attempt %d): %s", attempt + 1, e)
                await asyncio.sleep(wait)
                continue

        return None


# ── Geocoding Orchestrator ─────────────────────────────────────────────

def get_geocoder():
    """Factory: return the configured geocoder instance."""
    provider = get_settings().geocoding.provider
    if provider == "google":
        return GoogleGeocoder()
    return NominatimGeocoder()


async def geocode_location(location_name: str) -> GeocodeResult:
    """
    Geocode a location name with cache-first strategy.
    1. Normalize the name
    2. Check DB cache
    3. If miss, call geocoder API
    4. Store in cache
    Returns GeocodeResult (may have None lat/lon if not found).
    """
    settings = get_settings().geocoding
    normalized = normalize_location_name(location_name)

    # Check cache
    cached = await get_cached_geocode(normalized)
    if cached is not None:
        logger.debug("Geocode cache HIT: '%s'", normalized)
        return GeocodeResult(
            query=normalized,
            latitude=cached["latitude"],
            longitude=cached["longitude"],
            display_name=cached.get("display_name"),
            source=cached.get("source", "cache"),
            from_cache=True,
        )

    logger.debug("Geocode cache MISS: '%s'", normalized)

    # Call geocoder
    geocoder = get_geocoder()
    result = await geocoder.geocode(normalized)

    if result is None:
        result = GeocodeResult(query=normalized, source=settings.provider)

    # Store in cache (even negative results to avoid re-querying)
    await set_cached_geocode(
        query_normalized=normalized,
        latitude=result.latitude,
        longitude=result.longitude,
        display_name=result.display_name,
        source=result.source,
        raw_response=result.raw,
        ttl_days=settings.cache_ttl_days,
    )

    return result


async def geocode_pending_locations(limit: int = 500) -> dict:
    """
    Fetch all ungeooded location rows and geocode them.
    Returns stats dict with cache hits/misses.
    """
    stats = {"total": 0, "geocoded": 0, "failed": 0, "cache_hits": 0, "cache_misses": 0}

    locations = await get_ungeooded_locations(limit)
    stats["total"] = len(locations)

    for loc in locations:
        location_id = loc["id"]
        location_name = loc["location_name"]

        # Skip global locations
        if loc["location_type"] == "global":
            continue

        result = await geocode_location(location_name)

        if result.from_cache:
            stats["cache_hits"] += 1
        else:
            stats["cache_misses"] += 1

        if result.latitude is not None and result.longitude is not None:
            await update_location_geocode(
                location_id=location_id,
                latitude=result.latitude,
                longitude=result.longitude,
                source=result.source,
                raw_response=result.raw,
            )
            stats["geocoded"] += 1
        else:
            stats["failed"] += 1
            logger.warning("Failed to geocode location '%s' (id=%d)",
                           location_name, location_id)

    logger.info("Geocoding complete: %s", stats)
    return stats
