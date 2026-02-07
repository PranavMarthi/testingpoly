"""
Pipeline orchestrator.
Ties together ingest -> infer -> geocode in a single idempotent run.
Called by the scheduler or invoked manually via CLI.
"""

from __future__ import annotations

import logging
import time

from polymarket_geo.config import get_settings
from polymarket_geo.db import (
    finish_pipeline_run,
    get_unprocessed_markets,
    save_inference_result,
    start_pipeline_run,
)
from polymarket_geo.geocode import geocode_pending_locations
from polymarket_geo.infer import infer_locations_batch
from polymarket_geo.ingest import ingest_markets

logger = logging.getLogger(__name__)


async def run_pipeline() -> dict:
    """
    Execute the full pipeline:
      1. Ingest: Fetch markets from Polymarket API and upsert into DB
      2. Infer: Run NLP + heuristics on unprocessed markets
      3. Geocode: Geocode any new location candidates

    Idempotent: safe to run repeatedly.
    - Markets are upserted (ON CONFLICT UPDATE), so re-fetching is harmless.
    - Inference checks geo_version, so already-processed markets are skipped.
    - Geocoding checks the geocoded flag, so already-geocoded locations are skipped.

    Returns a stats dict summarizing the run.
    """
    settings = get_settings()
    geo_version = settings.inference.inference_version
    batch_size = settings.scheduler.batch_size

    stats = {
        "markets_fetched": 0,
        "markets_new": 0,
        "markets_updated": 0,
        "locations_inferred": 0,
        "locations_geocoded": 0,
        "geocode_cache_hits": 0,
        "geocode_cache_misses": 0,
        "avg_confidence": None,
        "duration_seconds": 0,
    }

    run_id = await start_pipeline_run()
    start_time = time.monotonic()

    try:
        # ── Stage 1: Ingest ────────────────────────────────────────────
        logger.info("=== Pipeline Stage 1: Ingestion ===")
        ingest_stats = await ingest_markets()
        stats["markets_fetched"] = ingest_stats.get("fetched", 0)
        stats["markets_new"] = ingest_stats.get("new", 0)
        stats["markets_updated"] = ingest_stats.get("updated", 0)
        logger.info("Ingestion done: %d fetched, %d new, %d updated",
                     stats["markets_fetched"], stats["markets_new"], stats["markets_updated"])

        # ── Stage 2: Inference ─────────────────────────────────────────
        logger.info("=== Pipeline Stage 2: Inference (version=%d) ===", geo_version)
        unprocessed = await get_unprocessed_markets(geo_version, limit=batch_size)
        logger.info("Found %d markets to process", len(unprocessed))

        if unprocessed:
            results = await infer_locations_batch(unprocessed)

            total_locs = 0
            total_conf = 0.0
            conf_count = 0

            for result in results:
                await save_inference_result(result, geo_version)
                n_locs = len(result.locations)
                total_locs += n_locs
                for loc in result.locations:
                    total_conf += loc.confidence
                    conf_count += 1

            stats["locations_inferred"] = total_locs
            stats["avg_confidence"] = round(total_conf / conf_count, 4) if conf_count > 0 else None
            logger.info("Inference done: %d locations from %d markets (avg conf: %s)",
                         total_locs, len(unprocessed), stats["avg_confidence"])

        # ── Stage 3: Geocoding ─────────────────────────────────────────
        logger.info("=== Pipeline Stage 3: Geocoding ===")
        geo_stats = await geocode_pending_locations(limit=batch_size)
        stats["locations_geocoded"] = geo_stats.get("geocoded", 0)
        stats["geocode_cache_hits"] = geo_stats.get("cache_hits", 0)
        stats["geocode_cache_misses"] = geo_stats.get("cache_misses", 0)
        logger.info("Geocoding done: %d geocoded (%d cache hits, %d misses)",
                     stats["locations_geocoded"],
                     stats["geocode_cache_hits"],
                     stats["geocode_cache_misses"])

        # ── Complete ───────────────────────────────────────────────────
        elapsed = time.monotonic() - start_time
        stats["duration_seconds"] = round(elapsed, 2)
        await finish_pipeline_run(run_id, stats)
        logger.info("=== Pipeline complete in %.1fs ===", elapsed)
        return stats

    except Exception as e:
        elapsed = time.monotonic() - start_time
        stats["duration_seconds"] = round(elapsed, 2)
        await finish_pipeline_run(run_id, stats, error=str(e))
        logger.error("Pipeline failed after %.1fs: %s", elapsed, e, exc_info=True)
        raise
