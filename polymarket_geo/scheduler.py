"""
Scheduler module using APScheduler.
Runs the pipeline at configurable intervals.
Can run standalone or be embedded in the FastAPI app.
"""

from __future__ import annotations

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from polymarket_geo.config import get_settings
from polymarket_geo.pipeline import run_pipeline

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _pipeline_job():
    """Wrapper that catches exceptions so the scheduler doesn't die on failure."""
    try:
        logger.info("Scheduled pipeline run starting...")
        stats = await run_pipeline()
        logger.info("Scheduled pipeline run completed: %s", stats)
    except Exception as e:
        logger.error("Scheduled pipeline run failed: %s", e, exc_info=True)


def create_scheduler() -> AsyncIOScheduler:
    """Create and configure the APScheduler instance."""
    global _scheduler
    settings = get_settings().scheduler

    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        _pipeline_job,
        trigger=IntervalTrigger(minutes=settings.interval_minutes),
        id="polymarket_geo_pipeline",
        name="Polymarket Geo Pipeline",
        replace_existing=True,
        max_instances=1,  # prevent overlapping runs
    )

    logger.info("Scheduler configured: pipeline runs every %d minutes",
                settings.interval_minutes)
    return _scheduler


def start_scheduler() -> None:
    """Start the scheduler (non-blocking)."""
    settings = get_settings().scheduler
    if not settings.enabled:
        logger.info("Scheduler disabled via config")
        return

    scheduler = create_scheduler()
    scheduler.start()
    logger.info("Scheduler started")


def stop_scheduler() -> None:
    """Stop the scheduler gracefully."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped")
        _scheduler = None


async def run_once():
    """Run the pipeline once (for CLI / testing)."""
    from polymarket_geo.db import close_pool, get_pool

    await get_pool()
    try:
        stats = await run_pipeline()
        return stats
    finally:
        await close_pool()
