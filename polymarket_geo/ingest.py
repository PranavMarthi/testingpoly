"""
Polymarket API ingestion module.
Fetches markets from the Polymarket CLOB/Gamma API with pagination,
validates with Pydantic, and upserts into Postgres.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import httpx

from polymarket_geo.config import get_settings
from polymarket_geo.models import RawMarket

logger = logging.getLogger(__name__)


class PolymarketClient:
    """
    Async client for the Polymarket markets API.
    Handles pagination, rate limiting, and retries.
    """

    def __init__(self) -> None:
        self.settings = get_settings().polymarket
        self._semaphore = asyncio.Semaphore(int(self.settings.rate_limit_rps))

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        offset: int,
    ) -> list[dict]:
        """Fetch a single page of markets."""
        async with self._semaphore:
            params = {
                "limit": self.settings.page_size,
                "offset": offset,
                "order": "id",
                "ascending": "false",  # newest first
            }
            try:
                resp = await client.get(
                    f"{self.settings.base_url}{self.settings.markets_endpoint}",
                    params=params,
                    timeout=self.settings.request_timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                # The API returns a list directly
                if isinstance(data, list):
                    return data
                # Or it may return {"data": [...], "next_cursor": ...}
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                return data if isinstance(data, list) else []

            except httpx.HTTPStatusError as e:
                logger.error("HTTP %d fetching markets (offset=%d): %s",
                             e.response.status_code, offset, e)
                if e.response.status_code == 429:
                    # Rate limited — back off
                    retry_after = int(e.response.headers.get("Retry-After", "5"))
                    logger.warning("Rate limited, sleeping %ds", retry_after)
                    await asyncio.sleep(retry_after)
                    return await self._fetch_page(client, offset)
                raise
            except httpx.RequestError as e:
                logger.error("Request error fetching markets (offset=%d): %s", offset, e)
                raise

    async def fetch_all_markets(self) -> AsyncIterator[list[dict]]:
        """
        Paginate through all markets, yielding pages.
        Stops when a page returns fewer than page_size results or max_pages hit.
        """
        async with httpx.AsyncClient() as client:
            offset = 0
            page_num = 0

            while page_num < self.settings.max_pages:
                logger.info("Fetching markets page %d (offset=%d)", page_num, offset)
                page = await self._fetch_page(client, offset)

                if not page:
                    logger.info("Empty page at offset %d, stopping", offset)
                    break

                yield page
                page_num += 1

                if len(page) < self.settings.page_size:
                    logger.info("Partial page (%d/%d), reached end",
                                len(page), self.settings.page_size)
                    break

                offset += self.settings.page_size

                # Small delay between pages to be polite
                await asyncio.sleep(1.0 / self.settings.rate_limit_rps)

        logger.info("Finished fetching markets: %d pages", page_num)

    async def fetch_all_markets_flat(self) -> list[dict]:
        """Fetch all markets as a flat list."""
        all_markets: list[dict] = []
        async for page in self.fetch_all_markets():
            all_markets.extend(page)
        return all_markets


def parse_raw_markets(raw_list: list[dict]) -> list[tuple[RawMarket, dict]]:
    """
    Validate raw API dicts into RawMarket models.
    Returns list of (parsed_model, original_dict) tuples.
    Skips invalid entries with a warning.
    """
    results = []
    for raw in raw_list:
        try:
            market = RawMarket.model_validate(raw)
            results.append((market, raw))
        except Exception as e:
            cid = raw.get("conditionId", raw.get("condition_id", "unknown"))
            logger.warning("Failed to parse market %s: %s", cid, e)
    return results


async def ingest_markets() -> dict:
    """
    Full ingestion: fetch from API, parse, upsert to DB.
    Returns stats dict with counts.
    """
    from polymarket_geo.db import upsert_markets_batch

    client = PolymarketClient()
    all_raw = await client.fetch_all_markets_flat()
    logger.info("Fetched %d raw markets from API", len(all_raw))

    parsed = parse_raw_markets(all_raw)
    logger.info("Parsed %d valid markets", len(parsed))

    stats = await upsert_markets_batch(parsed)
    stats["fetched"] = len(all_raw)
    stats["parsed"] = len(parsed)

    logger.info("Ingestion complete: %s", stats)
    return stats
