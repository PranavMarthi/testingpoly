"""
Event venue resolver for prediction-market prompts.

Design goals:
- Resolve near-term event venues from public web sources.
- Cache results aggressively to avoid repeated network calls.
- Return strict `not_available` when venue is unconfirmed or too far out.

This module is synchronous so it can be used directly inside the inference engine.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx


@dataclass(frozen=True)
class EventIntent:
    event_key: str
    event_year: Optional[int]
    query: str


@dataclass(frozen=True)
class EventVenueResult:
    status: str  # confirmed | uncertain | not_available
    event_key: str
    event_year: Optional[int]
    venue_name: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    source_url: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""


EVENT_PATTERNS: dict[str, list[str]] = {
    "oscars": ["oscars", "academy awards"],
    "grammys": ["grammys", "grammy awards", "recording academy"],
    "golden_globes": ["golden globes", "golden globe awards"],
    "emmys": ["emmys", "emmy awards", "primetime emmys"],
    "tonys": ["tonys", "tony awards"],
    "cannes": ["cannes", "palme d'or", "cannes film festival"],
    "met_gala": ["met gala", "metropolitan museum costume institute"],
    "super_bowl": ["super bowl"],
    "world_cup": ["world cup", "fifa world cup"],
    "olympics": ["olympics", "olympic games"],
}


WIKI_QUERY_TEMPLATES: dict[str, list[str]] = {
    "oscars": ["{year} Academy Awards", "{ord} Academy Awards"],
    "grammys": ["{year} Grammy Awards", "{ord} Grammy Awards"],
    "golden_globes": ["{year} Golden Globe Awards", "{ord} Golden Globe Awards"],
    "emmys": ["{year} Primetime Emmy Awards", "{ord} Primetime Emmy Awards"],
    "tonys": ["{year} Tony Awards", "{ord} Tony Awards"],
    "cannes": ["{year} Cannes Film Festival"],
    "met_gala": ["{year} Met Gala"],
    "super_bowl": ["Super Bowl {roman}", "{year} Super Bowl"],
    "world_cup": ["{year} FIFA World Cup"],
    "olympics": ["{year} Summer Olympics", "{year} Winter Olympics"],
}


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _to_roman(num: int) -> str:
    vals = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    result = []
    n = max(1, num)
    for v, s in vals:
        while n >= v:
            result.append(s)
            n -= v
    return "".join(result)


class EventVenueCache:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent / "event_venue_cache.sqlite"
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_venue_cache (
                    event_key TEXT NOT NULL,
                    event_year INTEGER,
                    status TEXT NOT NULL,
                    venue_name TEXT,
                    city TEXT,
                    country TEXT,
                    latitude REAL,
                    longitude REAL,
                    source_url TEXT,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    reason TEXT,
                    raw_payload TEXT,
                    fetched_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    PRIMARY KEY (event_key, event_year)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, event_key: str, event_year: Optional[int]) -> Optional[EventVenueResult]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT * FROM event_venue_cache
                WHERE event_key = ? AND event_year IS ?
                """,
                (event_key, event_year),
            ).fetchone()
            if row is None:
                return None
            now = datetime.now(timezone.utc)
            expires_at = datetime.fromisoformat(row["expires_at"])
            if expires_at < now:
                return None
            return EventVenueResult(
                status=row["status"],
                event_key=row["event_key"],
                event_year=row["event_year"],
                venue_name=row["venue_name"],
                city=row["city"],
                country=row["country"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                source_url=row["source_url"],
                confidence=float(row["confidence"] or 0.0),
                reason=row["reason"] or "",
            )
        finally:
            conn.close()

    def set(self, result: EventVenueResult, raw_payload: Optional[dict] = None) -> None:
        now = datetime.now(timezone.utc)
        ttl_days = 14 if result.status == "confirmed" else 3
        expires = now + timedelta(days=ttl_days)
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """
                INSERT INTO event_venue_cache (
                    event_key, event_year, status, venue_name, city, country,
                    latitude, longitude, source_url, confidence, reason,
                    raw_payload, fetched_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_key, event_year) DO UPDATE SET
                    status = excluded.status,
                    venue_name = excluded.venue_name,
                    city = excluded.city,
                    country = excluded.country,
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    source_url = excluded.source_url,
                    confidence = excluded.confidence,
                    reason = excluded.reason,
                    raw_payload = excluded.raw_payload,
                    fetched_at = excluded.fetched_at,
                    expires_at = excluded.expires_at
                """,
                (
                    result.event_key,
                    result.event_year,
                    result.status,
                    result.venue_name,
                    result.city,
                    result.country,
                    result.latitude,
                    result.longitude,
                    result.source_url,
                    result.confidence,
                    result.reason,
                    json.dumps(raw_payload) if raw_payload else None,
                    now.isoformat(),
                    expires.isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()


class EventVenueResolver:
    def __init__(self, horizon_months: int = 18):
        self.horizon_months = horizon_months
        self.cache = EventVenueCache()

    def detect_event(self, text: str) -> Optional[EventIntent]:
        text_lower = text.lower()
        event_key = None
        for key, aliases in EVENT_PATTERNS.items():
            if any(alias in text_lower for alias in aliases):
                event_key = key
                break
        if event_key is None:
            return None

        year = self._extract_year(text)
        return EventIntent(event_key=event_key, event_year=year, query=text)

    def resolve(self, text: str) -> Optional[EventVenueResult]:
        intent = self.detect_event(text)
        if intent is None:
            return None

        cached = self.cache.get(intent.event_key, intent.event_year)
        if cached is not None:
            return cached

        if not self._is_short_term(intent.event_year):
            result = EventVenueResult(
                status="not_available",
                event_key=intent.event_key,
                event_year=intent.event_year,
                confidence=0.35,
                reason="Event venue not publicly confirmed yet (outside short-term horizon)",
            )
            self.cache.set(result)
            return result

        resolved, raw = self._resolve_from_web(intent)
        self.cache.set(resolved, raw_payload=raw)
        return resolved

    def _extract_year(self, text: str) -> Optional[int]:
        years = re.findall(r"\b(20\d{2})\b", text)
        if not years:
            return None
        return int(years[0])

    def _is_short_term(self, event_year: Optional[int]) -> bool:
        if event_year is None:
            return True
        now = datetime.now(timezone.utc)
        try:
            event_date = datetime(event_year, 12, 31, tzinfo=timezone.utc)
        except ValueError:
            return False
        horizon = now + timedelta(days=30 * self.horizon_months)
        return event_date <= horizon

    def _resolve_from_web(self, intent: EventIntent) -> tuple[EventVenueResult, dict]:
        queries = self._candidate_queries(intent)
        raw = {"queries": [], "pages": []}

        for query in queries:
            page = self._search_wikipedia_page(query)
            raw["queries"].append({"query": query, "page": page})
            if page is None:
                continue
            extract = self._fetch_wikipedia_extract(page)
            if extract is None:
                continue
            raw["pages"].append({"title": page, "extract_head": extract[:500]})

            parsed = self._parse_location_from_text(extract)
            if parsed is not None:
                venue_name, city, country = parsed
                return (
                    EventVenueResult(
                        status="confirmed",
                        event_key=intent.event_key,
                        event_year=intent.event_year,
                        venue_name=venue_name,
                        city=city,
                        country=country,
                        source_url=f"https://en.wikipedia.org/wiki/{page.replace(' ', '_')}",
                        confidence=0.72,
                        reason="Event venue resolved from public event page",
                    ),
                    raw,
                )

        return (
            EventVenueResult(
                status="not_available",
                event_key=intent.event_key,
                event_year=intent.event_year,
                confidence=0.35,
                reason="Event venue not publicly confirmed yet",
            ),
            raw,
        )

    def _candidate_queries(self, intent: EventIntent) -> list[str]:
        year = intent.event_year or datetime.now(timezone.utc).year
        templates = WIKI_QUERY_TEMPLATES.get(intent.event_key, [f"{year} {intent.event_key}"])

        # Event-specific ordinal estimate for recurring annual events.
        # Academy Awards started 1929 ceremony cycle.
        academy_ord = max(1, year - 1928)
        grammy_ord = max(1, year - 1958)
        globe_ord = max(1, year - 1943)
        emmy_ord = max(1, year - 1948)
        tony_ord = max(1, year - 1946)

        ord_map = {
            "oscars": academy_ord,
            "grammys": grammy_ord,
            "golden_globes": globe_ord,
            "emmys": emmy_ord,
            "tonys": tony_ord,
        }
        n = ord_map.get(intent.event_key, max(1, year - 2000))
        roman = _to_roman(max(1, year - 1966))  # rough mapping for Super Bowl era

        queries: list[str] = []
        for t in templates:
            q = t.format(year=year, ord=_ordinal(n), roman=roman)
            queries.append(q)
        # Include raw title as fallback search query
        queries.append(intent.query)
        return list(dict.fromkeys(queries))

    def _search_wikipedia_page(self, query: str) -> Optional[str]:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 5,
        }
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return None

        results = data.get("query", {}).get("search", [])
        for item in results:
            title = item.get("title", "")
            if not title:
                continue
            return title
        return None

    def _fetch_wikipedia_extract(self, title: str) -> Optional[str]:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "redirects": 1,
        }
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return None

        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            extract = page.get("extract")
            if extract:
                return extract
        return None

    def _parse_location_from_text(self, text: str) -> Optional[tuple[str, str, str]]:
        # Common pattern: "... at the Dolby Theatre in Los Angeles, California ..."
        patterns = [
            r"at\s+the\s+([A-Z][A-Za-z0-9'&\-\.\s]+?)\s+in\s+([A-Z][A-Za-z\-\s]+),\s*([A-Z][A-Za-z\-\s]+)",
            r"held\s+at\s+([A-Z][A-Za-z0-9'&\-\.\s]+?)\s+in\s+([A-Z][A-Za-z\-\s]+),\s*([A-Z][A-Za-z\-\s]+)",
            r"in\s+([A-Z][A-Za-z\-\s]+),\s*([A-Z][A-Za-z\-\s]+)",
        ]

        snippet = text[:2000]
        for p in patterns:
            m = re.search(p, snippet)
            if not m:
                continue
            if len(m.groups()) == 3:
                venue = re.sub(r"\s+", " ", m.group(1)).strip(" .,")
                city = re.sub(r"\s+", " ", m.group(2)).strip(" .,")
                country = re.sub(r"\s+", " ", m.group(3)).strip(" .,")
                return venue, city, country
            if len(m.groups()) == 2:
                city = re.sub(r"\s+", " ", m.group(1)).strip(" .,")
                country = re.sub(r"\s+", " ", m.group(2)).strip(" .,")
                return city, city, country
        return None


_resolver: Optional[EventVenueResolver] = None


def get_event_resolver() -> EventVenueResolver:
    global _resolver
    if _resolver is None:
        _resolver = EventVenueResolver(horizon_months=18)
    return _resolver
