"""
Pydantic models used across the pipeline for validation and serialization.
These are pure data objects — no database coupling.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────

class LocationType(str, Enum):
    CITY = "city"
    STATE = "state"
    COUNTRY = "country"
    BUILDING = "building"
    ARENA = "arena"
    GLOBAL = "global"


class InferenceMethod(str, Enum):
    NLP = "nlp"
    HEURISTIC = "heuristic"
    LLM = "llm"
    MANUAL = "manual"


# ── Polymarket raw models ─────────────────────────────────────────────

class RawMarket(BaseModel):
    """Represents a market as returned by the Polymarket API."""
    condition_id: str = Field(..., alias="conditionId")
    question: str
    description: Optional[str] = None
    market_slug: Optional[str] = Field(None, alias="slug")
    category: Optional[str] = None
    end_date_iso: Optional[str] = Field(None, alias="endDate")
    active: bool = True
    closed: bool = False
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    outcomes: Optional[list[str]] = None
    outcome_prices: Optional[list[str]] = Field(None, alias="outcomePrices")
    tags: Optional[list[dict]] = None

    model_config = {"populate_by_name": True, "extra": "allow"}

    @field_validator("outcome_prices", mode="before")
    @classmethod
    def parse_outcome_prices(cls, v):
        """outcome_prices can arrive as a JSON string or list."""
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return None
        return v


# ── Location inference models ─────────────────────────────────────────

class LocationCandidate(BaseModel):
    """A single inferred location for a market."""
    location_name: str = Field(..., description="Human-readable location, e.g. 'Atlanta, GA'")
    location_type: LocationType = LocationType.CITY
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., description="Why this location was inferred")
    inference_method: InferenceMethod = InferenceMethod.NLP
    # Filled after geocoding
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class MarketInferenceResult(BaseModel):
    """Full inference result for a single market."""
    condition_id: str
    locations: list[LocationCandidate] = Field(default_factory=list)
    has_location: bool = False
    is_global: bool = False  # True if market has no specific geography


# ── Geocoding models ─────────────────────────────────────────────────

class GeocodeResult(BaseModel):
    query: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    display_name: Optional[str] = None
    source: str = "nominatim"
    from_cache: bool = False
    raw: Optional[dict] = None


# ── API response models ───────────────────────────────────────────────

class MarketLocationResponse(BaseModel):
    location_name: str
    location_type: str
    confidence: float
    reason: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    inference_method: str


class MarketResponse(BaseModel):
    id: int
    condition_id: str
    question: str
    description: Optional[str] = None
    category: Optional[str] = None
    active: bool = True
    volume: Optional[float] = None
    locations: list[MarketLocationResponse] = Field(default_factory=list)


class NearbyResponse(BaseModel):
    markets: list[MarketResponse]
    total: int
    center_lat: float
    center_lon: float
    radius_km: float


class SearchResponse(BaseModel):
    markets: list[MarketResponse]
    total: int
    query: str
    resolved_location: Optional[str] = None
    resolved_lat: Optional[float] = None
    resolved_lon: Optional[float] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    total_markets: int = 0
    processed_markets: int = 0
    pct_with_location: float = 0.0
    avg_confidence: Optional[float] = None
    last_pipeline_run: Optional[datetime] = None
