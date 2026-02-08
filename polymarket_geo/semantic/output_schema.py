from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


GeoType = Literal["explicit", "inferred", "multi", "global", "ambiguous", "none"]
EventType = Literal[
    "weather",
    "sports",
    "election",
    "geopolitics",
    "entertainment",
    "finance",
    "global",
    "unknown",
]
Granularity = Literal["city", "state", "country", "region", "global"]
EvidenceField = Literal["title", "description", "choices"]


class EvidenceItem(BaseModel):
    field: EvidenceField
    snippet: str
    retrieval_hit: str
    score: float = Field(ge=0.0, le=1.0)


class LocationHypothesis(BaseModel):
    place_id: str
    name: str
    lat: float
    lon: float
    granularity: Granularity
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class GeoInferenceOutput(BaseModel):
    geo_type: GeoType
    event_type: EventType
    locations: list[LocationHypothesis] = Field(default_factory=list)
