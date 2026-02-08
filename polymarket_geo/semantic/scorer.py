from __future__ import annotations

from dataclasses import dataclass

from polymarket_geo.semantic.calibrator import Calibrator
from polymarket_geo.semantic.retriever import RetrievalHit


@dataclass(frozen=True)
class ScoredCandidate:
    place_id: str
    name: str
    granularity: str
    lat: float
    lon: float
    confidence: float
    evidence: list[RetrievalHit]


class Scorer:
    def __init__(self, calibrator: Calibrator | None = None):
        self.calibrator = calibrator or Calibrator()

    def score(self, hits_by_field: dict[str, list[RetrievalHit]], top_k: int = 5) -> list[ScoredCandidate]:
        bucket: dict[str, list[RetrievalHit]] = {}
        for hits in hits_by_field.values():
            for h in hits:
                bucket.setdefault(h.record.place_id, []).append(h)

        out: list[ScoredCandidate] = []
        for place_id, hits in bucket.items():
            by_field: dict[str, float] = {
                "title": 0.0,
                "description": 0.0,
                "choices": 0.0,
                "combined": 0.0,
            }
            for h in hits:
                by_field[h.field] = max(by_field.get(h.field, 0.0), h.score)

            rec = hits[0].record
            agreement = sum(1 for k in ("title", "description", "choices") if by_field[k] > 0.4) / 3.0
            confidence = self.calibrator.confidence(
                s_combined=by_field["combined"],
                s_title=by_field["title"],
                s_desc=by_field["description"],
                s_choices=by_field["choices"],
                agreement=agreement,
                importance=max(0.0, min(1.0, rec.importance)),
            )

            out.append(
                ScoredCandidate(
                    place_id=place_id,
                    name=rec.place_name,
                    granularity=rec.granularity,
                    lat=rec.lat,
                    lon=rec.lon,
                    confidence=confidence,
                    evidence=sorted(hits, key=lambda x: x.score, reverse=True)[:6],
                )
            )

        out.sort(key=lambda c: c.confidence, reverse=True)
        return out[:top_k]
