"""Semantic-only geographic inference engine (offline)."""

from __future__ import annotations

from dataclasses import dataclass
import re

from polymarket_geo.models import InferenceMethod, LocationCandidate, LocationType, MarketInferenceResult
from polymarket_geo.semantic.composer import TextComposer
from polymarket_geo.semantic.decider import GeoTypeDecider
from polymarket_geo.semantic.event_type import EventTypeClassifier
from polymarket_geo.semantic.indexes import LocalIndexes
from polymarket_geo.semantic.output_schema import EvidenceItem, GeoInferenceOutput, LocationHypothesis
from polymarket_geo.semantic.retriever import Retriever
from polymarket_geo.semantic.scorer import Scorer


@dataclass
class SemanticPipeline:
    indexes: LocalIndexes
    retriever: Retriever
    scorer: Scorer
    event_classifier: EventTypeClassifier
    geo_decider: GeoTypeDecider

    @classmethod
    def build(cls) -> "SemanticPipeline":
        indexes = LocalIndexes()
        return cls(
            indexes=indexes,
            retriever=Retriever(indexes),
            scorer=Scorer(),
            event_classifier=EventTypeClassifier(indexes.embedder),
            geo_decider=GeoTypeDecider(),
        )


class LocationInferenceEngine:
    """Backwards-compatible engine wrapper around semantic retrieval pipeline."""

    def __init__(self):
        self.pipeline = SemanticPipeline.build()

    def infer_semantic(
        self,
        title: str,
        description: str | None = None,
        choices: list[str] | None = None,
        top_k: int = 5,
    ) -> GeoInferenceOutput:
        composed = TextComposer.compose(title=title, description=description, choices=choices)
        field_texts = {
            "title": composed.title_text,
            "description": composed.description_text,
            "choices": composed.choices_text,
            "combined": composed.combined_text,
        }

        hits = self.pipeline.retriever.retrieve(field_texts, top_n=10)
        scored = self.pipeline.scorer.score(hits, top_k=top_k)

        locations: list[LocationHypothesis] = []
        multi_query = False
        for candidate in scored:
            evidence: list[EvidenceItem] = []
            for h in candidate.evidence:
                if h.field not in ("title", "description", "choices"):
                    continue
                snippet = composed.snippets.get(h.field, "")
                if not snippet:
                    continue
                evidence.append(
                    EvidenceItem(
                        field=h.field,
                        snippet=snippet,
                        retrieval_hit=h.record.searchable_text[:180],
                        score=h.score,
                    )
                )

            if not evidence:
                continue

            locations.append(
                LocationHypothesis(
                    place_id=candidate.place_id,
                    name=candidate.name,
                    lat=candidate.lat,
                    lon=candidate.lon,
                    granularity=candidate.granularity,
                    confidence=candidate.confidence,
                    evidence=evidence,
                )
            )

        if locations:
            locations.sort(key=lambda x: x.confidence, reverse=True)
            top_conf = locations[0].confidence
            q = composed.combined_text.lower()
            multi_query = bool(
                re.search(r"\b(vs\.?|versus|against|between)\b", q)
                or re.search(r"\b(strike|attack|invade|sanctions?)\b.*\b(on|against)\b", q)
            )
            cutoff = 0.15 if multi_query else min(0.28, max(0.15, top_conf - 0.09))
            filtered: list[LocationHypothesis] = []
            for loc in locations:
                best_ev = max((ev.score for ev in loc.evidence), default=0.0)
                if loc.confidence < cutoff:
                    continue
                if best_ev < 0.1:
                    continue
                filtered.append(loc)
            locations = filtered

        event_type = self.pipeline.event_classifier.predict(composed)
        geo_type = self.pipeline.geo_decider.decide([l.confidence for l in locations], event_type)

        if locations:
            top = locations[0]
            top_hit = max(top.evidence, key=lambda e: e.score) if top.evidence else None
            if (
                top_hit is not None
                and top_hit.field == "title"
                and top_hit.score >= 0.82
                and geo_type in {"inferred", "multi"}
            ):
                geo_type = "explicit"
            if (
                len(locations) > 1
                and locations[0].confidence >= 0.45
                and abs(locations[0].confidence - locations[1].confidence) < 0.08
            ):
                geo_type = "multi"
            if multi_query and len(locations) > 1:
                geo_type = "multi"

        if geo_type == "none":
            locations = []

        return GeoInferenceOutput(geo_type=geo_type, event_type=event_type, locations=locations)

    def infer(
        self,
        condition_id: str,
        question: str,
        description: str | None = None,
        choices: list[str] | None = None,
    ) -> MarketInferenceResult:
        """Compatibility output for legacy API/database pipeline callers."""
        semantic = self.infer_semantic(question, description=description, choices=choices)

        type_map = {
            "city": LocationType.CITY,
            "state": LocationType.STATE,
            "country": LocationType.COUNTRY,
            "region": LocationType.COUNTRY,
            "global": LocationType.GLOBAL,
        }
        locs: list[LocationCandidate] = []
        for h in semantic.locations:
            reason = "; ".join(
                f"{e.field}:{e.snippet[:70]} -> {e.score:.2f}" for e in h.evidence[:3]
            )
            locs.append(
                LocationCandidate(
                    location_name=h.name,
                    location_type=type_map.get(h.granularity, LocationType.CITY),
                    confidence=h.confidence,
                    reason=reason,
                    inference_method=InferenceMethod.HEURISTIC,
                    latitude=h.lat,
                    longitude=h.lon,
                )
            )

        return MarketInferenceResult(
            condition_id=condition_id,
            locations=locs,
            has_location=bool(locs),
            is_global=(semantic.geo_type == "global"),
        )
