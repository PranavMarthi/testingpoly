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
        locations = self._refine_specificity(locations, event_type, composed.combined_text)
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

    def _refine_specificity(
        self,
        locations: list[LocationHypothesis],
        event_type: str,
        combined_text: str,
    ) -> list[LocationHypothesis]:
        if not locations:
            return locations

        refined = locations[:]

        # Prefer narrower granularity if both broad and specific candidates are
        # tied to the same country and confidence is comparable.
        cities = [l for l in refined if l.granularity == "city"]
        if cities:
            city_countries = {self._country_for_location(c) for c in cities}
            city_countries.discard(None)
            refined = [
                l
                for l in refined
                if not (l.granularity == "country" and l.name in city_countries)
            ]

        # For policy-oriented prompts, if only country-level results are found,
        # default to that country's capital city from local data when available.
        policy_like = self._is_policy_like(event_type, combined_text)
        if policy_like and refined and all(l.granularity == "country" for l in refined):
            upgraded: list[LocationHypothesis] = []
            for loc in refined:
                capital = self.pipeline.indexes.capital_for_country(loc.name)
                if capital is None:
                    upgraded.append(loc)
                    continue
                upgraded.append(
                    LocationHypothesis(
                        place_id=capital.place_id,
                        name=capital.place_name,
                        lat=capital.lat,
                        lon=capital.lon,
                        granularity=capital.granularity,
                        confidence=max(0.0, min(1.0, loc.confidence * 0.95)),
                        evidence=loc.evidence,
                    )
                )
            refined = upgraded

        refined.sort(key=lambda x: x.confidence, reverse=True)
        return refined

    def _is_policy_like(self, event_type: str, combined_text: str) -> bool:
        if event_type in {"election", "geopolitics", "finance"}:
            return True
        model = self.pipeline.indexes.embedder
        qv = model.embed(combined_text)
        pv = model.embed(
            "government policy legislation regulation court election federal parliament congress "
            "immigration deportation border enforcement sanctions"
        )
        sim = model.cosine(qv, pv)
        return sim >= 0.08

    def _country_for_location(self, location: LocationHypothesis) -> str | None:
        records = self.pipeline.indexes.records_for_place(location.place_id)
        if not records:
            return None
        for rec in records:
            if rec.country:
                return rec.country
        return None

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
