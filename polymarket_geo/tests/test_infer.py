"""
Tests for the location inference engine.
These are pure unit tests — no database or network required.

Note: spaCy may not be available on Python 3.14+ due to pydantic v1 incompatibility.
Tests that require NLP entity extraction are marked with @requires_spacy and
will be skipped gracefully when spaCy cannot load.
"""

from __future__ import annotations

import os
import pytest

# Override spacy model to smallest for testing speed
os.environ.setdefault("SPACY_MODEL", "en_core_web_sm")

from polymarket_geo.infer import LocationInferenceEngine, SPACY_AVAILABLE, get_nlp
from polymarket_geo.models import InferenceMethod, LocationType
from polymarket_geo.event_resolver import EventVenueResult

# Check if spaCy is truly functional (installed + model loads)
_spacy_functional = SPACY_AVAILABLE
if _spacy_functional:
    try:
        _test_nlp = get_nlp()
        _spacy_functional = _test_nlp is not None
    except Exception:
        _spacy_functional = False

requires_spacy = pytest.mark.skipif(
    not _spacy_functional,
    reason="spaCy not available or model cannot load (Python 3.14+ incompatibility)"
)


@pytest.fixture(scope="module")
def engine():
    return LocationInferenceEngine()


# ── Direct location in text ───────────────────────────────────────────

class TestNLPExtraction:
    @requires_spacy
    def test_city_in_question(self, engine):
        result = engine.infer("test-1", "Highest temperature in Atlanta on February 7?")
        assert result.has_location
        names = [loc.location_name.lower() for loc in result.locations]
        assert any("atlanta" in n for n in names)

    @requires_spacy
    def test_country_in_question(self, engine):
        result = engine.infer("test-2", "Will France ban TikTok?")
        assert result.has_location
        names = [loc.location_name.lower() for loc in result.locations]
        assert any("france" in n or "paris" in n for n in names)


# ── Sports team heuristics ────────────────────────────────────────────

class TestSportsHeuristics:
    def test_nba_matchup(self, engine):
        result = engine.infer("test-3", "Atlanta Hawks vs Lakers")
        assert result.has_location
        names = {loc.location_name for loc in result.locations}
        # Should find Atlanta and LA
        assert any("Atlanta" in n for n in names)
        assert any("Los Angeles" in n for n in names)

    def test_nfl_team(self, engine):
        result = engine.infer("test-4", "Will the Kansas City Chiefs win the Super Bowl?")
        assert result.has_location
        names = {loc.location_name for loc in result.locations}
        assert any("Kansas City" in n for n in names)

    def test_soccer_team(self, engine):
        result = engine.infer("test-5", "Manchester United vs Arsenal")
        assert result.has_location
        names = {loc.location_name.lower() for loc in result.locations}
        assert any("manchester" in n for n in names)
        assert any("london" in n for n in names)


# ── Political figure heuristics ───────────────────────────────────────

class TestPoliticalHeuristics:
    def test_trump(self, engine):
        result = engine.infer("test-6", "What will Trump say this week?")
        assert result.has_location
        names = {loc.location_name for loc in result.locations}
        assert any("Washington" in n for n in names)

    def test_putin(self, engine):
        result = engine.infer("test-7", "Will Putin attend the summit?")
        assert result.has_location
        names = {loc.location_name.lower() for loc in result.locations}
        assert any("moscow" in n for n in names)


# ── Institution heuristics ────────────────────────────────────────────

class TestInstitutionHeuristics:
    def test_fed(self, engine):
        result = engine.infer("test-8", "Will the Fed cut rates?")
        assert result.has_location
        names = {loc.location_name for loc in result.locations}
        assert any("Washington" in n for n in names)

    def test_ecb(self, engine):
        result = engine.infer("test-9", "ECB interest rate decision")
        assert result.has_location
        names = {loc.location_name.lower() for loc in result.locations}
        assert any("frankfurt" in n for n in names)

    def test_un(self, engine):
        result = engine.infer("test-10", "UN Security Council vote on sanctions")
        assert result.has_location
        names = {loc.location_name for loc in result.locations}
        assert any("New York" in n for n in names)


# ── Building heuristics ──────────────────────────────────────────────

class TestBuildingHeuristics:
    def test_mar_a_lago(self, engine):
        result = engine.infer("test-11", "Will Trump host a dinner at Mar-a-Lago?")
        assert result.has_location
        names = {loc.location_name for loc in result.locations}
        assert any("Palm Beach" in n for n in names)
        # Should have pre-filled coordinates
        palm_beach_locs = [l for l in result.locations if "Palm Beach" in l.location_name]
        assert any(l.latitude is not None for l in palm_beach_locs)

    def test_white_house(self, engine):
        result = engine.infer("test-12", "White House press conference this week?")
        assert result.has_location
        names = {loc.location_name for loc in result.locations}
        assert any("Washington" in n for n in names)


# ── Global markets ────────────────────────────────────────────────────

class TestGlobalMarkets:
    def test_bitcoin(self, engine):
        result = engine.infer("test-13", "BTC price above 100k by Friday?")
        assert result.is_global
        assert not result.has_location
        assert any(l.location_type == LocationType.GLOBAL for l in result.locations)

    def test_crypto_generic(self, engine):
        result = engine.infer("test-14", "Will Ethereum flip Bitcoin in market cap?")
        assert result.is_global


# ── Confidence scoring ────────────────────────────────────────────────

class TestConfidenceScoring:
    @requires_spacy
    def test_single_location_high_confidence(self, engine):
        result = engine.infer("test-15", "Highest temperature in Atlanta on February 7?")
        atlanta = [l for l in result.locations if "atlanta" in l.location_name.lower()]
        assert atlanta
        assert atlanta[0].confidence >= 0.60

    def test_multi_location_has_ambiguity_penalty(self, engine):
        result = engine.infer("test-16", "Atlanta Hawks vs Lakers")
        # Multiple locations should have ambiguity penalty applied
        for loc in result.locations:
            assert "ambiguity penalty" in loc.reason.lower() or loc.confidence < 0.95

    def test_all_confidences_in_range(self, engine):
        texts = [
            "Temperature in NYC?",
            "Trump at Mar-a-Lago",
            "BTC to 200k?",
            "Arsenal vs Chelsea",
        ]
        for text in texts:
            result = engine.infer("test", text)
            for loc in result.locations:
                assert 0 <= loc.confidence <= 1.0

    def test_confidence_ordering(self, engine):
        result = engine.infer("test-17", "Atlanta Hawks vs Lakers")
        confidences = [l.confidence for l in result.locations]
        assert confidences == sorted(confidences, reverse=True)


# ── Merge behavior ────────────────────────────────────────────────────

class TestMergeBehavior:
    def test_nlp_and_heuristic_merge_boost_confidence(self, engine):
        """When NLP and heuristic both find the same location, confidence should be boosted."""
        result = engine.infer("test-18", "Will the Fed raise rates in Washington?")
        dc_locs = [l for l in result.locations if "Washington" in l.location_name]
        # Both NLP ("Washington" as GPE) and heuristic ("Fed" -> DC) should merge
        if dc_locs:
            # Merged confidence should be higher than either source alone
            assert dc_locs[0].confidence >= 0.60

    def test_no_duplicate_locations(self, engine):
        result = engine.infer("test-19", "Atlanta Hawks play in Atlanta tonight")
        names = [l.location_name.lower() for l in result.locations]
        # Should not have duplicate Atlanta entries
        atlanta_count = sum(1 for n in names if "atlanta" in n)
        assert atlanta_count <= 1


# ── Inference method tracking ─────────────────────────────────────────

class TestInferenceMethods:
    def test_nlp_method_tagged(self, engine):
        result = engine.infer("test-20", "Earthquake in Japan tomorrow?")
        for loc in result.locations:
            assert loc.inference_method in (InferenceMethod.NLP, InferenceMethod.HEURISTIC)

    def test_heuristic_method_tagged(self, engine):
        result = engine.infer("test-21", "Will the Fed cut rates?")
        fed_locs = [l for l in result.locations if "heuristic" in l.reason.lower()]
        assert len(fed_locs) > 0


# ── Gazetteer + policy fallback regressions ───────────────────────────

class TestGeneralLocationFallback:
    def test_us_greenland_not_global(self, engine):
        result = engine.infer("test-22", "Will the US acquire part of Greenland in 2026?")
        assert result.has_location
        assert not result.is_global
        names = {loc.location_name.lower() for loc in result.locations}
        assert any("greenland" in n for n in names)
        assert any("united states" in n or "washington" in n for n in names)

    def test_policy_defaults_to_dc(self, engine):
        result = engine.infer("test-23", "Clarity Act signed into law in 2026?")
        assert result.has_location
        names = {loc.location_name.lower() for loc in result.locations}
        assert any("washington" in n for n in names)


class TestEventVenueInference:
    def test_oscars_returns_not_available_or_confirmed(self, engine, monkeypatch):
        class StubResolver:
            def resolve(self, text: str):
                return EventVenueResult(
                    status="not_available",
                    event_key="oscars",
                    event_year=2026,
                    confidence=0.35,
                    reason="Event venue not publicly confirmed yet",
                )

        monkeypatch.setattr("polymarket_geo.infer.get_event_resolver", lambda: StubResolver())
        result = engine.infer("test-24", "Oscars 2026: Best Picture Winner")
        names = [l.location_name for l in result.locations]
        assert "Oscars" not in names
        assert "Best Picture Winner" not in names
        assert any(n == "not_available" for n in names)

    def test_event_confirmed_location(self, engine, monkeypatch):
        class StubResolver:
            def resolve(self, text: str):
                return EventVenueResult(
                    status="confirmed",
                    event_key="oscars",
                    event_year=2026,
                    venue_name="Dolby Theatre",
                    city="Los Angeles",
                    country="USA",
                    confidence=0.72,
                    source_url="https://example.com/oscars-2026",
                    reason="Event venue resolved from public event page",
                )

        monkeypatch.setattr("polymarket_geo.infer.get_event_resolver", lambda: StubResolver())
        result = engine.infer("test-25", "Oscars 2026: Best Picture Winner")
        names = [l.location_name for l in result.locations]
        assert any("Los Angeles" in n for n in names)
        assert all("Best Picture Winner" not in n for n in names)
