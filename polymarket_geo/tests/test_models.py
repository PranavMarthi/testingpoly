"""
Tests for Pydantic model validation.
"""

from __future__ import annotations

import pytest

from polymarket_geo.models import (
    LocationCandidate,
    LocationType,
    InferenceMethod,
    MarketInferenceResult,
    RawMarket,
)


class TestRawMarket:
    def test_basic_parsing(self):
        data = {
            "conditionId": "abc123",
            "question": "Will it rain?",
            "active": True,
            "closed": False,
        }
        market = RawMarket.model_validate(data)
        assert market.condition_id == "abc123"
        assert market.question == "Will it rain?"
        assert market.active is True

    def test_alias_fields(self):
        data = {
            "conditionId": "abc123",
            "question": "Test?",
            "slug": "test-market",
            "endDate": "2025-12-31T00:00:00Z",
            "outcomePrices": '["0.65", "0.35"]',
        }
        market = RawMarket.model_validate(data)
        assert market.market_slug == "test-market"
        assert market.end_date_iso == "2025-12-31T00:00:00Z"
        assert market.outcome_prices == ["0.65", "0.35"]

    def test_outcome_prices_as_list(self):
        data = {
            "conditionId": "abc123",
            "question": "Test?",
            "outcomePrices": ["0.65", "0.35"],
        }
        market = RawMarket.model_validate(data)
        assert market.outcome_prices == ["0.65", "0.35"]

    def test_extra_fields_allowed(self):
        data = {
            "conditionId": "abc123",
            "question": "Test?",
            "some_unknown_field": "value",
        }
        market = RawMarket.model_validate(data)
        assert market.condition_id == "abc123"


class TestLocationCandidate:
    def test_valid_candidate(self):
        loc = LocationCandidate(
            location_name="Atlanta, GA",
            location_type=LocationType.CITY,
            confidence=0.85,
            reason="NLP entity",
            inference_method=InferenceMethod.NLP,
        )
        assert loc.confidence == 0.85
        assert loc.latitude is None

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            LocationCandidate(
                location_name="Test",
                confidence=1.5,  # > 1.0
                reason="test",
            )

        with pytest.raises(Exception):
            LocationCandidate(
                location_name="Test",
                confidence=-0.1,  # < 0.0
                reason="test",
            )

    def test_with_coordinates(self):
        loc = LocationCandidate(
            location_name="White House",
            location_type=LocationType.BUILDING,
            confidence=0.90,
            reason="Building heuristic",
            inference_method=InferenceMethod.HEURISTIC,
            latitude=38.8977,
            longitude=-77.0365,
        )
        assert loc.latitude == pytest.approx(38.8977)
        assert loc.longitude == pytest.approx(-77.0365)


class TestMarketInferenceResult:
    def test_empty_result(self):
        result = MarketInferenceResult(condition_id="test")
        assert result.locations == []
        assert result.has_location is False
        assert result.is_global is False

    def test_with_locations(self):
        result = MarketInferenceResult(
            condition_id="test",
            locations=[
                LocationCandidate(
                    location_name="Atlanta, GA",
                    confidence=0.85,
                    reason="test",
                ),
            ],
            has_location=True,
        )
        assert len(result.locations) == 1
        assert result.has_location is True
