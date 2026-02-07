"""
Tests for the geocoding module.
Unit tests for normalization (no network/DB required).
"""

from __future__ import annotations

import pytest

from polymarket_geo.geocode import normalize_location_name


class TestNormalization:
    def test_known_city(self):
        assert normalize_location_name("Atlanta") == "Atlanta, GA, USA"
        assert normalize_location_name("atlanta") == "Atlanta, GA, USA"
        assert normalize_location_name("  Atlanta  ") == "Atlanta, GA, USA"

    def test_city_with_state_abbrev(self):
        assert normalize_location_name("Atlanta, GA") == "Atlanta, GA, USA"

    def test_nyc_aliases(self):
        assert normalize_location_name("NYC") == "New York, NY, USA"
        assert normalize_location_name("New York") == "New York, NY, USA"
        assert normalize_location_name("new york, ny") == "New York, NY, USA"

    def test_dc_aliases(self):
        assert normalize_location_name("Washington, DC") == "Washington, DC, USA"
        assert normalize_location_name("DC") == "Washington, DC, USA"
        assert normalize_location_name("washington") == "Washington, DC, USA"

    def test_international_city(self):
        assert normalize_location_name("London") == "London, United Kingdom"
        assert normalize_location_name("london, uk") == "London, United Kingdom"
        assert normalize_location_name("Paris") == "Paris, France"
        assert normalize_location_name("Tokyo") == "Tokyo, Japan"

    def test_state_abbreviation_expansion(self):
        result = normalize_location_name("Portland, OR")
        assert "Oregon" in result or "OR" in result

    def test_unknown_location_lowercase(self):
        result = normalize_location_name("Timbuktu")
        assert result == "timbuktu"

    def test_whitespace_collapse(self):
        result = normalize_location_name("  San   Francisco  ")
        assert "san francisco" in result.lower()

    def test_palm_beach(self):
        assert normalize_location_name("Palm Beach, FL") == "Palm Beach, FL, USA"


class TestNormalizationEdgeCases:
    def test_empty_string(self):
        result = normalize_location_name("")
        assert result == ""

    def test_single_word(self):
        result = normalize_location_name("Denver")
        assert result == "Denver, CO, USA"

    def test_la_disambiguation(self):
        # "LA" should resolve to Los Angeles, not Louisiana
        assert normalize_location_name("LA") == "Los Angeles, CA, USA"

    def test_sf_disambiguation(self):
        assert normalize_location_name("SF") == "San Francisco, CA, USA"
