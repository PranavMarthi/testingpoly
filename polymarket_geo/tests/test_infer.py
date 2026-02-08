from __future__ import annotations

from polymarket_geo.infer import LocationInferenceEngine


def test_output_contains_evidence_objects() -> None:
    engine = LocationInferenceEngine()
    out = engine.infer_semantic(
        title="Will they rebuild scotiabank arena?",
        description="Toronto city council is reviewing reconstruction permits.",
        choices=["Yes", "No"],
    )
    assert out.locations
    first = out.locations[0]
    assert first.evidence
    assert all(e.field in {"title", "description", "choices"} for e in first.evidence)
    assert all(0.0 <= e.score <= 1.0 for e in first.evidence)


def test_description_disambiguates_ambiguous_title() -> None:
    engine = LocationInferenceEngine()

    base = engine.infer_semantic(title="Presidential Election Winner 2028")
    with_desc = engine.infer_semantic(
        title="Presidential Election Winner 2028",
        description="This market refers to the Canadian federal election.",
        choices=["Liberal Party", "Conservative Party"],
    )

    assert with_desc.locations
    assert any("canada" in loc.name.lower() for loc in with_desc.locations)
    # Description/choices should materially improve certainty compared to title-only.
    base_top = base.locations[0].confidence if base.locations else 0.0
    assert with_desc.locations[0].confidence >= base_top


def test_multi_location_case_keeps_multiple_candidates() -> None:
    engine = LocationInferenceEngine()
    out = engine.infer_semantic(
        title="U.S. strike on Somalia by February 7?",
        description="Military action involving US forces and Somalia.",
    )
    names = [x.name.lower() for x in out.locations]
    assert any(("united states" in n) or ("washington" in n) for n in names)
    assert any(("somalia" in n) or ("mogadishu" in n) for n in names)
    assert out.geo_type in {"multi", "inferred", "ambiguous"}


def test_policy_defaults_country_to_capital_when_city_not_explicit() -> None:
    engine = LocationInferenceEngine()
    out = engine.infer_semantic(
        title="How many people will Trump deport in 2025?",
        description="Federal immigration policy market in the United States.",
    )
    assert out.locations
    top = out.locations[0].name.lower()
    assert ("washington" in top) or ("united states" in top)
