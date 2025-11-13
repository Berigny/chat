from flow_rules import assess_enrichment_path, assess_write_path


SCHEMA = {
    2: {"tier": "S"},
    3: {"tier": "S"},
    11: {"tier": "A"},
    37: {"tier": "C"},
}


def test_assess_write_path_requires_conductor() -> None:
    assessment = assess_write_path([{"prime": 2}, {"prime": 11}], SCHEMA)
    assert not assessment.ok
    assert assessment.messages()


def test_assess_write_path_passes_with_conductor() -> None:
    assessment = assess_write_path([{"prime": 2}, {"prime": 37}, {"prime": 11}], SCHEMA)
    assert assessment.ok


def test_assess_enrichment_path_flags_ref_transition() -> None:
    assessment = assess_enrichment_path(2, [{"prime": 11, "delta": 1}], SCHEMA)
    assert not assessment.ok
    assert assessment.violations[0].prime == 11
