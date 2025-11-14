from flow_rules import FlowAssessment, assess_enrichment_path, assess_write_path


SCHEMA = {
    2: {"tier": "S", "name": "Subject"},
    3: {"tier": "S", "name": "Action"},
    11: {"tier": "A", "name": "Time"},
    37: {"tier": "C", "name": "Mediator"},
}


def test_assess_write_path_blocks_unmediated_transition() -> None:
    assessment = assess_write_path(
        [{"prime": 2, "delta": 1}, {"prime": 11, "delta": 1}],
        SCHEMA,
    )
    assert not assessment.ok
    assert "C-tier mediator" in assessment.messages()[0]


def test_assess_write_path_allows_mediated_flow() -> None:
    assessment = assess_write_path(
        [
            {"prime": 2, "delta": 1},
            {"prime": 37, "delta": 1},
            {"prime": 11, "delta": 1},
        ],
        SCHEMA,
    )
    assert assessment.ok
    assert assessment.messages() == []


def test_assess_enrichment_path_respects_same_rules() -> None:
    assessment = assess_enrichment_path(
        2,
        [{"prime": 37, "delta": 1}, {"prime": 11, "delta": 1}],
        SCHEMA,
    )
    assert assessment.ok

    violation = assess_enrichment_path(2, [{"prime": 11, "delta": 1}], SCHEMA)
    assert not violation.ok
    assert violation.violations[0].primes == (2, 11)


def test_flow_assessment_serializes() -> None:
    violation = assess_write_path([2, 11], SCHEMA)
    assert isinstance(violation, FlowAssessment)
    payload = violation.asdict()
    assert payload["ok"] is False
    assert payload["violations"][0]["code"] == "flow.write.mediator"
