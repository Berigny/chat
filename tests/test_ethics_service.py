from services.ethics_service import EthicsService


def test_ethics_service_positive_signal():
    schema = {
        2: {"tier": "S1"},
        3: {"tier": "S1"},
    }
    service = EthicsService(schema=schema)
    snapshot = {"factors": [{"prime": 2, "timestamp": 1}]}
    assessment = service.evaluate(
        snapshot,
        deltas=[{"prime": 3, "delta": 1}],
        minted_bodies=[{"body": "Documented policy compliance."}],
    )

    assert assessment.lawfulness >= 0.88
    assert assessment.evidence >= 0.5
    assert assessment.non_harm >= 0.9
    assert any("body" in note.lower() for note in assessment.notes)


def test_ethics_service_flags_harmful_language():
    service = EthicsService(schema={2: {"tier": "S1"}})
    assessment = service.evaluate(
        {"factors": [{"prime": 2}]},
        minted_bodies=[{"body": "Plan to attack the target."}],
    )

    assert assessment.non_harm < 0.9
    assert any("harmful" in note.lower() for note in assessment.notes)
