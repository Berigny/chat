from prime_pipeline import build_anchor_factors

SCHEMA = {
    2: {"name": "Subject"},
    3: {"name": "Action"},
    5: {"name": "Object"},
    7: {"name": "Channel"},
    11: {"name": "Time"},
}
FALLBACK_PRIME = 2


def test_build_anchor_factors_from_text():
    text = "I met with the team at the office at 9am"
    factors = build_anchor_factors(text, SCHEMA, fallback_prime=FALLBACK_PRIME)
    assert factors == [
        {"prime": 2, "delta": 1},
        {"prime": 3, "delta": 1},
        {"prime": 5, "delta": 1},
        {"prime": 7, "delta": 1},
        {"prime": 11, "delta": 1},
    ]


def test_build_anchor_factors_override_deduplicates():
    override = [
        {"prime": 11, "delta": 3},
        {"prime": 11, "delta": 7},
        {"prime": 3, "delta": -2},
        {"prime": 99, "delta": 5},
    ]
    factors = build_anchor_factors(
        "",
        SCHEMA,
        fallback_prime=FALLBACK_PRIME,
        factors_override=override,
    )
    assert factors == [
        {"prime": 11, "delta": 3},
        {"prime": 3, "delta": -2},
    ]


def test_build_anchor_factors_fallback_prime():
    factors = build_anchor_factors("", SCHEMA, fallback_prime=FALLBACK_PRIME)
    assert factors == [{"prime": FALLBACK_PRIME, "delta": 1}]
