from flow_safe import sequence


def test_flow_safe_sequence_basic():
    primes = [2, 3, 5]
    expected = [
        {"prime": 2, "delta": 1},
        {"prime": 2, "delta": 1},
        {"prime": 3, "delta": 1},
        {"prime": 3, "delta": 1},
        {"prime": 5, "delta": 1},
        {"prime": 5, "delta": 1},
    ]
    assert sequence(primes) == expected


def test_flow_safe_sequence_empty():
    expected = [
        {"prime": 2, "delta": 1},
        {"prime": 2, "delta": 1},
    ]
    assert sequence([]) == expected

