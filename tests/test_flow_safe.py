from flow_safe import sequence


def test_flow_safe_sequence_basic():
    primes = [2, 3, 5]
    expected = [
        {"p": 2, "d": 1},
        {"p": 2, "d": 1},
        {"p": 3, "d": 1},
        {"p": 3, "d": 1},
        {"p": 5, "d": 1},
        {"p": 5, "d": 1},
    ]
    assert sequence(primes) == expected


def test_flow_safe_sequence_empty():
    expected = [
        {"p": 2, "d": 1},
        {"p": 2, "d": 1},
    ]
    assert sequence([]) == expected

