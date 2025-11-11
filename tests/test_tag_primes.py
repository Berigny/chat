from prime_schema import DEFAULT_PRIME_SCHEMA
from prime_tagger import tag_primes


def test_tag_primes_expected_sequence():
    schema = DEFAULT_PRIME_SCHEMA
    sentence = "I met Alice at 3 pm to discuss ethics"
    assert tag_primes(sentence, schema) == [2, 3, 7, 11, 13, 17, 37]


def test_tag_primes_multiple_cases():
    schema = DEFAULT_PRIME_SCHEMA
    cases = [
        ("We planned a review call", [2, 3]),
        ("The blocker worried the team", [2, 17, 19]),
        ("Remember the moral insight", [23, 29, 31]),
        ("", [2]),
    ]
    for text, expected in cases:
        assert tag_primes(text, schema) == expected

