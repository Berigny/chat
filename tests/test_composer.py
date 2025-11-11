from composer import compose_summary


def test_composer_without_source_url():
    shard = {
        "theses": ["Insight without citation."],
        "snippets": [{"type": "quote", "text": "Do not fabricate."}],
        "provenance": {"title": "Notebook", "author": "Analyst", "year": 2024},
    }
    reply = compose_summary([shard], "question")
    assert "“" not in reply.split("\n\n")[0]


def test_composer_empty_list():
    reply = compose_summary([], "question")
    assert reply.startswith("I couldn’t find enough grounded material")

