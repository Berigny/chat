from types import SimpleNamespace

from services.memory_service import MOBIUS_REFRESH_INTERVAL, MemoryService, is_recall_query


class DummyAssemblyApi:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_assembly(self, *_, **__):
        self.calls += 1
        return {"summaries": [], "bodies": [], "claims": []}


def test_select_context_forwards_search_params() -> None:
    calls: list[dict] = []

    class ApiStub(SimpleNamespace):
        def search_slots(self, entity, query, **kwargs):
            calls.append({"entity": entity, "query": query, **kwargs})
            return [
                {"prime": 2, "summary": "Met with Alice", "score": 0.9},
                {"prime": 11, "summary": "Follow-up scheduled", "score": 0.5},
            ]

    service = MemoryService(api_service=ApiStub(), prime_weights={})

    results = service.select_context(
        "demo",
        "meeting recap",
        {},
        ledger_id="ledger-alpha",
        limit=2,
    )

    assert len(results) == 2
    assert results[0]["summary"] == "Met with Alice"
    assert calls and calls[0]["mode"] == "slots"
    assert calls[0]["limit"] == 2
    assert calls[0]["ledger_id"] == "ledger-alpha"


def test_build_recall_response_returns_engine_text() -> None:
    class ApiStub(SimpleNamespace):
        def __init__(self) -> None:
            super().__init__(
                payload={"response": "Here’s what the ledger currently recalls:\n• Meeting notes"}
            )

        def search(self, entity, query, **kwargs):
            self.last_call = {"entity": entity, "query": query, **kwargs}
            return self.payload

    api = ApiStub()
    service = MemoryService(api_service=api, prime_weights={})

    response = service.build_recall_response("demo", "recall meeting", {}, ledger_id="ledger-alpha")

    assert response == "Here’s what the ledger currently recalls:\n• Meeting notes"
    assert api.last_call["mode"] == "all"
    assert api.last_call["limit"] >= 1
    assert api.last_call["ledger_id"] == "ledger-alpha"


def test_build_recall_response_uses_custom_mode() -> None:
    class ApiStub(SimpleNamespace):
        def __init__(self) -> None:
            self.calls: list[str] = []
            super().__init__(payload={"response": "body mode"})

        def search(self, entity, query, **kwargs):
            self.last_call = {"entity": entity, "query": query, **kwargs}
            mode = kwargs.get("mode")
            if mode:
                self.calls.append(mode)
            return self.payload

    api = ApiStub()
    service = MemoryService(api_service=api, prime_weights={})

    response = service.build_recall_response(
        "demo",
        "recall meeting",
        {},
        ledger_id="ledger-beta",
        mode="body",
    )

    assert response == "body mode"
    assert api.last_call["mode"] == "body"
    assert api.last_call["ledger_id"] == "ledger-beta"
    assert api.calls == ["body"]


def test_build_recall_response_retries_body_mode_when_empty() -> None:
    class ApiStub(SimpleNamespace):
        def __init__(self) -> None:
            self.modes: list[str] = []

        def search(self, entity, query, **kwargs):
            mode = kwargs.get("mode")
            self.modes.append(mode)
            if mode == "all":
                return {"results": []}
            return {"results": [{"snippet": "Ledger body hit"}]}

    api = ApiStub()
    service = MemoryService(api_service=api, prime_weights={})

    response = service.build_recall_response("demo", "recall topic", {})

    assert response == "Ledger body hit"
    assert api.modes == ["all", "body"]


def test_build_recall_response_skips_prompt_echo_snippet() -> None:
    class ApiStub(SimpleNamespace):
        def __init__(self) -> None:
            self.modes: list[str] = []

        def search(self, entity, query, **kwargs):
            mode = kwargs.get("mode")
            self.modes.append(mode)
            if len(self.modes) == 1:
                return {"results": [{"snippet": "Do you have any quotes about God?"}]}
            return {"results": [{"snippet": "Ledger answer"}]}

    api = ApiStub()
    service = MemoryService(api_service=api, prime_weights={})

    response = service.build_recall_response("demo", "Do you have any quotes about God?", {})

    assert response == "Ledger answer"
    assert api.modes == ["all", "body"]


def test_build_recall_response_ignores_prompt_echo_response_payload() -> None:
    class ApiStub(SimpleNamespace):
        def __init__(self) -> None:
            self.modes: list[str] = []

        def search(self, entity, query, **kwargs):
            mode = kwargs.get("mode")
            self.modes.append(mode)
            if mode == "all":
                return {"response": "Do you have any quotes about God?"}
            return {"results": [{"snippet": "Ledger echo free result"}]}

    api = ApiStub()
    service = MemoryService(api_service=api, prime_weights={})

    response = service.build_recall_response("demo", "Do you have any quotes about God?", {})

    assert response == "Ledger echo free result"
    assert api.modes == ["all", "body"]


def test_build_recall_response_returns_message_when_no_results() -> None:
    class ApiStub(SimpleNamespace):
        def search(self, *_, **__):
            return {}

    service = MemoryService(api_service=ApiStub(), prime_weights={})

    assert (
        service.build_recall_response("demo", "recall meeting", {})
        == "- No ledger memories matched the topic (recall, meeting)."
    )


def test_build_recall_response_falls_back_to_memory_lookup_entries() -> None:
    class ApiStub(SimpleNamespace):
        def __init__(self) -> None:
            self.search_calls = 0

        def search(self, *_, **__):
            self.search_calls += 1
            return {"results": [{"snippet": "Do you have any quotes about God?"}]}

        def fetch_memories(self, *_, **__):
            return [
                {"summary": "Light for the million – quote about God", "meta": {"source": "chat_demo"}},
                {"summary": "Irrelevant entry", "meta": {"source": "chat_demo"}},
            ]

    service = MemoryService(api_service=ApiStub(), prime_weights={})

    response = service.build_recall_response("demo", "Do you have any quotes about God?", {})

    assert response.startswith("- Light for the million – quote about God")


def test_mobius_refresh_waits_for_rotation() -> None:
    api = DummyAssemblyApi()
    service = MemoryService(api_service=api, prime_weights={})

    triggered = service.maybe_refresh_mobius_alignment("demo", ledger_id="alpha")
    assert not triggered
    assert api.calls == 0


def test_mobius_refresh_triggers_after_interval() -> None:
    api = DummyAssemblyApi()
    service = MemoryService(api_service=api, prime_weights={})

    service.note_mobius_rotation("demo", ledger_id="alpha", timestamp=1.0)

    triggered = service.maybe_refresh_mobius_alignment(
        "demo", ledger_id="alpha", now=MOBIUS_REFRESH_INTERVAL + 1
    )

    assert triggered
    assert api.calls == 1


def test_is_recall_query_matches_ledger_sentence_prompt() -> None:
    prompt = (
        "In the ledger this sentence exists: 'Memory is not quoted through a fuzzy "
        "similarity search but through an exact p-adic valuation check' what's the paragraph after that sentence?"
    )

    assert is_recall_query(prompt)


def test_is_recall_query_matches_plural_quotes() -> None:
    assert is_recall_query("do you have any quotes about God?")
