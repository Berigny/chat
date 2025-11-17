"""Tests for :mod:`ui_components`."""

from __future__ import annotations

from ui_components import prepare_metric_rows, sanitize_json_payload


def test_sanitize_json_payload_from_string() -> None:
    payload = sanitize_json_payload('{"key": "value"}')
    assert isinstance(payload, dict)
    assert payload["key"] == "value"


def test_prepare_metric_rows_from_mapping() -> None:
    rows = prepare_metric_rows({"ΔE": 1.234, "Status": "ok"})
    assert rows[0] == ("ΔE", "1.23")
    assert rows[1] == ("Status", "ok")
