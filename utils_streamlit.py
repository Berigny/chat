"""Streamlit helpers shared by the coherence and ethics dashboards."""

from __future__ import annotations

import json
from typing import Any

import requests
import streamlit as st


def parse_json_input(raw_value: str) -> tuple[Any | None, str | None]:
    """Parse JSON from user input returning the payload and an error message if any."""

    cleaned = (raw_value or "").strip()
    if not cleaned:
        return None, None
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"


def show_api_error(error: Exception | requests.Response, *, st_module=st) -> None:
    """Render a consistent API error block in Streamlit."""

    response: requests.Response | None = None
    if isinstance(error, requests.HTTPError):
        response = error.response
    elif isinstance(error, requests.Response):
        response = error

    if response is None:
        st_module.error(f"Request failed: {error}")
        return

    message = _response_message(response)
    status = response.status_code
    st_module.error(f"API request failed ({status}): {message}")
    if response.url:
        st_module.caption(response.url)


def _response_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message") or payload.get("error")
        if detail:
            return str(detail)
        return json.dumps(payload)
    return str(payload)


__all__ = ["parse_json_input", "show_api_error"]
