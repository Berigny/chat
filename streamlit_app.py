"""Streamlit Cloud entry point.

This module exists for backwards compatibility with older deployments that still
launch ``streamlit_app.py`` as the main module. The public demo lives in
:mod:`chat_demo_app`, so we simply forward ``main`` here.
"""

from chat_demo_app import main as chat_demo_main


def main() -> None:
    """Invoke the chat demo application."""

    chat_demo_main()


if __name__ == "__main__":  # pragma: no cover
    main()
