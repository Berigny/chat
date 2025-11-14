from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from services import migration_cli


def test_run_ledger_migration_invokes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def fake_run(cmd, capture_output, text, check, env):
        captured.append(cmd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(migration_cli.subprocess, "run", fake_run)

    result = migration_cli.run_ledger_migration(
        "demo-entity",
        ledger_id="ledger-42",
        extra_args=["--dry-run"],
        env={"EXTRA": "1"},
    )

    assert captured, "Expected subprocess invocation"
    command = captured[0]
    assert command[:3] == [sys.executable, "-m", "tools.migrate_ledger_v11"]
    assert "--entity" in command and "demo-entity" in command
    assert "--ledger" in command and "ledger-42" in command
    assert "--dry-run" in command
    assert result.ok
    assert result.stdout == "ok"
    assert result.stderr == ""
