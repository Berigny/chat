"""Wrappers for invoking shared ledger migration tooling."""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import sys
from typing import Mapping, Sequence


@dataclass(slots=True)
class MigrationResult:
    """Represents the outcome from running the migration CLI."""

    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def asdict(self) -> dict[str, object]:
        return {
            "args": list(self.args),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "ok": self.ok,
        }


def run_ledger_migration(
    entity: str,
    *,
    ledger_id: str | None = None,
    extra_args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> MigrationResult:
    """Invoke ``tools.migrate_ledger_v11`` with the provided ledger context."""

    if not isinstance(entity, str) or not entity.strip():
        raise ValueError("entity must be a non-empty string")

    command: list[str] = [sys.executable, "-m", "tools.migrate_ledger_v11", "--entity", entity]
    if ledger_id:
        command.extend(["--ledger", ledger_id])
    if extra_args:
        command.extend(list(extra_args))

    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=env_vars,
    )
    return MigrationResult(
        args=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


__all__ = ["MigrationResult", "run_ledger_migration"]
