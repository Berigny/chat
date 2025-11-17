"""Developer RocksDB probe utilities for inspecting ledger state."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rocksdict import AccessType, Options, Rdict

DEFAULT_ROCKSDB_DATA_PATH = Path(os.getenv("ROCKSDB_DATA_PATH", "/app/rocksdb-data"))
_DEFAULT_PATTERN = (2, 3, 5, 7)
_PATTERN_RE = re.compile(r"\d+")
_PROBE_PREFIX = "__probe__"


@dataclass
class RocksDBClient:
    """Lightweight wrapper around ``rocksdict.Rdict`` for debugging."""

    data_path: str | Path
    create_if_missing: bool = False
    _db: Rdict | None = None

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)

    def __enter__(self) -> "RocksDBClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def connect(self) -> Rdict:
        """Open the RocksDB instance if it is not already available."""

        if self._db is None:
            if not self.create_if_missing and not self.data_path.exists():
                raise FileNotFoundError(f"RocksDB path '{self.data_path}' not found")
            options = Options()
            options.create_if_missing = self.create_if_missing
            self._db = Rdict(
                str(self.data_path),
                options=options,
                access_type=AccessType.read_write(),
            )
        return self._db

    def close(self) -> None:
        """Close the RocksDB handle and release resources."""

        if self._db is not None:
            self._db.close()
            self._db = None

    def anchor_probe(self, entity: str, prompt: str, pattern: Sequence[int]) -> str | None:
        """Write a synthetic entry under a dedicated probe prefix."""

        normalized_prompt = (prompt or "").strip()
        if not normalized_prompt:
            return None
        pattern_label = "x".join(str(p) for p in pattern) or "baseline"
        key = f"{_PROBE_PREFIX}:{entity}:{pattern_label}:{int(time.time()*1000)}:{uuid.uuid4().hex}"
        payload = json.dumps(
            {
                "entity": entity,
                "pattern": list(pattern),
                "text": normalized_prompt,
                "ts": int(time.time()),
            },
            ensure_ascii=False,
        )
        self.connect()[key] = payload
        return key

    def collect(self, entity: str, pattern_products: Sequence[int], limit: int) -> list[dict[str, str]]:
        """Collect raw key/value pairs using the multiplicative pattern."""

        if limit <= 0:
            return []
        head_limit = max(limit, max(pattern_products) if pattern_products else 1)
        head = self._collect_entity_entries(entity, head_limit * 2)
        if not head:
            return []
        hits: list[dict[str, str]] = []
        used_indices: set[int] = set()
        for stride in pattern_products or (1,):
            idx = max(0, stride - 1)
            if idx < len(head):
                hits.append(head[idx])
                used_indices.add(idx)
            if len(hits) >= limit:
                break
        if len(hits) < limit:
            for idx, item in enumerate(head):
                if idx in used_indices:
                    continue
                hits.append(item)
                if len(hits) >= limit:
                    break
        return hits[:limit]

    def _collect_entity_entries(self, entity: str, limit: int) -> list[dict[str, str]]:
        if limit <= 0:
            return []
        entity_bytes = entity.encode("utf-8") if entity else b""
        iterator = self.connect().iter()
        iterator.seek_to_first()
        matches: list[dict[str, str]] = []
        try:
            while iterator.valid():
                key_raw = iterator.key()
                value_raw = iterator.value()
                if not entity_bytes or entity_bytes in self._to_bytes(key_raw):
                    matches.append(
                        {
                            "key": self._to_text(key_raw),
                            "value": self._to_text(value_raw),
                        }
                    )
                    if len(matches) >= limit:
                        break
                iterator.next()
        finally:
            del iterator
        return matches

    @staticmethod
    def _to_bytes(value: object) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8", "ignore")
        return str(value).encode("utf-8", "ignore")

    @staticmethod
    def _to_text(value: object) -> str:
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.hex()
        return str(value)


def _parse_pattern(prime_pattern: str | Sequence[int] | None) -> list[int]:
    if not prime_pattern:
        return list(_DEFAULT_PATTERN)
    if isinstance(prime_pattern, Sequence) and not isinstance(prime_pattern, (str, bytes, bytearray)):
        values: list[int] = []
        for entry in prime_pattern:
            try:
                candidate = int(entry)
            except (TypeError, ValueError):
                continue
            if candidate > 1:
                values.append(candidate)
        return values or list(_DEFAULT_PATTERN)
    matches = _PATTERN_RE.findall(str(prime_pattern))
    parsed = [int(match) for match in matches if int(match) > 1]
    return parsed or list(_DEFAULT_PATTERN)


def _expand_pattern(primes: Sequence[int]) -> list[int]:
    products: list[int] = []
    running = 1
    for prime in primes or (1,):
        running *= max(1, prime)
        products.append(running)
    return products


def run_probe(entity: str, prompt: str, prime_pattern: str, top_n: int = 20) -> dict[str, object]:
    """Anchor a synthetic prompt and walk the RocksDB keyspace."""

    entity_id = (entity or "").strip()
    if not entity_id:
        raise ValueError("Entity is required for a RocksDB probe.")
    primes = _parse_pattern(prime_pattern)
    pattern_products = _expand_pattern(primes)
    limit = max(1, int(top_n or 1))
    with RocksDBClient(DEFAULT_ROCKSDB_DATA_PATH) as client:
        anchor_key = client.anchor_probe(entity_id, prompt, primes)
        hits = client.collect(entity_id, pattern_products, limit)
    return {
        "entity": entity_id,
        "prime_pattern": primes,
        "multiplicative_walk": pattern_products,
        "anchor_key": anchor_key,
        "results": hits,
    }


__all__ = ["run_probe", "RocksDBClient"]
