"""Simple RocksDB-backed ledger adapter."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

SEED_KEY = "shard::theology/god/berigny::core"
SEED_VALUE = {
    "topic": "theology/god/berigny",
    "handle": "core",
    "primes": [3, 5, 11, 13, 19, 37],
    "theses": [
        "God as described by Berigny emphasises relational mastery.",
    ],
    "snippets": [
        {
            "type": "summary",
            "text": "Berigny frames God as the nexus of connection, memory, and action.",
        }
    ],
    "provenance": {
        "title": "Berigny Treatise",
        "author": "C. Berigny",
        "year": 1764,
        "source_url": "https://example.com/berigny",
    },
}


@dataclass
class KV:
    store: object
    backend: str

    def get(self, key: str) -> Optional[str]:
        if self.backend == "rocksdict":
            raw = self.store.get(key.encode("utf-8"))
            return raw.decode("utf-8") if raw else None
        return self.store.get(key)

    def put(self, key: str, value: str) -> None:
        if self.backend == "rocksdict":
            self.store[key.encode("utf-8")] = value.encode("utf-8")
        else:
            self.store[key] = value

    def items(self):
        if self.backend == "rocksdict":
            for key, value in self.store.items():
                yield key.decode("utf-8"), value.decode("utf-8")
        else:
            yield from self.store.items()

    def close(self):
        if self.backend == "rocksdict":
            self.store.close()


def _import_rocksdict():
    spec = importlib.util.find_spec("rocksdict")
    if not spec:
        return None
    return importlib.import_module("rocksdict")


def open_kv(path: str) -> KV:
    os.makedirs(path, exist_ok=True)
    backend = "dict"
    store: object = {}
    module = _import_rocksdict()
    if module is not None:
        try:
            store = module.Rdict(path)
            backend = "rocksdict"
        except Exception:
            store = {}
            backend = "dict"
    kv = KV(store=store, backend=backend)
    if kv.get(SEED_KEY) is None:
        kv.put(SEED_KEY, json.dumps(SEED_VALUE))
    return kv


def put_shard(topic: str, handle: str, shard: Dict, *, kv: Optional[KV] = None, path: str = "ledger_db") -> None:
    close_when_done = False
    if kv is None:
        kv = open_kv(path)
        close_when_done = True
    key = f"shard::{topic}::{handle}"
    payload = json.dumps(shard)
    kv.put(key, payload)
    if close_when_done:
        kv.close()


def _score_shard(shard: Dict, required: Iterable[int], preferred: Iterable[int], modifiers: Iterable[int]) -> Optional[float]:
    primes = shard.get("primes") or []
    if required and not set(required).issubset(primes):
        return None
    base = 0.0
    base += 10.0 * sum(1 for prime in primes if prime in set(preferred))
    base += 2.0 * sum(1 for prime in primes if prime in set(modifiers))
    base += 1.0 * len(primes)
    return base


def query_shards(
    topic_prefix: str,
    required: Iterable[int],
    preferred: Iterable[int],
    modifiers: Iterable[int],
    limit: int,
    *,
    kv: Optional[KV] = None,
    path: str = "ledger_db",
) -> List[Dict]:
    close_when_done = False
    if kv is None:
        kv = open_kv(path)
        close_when_done = True

    shards: List[Dict] = []
    prefix = f"shard::{topic_prefix}" if topic_prefix else "shard::"
    for key, value in kv.items():
        if not key.startswith(prefix):
            continue
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            continue
        if data.get("tombstone"):
            continue
        score = _score_shard(data, required, preferred, modifiers)
        if score is None:
            continue
        data["_score"] = score
        shards.append(data)

    shards.sort(key=lambda item: item.get("_score", 0), reverse=True)
    if close_when_done:
        kv.close()
    return shards[: max(1, limit)] if shards else []

