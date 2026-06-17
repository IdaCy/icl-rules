"""Shared test fakes. Zero network: the OpenAI API surface is faked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def fake_response_data(
    text: str = "True",
    prompt_tokens: int = 100,
    completion_tokens: int = 1,
    model: str = "gpt-4.1",
) -> dict[str, Any]:
    return {
        "id": "fake",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
                "logprobs": {
                    "content": [
                        {
                            "token": text,
                            "logprob": -0.01,
                            "top_logprobs": [
                                {"token": text, "logprob": -0.01},
                                {"token": " False", "logprob": -4.5},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class _FakeModelDump:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return self._data


class _FakeCompletions:
    def __init__(self, owner: "FakeAPI") -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> _FakeModelDump:
        self._owner.calls.append(kwargs)
        if self._owner.errors_to_raise:
            raise self._owner.errors_to_raise.pop(0)
        return _FakeModelDump(self._owner.next_data(kwargs))


class _FakeChat:
    def __init__(self, owner: "FakeAPI") -> None:
        self.completions = _FakeCompletions(owner)


class FakeAPI:
    """Minimal stand-in for openai.AsyncOpenAI for OpenAIClient(api=...)."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.errors_to_raise: list[Exception] = []
        self._data = data or fake_response_data()
        self.chat = _FakeChat(self)

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        data = dict(self._data)
        data["model"] = kwargs.get("model", data.get("model"))
        return data


# --- synthetic step-1 dataset (locked item schema) ----------------------------


def make_rule_items(
    rule_id: str = "toy_rule",
    n_pool_bases: int = 100,
    n_held_out: int = 120,
    n_confirmation: int = 100,
    n_spare: int = 20,
) -> list[dict[str, Any]]:
    """A schema-conformant synthetic dataset. The toy 'rule': label True iff
    the text contains 'alpha' (so a fake API can answer perfectly)."""
    items: list[dict[str, Any]] = []

    def add(item_id: str, base_id: str, label: bool, text: str, split: str) -> None:
        items.append(
            {
                "item_id": item_id,
                "base_id": base_id,
                "rule_id": rule_id,
                "label": label,
                "text": text,
                "slots_meta": {},
                "split": split,
            }
        )

    for i in range(n_pool_bases):  # pool: both variants per base
        base = f"pool{i:04d}"
        add(f"{base}-T", base, True, f"pool sentence {i} alpha", "few_shot_pool")
        add(f"{base}-F", base, False, f"pool sentence {i} beta", "few_shot_pool")
    for split, n, prefix in (
        ("held_out", n_held_out, "held"),
        ("confirmation", n_confirmation, "conf"),
        ("spare", n_spare, "spare"),
    ):
        for i in range(n):  # one variant per base, alternating labels (exact 50/50)
            label = i % 2 == 0
            base = f"{prefix}{i:04d}"
            word = "alpha" if label else "beta"
            add(f"{base}-q", base, label, f"{prefix} sentence {i} {word}", split)
    return items


def write_rule_dataset(data_dir: str | Path, rule_id: str = "toy_rule", **kwargs: Any) -> Path:
    """Write make_rule_items() to <data_dir>/<rule_id>/items.jsonl."""
    items = make_rule_items(rule_id=rule_id, **kwargs)
    rule_dir = Path(data_dir) / rule_id
    rule_dir.mkdir(parents=True, exist_ok=True)
    path = rule_dir / "items.jsonl"
    path.write_text("\n".join(json.dumps(it) for it in items) + "\n", encoding="utf-8")
    return path
