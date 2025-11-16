from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Sequence, Tuple


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


@dataclass(frozen=True)
class DatasetFieldConfig:
    key: str
    label: str


@dataclass(frozen=True)
class ContextFieldConfig:
    key: str
    label: str
    token: str
    placement: str = "after_prompt"
    canonical_tag: str | None = None

    def normalized_token(self, value: str) -> str:
        lowered = value.strip().lower()
        if not lowered:
            return "unknown"
        collapsed = re.sub(r"\s+", "_", lowered)
        sanitized = re.sub(r"[^a-z0-9_:-]", "", collapsed)
        sanitized = sanitized.strip("_:-")
        return sanitized or "unknown"

    def should_prepend(self) -> bool:
        return self.placement == "before_prompt"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    prompt: DatasetFieldConfig
    response: DatasetFieldConfig
    context_fields: Tuple[ContextFieldConfig, ...]
    source_path: Path | None = None

    @staticmethod
    def default() -> DatasetConfig:
        return DatasetConfig(
            name="default",
            prompt=DatasetFieldConfig("prompt", "|USER|"),
            response=DatasetFieldConfig("response", "|RESPONSE|"),
            context_fields=tuple(),
            source_path=None,
        )

    def extract_prompt(self, payload: dict[str, Any]) -> str:
        return _stringify(payload.get(self.prompt.key, "")).strip()

    def extract_response(self, payload: dict[str, Any]) -> str:
        return _stringify(payload.get(self.response.key, "")).strip()

    def iter_context_values(self, payload: dict[str, Any]) -> Iterator[tuple[ContextFieldConfig, str]]:
        for field in self.context_fields:
            raw_value = _stringify(payload.get(field.key, "")).strip()
            if not raw_value:
                continue
            yield field, raw_value

    def context_map(self, payload: dict[str, Any]) -> dict[str, str]:
        return {field.token: value for field, value in self.iter_context_values(payload)}

    def partition_context_values(
        self, context_values: Sequence[tuple[ContextFieldConfig, str]]
    ) -> tuple[list[tuple[ContextFieldConfig, str]], list[tuple[ContextFieldConfig, str]]]:
        preface: list[tuple[ContextFieldConfig, str]] = []
        post: list[tuple[ContextFieldConfig, str]] = []
        for field, value in context_values:
            if field.should_prepend():
                preface.append((field, value))
            else:
                post.append((field, value))
        return preface, post

    def prompt_tag_labels(self) -> Tuple[str, ...]:
        labels: list[str] = []
        seen: set[str] = set()
        for candidate in (
            _normalize_pipe_tag(self.prompt.label),
            _normalize_pipe_tag(self.response.label),
        ):
            if candidate and candidate not in seen:
                labels.append(candidate)
                seen.add(candidate)
        for field in self.context_fields:
            for candidate in (
                _normalize_pipe_tag(field.label),
                _normalize_pipe_tag(field.canonical_tag),
            ):
                if candidate and candidate not in seen:
                    labels.append(candidate)
                    seen.add(candidate)
        return tuple(labels)

    def prompt_tag_tokens(self) -> Tuple[str, ...]:
        return tuple(f"{label}:" for label in self.prompt_tag_labels())

    def compose_prompt(
        self,
        payload: dict[str, Any],
        *,
        raw_prompt: str | None = None,
        context_values: Sequence[tuple[ContextFieldConfig, str]] | None = None,
    ) -> str:
        prompt_value = _stringify(raw_prompt) if raw_prompt is not None else self.extract_prompt(payload)
        prompt_value = prompt_value.strip()
        ctx_values = (
            list(context_values)
            if context_values is not None
            else list(self.iter_context_values(payload))
        )
        lines: list[str] = []
        for field, ctx_val in ctx_values:
            if not field.should_prepend():
                continue
            ctx_line = ctx_val.strip()
            if not ctx_line:
                continue
            if field.label:
                lines.append(f"{field.label}: {ctx_line}")
            else:
                lines.append(ctx_line)
        if prompt_value:
            prompt_line = f"{self.prompt.label}: {prompt_value}" if self.prompt.label else prompt_value
            lines.append(prompt_line.strip())
        return "\n".join(lines).strip()


def infer_config_path(dataset_path: Path) -> Path:
    parent = dataset_path.parent if dataset_path.parent else Path(".")
    stem = dataset_path.stem or dataset_path.name
    return parent / f"{stem}.config.json"


def load_dataset_config(dataset_path: str | Path | None, *, override: str | None = None) -> DatasetConfig:
    path_obj = Path(dataset_path) if dataset_path else None
    override_path = Path(override) if override else None
    env_override = os.environ.get("DBSLM_DATASET_CONFIG_PATH")
    env_path = Path(env_override) if env_override else None

    candidates: List[Path] = []
    for candidate in (override_path, env_path):
        if candidate and candidate.exists():
            candidates.append(candidate)
    if path_obj:
        inferred = infer_config_path(path_obj)
        if inferred.exists():
            candidates.append(inferred)
    for candidate in candidates:
        try:
            return _parse_dataset_config(candidate)
        except Exception:
            continue
    return DatasetConfig.default()


def _parse_dataset_config(path: Path) -> DatasetConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = _stringify(payload.get("name") or path.stem or "dataset")
    prompt_key = _stringify(payload.get("prompt_field") or "prompt") or "prompt"
    prompt_label = _stringify(payload.get("prompt_label") or "|USER|") or "|USER|"
    response_key = _stringify(payload.get("response_field") or "response") or "response"
    response_label = _stringify(payload.get("response_label") or "|RESPONSE|") or "|RESPONSE|"
    prompt_cfg = DatasetFieldConfig(prompt_key, prompt_label)
    response_cfg = DatasetFieldConfig(response_key, response_label)

    context_fields: List[ContextFieldConfig] = []
    for entry in payload.get("context_fields", []):
        if not isinstance(entry, dict):
            continue
        field_key = _stringify(entry.get("field")).strip()
        if not field_key:
            continue
        token_name = _stringify(entry.get("token_name") or field_key).strip() or field_key
        label = _stringify(entry.get("label") or token_name).strip() or token_name
        placement = _normalize_context_placement(entry.get("placement"))
        canonical_tag = _normalize_pipe_tag(entry.get("canonical_tag"))
        context_fields.append(
            ContextFieldConfig(field_key, label, token_name, placement, canonical_tag)
        )

    return DatasetConfig(
        name=name,
        prompt=prompt_cfg,
        response=response_cfg,
        context_fields=tuple(context_fields),
        source_path=path,
    )


def _normalize_context_placement(raw: Any) -> str:
    normalized = _stringify(raw).strip().lower()
    if not normalized:
        return "after_prompt"
    if normalized in {"before", "pre", "preface", "before_prompt", "prompt_preface"}:
        return "before_prompt"
    return "after_prompt"


def _normalize_pipe_tag(value: Any) -> str | None:
    candidate = _stringify(value).strip()
    if (
        candidate
        and candidate.startswith("|")
        and candidate.endswith("|")
        and len(candidate) > 2
    ):
        return candidate
    return None
