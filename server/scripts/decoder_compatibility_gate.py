#!/usr/bin/env python3
"""Fail-closed offline acceptance gate for the GigaAM CTC decoder spike.

The gate compares two immutable benchmark JSON reports produced from the same
real frozen manifest:

* the current GigaAM greedy decoder;
* an NGPU-LM candidate run with an explicitly reported LM weight of zero.

No audio or transcript text is read.  At any failed check the decision selects
the separately pinned Linux Flashlight+KenLM fallback and exits with status 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

GATE_SCHEMA_VERSION = "voicemed.decoder-compatibility-gate.v1"
BENCHMARK_SCHEMA_VERSION = "voicemed.asr-benchmark.v2"
FALLBACK_LOCK_SCHEMA_VERSION = "voicemed.decoder-fallback-lock.v1"
FROZEN_DATASET_KIND = "real_frozen_test"
MAX_P95_RTF = 0.5
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _at_path(value: dict[str, Any], *parts: str) -> Any:
    current: Any = value
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def decoder_metadata(report: dict[str, Any]) -> dict[str, Any]:
    candidates = (
        _at_path(report, "server", "configuration", "ctc_decoder"),
        _at_path(report, "server", "configuration", "ctcDecoder"),
        _at_path(report, "server", "model", "ctcDecoder"),
        _at_path(
            report,
            "server",
            "runtime",
            "configuration",
            "ctcDecoder",
        ),
    )
    return next(
        (candidate for candidate in candidates if isinstance(candidate, dict)),
        {},
    )


def _flatten_metadata_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        flattened: list[str] = []
        for nested in value.values():
            flattened.extend(_flatten_metadata_strings(nested))
        return flattened
    if isinstance(value, list):
        flattened = []
        for nested in value:
            flattened.extend(_flatten_metadata_strings(nested))
        return flattened
    return []


def reported_lm_weight(decoder: dict[str, Any]) -> float | None:
    language_model = decoder.get("languageModel")
    containers = [
        decoder,
        language_model if isinstance(language_model, dict) else {},
        (
            decoder.get("implementationMetadata")
            if isinstance(decoder.get("implementationMetadata"), dict)
            else {}
        ),
    ]
    for container in containers:
        for name in ("lm_weight", "lmWeight", "weight", "alpha"):
            value = container.get(name)
            if _is_number(value):
                return float(value)
    return None


def is_ngpu_lm_decoder(decoder: dict[str, Any]) -> bool:
    if decoder.get("mode") != "beam" or decoder.get("active") is not True:
        return False
    identity = " ".join(_flatten_metadata_strings(decoder)).casefold()
    return "ngpu" in identity


def validate_fallback_lock(
    lock: dict[str, Any],
    lock_path: Path,
) -> dict[str, Any]:
    if lock.get("schema_version") != FALLBACK_LOCK_SCHEMA_VERSION:
        raise ValueError(
            "fallback lock schema_version must be "
            f"{FALLBACK_LOCK_SCHEMA_VERSION}"
        )
    if lock.get("backend") != "flashlight+kenlm":
        raise ValueError("fallback lock backend must be flashlight+kenlm")
    container_image = lock.get("container_image")
    if not isinstance(container_image, str) or not IMAGE_DIGEST_RE.search(
        container_image
    ):
        raise ValueError(
            "fallback lock container_image must end with @sha256:<64 hex>"
        )
    for field in ("decoder_artifact_sha256", "kenlm_runtime_sha256"):
        value = lock.get(field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValueError(f"fallback lock {field} must be a SHA-256")
    return {
        "backend": "flashlight+kenlm",
        "runtime": "linux-cuda-container",
        "container_image": container_image,
        "decoder_artifact_sha256": lock["decoder_artifact_sha256"],
        "kenlm_runtime_sha256": lock["kenlm_runtime_sha256"],
        "lock_file": lock_path.name,
        "lock_sha256": sha256_file(lock_path),
    }


def _case_inventory(
    report: dict[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], bool]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        return [], {}, False
    ordered_ids: list[str] = []
    inventory: dict[str, dict[str, Any]] = {}
    valid = True
    for case in cases:
        if not isinstance(case, dict):
            valid = False
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in inventory:
            valid = False
            continue
        ordered_ids.append(case_id)
        inventory[case_id] = case
    return ordered_ids, inventory, valid


def _complete_word_evidence(case: dict[str, Any]) -> bool:
    evidence = case.get("word_evidence")
    if not isinstance(evidence, dict):
        return False
    count = evidence.get("word_count")
    return bool(
        isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        and evidence.get("timestamped_words") == count
        and evidence.get("acoustic_confidence_words") == count
        and evidence.get("timestamps_valid") is True
        and evidence.get("timestamps_complete") is True
        and evidence.get("acoustic_confidence_valid") is True
        and evidence.get("acoustic_confidence_complete") is True
        and evidence.get("utterance_acoustic_confidence") is True
        and evidence.get("all_repeats_complete") is True
        and evidence.get("repeat_deterministic") is True
    )


def evaluate_decoder_compatibility(
    greedy: dict[str, Any],
    zero_lm: dict[str, Any],
    fallback: dict[str, Any],
    *,
    greedy_report_sha256: str | None = None,
    zero_lm_report_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate immutable report objects and return an auditable decision."""

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, required: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "required": required,
            }
        )

    add(
        "benchmark_schema",
        greedy.get("schema_version") == BENCHMARK_SCHEMA_VERSION
        and zero_lm.get("schema_version") == BENCHMARK_SCHEMA_VERSION,
        {
            "greedy": greedy.get("schema_version"),
            "zero_lm": zero_lm.get("schema_version"),
        },
        BENCHMARK_SCHEMA_VERSION,
    )
    greedy_manifest = greedy.get("manifest")
    candidate_manifest = zero_lm.get("manifest")
    greedy_manifest = (
        greedy_manifest if isinstance(greedy_manifest, dict) else {}
    )
    candidate_manifest = (
        candidate_manifest if isinstance(candidate_manifest, dict) else {}
    )
    same_manifest = (
        isinstance(greedy_manifest.get("sha256"), str)
        and bool(greedy_manifest.get("sha256"))
        and greedy_manifest.get("sha256") == candidate_manifest.get("sha256")
        and greedy_manifest.get("dataset_kind") == FROZEN_DATASET_KIND
        and candidate_manifest.get("dataset_kind") == FROZEN_DATASET_KIND
        and greedy.get("limited") is False
        and zero_lm.get("limited") is False
        and greedy.get("purpose") == "baseline"
        and zero_lm.get("purpose") == "candidate"
    )
    add(
        "same_real_frozen_manifest",
        same_manifest,
        {
            "greedy_sha256": greedy_manifest.get("sha256"),
            "zero_lm_sha256": candidate_manifest.get("sha256"),
            "greedy_dataset_kind": greedy_manifest.get("dataset_kind"),
            "zero_lm_dataset_kind": candidate_manifest.get("dataset_kind"),
            "greedy_purpose": greedy.get("purpose"),
            "zero_lm_purpose": zero_lm.get("purpose"),
            "greedy_limited": greedy.get("limited"),
            "zero_lm_limited": zero_lm.get("limited"),
        },
        {
            "same_nonempty_sha256": True,
            "dataset_kind": FROZEN_DATASET_KIND,
            "greedy_purpose": "baseline",
            "zero_lm_purpose": "candidate",
            "limited": False,
        },
    )

    greedy_ids, greedy_cases, greedy_inventory_valid = _case_inventory(greedy)
    candidate_ids, candidate_cases, candidate_inventory_valid = (
        _case_inventory(zero_lm)
    )
    same_case_bindings = bool(
        greedy_inventory_valid
        and candidate_inventory_valid
        and greedy_ids
        and greedy_ids == candidate_ids
        and all(
            greedy_cases[case_id].get("audio_sha256")
            == candidate_cases[case_id].get("audio_sha256")
            and bool(greedy_cases[case_id].get("audio_sha256"))
            and greedy_cases[case_id].get("reference_sha256")
            == candidate_cases[case_id].get("reference_sha256")
            and bool(greedy_cases[case_id].get("reference_sha256"))
            for case_id in greedy_ids
        )
    )
    add(
        "same_case_inventory",
        same_case_bindings,
        {
            "greedy_ids": greedy_ids,
            "zero_lm_ids": candidate_ids,
            "bindings_match": same_case_bindings,
        },
        "same ordered IDs, audio SHA-256 and reference SHA-256",
    )

    decoder = decoder_metadata(zero_lm)
    weight = reported_lm_weight(decoder)
    context_scope = zero_lm.get("context_scope")
    zero_weight_ngpu = bool(
        is_ngpu_lm_decoder(decoder)
        and weight == 0.0
        and context_scope in (None, "")
    )
    add(
        "ngpu_lm_zero_weight_identity",
        zero_weight_ngpu,
        {
            "mode": decoder.get("mode"),
            "active": decoder.get("active"),
            "implementation": decoder.get("implementation"),
            "reported_lm_weight": weight,
            "context_scope": context_scope,
        },
        {
            "backend_identity_contains": "ngpu",
            "mode": "beam",
            "active": True,
            "reported_lm_weight": 0.0,
            "context_scope": None,
        },
    )

    equal_predictions = bool(
        same_case_bindings
        and all(
            bool(greedy_cases[case_id].get("prediction_sha256"))
            and greedy_cases[case_id].get("prediction_sha256")
            == candidate_cases[case_id].get("prediction_sha256")
            for case_id in greedy_ids
        )
    )
    add(
        "zero_weight_predictions_equal_greedy",
        equal_predictions,
        {
            "matching_cases": (
                sum(
                    greedy_cases[case_id].get("prediction_sha256")
                    == candidate_cases[case_id].get("prediction_sha256")
                    for case_id in greedy_ids
                )
                if same_case_bindings
                else 0
            ),
            "total_cases": len(greedy_ids),
        },
        "exact prediction SHA-256 equality for every case",
    )

    evidence_preserved = bool(
        same_case_bindings
        and all(
            _complete_word_evidence(greedy_cases[case_id])
            and _complete_word_evidence(candidate_cases[case_id])
            and _at_path(
                greedy_cases[case_id], "word_evidence", "word_count"
            )
            == _at_path(
                candidate_cases[case_id], "word_evidence", "word_count"
            )
            for case_id in greedy_ids
        )
    )
    add(
        "timestamps_and_acoustic_confidence_preserved",
        evidence_preserved,
        {
            "complete_greedy_cases": sum(
                _complete_word_evidence(case)
                for case in greedy_cases.values()
            ),
            "complete_zero_lm_cases": sum(
                _complete_word_evidence(case)
                for case in candidate_cases.values()
            ),
            "total_cases": len(greedy_ids),
        },
        (
            "all repeated responses have deterministic, valid timestamps and "
            "word/utterance acoustic confidence; word counts equal greedy"
        ),
    )

    candidate_repeat = zero_lm.get("repeat")
    deterministic_cases = _at_path(
        zero_lm, "metrics", "deterministic_cases"
    )
    candidate_deterministic = bool(
        isinstance(candidate_repeat, int)
        and not isinstance(candidate_repeat, bool)
        and candidate_repeat >= 2
        and candidate_ids
        and deterministic_cases == len(candidate_ids)
        and all(
            candidate_cases[case_id].get("deterministic") is True
            and _at_path(
                candidate_cases[case_id],
                "word_evidence",
                "repeat_deterministic",
            )
            is True
            for case_id in candidate_ids
        )
    )
    add(
        "candidate_deterministic",
        candidate_deterministic,
        {
            "repeat": candidate_repeat,
            "deterministic_cases": deterministic_cases,
            "total_cases": len(candidate_ids),
        },
        "repeat>=2 and deterministic text/timing/acoustic evidence for every case",
    )

    p95_rtf = _at_path(zero_lm, "metrics", "rtf", "p95")
    rtf_passed = bool(_is_number(p95_rtf) and 0 <= float(p95_rtf) <= MAX_P95_RTF)
    add(
        "candidate_p95_rtf",
        rtf_passed,
        p95_rtf,
        {"minimum": 0.0, "maximum": MAX_P95_RTF},
    )

    failed = [check["name"] for check in checks if not check["passed"]]
    ngpu_eligible = not failed
    selected = (
        {
            "backend": "ngpu-lm",
            "runtime_id": _at_path(zero_lm, "server", "runtime_id"),
            "benchmark_report_sha256": zero_lm_report_sha256,
            "reason": "all compatibility checks passed",
        }
        if ngpu_eligible
        else {
            **fallback,
            "reason": "NGPU-LM compatibility checks failed",
        }
    )
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "created_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "eligible": ngpu_eligible,
        "failed": failed,
        "checks": checks,
        "inputs": {
            "greedy_report_sha256": greedy_report_sha256,
            "zero_lm_report_sha256": zero_lm_report_sha256,
            "manifest_sha256": greedy_manifest.get("sha256"),
        },
        "selected": selected,
        "fallback": fallback,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--greedy-report", required=True, type=Path)
    parser.add_argument("--zero-lm-report", required=True, type=Path)
    parser.add_argument(
        "--fallback-lock",
        required=True,
        type=Path,
        help="lock for the Linux CUDA Flashlight+KenLM fallback",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    greedy_path = args.greedy_report.resolve()
    zero_lm_path = args.zero_lm_report.resolve()
    fallback_path = args.fallback_lock.resolve()
    fallback = validate_fallback_lock(
        read_json_object(fallback_path, "fallback lock"),
        fallback_path,
    )
    decision = evaluate_decoder_compatibility(
        read_json_object(greedy_path, "greedy report"),
        read_json_object(zero_lm_path, "zero-LM report"),
        fallback,
        greedy_report_sha256=sha256_file(greedy_path),
        zero_lm_report_sha256=sha256_file(zero_lm_path),
    )
    write_json_atomic(args.output.resolve(), decision)
    print(
        json.dumps(
            {
                "eligible": decision["eligible"],
                "failed": decision["failed"],
                "selected": decision["selected"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if decision["eligible"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"decoder compatibility gate failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
