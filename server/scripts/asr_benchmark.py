#!/usr/bin/env python3
"""Auditable ASR benchmark for validation, frozen tests, and synthetic regressions.

This runner intentionally refuses to mix real held-out audio with synthetic TTS
and refuses to use a synthetic/browser/partial manifest for a release decision.
"""

from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from validate_asr_manifest import (
    ENTITY_TYPES,
    audio_duration_seconds,
    parse_entities,
)

BENCHMARK_KINDS = {
    "real_validation",
    "real_frozen_test",
    "synthetic_regression",
}
REAL_BENCHMARK_PURPOSES = {"baseline", "candidate", "release"}
REQUIRED_COLUMNS = {
    "id",
    "path",
    "duration",
    "transcription",
    "dataset_kind",
    "split",
    "audio_sha256",
    "source_recording_id",
    "transcript_source",
    "entities_json",
}
WORD_RE = re.compile(r"[0-9a-zа-яё]+(?:[.,][0-9]+)?", re.IGNORECASE)
NUMBER_UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:мм|см|мл|л|hu|ед(?:/л)?|мг|г|%)"
    r"(?=$|\s|[.,;:])",
    re.IGNORECASE,
)
NEGATIONS = {"нет", "не", "без", "отсутствует", "отсутствуют"}
LATERALITY = {"справа", "слева", "правый", "правая", "левый", "левая"}
CONTRAST = {"контраст", "контрастом", "контрастирования", "без контраста"}
SAFETY_ENTITY_TYPES = ("number_unit", "negation", "laterality", "contrast")
RELEASE_THRESHOLDS = {
    "medical_wer_relative_improvement": 0.25,
    "overall_wer_absolute_regression": 0.01,
    "critical_entity_recall": 0.98,
    "entity_exact_accuracy": 0.99,
    "p95_rtf": 0.5,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def summarize_word_evidence(response: dict[str, Any]) -> dict[str, Any]:
    """Return a non-PHI summary of timestamp/acoustic word evidence.

    Text and individual word values are intentionally omitted.  The two
    signatures cover only numeric timing/acoustic evidence and may therefore be
    used to prove repeat determinism without copying dictated content into the
    benchmark report.
    """

    raw_words_value = response.get("words")
    words = raw_words_value if isinstance(raw_words_value, list) else []
    timestamped_words = 0
    acoustic_confidence_words = 0
    timestamps_valid = bool(words)
    acoustic_confidence_valid = bool(words)
    previous_start = -math.inf
    timing_signature: list[tuple[float, float]] = []
    acoustic_signature: list[tuple[float, float, str]] = []
    score_types: set[str] = set()

    for item in words:
        if not isinstance(item, dict):
            timestamps_valid = False
            acoustic_confidence_valid = False
            continue
        start = item.get("start")
        end = item.get("end")
        if _finite_number(start) and _finite_number(end):
            numeric_start = float(start)
            numeric_end = float(end)
            timestamped_words += 1
            timing_signature.append((numeric_start, numeric_end))
            if (
                numeric_start < 0
                or numeric_end < numeric_start
                or numeric_start < previous_start
            ):
                timestamps_valid = False
            previous_start = numeric_start
        else:
            timestamps_valid = False

        avg_logprob = item.get("avg_logprob")
        confidence = item.get("confidence")
        score_type = str(item.get("score_type") or "").strip()
        normalized_score_type = score_type.casefold()
        is_acoustic_score = bool(score_type) and not any(
            marker in normalized_score_type
            for marker in ("fused", "language_model", "lm_score")
        )
        if (
            _finite_number(avg_logprob)
            and _finite_number(confidence)
            and 0.0 <= float(confidence) <= 1.0
            and is_acoustic_score
        ):
            acoustic_confidence_words += 1
            score_types.add(score_type)
            acoustic_signature.append(
                (float(avg_logprob), float(confidence), score_type)
            )
        else:
            acoustic_confidence_valid = False

    confidence_value = response.get("confidence")
    utterance_confidence = (
        confidence_value if isinstance(confidence_value, dict) else {}
    )
    confidence_method = str(utterance_confidence.get("method") or "").strip()
    normalized_method = confidence_method.casefold()
    utterance_acoustic_confidence = bool(
        utterance_confidence.get("available")
        and _finite_number(utterance_confidence.get("avg_logprob"))
        and confidence_method
        and not any(
            marker in normalized_method
            for marker in ("fused", "language_model", "lm_score")
        )
    )

    word_count = len(words)
    timestamps_complete = bool(
        word_count
        and timestamped_words == word_count
        and timestamps_valid
    )
    acoustic_confidence_complete = bool(
        word_count
        and acoustic_confidence_words == word_count
        and acoustic_confidence_valid
        and utterance_acoustic_confidence
    )
    return {
        "available": bool(word_count),
        "word_count": word_count,
        "timestamped_words": timestamped_words,
        "timestamps_valid": timestamps_valid,
        "timestamps_complete": timestamps_complete,
        "acoustic_confidence_words": acoustic_confidence_words,
        "acoustic_confidence_valid": acoustic_confidence_valid,
        "acoustic_confidence_complete": acoustic_confidence_complete,
        "utterance_acoustic_confidence": utterance_acoustic_confidence,
        "score_types": sorted(score_types),
        "timing_signature_sha256": sha256_text(
            json.dumps(timing_signature, separators=(",", ":"))
        ),
        "acoustic_signature_sha256": sha256_text(
            json.dumps(acoustic_signature, separators=(",", ":"))
        ),
    }


def normalized_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return [token.replace(",", ".") for token in WORD_RE.findall(normalized)]


def raw_words(text: str) -> list[str]:
    return unicodedata.normalize("NFC", text).split()


def raw_chars(text: str) -> list[str]:
    return list(unicodedata.normalize("NFC", text))


def normalized_chars(text: str) -> list[str]:
    return list(" ".join(normalized_words(text)))


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: list[str], hypothesis: list[str]) -> tuple[int, int]:
    return edit_distance(reference, hypothesis), max(1, len(reference))


def extract_number_units(text: str) -> collections.Counter[str]:
    normalized = (
        unicodedata.normalize("NFKC", text)
        .casefold()
        .replace("ё", "е")
        .replace(",", ".")
    )
    return collections.Counter(
        re.sub(r"\s+", "", match.group(0)).replace(",", ".")
        for match in NUMBER_UNIT_RE.finditer(normalized)
    )


def exact_counter_recall(
    reference: collections.Counter[str], hypothesis: collections.Counter[str]
) -> tuple[int, int]:
    true_positive = sum((reference & hypothesis).values())
    return true_positive, sum(reference.values())


def phrase_present(phrase: str, text: str) -> bool:
    phrase_tokens = normalized_words(phrase)
    text_tokens = normalized_words(text)
    if not phrase_tokens:
        return False
    width = len(phrase_tokens)
    return any(
        text_tokens[index : index + width] == phrase_tokens
        for index in range(0, len(text_tokens) - width + 1)
    )


def phrase_error(reference_phrase: str, hypothesis: str) -> tuple[int, int]:
    """Minimum local word edit distance for an annotated medical phrase."""
    reference = normalized_words(reference_phrase)
    hypothesis_words = normalized_words(hypothesis)
    if not reference:
        return 0, 0
    best = len(reference)
    minimum_width = max(1, len(reference) - 2)
    maximum_width = min(len(hypothesis_words), len(reference) + 2)
    for width in range(minimum_width, maximum_width + 1):
        for index in range(0, len(hypothesis_words) - width + 1):
            best = min(
                best,
                edit_distance(
                    reference,
                    hypothesis_words[index : index + width],
                ),
            )
    return best, len(reference)


def alignment_operations(
    reference: list[str],
    hypothesis: list[str],
) -> list[tuple[str, int, int | None]]:
    """Return deterministic Levenshtein operations in forward token order.

    Insertions carry the reference boundary at which they occur. This lets
    span scoring include words inserted *inside* a multi-word entity instead
    of accidentally calling the entity exact.
    """

    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    for ref_index in range(rows):
        costs[ref_index][0] = ref_index
    for hyp_index in range(columns):
        costs[0][hyp_index] = hyp_index
    for ref_index in range(1, rows):
        for hyp_index in range(1, columns):
            costs[ref_index][hyp_index] = min(
                costs[ref_index - 1][hyp_index] + 1,
                costs[ref_index][hyp_index - 1] + 1,
                costs[ref_index - 1][hyp_index - 1]
                + (reference[ref_index - 1] != hypothesis[hyp_index - 1]),
            )

    reversed_operations: list[tuple[str, int, int | None]] = []
    ref_index = len(reference)
    hyp_index = len(hypothesis)
    while ref_index > 0 or hyp_index > 0:
        if ref_index > 0 and hyp_index > 0:
            substitution_cost = (
                reference[ref_index - 1] != hypothesis[hyp_index - 1]
            )
            if (
                costs[ref_index][hyp_index]
                == costs[ref_index - 1][hyp_index - 1] + substitution_cost
            ):
                reversed_operations.append(
                    ("align", ref_index - 1, hyp_index - 1)
                )
                ref_index -= 1
                hyp_index -= 1
                continue
        if (
            ref_index > 0
            and costs[ref_index][hyp_index]
            == costs[ref_index - 1][hyp_index] + 1
        ):
            reversed_operations.append(("delete", ref_index - 1, None))
            ref_index -= 1
            continue
        reversed_operations.append(("insert", ref_index, hyp_index - 1))
        hyp_index -= 1
    return list(reversed(reversed_operations))


def align_reference_tokens(
    reference: list[str],
    hypothesis: list[str],
) -> list[str | None]:
    """Map each reference token to its globally aligned hypothesis token."""

    mapping: list[str | None] = [None] * len(reference)
    for operation, ref_index, hyp_index in alignment_operations(
        reference,
        hypothesis,
    ):
        if operation == "align" and hyp_index is not None:
            mapping[ref_index] = hypothesis[hyp_index]
    return mapping


def score_annotated_entities(
    reference: str,
    hypothesis: str,
    entities: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    reference_tokens = normalized_words(reference)
    hypothesis_tokens = normalized_words(hypothesis)
    operations = alignment_operations(reference_tokens, hypothesis_tokens)
    scores = {
        entity_type: {
            "correct": 0,
            "total": 0,
            "word_edits": 0,
            "word_count": 0,
        }
        for entity_type in ENTITY_TYPES
    }
    for entity in entities:
        entity_type = str(entity["type"])
        start_word = int(entity["start_word"])
        end_word = int(entity["end_word"])
        expected = reference_tokens[start_word:end_word]
        observed: list[str] = []
        for operation, ref_index, hyp_index in operations:
            if operation == "align" and start_word <= ref_index < end_word:
                assert hyp_index is not None
                observed.append(hypothesis_tokens[hyp_index])
            elif (
                operation == "insert"
                and start_word < ref_index < end_word
            ):
                assert hyp_index is not None
                observed.append(hypothesis_tokens[hyp_index])
        bucket = scores[entity_type]
        bucket["correct"] += int(expected == observed)
        bucket["total"] += 1
        bucket["word_edits"] += edit_distance(expected, observed)
        bucket["word_count"] += len(expected)
    return scores


def parse_terms(value: str) -> list[str]:
    return [term.strip() for term in value.split("|") if term.strip()]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("manifest has no cases")
    return rows


def validate_benchmark_manifest(
    manifest_path: Path, rows: list[dict[str, str]], purpose: str
) -> list[dict[str, Any]]:
    kinds = {row["dataset_kind"].strip() for row in rows}
    if not kinds <= BENCHMARK_KINDS:
        raise ValueError(f"unsupported benchmark dataset_kind values: {sorted(kinds)}")
    if len(kinds) != 1:
        raise ValueError("real frozen and synthetic regression rows must never be mixed")
    kind = next(iter(kinds))
    if purpose in REAL_BENCHMARK_PURPOSES and kind != "real_frozen_test":
        raise ValueError(
            f"{purpose} benchmarks require dataset_kind=real_frozen_test"
        )
    if purpose == "validation" and kind != "real_validation":
        raise ValueError(
            "validation benchmarks require dataset_kind=real_validation"
        )
    if purpose == "regression" and kind != "synthetic_regression":
        raise ValueError("regression benchmarks require dataset_kind=synthetic_regression")

    resolved: list[dict[str, Any]] = []
    ids: set[str] = set()
    audio_hashes: set[str] = set()
    for line, row in enumerate(rows, start=2):
        case_id = row["id"].strip()
        if not case_id or case_id in ids:
            raise ValueError(f"line {line}: missing or duplicate id")
        ids.add(case_id)
        expected_split = {
            "real_validation": "validation",
            "real_frozen_test": "test",
            "synthetic_regression": "regression",
        }[kind]
        if row["split"].strip() != expected_split:
            raise ValueError(f"line {line}: split must be {expected_split}")
        if not row["transcription"].strip():
            raise ValueError(f"line {line}: empty verbatim transcription")
        transcript_source = row["transcript_source"].strip()
        if transcript_source == "browser_asr":
            raise ValueError(
                f"line {line}: browser_asr transcripts are prohibited in benchmarks"
            )
        expected_source = (
            "human_verbatim"
            if kind in {"real_validation", "real_frozen_test"}
            else "tts_script"
        )
        if transcript_source != expected_source:
            raise ValueError(
                f"line {line}: {kind} requires "
                f"transcript_source={expected_source}"
            )
        if not row["source_recording_id"].strip():
            raise ValueError(f"line {line}: source_recording_id is required")
        entities = parse_entities(
            row["entities_json"].strip(),
            row["transcription"],
            line,
        )
        try:
            duration = float(row["duration"])
        except ValueError as exc:
            raise ValueError(f"line {line}: invalid duration") from exc
        if duration <= 0:
            raise ValueError(f"line {line}: duration must be positive")

        audio_path = (manifest_path.parent / row["path"]).resolve()
        if not audio_path.is_file():
            raise ValueError(f"line {line}: audio does not exist: {audio_path}")
        actual_hash = sha256_file(audio_path)
        declared_hash = row["audio_sha256"].strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
            raise ValueError(f"line {line}: audio_sha256 is required")
        if actual_hash != declared_hash:
            raise ValueError(f"line {line}: audio_sha256 mismatch")
        if actual_hash in audio_hashes:
            raise ValueError(f"line {line}: duplicate audio content")
        audio_hashes.add(actual_hash)
        actual_duration = audio_duration_seconds(audio_path)
        duration_tolerance = max(0.05, actual_duration * 0.01)
        if abs(actual_duration - duration) > duration_tolerance:
            raise ValueError(
                f"line {line}: duration mismatch: declared={duration:.3f}s, "
                f"actual={actual_duration:.3f}s"
            )

        if kind in {"real_validation", "real_frozen_test"}:
            required_fields = [
                "speaker_id",
                "patient_id",
                "study_id",
                "reviewed_by",
            ]
            if kind == "real_frozen_test":
                required_fields.append("frozen_at")
            for required in required_fields:
                if not row.get(required, "").strip():
                    raise ValueError(
                        f"line {line}: {required} is required for reviewed real data"
                    )

        resolved.append(
            {
                **row,
                "id": case_id,
                "audio_path": audio_path,
                "duration_seconds": actual_duration,
                "declared_duration_seconds": duration,
                "audio_sha256": actual_hash,
                "_entities": entities,
            }
        )
    return resolved


def http_json(
    url: str, payload: dict[str, Any] | None, timeout: float
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def score_case(
    reference: str,
    hypothesis: str,
    terms: list[str],
    entities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw_edits, raw_count = error_rate(raw_words(reference), raw_words(hypothesis))
    raw_char_edits, raw_char_count = error_rate(
        raw_chars(reference),
        raw_chars(hypothesis),
    )
    word_edits, word_count = error_rate(
        normalized_words(reference), normalized_words(hypothesis)
    )
    char_edits, char_count = error_rate(
        normalized_chars(reference), normalized_chars(hypothesis)
    )
    ref_numbers = extract_number_units(reference)
    hyp_numbers = extract_number_units(hypothesis)
    number_tp, number_total = exact_counter_recall(ref_numbers, hyp_numbers)
    medical_phrase_scores = [phrase_error(term, hypothesis) for term in terms]
    entity_scores = score_annotated_entities(
        reference,
        hypothesis,
        entities or [],
    )
    ref_words = normalized_words(reference)
    hyp_words = normalized_words(hypothesis)
    critical: dict[str, dict[str, int]] = {}
    for name, vocabulary in (
        ("negation", NEGATIONS),
        ("laterality", LATERALITY),
        ("contrast", CONTRAST),
    ):
        category_tokens = set(normalized_words(" ".join(vocabulary)))
        expected = collections.Counter(
            token for token in ref_words if token in category_tokens
        )
        observed = collections.Counter(
            token for token in hyp_words if token in category_tokens
        )
        correct = sum((expected & observed).values())
        critical[name] = {
            "correct": correct,
            "reference_total": sum(expected.values()),
            "hypothesis_total": sum(observed.values()),
            "exact": int(expected == observed),
        }
    return {
        "raw_word_edits": raw_edits,
        "raw_word_count": raw_count,
        "raw_char_edits": raw_char_edits,
        "raw_char_count": raw_char_count,
        "word_edits": word_edits,
        "word_count": word_count,
        "char_edits": char_edits,
        "char_count": char_count,
        "entities": entity_scores,
        "diagnostics": {
            "legacy_phrase_correct": sum(
                phrase_present(term, hypothesis) for term in terms
            ),
            "legacy_phrase_total": len(terms),
            "legacy_phrase_word_edits": sum(
                score[0] for score in medical_phrase_scores
            ),
            "legacy_phrase_word_count": sum(
                score[1] for score in medical_phrase_scores
            ),
            "number_unit_bag_correct": number_tp,
            "number_unit_bag_reference_total": number_total,
            "number_unit_bag_hypothesis_total": sum(hyp_numbers.values()),
            "number_unit_bag_exact": int(ref_numbers == hyp_numbers),
            "critical_token_bags": critical,
            "promotion_eligible": False,
            "reason": "bag diagnostics do not preserve entity association",
        },
    }


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def aggregate(
    case_results: list[dict[str, Any]], dataset_kind: str
) -> dict[str, Any]:
    def total(name: str) -> int:
        return sum(int(case["scores"][name]) for case in case_results)

    def entity_total(entity_type: str, name: str) -> int:
        return sum(
            int(case["scores"]["entities"][entity_type][name])
            for case in case_results
        )

    rtf_values = [
        float(case["inference_seconds"]) / float(case["duration_seconds"])
        for case in case_results
        if case["duration_seconds"] > 0
    ]
    wall_values = [
        float(wall_time)
        for case in case_results
        for wall_time in case["wall_seconds"]
    ]
    entity_metrics: dict[str, Any] = {}
    for entity_type in sorted(ENTITY_TYPES):
        correct = entity_total(entity_type, "correct")
        expected = entity_total(entity_type, "total")
        word_edits = entity_total(entity_type, "word_edits")
        word_count = entity_total(entity_type, "word_count")
        entity_metrics[entity_type] = {
            "available": expected > 0,
            "correct": correct,
            "eligible_annotations": expected,
            "exact_accuracy": ratio(correct, expected),
            "word_error_rate": ratio(word_edits, word_count),
            "word_edits": word_edits,
            "word_count": word_count,
            "unavailable_reason": (
                None
                if expected > 0
                else "no reviewed span annotations of this entity type"
            ),
        }
    critical_correct = sum(
        entity_metrics[entity_type]["correct"]
        for entity_type in SAFETY_ENTITY_TYPES
    )
    critical_total = sum(
        entity_metrics[entity_type]["eligible_annotations"]
        for entity_type in SAFETY_ENTITY_TYPES
    )
    return {
        "dataset_kind": dataset_kind,
        "cases": len(case_results),
        "raw_wer": ratio(total("raw_word_edits"), total("raw_word_count")),
        "raw_cer": ratio(total("raw_char_edits"), total("raw_char_count")),
        "normalized_wer": ratio(total("word_edits"), total("word_count")),
        "normalized_cer": ratio(total("char_edits"), total("char_count")),
        "medical_wer": entity_metrics["medical_term"]["word_error_rate"],
        "medical_term_recall": entity_metrics["medical_term"]["exact_accuracy"],
        "critical_entity_recall": {
            "available": critical_total > 0,
            "correct": critical_correct,
            "eligible_annotations": critical_total,
            "recall": ratio(critical_correct, critical_total),
            "unavailable_reason": (
                None
                if critical_total > 0
                else "no reviewed critical entity span annotations"
            ),
        },
        "entities": entity_metrics,
        "rtf": {
            "median": statistics.median(rtf_values) if rtf_values else None,
            "p95": percentile(rtf_values, 0.95),
        },
        "wall_latency_seconds": {
            "p50": statistics.median(wall_values) if wall_values else None,
            "p95": percentile(wall_values, 0.95),
        },
        "deterministic_cases": sum(case["deterministic"] for case in case_results),
    }


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def prediction_inventory_sha256(cases: list[dict[str, Any]]) -> str:
    inventory = [
        {
            "id": case.get("id"),
            "audio_sha256": case.get("audio_sha256"),
            "prediction_sha256": case.get("prediction_sha256"),
        }
        for case in cases
    ]
    canonical = json.dumps(
        sorted(inventory, key=lambda item: str(item["id"])),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256_text(canonical)


def evaluate_release_gates(
    report: dict[str, Any],
    baseline: dict[str, Any],
    sequential: dict[str, Any],
    safety: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(
        name: str,
        passed: bool,
        actual: Any,
        requirement: Any,
    ) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "actual": actual,
                "requirement": requirement,
            }
        )

    manifest_hash = (report.get("manifest") or {}).get("sha256")
    server_metadata = report.get("server") or {}
    runtime_id = server_metadata.get("runtime_id")
    server_configuration = server_metadata.get("configuration") or {}
    server_model = server_metadata.get("model") or {}
    server_runtime = server_metadata.get("runtime") or {}
    current_cases = report.get("cases") or []
    current_case_ids = [case.get("id") for case in current_cases]
    current_prediction_inventory = prediction_inventory_sha256(current_cases)
    add(
        "release_report_contract",
        report.get("purpose") == "release"
        and report.get("repeat") == 10
        and report.get("concurrency") == 10
        and report.get("limited") is False,
        {
            "purpose": report.get("purpose"),
            "repeat": report.get("repeat"),
            "concurrency": report.get("concurrency"),
            "limited": report.get("limited"),
        },
        {
            "purpose": "release",
            "repeat": 10,
            "concurrency": 10,
            "limited": False,
        },
    )
    add(
        "prediction_inventory_integrity",
        bool(current_cases)
        and report.get("prediction_inventory_sha256")
        == current_prediction_inventory,
        report.get("prediction_inventory_sha256"),
        current_prediction_inventory,
    )
    add(
        "runtime_identity_present",
        isinstance(runtime_id, str) and bool(runtime_id),
        runtime_id,
        "non-empty runtime_id",
    )
    add(
        "strict_runtime_lock",
        server_configuration.get("strict_runtime_lock") is True,
        server_configuration.get("strict_runtime_lock"),
        True,
    )
    add(
        "checkpoint_verified",
        (server_model.get("checkpoint") or {}).get("verified") is True,
        (server_model.get("checkpoint") or {}).get("verified"),
        True,
    )
    add(
        "clean_runtime_source",
        (server_runtime.get("source") or {}).get("projectDirty") is False,
        (server_runtime.get("source") or {}).get("projectDirty"),
        False,
    )
    snapshot_output = (
        ((snapshot.get("outputs") or {}).get("real_frozen_benchmark") or {})
    )
    add(
        "snapshot_schema",
        snapshot.get("schema_version") == "voicemed.asr-dataset-snapshot.v1",
        snapshot.get("schema_version"),
        "voicemed.asr-dataset-snapshot.v1",
    )
    add(
        "snapshot_promotion_eligible",
        snapshot.get("audio_hashes_verified") is True
        and (snapshot.get("promotion") or {}).get("eligible") is True
        and not ((snapshot.get("promotion") or {}).get("blockers") or []),
        snapshot.get("promotion"),
        "audio_hashes_verified=true and eligible=true with no blockers",
    )
    add(
        "snapshot_manifest_binding",
        snapshot_output.get("sha256") == manifest_hash,
        snapshot_output.get("sha256"),
        manifest_hash,
    )

    add(
        "baseline_contract",
        baseline.get("purpose") == "baseline"
        and baseline.get("schema_version") == report.get("schema_version")
        and baseline.get("repeat") == 10
        and baseline.get("concurrency") == 1
        and baseline.get("limited") is False,
        {
            "purpose": baseline.get("purpose"),
            "schema_version": baseline.get("schema_version"),
            "repeat": baseline.get("repeat"),
            "concurrency": baseline.get("concurrency"),
            "limited": baseline.get("limited"),
        },
        {
            "purpose": "baseline",
            "schema_version": report.get("schema_version"),
            "repeat": 10,
            "concurrency": 1,
            "limited": False,
        },
    )
    add(
        "baseline_manifest_binding",
        (baseline.get("manifest") or {}).get("sha256") == manifest_hash,
        (baseline.get("manifest") or {}).get("sha256"),
        manifest_hash,
    )
    baseline_metrics = baseline.get("metrics") or {}
    baseline_cases = baseline.get("cases") or []
    baseline_case_ids = [case.get("id") for case in baseline_cases]
    add(
        "baseline_complete_deterministic_coverage",
        bool(current_case_ids)
        and baseline_case_ids == current_case_ids
        and baseline_metrics.get("cases") == len(current_case_ids)
        and baseline_metrics.get("deterministic_cases") == len(current_case_ids),
        {
            "case_ids_match": baseline_case_ids == current_case_ids,
            "cases": baseline_metrics.get("cases"),
            "deterministic_cases": baseline_metrics.get(
                "deterministic_cases"
            ),
        },
        {
            "same_ordered_case_ids": True,
            "cases": len(current_case_ids),
            "deterministic_cases": len(current_case_ids),
        },
    )
    add(
        "sequential_candidate_contract",
        sequential.get("purpose") == "candidate"
        and sequential.get("schema_version") == report.get("schema_version")
        and sequential.get("repeat") == 10
        and sequential.get("concurrency") == 1
        and sequential.get("limited") is False
        and "context_scope" in sequential
        and sequential.get("context_scope") == report.get("context_scope"),
        {
            "purpose": sequential.get("purpose"),
            "schema_version": sequential.get("schema_version"),
            "repeat": sequential.get("repeat"),
            "concurrency": sequential.get("concurrency"),
            "limited": sequential.get("limited"),
            "context_scope": sequential.get("context_scope"),
        },
        {
            "purpose": "candidate",
            "schema_version": report.get("schema_version"),
            "repeat": 10,
            "concurrency": 1,
            "limited": False,
            "context_scope": report.get("context_scope"),
        },
    )
    add(
        "sequential_manifest_binding",
        (sequential.get("manifest") or {}).get("sha256") == manifest_hash,
        (sequential.get("manifest") or {}).get("sha256"),
        manifest_hash,
    )
    add(
        "sequential_runtime_binding",
        (sequential.get("server") or {}).get("runtime_id") == runtime_id,
        (sequential.get("server") or {}).get("runtime_id"),
        runtime_id,
    )

    current_metrics = report.get("metrics") or {}
    baseline_medical_wer = finite_number(baseline_metrics.get("medical_wer"))
    current_medical_wer = finite_number(current_metrics.get("medical_wer"))
    if baseline_medical_wer is None or current_medical_wer is None:
        relative_improvement = None
        medical_passed = False
    elif baseline_medical_wer == 0:
        relative_improvement = None
        medical_passed = current_medical_wer == 0
    else:
        relative_improvement = (
            baseline_medical_wer - current_medical_wer
        ) / baseline_medical_wer
        medical_passed = (
            relative_improvement
            >= RELEASE_THRESHOLDS["medical_wer_relative_improvement"]
        )
    add(
        "medical_wer_relative_improvement",
        medical_passed,
        {
            "baseline": baseline_medical_wer,
            "candidate": current_medical_wer,
            "relative_improvement": relative_improvement,
        },
        {
            "minimum": RELEASE_THRESHOLDS[
                "medical_wer_relative_improvement"
            ],
            "baseline_zero_rule": "candidate must also be zero",
        },
    )

    baseline_wer = finite_number(baseline_metrics.get("normalized_wer"))
    current_wer = finite_number(current_metrics.get("normalized_wer"))
    wer_regression = (
        current_wer - baseline_wer
        if current_wer is not None and baseline_wer is not None
        else None
    )
    add(
        "overall_wer_regression",
        wer_regression is not None
        and wer_regression
        <= RELEASE_THRESHOLDS["overall_wer_absolute_regression"],
        {
            "baseline": baseline_wer,
            "candidate": current_wer,
            "absolute_regression": wer_regression,
        },
        {
            "maximum": RELEASE_THRESHOLDS[
                "overall_wer_absolute_regression"
            ]
        },
    )

    critical = current_metrics.get("critical_entity_recall") or {}
    critical_recall = finite_number(critical.get("recall"))
    critical_denominator = int(critical.get("eligible_annotations") or 0)
    add(
        "critical_entity_recall",
        critical.get("available") is True
        and critical_denominator > 0
        and critical_recall is not None
        and critical_recall >= RELEASE_THRESHOLDS["critical_entity_recall"],
        {
            "recall": critical_recall,
            "eligible_annotations": critical_denominator,
        },
        {
            "minimum": RELEASE_THRESHOLDS["critical_entity_recall"],
            "eligible_annotations": ">0",
        },
    )
    for entity_type in SAFETY_ENTITY_TYPES:
        metric = (current_metrics.get("entities") or {}).get(entity_type) or {}
        accuracy = finite_number(metric.get("exact_accuracy"))
        denominator = int(metric.get("eligible_annotations") or 0)
        add(
            f"{entity_type}_exact_accuracy",
            metric.get("available") is True
            and denominator > 0
            and accuracy is not None
            and accuracy >= RELEASE_THRESHOLDS["entity_exact_accuracy"],
            {
                "accuracy": accuracy,
                "eligible_annotations": denominator,
            },
            {
                "minimum": RELEASE_THRESHOLDS["entity_exact_accuracy"],
                "eligible_annotations": ">0",
            },
        )

    p95_rtf = finite_number((current_metrics.get("rtf") or {}).get("p95"))
    add(
        "p95_rtf",
        p95_rtf is not None and p95_rtf <= RELEASE_THRESHOLDS["p95_rtf"],
        p95_rtf,
        {"maximum": RELEASE_THRESHOLDS["p95_rtf"]},
    )
    add(
        "concurrent_determinism",
        current_metrics.get("cases", 0) > 0
        and current_metrics.get("deterministic_cases")
        == current_metrics.get("cases"),
        {
            "deterministic_cases": current_metrics.get("deterministic_cases"),
            "cases": current_metrics.get("cases"),
            "repeat": report.get("repeat"),
            "concurrency": report.get("concurrency"),
        },
        {"all_cases": True, "repeat": 10, "concurrency": 10},
    )
    sequential_metrics = sequential.get("metrics") or {}
    add(
        "sequential_determinism",
        sequential_metrics.get("cases", 0) > 0
        and sequential_metrics.get("deterministic_cases")
        == sequential_metrics.get("cases"),
        {
            "deterministic_cases": sequential_metrics.get(
                "deterministic_cases"
            ),
            "cases": sequential_metrics.get("cases"),
        },
        {"all_cases": True, "repeat": 10, "concurrency": 1},
    )
    current_predictions = {
        case.get("id"): case.get("prediction_sha256")
        for case in report.get("cases") or []
    }
    sequential_predictions = {
        case.get("id"): case.get("prediction_sha256")
        for case in sequential.get("cases") or []
    }
    add(
        "sequential_concurrent_prediction_match",
        bool(current_predictions)
        and current_predictions == sequential_predictions,
        {
            "current_cases": len(current_predictions),
            "sequential_cases": len(sequential_predictions),
            "equal": current_predictions == sequential_predictions,
        },
        "identical per-case raw transcript SHA-256",
    )

    safety_schema = safety.get("schema_version")
    add(
        "structuring_safety_schema",
        safety_schema == "voicemed.structuring-safety.v1",
        safety_schema,
        "voicemed.structuring-safety.v1",
    )
    add(
        "structuring_safety_manifest_binding",
        safety.get("manifest_sha256") == manifest_hash,
        safety.get("manifest_sha256"),
        manifest_hash,
    )
    add(
        "structuring_safety_runtime_binding",
        safety.get("runtime_id") == runtime_id,
        safety.get("runtime_id"),
        runtime_id,
    )
    add(
        "structuring_safety_prediction_binding",
        safety.get("prediction_inventory_sha256")
        == current_prediction_inventory
        and safety.get("cases_total") == len(current_case_ids)
        and "context_scope" in safety
        and safety.get("context_scope") == report.get("context_scope"),
        {
            "prediction_inventory_sha256": safety.get(
                "prediction_inventory_sha256"
            ),
            "cases_total": safety.get("cases_total"),
            "context_scope": safety.get("context_scope"),
        },
        {
            "prediction_inventory_sha256": current_prediction_inventory,
            "cases_total": len(current_case_ids),
            "context_scope": report.get("context_scope"),
        },
    )
    critical_facts_total = int(safety.get("critical_facts_total") or 0)
    critical_facts_with_provenance = int(
        safety.get("critical_facts_with_provenance") or 0
    )
    unsupported_critical_facts = int(
        safety.get("unsupported_critical_facts") or 0
    )
    add(
        "critical_fact_provenance",
        critical_facts_total > 0
        and critical_facts_with_provenance == critical_facts_total,
        {
            "critical_facts_total": critical_facts_total,
            "critical_facts_with_provenance": critical_facts_with_provenance,
        },
        "all critical facts have source-span provenance and denominator >0",
    )
    add(
        "unsupported_critical_facts",
        critical_facts_total > 0 and unsupported_critical_facts == 0,
        {
            "critical_facts_total": critical_facts_total,
            "unsupported_critical_facts": unsupported_critical_facts,
        },
        {"maximum": 0, "critical_facts_total": ">0"},
    )

    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "eligible": not failed,
        "failed": failed,
        "checks": checks,
        "thresholds": RELEASE_THRESHOLDS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:9002")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--purpose",
        choices=("validation", "baseline", "candidate", "release", "regression"),
        required=True,
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--context-scope",
        help="approved server-side context scope (never a free-form hotword list)",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="include deidentified reference/prediction text in the report",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="required in release mode: frozen baseline benchmark JSON",
    )
    parser.add_argument(
        "--sequential-report",
        type=Path,
        help="required in release mode: candidate repeat=10 concurrency=1 JSON",
    )
    parser.add_argument(
        "--structuring-safety-report",
        type=Path,
        help="required in release mode: provenance/unsupported-fact report JSON",
    )
    parser.add_argument(
        "--dataset-snapshot",
        type=Path,
        help="required in release mode: governed dataset-snapshot.json",
    )
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.repeat > 10:
        parser.error("--repeat must be in the range 1..10")
    if args.concurrency < 1 or args.concurrency > 10:
        parser.error("--concurrency must be in the range 1..10")
    release_inputs: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ] | None = None
    if args.purpose == "release":
        if args.limit is not None:
            parser.error("--limit is prohibited for a release benchmark")
        if args.repeat != 10 or args.concurrency != 10:
            parser.error(
                "release requires --repeat 10 --concurrency 10; "
                "provide the separate sequential candidate report"
            )
        required_paths = {
            "--baseline-report": args.baseline_report,
            "--sequential-report": args.sequential_report,
            "--structuring-safety-report": args.structuring_safety_report,
            "--dataset-snapshot": args.dataset_snapshot,
        }
        missing_paths = [
            option for option, path in required_paths.items() if path is None
        ]
        if missing_paths:
            parser.error(
                "release is missing required inputs: " + ", ".join(missing_paths)
            )
        assert args.baseline_report is not None
        assert args.sequential_report is not None
        assert args.structuring_safety_report is not None
        assert args.dataset_snapshot is not None
        release_inputs = (
            read_json_object(args.baseline_report.resolve(), "baseline report"),
            read_json_object(
                args.sequential_report.resolve(), "sequential candidate report"
            ),
            read_json_object(
                args.structuring_safety_report.resolve(),
                "structuring safety report",
            ),
            read_json_object(
                args.dataset_snapshot.resolve(),
                "dataset snapshot",
            ),
        )

    manifest_path = args.manifest.resolve()
    rows = read_manifest(manifest_path)
    cases = validate_benchmark_manifest(manifest_path, rows, args.purpose)
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    if not cases:
        raise ValueError("no benchmark cases selected")

    server_url = args.server_url.rstrip("/")
    metadata = http_json(f"{server_url}/metadata", None, args.timeout)
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        audio = base64.b64encode(case["audio_path"].read_bytes()).decode("ascii")
        responses: list[dict[str, Any]] = []
        wall_times: list[float] = []

        def transcribe_once() -> tuple[dict[str, Any], float]:
            started = time.perf_counter()
            response = http_json(
                f"{server_url}/transcribe",
                {
                    "audio_base64": audio,
                    **(
                        {"context_scope": args.context_scope}
                        if args.context_scope
                        else {}
                    ),
                },
                args.timeout,
            )
            return response, time.perf_counter() - started

        if args.concurrency == 1:
            completed = [transcribe_once() for _ in range(args.repeat)]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as executor:
                completed = list(
                    executor.map(lambda _: transcribe_once(), range(args.repeat))
                )
        for response, wall_time in completed:
            responses.append(response)
            wall_times.append(wall_time)
        response_texts: list[str] = []
        hashes: list[str] = []
        expected_runtime_id = metadata.get("runtime_id")
        for response in responses:
            if response.get("source") != "gigaam":
                raise RuntimeError(
                    f"{case['id']}: benchmark server returned non-GigaAM source"
                )
            if response.get("runtime_id") != expected_runtime_id:
                raise RuntimeError(
                    f"{case['id']}: runtime_id changed during benchmark"
                )
            response_hashes = response.get("hashes") or {}
            if response_hashes.get("audio_sha256") != case["audio_sha256"]:
                raise RuntimeError(
                    f"{case['id']}: server audio SHA-256 does not match manifest"
                )
            response_text = str(
                response.get("raw_text", response.get("text", ""))
            )
            actual_text_hash = sha256_text(response_text)
            reported_text_hash = response_hashes.get("raw_text_sha256")
            if reported_text_hash and reported_text_hash != actual_text_hash:
                raise RuntimeError(
                    f"{case['id']}: server raw text SHA-256 is inconsistent"
                )
            response_texts.append(response_text)
            hashes.append(actual_text_hash)
        first = responses[0]
        hypothesis = response_texts[0]
        repeat_word_evidence = [
            summarize_word_evidence(response) for response in responses
        ]
        first_word_evidence = dict(repeat_word_evidence[0])
        first_word_evidence["all_repeats_complete"] = all(
            evidence["timestamps_complete"]
            and evidence["acoustic_confidence_complete"]
            for evidence in repeat_word_evidence
        )
        first_word_evidence["repeat_deterministic"] = (
            len(
                {
                    (
                        evidence["word_count"],
                        evidence["timing_signature_sha256"],
                        evidence["acoustic_signature_sha256"],
                    )
                    for evidence in repeat_word_evidence
                }
            )
            == 1
        )
        scores = score_case(
            case["transcription"],
            hypothesis,
            parse_terms(case.get("terms", "")),
            case["_entities"],
        )
        inference_seconds = float(
            (first.get("timings") or {}).get("inference_seconds")
            or first.get("elapsed")
            or wall_times[0]
        )
        result: dict[str, Any] = {
            "id": case["id"],
            "audio_sha256": case["audio_sha256"],
            "reference_sha256": sha256_text(case["transcription"]),
            "prediction_sha256": hashes[0],
            "runtime_id": first.get("runtime_id"),
            "duration_seconds": case["duration_seconds"],
            "inference_seconds": inference_seconds,
            "wall_seconds": wall_times,
            "deterministic": len(set(hashes)) == 1,
            "confidence": first.get("confidence"),
            "word_evidence": first_word_evidence,
            "scores": scores,
        }
        if args.include_text:
            result["reference"] = case["transcription"]
            result["prediction"] = hypothesis
        results.append(result)
        print(
            f"[{index}/{len(cases)}] {case['id']}: "
            f"WER={scores['word_edits'] / scores['word_count']:.3f} "
            f"deterministic={result['deterministic']}",
            flush=True,
        )

    dataset_kind = cases[0]["dataset_kind"].strip()
    report = {
        "schema_version": "voicemed.asr-benchmark.v2",
        "purpose": args.purpose,
        "context_scope": args.context_scope,
        "repeat": args.repeat,
        "concurrency": args.concurrency,
        "limited": args.limit is not None,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": {
            "file": manifest_path.name,
            "sha256": sha256_file(manifest_path),
            "dataset_kind": dataset_kind,
        },
        "server": metadata,
        "metrics": aggregate(results, dataset_kind),
        "cases": results,
        "prediction_inventory_sha256": prediction_inventory_sha256(results),
    }
    exit_code = 0
    if release_inputs is not None:
        baseline, sequential, safety, snapshot = release_inputs
        report["release_gate"] = evaluate_release_gates(
            report,
            baseline,
            sequential,
            safety,
            snapshot,
        )
        if not report["release_gate"]["eligible"]:
            exit_code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    if "release_gate" in report:
        print(json.dumps(report["release_gate"], ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
