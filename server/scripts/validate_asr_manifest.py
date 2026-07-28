#!/usr/bin/env python3
"""Validate a governed ASR dataset and emit official GigaAM TSV manifests."""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

KIND_TO_SPLIT = {
    "real_train": "train",
    "synthetic_train": "train",
    "general_replay": "train",
    "real_validation": "validation",
    "real_frozen_test": "test",
    "synthetic_regression": "regression",
}
REAL_KINDS = {"real_train", "real_validation", "real_frozen_test", "general_replay"}
ENTITY_TYPES = {
    "medical_term",
    "number_unit",
    "negation",
    "laterality",
    "contrast",
}
EXPECTED_TRANSCRIPT_SOURCE = {
    "real_train": "human_verbatim",
    "general_replay": "human_verbatim",
    "real_validation": "human_verbatim",
    "real_frozen_test": "human_verbatim",
    "synthetic_train": "tts_script",
    "synthetic_regression": "tts_script",
}
MANIFEST_COLUMNS = (
    "id",
    "path",
    "duration",
    "transcription",
    "dataset_kind",
    "split",
    "speaker_id",
    "patient_id",
    "study_id",
    "recorded_at",
    "modality",
    "anatomy",
    "reviewed_by",
    "frozen_at",
    "audio_sha256",
    "source_recording_id",
    "transcript_source",
    "entities_json",
    "terms",
)
REQUIRED_COLUMNS = set(MANIFEST_COLUMNS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def audio_duration_seconds(path: Path) -> float:
    """Read duration from encoded audio instead of trusting manifest metadata."""

    try:
        import soundfile

        duration = float(soundfile.info(str(path)).duration)
    except Exception as exc:
        raise ValueError(f"audio cannot be decoded for duration: {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"audio has invalid duration: {path}")
    return duration


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("manifest has no rows")
    return rows


def normalized_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return re.findall(r"[0-9a-zа-я]+(?:[.,][0-9]+)?", normalized)


def parse_entities(
    value: str,
    transcription: str,
    line: int,
) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {line}: entities_json is not valid JSON") from exc
    if not isinstance(raw, list):
        raise ValueError(f"line {line}: entities_json must be a JSON array")

    reference_words = normalized_words(transcription)
    entities: list[dict[str, Any]] = []
    entity_ids: set[str] = set()
    for index, entity in enumerate(raw):
        prefix = f"line {line}: entities_json[{index}]"
        if not isinstance(entity, dict):
            raise ValueError(f"{prefix} must be an object")
        entity_id = entity.get("id")
        entity_type = entity.get("type")
        text = entity.get("text")
        start_word = entity.get("start_word")
        end_word = entity.get("end_word")
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ValueError(f"{prefix}.id must be a non-empty string")
        entity_id = entity_id.strip()
        if entity_id in entity_ids:
            raise ValueError(f"{prefix}.id is duplicated: {entity_id}")
        entity_ids.add(entity_id)
        if entity_type not in ENTITY_TYPES:
            raise ValueError(
                f"{prefix}.type must be one of {sorted(ENTITY_TYPES)}"
            )
        if not isinstance(start_word, int) or isinstance(start_word, bool):
            raise ValueError(f"{prefix}.start_word must be an integer")
        if not isinstance(end_word, int) or isinstance(end_word, bool):
            raise ValueError(f"{prefix}.end_word must be an integer")
        if not 0 <= start_word < end_word <= len(reference_words):
            raise ValueError(
                f"{prefix} has invalid word span [{start_word}, {end_word})"
            )
        if not isinstance(text, str) or not normalized_words(text):
            raise ValueError(f"{prefix}.text must be a non-empty string")
        span_words = reference_words[start_word:end_word]
        if normalized_words(text) != span_words:
            raise ValueError(
                f"{prefix}.text does not match transcription word span: "
                f"expected {' '.join(span_words)!r}"
            )
        entities.append(
            {
                "id": entity_id,
                "type": entity_type,
                "text": text,
                "start_word": start_word,
                "end_word": end_word,
            }
        )
    return entities


def parse_iso_date(value: str, field: str, line: int) -> None:
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"line {line}: {field} must be ISO-8601") from exc


def assert_disjoint(
    rows_by_split: dict[str, list[dict[str, Any]]], field: str
) -> None:
    owners: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for row in rows_by_split.get(split, []):
            if row["dataset_kind"] not in REAL_KINDS:
                continue
            value = str(row.get(field, "")).strip()
            if not value:
                continue
            previous = owners.get(value)
            if previous and previous != split:
                raise ValueError(
                    f"{field}={value!r} leaks between {previous} and {split}"
                )
            owners[value] = split


def validate(
    manifest_path: Path, rows: list[dict[str, str]], verify_audio: bool
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    resolved: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    ids: set[str] = set()
    hashes: set[str] = set()
    warnings: list[str] = []

    for line, raw in enumerate(rows, start=2):
        row = {key: (value or "").strip() for key, value in raw.items()}
        case_id = row["id"]
        if not case_id or case_id in ids:
            raise ValueError(f"line {line}: missing or duplicate id")
        ids.add(case_id)

        kind = row["dataset_kind"]
        if kind not in KIND_TO_SPLIT:
            raise ValueError(f"line {line}: unsupported dataset_kind={kind!r}")
        expected_split = KIND_TO_SPLIT[kind]
        if row["split"] != expected_split:
            raise ValueError(
                f"line {line}: {kind} must use split={expected_split}"
            )
        if not row["transcription"]:
            raise ValueError(f"line {line}: verbatim transcription is empty")
        expected_source = EXPECTED_TRANSCRIPT_SOURCE[kind]
        if row["transcript_source"] == "browser_asr":
            raise ValueError(
                f"line {line}: browser_asr transcripts are prohibited in governed data"
            )
        if row["transcript_source"] != expected_source:
            raise ValueError(
                f"line {line}: {kind} requires "
                f"transcript_source={expected_source}"
            )
        if not row["source_recording_id"]:
            raise ValueError(f"line {line}: source_recording_id is required")
        entities = parse_entities(row["entities_json"], row["transcription"], line)
        try:
            duration = float(row["duration"])
        except ValueError as exc:
            raise ValueError(f"line {line}: invalid duration") from exc
        if duration <= 0:
            raise ValueError(f"line {line}: duration must be positive")
        if expected_split in {"train", "validation", "test"} and not 2 <= duration <= 20:
            raise ValueError(
                f"line {line}: train/validation/test segments must be 2..20 seconds"
            )

        audio_path = (manifest_path.parent / row["path"]).resolve()
        if not audio_path.is_file():
            raise ValueError(f"line {line}: audio does not exist: {audio_path}")
        declared_hash = row["audio_sha256"].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
            raise ValueError(f"line {line}: audio_sha256 is required")
        if declared_hash in hashes:
            raise ValueError(f"line {line}: duplicate audio content/hash")
        hashes.add(declared_hash)
        verified_duration = duration
        if verify_audio:
            actual_hash = sha256_file(audio_path)
            if actual_hash != declared_hash:
                raise ValueError(f"line {line}: audio_sha256 mismatch")
            actual_duration = audio_duration_seconds(audio_path)
            duration_tolerance = max(0.05, actual_duration * 0.01)
            if abs(actual_duration - duration) > duration_tolerance:
                raise ValueError(
                    f"line {line}: duration mismatch: declared={duration:.3f}s, "
                    f"actual={actual_duration:.3f}s"
                )
            verified_duration = actual_duration

        if kind in REAL_KINDS:
            for field in (
                "speaker_id",
                "patient_id",
                "study_id",
                "recorded_at",
                "reviewed_by",
            ):
                if not row[field]:
                    raise ValueError(f"line {line}: {field} is required for real audio")
            parse_iso_date(row["recorded_at"], "recorded_at", line)
        if not row["modality"] or not row["anatomy"]:
            raise ValueError(f"line {line}: modality and anatomy are required")
        if kind == "real_frozen_test" and not row["frozen_at"]:
            raise ValueError(f"line {line}: frozen_at is required for test audio")
        if kind == "real_frozen_test":
            parse_iso_date(row["frozen_at"], "frozen_at", line)
        if expected_split == "test" and kind != "real_frozen_test":
            raise ValueError(f"line {line}: test may contain only real frozen audio")
        if expected_split == "validation" and kind != "real_validation":
            raise ValueError(f"line {line}: validation may contain only real audio")

        resolved[expected_split].append(
            {
                **row,
                "duration_seconds": verified_duration,
                "declared_duration_seconds": duration,
                "audio_path": audio_path,
                "_entities": entities,
            }
        )

    for required_split in ("train", "validation", "test"):
        if not resolved.get(required_split):
            raise ValueError(f"dataset has no {required_split} rows")

    assert_disjoint(resolved, "patient_id")
    assert_disjoint(resolved, "study_id")
    assert_disjoint(resolved, "recorded_at")
    assert_disjoint(resolved, "source_recording_id")
    # Keep the compound encounter check as a second guard when a source system
    # later switches recorded_at from a date to a timestamp.
    train_triplets = {
        (row["speaker_id"], row["patient_id"], row["recorded_at"])
        for row in resolved["train"]
        if row["dataset_kind"] in REAL_KINDS
    }
    for split in ("validation", "test"):
        for row in resolved[split]:
            triplet = (row["speaker_id"], row["patient_id"], row["recorded_at"])
            if triplet in train_triplets:
                raise ValueError(f"encounter leakage into {split}: {row['id']}")

    train_speakers = {
        row["speaker_id"]
        for row in resolved["train"]
        if row["dataset_kind"] in REAL_KINDS
    }
    test_speakers = {row["speaker_id"] for row in resolved["test"]}
    held_out = test_speakers - train_speakers
    if not held_out:
        raise ValueError("test must contain at least one fully speaker-held-out doctor")

    train_count = len(resolved["train"])
    synthetic_count = sum(
        row["dataset_kind"] == "synthetic_train" for row in resolved["train"]
    )
    replay_count = sum(
        row["dataset_kind"] == "general_replay" for row in resolved["train"]
    )
    synthetic_ratio = synthetic_count / train_count
    replay_ratio = replay_count / train_count
    if synthetic_ratio > 0.20:
        raise ValueError(
            f"synthetic_train is {synthetic_ratio:.1%}; maximum is 20% of train rows"
        )
    if not 0.20 <= replay_ratio <= 0.30:
        warnings.append(
            f"general_replay is {replay_ratio:.1%}; promotion target is 20-30% "
            "(enforce the same ratio in the sampler)"
        )
    if not resolved.get("regression"):
        warnings.append("no synthetic_regression rows; optional regression tier is empty")
    return resolved, warnings


def promotion_assessment(
    rows_by_split: dict[str, list[dict[str, Any]]],
    audio_hashes_verified: bool,
) -> dict[str, Any]:
    """Return explicit eligibility blockers; validation alone is not promotion."""

    train_rows = rows_by_split["train"]
    validation_rows = rows_by_split["validation"]
    test_rows = rows_by_split["test"]
    real_train_hours = sum(
        row["duration_seconds"]
        for row in train_rows
        if row["dataset_kind"] == "real_train"
    ) / 3600
    validation_hours = sum(row["duration_seconds"] for row in validation_rows) / 3600
    test_hours = sum(row["duration_seconds"] for row in test_rows) / 3600
    doctors = {
        row["speaker_id"]
        for split in ("train", "validation", "test")
        for row in rows_by_split[split]
        if row["dataset_kind"] in REAL_KINDS
    }
    train_count = len(train_rows)
    synthetic_ratio = (
        sum(row["dataset_kind"] == "synthetic_train" for row in train_rows)
        / train_count
    )
    replay_ratio = (
        sum(row["dataset_kind"] == "general_replay" for row in train_rows)
        / train_count
    )
    validation_entities = collections.Counter(
        entity["type"]
        for row in validation_rows
        for entity in row["_entities"]
    )
    test_entities = collections.Counter(
        entity["type"] for row in test_rows for entity in row["_entities"]
    )

    blockers: list[str] = []
    if not audio_hashes_verified:
        blockers.append("audio hashes were not verified")
    if real_train_hours < 10:
        blockers.append(
            f"real train is {real_train_hours:.3f}h; promotion minimum is 10h"
        )
    if validation_hours < 2:
        blockers.append(
            f"validation is {validation_hours:.3f}h; promotion minimum is 2h"
        )
    if test_hours < 2:
        blockers.append(f"test is {test_hours:.3f}h; promotion minimum is 2h")
    if len(doctors) < 3:
        blockers.append(
            f"dataset has {len(doctors)} doctors; promotion minimum is 3"
        )
    if synthetic_ratio > 0.20:
        blockers.append(
            f"synthetic train row ratio is {synthetic_ratio:.1%}; maximum is 20%"
        )
    if not 0.20 <= replay_ratio <= 0.30:
        blockers.append(
            f"general replay row ratio is {replay_ratio:.1%}; target is 20-30%"
        )
    if validation_entities["medical_term"] <= 0:
        blockers.append("validation has no medical_term entity annotations")
    for entity_type in ENTITY_TYPES:
        if test_entities[entity_type] <= 0:
            blockers.append(
                f"frozen test has no {entity_type} entity annotations"
            )

    return {
        "eligible": not blockers,
        "blockers": blockers,
        "targets": {
            "real_train_hours_minimum": 10,
            "validation_hours_minimum": 2,
            "frozen_test_hours_minimum": 2,
            "minimum_doctors": 3,
            "synthetic_train_row_maximum": 0.20,
            "general_replay_row_range": [0.20, 0.30],
        },
        "observed": {
            "real_train_hours": round(real_train_hours, 3),
            "validation_hours": round(validation_hours, 3),
            "frozen_test_hours": round(test_hours, 3),
            "doctors": len(doctors),
            "synthetic_train_row_ratio": synthetic_ratio,
            "general_replay_row_ratio": replay_ratio,
            "validation_entities": dict(sorted(validation_entities.items())),
            "test_entities": dict(sorted(test_entities.items())),
        },
    }


def audio_inventory_sha256(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> str:
    inventory = [
        {
            "id": row["id"],
            "split": split,
            "audio_sha256": row["audio_sha256"],
            "source_recording_id": row["source_recording_id"],
        }
        for split in ("train", "validation", "test", "regression")
        for row in rows_by_split.get(split, [])
    ]
    canonical = json.dumps(
        sorted(inventory, key=lambda item: (item["split"], item["id"])),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_gigaam_manifest(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("path", "duration", "transcription"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": str(row["audio_path"]),
                    "duration": f"{row['duration_seconds']:.3f}",
                    "transcription": row["transcription"],
                }
            )


def write_governed_benchmark_manifest(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=MANIFEST_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: (
                        str(row["audio_path"])
                        if column == "path"
                        else f"{row['duration_seconds']:.3f}"
                        if column == "duration"
                        else row.get(column, "")
                    )
                    for column in MANIFEST_COLUMNS
                }
            )


def write_atomically(path: Path, writer: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-audio-hash",
        action="store_true",
        help=(
            "skip audio hash/decode/duration integrity checks; only for fast "
            "local iteration and prohibited for a frozen snapshot"
        ),
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="replace an existing local draft; prohibited for an immutable snapshot",
    )
    args = parser.parse_args(argv)
    manifest = args.manifest.resolve()
    rows_by_split, warnings = validate(
        manifest,
        read_rows(manifest),
        verify_audio=not args.skip_audio_hash,
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_names = {
        "train.tsv",
        "validation.tsv",
        "test.tsv",
        "real-validation-benchmark.tsv",
        "real-frozen-benchmark.tsv",
        "governed-source.tsv",
        "dataset-snapshot.json",
    }
    if rows_by_split.get("regression"):
        target_names.add("synthetic-regression-benchmark.tsv")
    existing = sorted(name for name in target_names if (output / name).exists())
    if existing and not args.allow_overwrite:
        raise ValueError(
            "snapshot output already exists and is immutable by default: "
            + ", ".join(existing)
        )

    emitted: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        target = output / f"{split}.tsv"
        write_atomically(
            target,
            lambda temporary, split=split: write_gigaam_manifest(
                temporary, rows_by_split[split]
            ),
        )
        emitted[split] = {
            "file": target.name,
            "sha256": sha256_file(target),
            "rows": len(rows_by_split[split]),
            "hours": round(
                sum(row["duration_seconds"] for row in rows_by_split[split]) / 3600,
                3,
            ),
        }

    real_benchmark = output / "real-frozen-benchmark.tsv"
    write_atomically(
        real_benchmark,
        lambda temporary: write_governed_benchmark_manifest(
            temporary, rows_by_split["test"]
        ),
    )
    emitted["real_frozen_benchmark"] = {
        "file": real_benchmark.name,
        "sha256": sha256_file(real_benchmark),
        "rows": len(rows_by_split["test"]),
    }
    validation_benchmark = output / "real-validation-benchmark.tsv"
    write_atomically(
        validation_benchmark,
        lambda temporary: write_governed_benchmark_manifest(
            temporary,
            rows_by_split["validation"],
        ),
    )
    emitted["real_validation_benchmark"] = {
        "file": validation_benchmark.name,
        "sha256": sha256_file(validation_benchmark),
        "rows": len(rows_by_split["validation"]),
    }
    if rows_by_split.get("regression"):
        synthetic_benchmark = output / "synthetic-regression-benchmark.tsv"
        write_atomically(
            synthetic_benchmark,
            lambda temporary: write_governed_benchmark_manifest(
                temporary,
                rows_by_split["regression"],
            ),
        )
        emitted["synthetic_regression_benchmark"] = {
            "file": synthetic_benchmark.name,
            "sha256": sha256_file(synthetic_benchmark),
            "rows": len(rows_by_split["regression"]),
        }

    governed_source = output / "governed-source.tsv"
    write_atomically(
        governed_source,
        lambda temporary: temporary.write_bytes(manifest.read_bytes()),
    )
    promotion = promotion_assessment(
        rows_by_split,
        audio_hashes_verified=not args.skip_audio_hash,
    )
    snapshot = {
        "schema_version": "voicemed.asr-dataset-snapshot.v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_manifest": {
            "file": manifest.name,
            "sha256": sha256_file(manifest),
            "snapshot_copy": governed_source.name,
            "snapshot_copy_sha256": sha256_file(governed_source),
        },
        "audio_hashes_verified": not args.skip_audio_hash,
        "audio_inventory_sha256": audio_inventory_sha256(rows_by_split),
        "outputs": emitted,
        "speaker_held_out": sorted(
            {row["speaker_id"] for row in rows_by_split["test"]}
            - {
                row["speaker_id"]
                for row in rows_by_split["train"]
                if row["dataset_kind"] in REAL_KINDS
            }
        ),
        "transcript_sources": dict(
            sorted(
                collections.Counter(
                    row["transcript_source"]
                    for rows in rows_by_split.values()
                    for row in rows
                ).items()
            )
        ),
        "promotion": promotion,
        "warnings": warnings,
    }
    snapshot_path = output / "dataset-snapshot.json"
    write_atomically(
        snapshot_path,
        lambda temporary: temporary.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        ),
    )
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"manifest validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
