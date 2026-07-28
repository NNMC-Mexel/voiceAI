#!/usr/bin/env python3
"""Build a governed 5-gram KenLM artifact from reviewed radiology reports.

The builder is intentionally fail-closed:

* only explicitly approved, deidentified train reports are accepted;
* every source document and normalized text is content-addressed;
* validation/test material and obvious PHI are rejected;
* template defaults are removed only by exact, case-sensitive line match;
* KenLM executables are invoked directly, never through a shell;
* the corpus, binary model and metadata lock are written atomically.

The manifest contains no report text.  See
``radiology-lm-manifest.schema.tsv`` next to this file for its columns.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LOCK_SCHEMA_VERSION = "voicemed.radiology-kenlm-lock.v1"
TEMPLATE_DEFAULTS_SCHEMA_VERSION = "voicemed.radiology-template-defaults.v1"
BUILDER_VERSION = "radiology-kenlm-builder-v1"
NORMALIZATION_VERSION = "conservative-whitespace-v1"
KENLM_ORDER = 5
MAX_REPORT_BYTES = 16 * 1024 * 1024
EXPECTED_DATASET_KIND = "approved_deidentified_train_report"
EXPECTED_REFERENCE_STATUS = "verified"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
MANIFEST_COLUMNS = (
    "id",
    "path",
    "dataset_kind",
    "split",
    "domain",
    "reference_status",
    "approved",
    "deidentified",
    "reviewed_by",
    "approved_at",
    "document_sha256",
    "text_sha256",
)

# These checks are deliberately limited to high-signal patterns.  They are a
# final guardrail, not a replacement for the human deidentification workflow.
PHI_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "patient_name_label",
        re.compile(
            r"(?im)\b(?:фио|пациент(?:ка)?|patient(?:\s+name)?)\s*[:=]"
        ),
    ),
    (
        "birth_date_label",
        re.compile(
            r"(?im)\b(?:дата\s+рождения|д\.?\s*р\.?|date\s+of\s+birth|dob)"
            r"\s*[:=]"
        ),
    ),
    (
        "patient_identifier_label",
        re.compile(
            r"(?im)\b(?:иин|снилс|mrn|(?:номер\s+)?(?:карты|медкарты|"
            r"истории\s+болезни))\s*[:=#№]\s*[A-Za-zА-Яа-яЁё0-9-]{3,}"
        ),
    ),
    (
        "address_label",
        re.compile(r"(?im)\b(?:адрес|address)\s*[:=]"),
    ),
    (
        "email",
        re.compile(
            r"(?i)(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])"
        ),
    ),
    (
        "phone",
        re.compile(
            r"(?<!\d)(?:\+?7|8)[\s().-]*(?:\d[\s().-]*){10}(?!\d)"
        ),
    ),
    (
        "iin_12_digits",
        re.compile(r"(?<!\d)\d{12}(?!\d)"),
    ),
    (
        "russian_full_name",
        re.compile(
            r"\b[А-ЯЁ][а-яё-]{1,}\s+[А-ЯЁ][а-яё-]{1,}\s+"
            r"[А-ЯЁ][а-яё-]*(?:ович|евич|овна|евна)\b"
        ),
    ),
)


@dataclass(frozen=True)
class GovernedDocument:
    case_id: str
    path: Path
    document_sha256: str
    text_sha256: str
    normalized_text: str
    approved_at: str
    reviewed_by: str


@dataclass(frozen=True)
class TemplateDefaults:
    path: Path
    file_sha256: str
    reviewed_by: str
    approved_at: str
    lines: frozenset[str]


@dataclass(frozen=True)
class BuildRequest:
    manifest: Path
    domain: str
    lmplz: Path
    build_binary: Path
    kenlm_version: str
    output_corpus: Path
    output_model: Path
    output_lock: Path
    template_defaults: Path | None = None
    memory: str = "50%"
    timeout_seconds: int = 86_400
    overwrite: bool = False


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def conservative_whitespace(text: str) -> str:
    """Normalize whitespace without changing case, punctuation or characters."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines: list[str] = []
    for raw_line in text.split("\n"):
        chars = [
            " " if char == "\t" or char.isspace() and char != "\n" else char
            for char in raw_line
        ]
        line = re.sub(r" +", " ", "".join(chars)).strip()
        if line:
            normalized_lines.append(line)
    return "\n".join(normalized_lines)


def obvious_phi_findings(text: str) -> list[str]:
    return [
        label
        for label, pattern in PHI_PATTERNS
        if pattern.search(text) is not None
    ]


def parse_iso_datetime(value: str, label: str) -> None:
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO-8601") from exc


def decode_report(raw: bytes, label: str) -> str:
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError(f"{label} exceeds {MAX_REPORT_BYTES} bytes")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 text") from exc
    if "\x00" in text:
        raise ValueError(f"{label} contains a NUL byte")
    normalized = conservative_whitespace(text)
    if not normalized:
        raise ValueError(f"{label} is empty after whitespace normalization")
    return normalized


def _require_exact_columns(fieldnames: Iterable[str] | None) -> None:
    actual = tuple(fieldnames or ())
    if set(actual) != set(MANIFEST_COLUMNS) or len(actual) != len(
        MANIFEST_COLUMNS
    ):
        missing = sorted(set(MANIFEST_COLUMNS) - set(actual))
        extra = sorted(set(actual) - set(MANIFEST_COLUMNS))
        raise ValueError(
            f"manifest columns do not match schema; missing={missing}, extra={extra}"
        )


def _resolve_manifest_document(manifest: Path, raw_path: str, line: int) -> Path:
    relative = Path(raw_path)
    if not raw_path or relative.is_absolute():
        raise ValueError(f"line {line}: path must be a non-empty relative path")
    root = manifest.parent.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"line {line}: path escapes manifest directory") from exc
    if not resolved.is_file():
        raise ValueError(f"line {line}: report does not exist")
    if resolved.suffix.casefold() != ".txt":
        raise ValueError(f"line {line}: report must be a UTF-8 .txt file")
    return resolved


def load_governed_documents(
    manifest: Path,
    expected_domain: str,
) -> list[GovernedDocument]:
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            _require_exact_columns(reader.fieldnames)
            raw_rows = [dict(row) for row in reader]
    except OSError as exc:
        raise ValueError(f"manifest is not readable: {manifest}") from exc
    if not raw_rows:
        raise ValueError("manifest has no report rows")

    documents: list[GovernedDocument] = []
    seen_ids: set[str] = set()
    seen_document_hashes: set[str] = set()
    seen_text_hashes: set[str] = set()
    for line, raw_row in enumerate(raw_rows, start=2):
        row = {key: (value or "").strip() for key, value in raw_row.items()}
        case_id = row["id"]
        if not SAFE_ID_RE.fullmatch(case_id):
            raise ValueError(f"line {line}: id must be an opaque safe identifier")
        if case_id in seen_ids:
            raise ValueError(f"line {line}: duplicate id")
        seen_ids.add(case_id)

        if row["dataset_kind"] != EXPECTED_DATASET_KIND:
            raise ValueError(
                f"line {line}: dataset_kind must be {EXPECTED_DATASET_KIND}"
            )
        if row["split"] != "train":
            raise ValueError(
                f"line {line}: validation/test/non-train reports are prohibited"
            )
        if row["domain"] != expected_domain:
            raise ValueError(
                f"line {line}: domain must exactly match {expected_domain!r}"
            )
        if row["reference_status"] != EXPECTED_REFERENCE_STATUS:
            raise ValueError(
                f"line {line}: reference_status must be "
                f"{EXPECTED_REFERENCE_STATUS}"
            )
        if row["approved"] != "true":
            raise ValueError(f"line {line}: approved must be exactly true")
        if row["deidentified"] != "true":
            raise ValueError(f"line {line}: deidentified must be exactly true")
        if not SAFE_ID_RE.fullmatch(row["reviewed_by"]):
            raise ValueError(
                f"line {line}: reviewed_by must be an opaque safe identifier"
            )
        parse_iso_datetime(row["approved_at"], f"line {line}: approved_at")

        declared_document_hash = row["document_sha256"].casefold()
        declared_text_hash = row["text_sha256"].casefold()
        if not SHA256_RE.fullmatch(declared_document_hash):
            raise ValueError(f"line {line}: document_sha256 must be a SHA-256")
        if not SHA256_RE.fullmatch(declared_text_hash):
            raise ValueError(f"line {line}: text_sha256 must be a SHA-256")
        if declared_document_hash in seen_document_hashes:
            raise ValueError(f"line {line}: duplicate document_sha256")
        if declared_text_hash in seen_text_hashes:
            raise ValueError(f"line {line}: duplicate text_sha256")

        path = _resolve_manifest_document(manifest, row["path"], line)
        raw = path.read_bytes()
        actual_document_hash = sha256_bytes(raw)
        if actual_document_hash != declared_document_hash:
            raise ValueError(f"line {line}: document_sha256 mismatch")
        normalized = decode_report(raw, f"line {line}: report")
        actual_text_hash = sha256_text(normalized)
        if actual_text_hash != declared_text_hash:
            raise ValueError(f"line {line}: text_sha256 mismatch")
        phi_findings = obvious_phi_findings(normalized)
        if phi_findings:
            raise ValueError(
                f"line {line}: obvious PHI detected ({', '.join(phi_findings)})"
            )

        seen_document_hashes.add(declared_document_hash)
        seen_text_hashes.add(declared_text_hash)
        documents.append(
            GovernedDocument(
                case_id=case_id,
                path=path,
                document_sha256=actual_document_hash,
                text_sha256=actual_text_hash,
                normalized_text=normalized,
                approved_at=row["approved_at"],
                reviewed_by=row["reviewed_by"],
            )
        )
    return sorted(documents, key=lambda item: item.case_id)


def load_template_defaults(path: Path | None) -> TemplateDefaults | None:
    if path is None:
        return None
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("approved template-default file is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("approved template-default file must be a JSON object")
    if value.get("schema_version") != TEMPLATE_DEFAULTS_SCHEMA_VERSION:
        raise ValueError(
            "template defaults schema_version must be "
            f"{TEMPLATE_DEFAULTS_SCHEMA_VERSION}"
        )
    if value.get("approved") is not True:
        raise ValueError("template defaults must be explicitly approved")
    if value.get("deidentified") is not True:
        raise ValueError("template defaults must be explicitly deidentified")
    reviewed_by = value.get("reviewed_by")
    approved_at = value.get("approved_at")
    if not isinstance(reviewed_by, str) or not SAFE_ID_RE.fullmatch(reviewed_by):
        raise ValueError("template defaults reviewed_by must be an opaque safe ID")
    if not isinstance(approved_at, str):
        raise ValueError("template defaults approved_at is required")
    parse_iso_datetime(approved_at, "template defaults approved_at")
    raw_lines = value.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError("template defaults lines must be a non-empty array")

    lines: set[str] = set()
    for index, raw_line in enumerate(raw_lines):
        if not isinstance(raw_line, str):
            raise ValueError(f"template defaults lines[{index}] must be a string")
        normalized = conservative_whitespace(raw_line)
        if not normalized or "\n" in normalized:
            raise ValueError(
                f"template defaults lines[{index}] must normalize to one line"
            )
        if normalized in lines:
            raise ValueError(f"template defaults lines[{index}] is duplicated")
        findings = obvious_phi_findings(normalized)
        if findings:
            raise ValueError(
                "obvious PHI detected in template defaults "
                f"({', '.join(findings)})"
            )
        lines.add(normalized)
    return TemplateDefaults(
        path=path.resolve(),
        file_sha256=sha256_bytes(raw),
        reviewed_by=reviewed_by,
        approved_at=approved_at,
        lines=frozenset(lines),
    )


def build_corpus_bytes(
    documents: list[GovernedDocument],
    defaults: TemplateDefaults | None,
) -> tuple[bytes, list[dict[str, Any]], int, int]:
    default_lines = defaults.lines if defaults else frozenset()
    corpus_lines: list[str] = []
    inventory: list[dict[str, Any]] = []
    seen_training_hashes: set[str] = set()
    stripped_line_count = 0
    for document in documents:
        source_lines = document.normalized_text.split("\n")
        retained_lines = [line for line in source_lines if line not in default_lines]
        stripped_line_count += len(source_lines) - len(retained_lines)
        if not retained_lines:
            raise ValueError(
                f"document {document.case_id!r} is empty after exact default stripping"
            )
        training_text = "\n".join(retained_lines)
        training_hash = sha256_text(training_text)
        if training_hash in seen_training_hashes:
            raise ValueError(
                "duplicate training text after exact template-default stripping"
            )
        seen_training_hashes.add(training_hash)
        corpus_lines.extend(retained_lines)
        inventory.append(
            {
                "id": document.case_id,
                "document_sha256": document.document_sha256,
                "text_sha256": document.text_sha256,
                "training_text_sha256": training_hash,
                "reviewed_by": document.reviewed_by,
                "approved_at": document.approved_at,
                "sentence_count": len(retained_lines),
            }
        )
    corpus = ("\n".join(corpus_lines) + "\n").encode("utf-8")
    return corpus, inventory, stripped_line_count, len(corpus_lines)


def validate_executable(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} executable does not exist: {resolved}")
    if os.name != "nt" and not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} is not executable: {resolved}")
    return resolved


def _run_tool(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    stdin: Any = subprocess.DEVNULL,
    stdout: Any = subprocess.PIPE,
) -> None:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=environment,
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{Path(command[0]).name} could not complete") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"{Path(command[0]).name} failed with exit code {result.returncode}"
        )


def _ensure_output_targets(request: BuildRequest) -> None:
    targets = (
        request.output_corpus.resolve(),
        request.output_model.resolve(),
        request.output_lock.resolve(),
    )
    if len(set(targets)) != len(targets):
        raise ValueError("output corpus, model and lock paths must be distinct")
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not request.overwrite:
            raise ValueError(f"output already exists (use --overwrite): {target}")


def _protect_inputs_from_output_overwrite(
    request: BuildRequest,
    documents: list[GovernedDocument],
    *,
    manifest: Path,
    lmplz: Path,
    build_binary: Path,
    defaults: TemplateDefaults | None,
) -> None:
    protected = {
        manifest,
        lmplz,
        build_binary,
        *(document.path.resolve() for document in documents),
    }
    if defaults is not None:
        protected.add(defaults.path)
    for output in (
        request.output_corpus.resolve(),
        request.output_model.resolve(),
        request.output_lock.resolve(),
    ):
        if output in protected:
            raise ValueError("an output path must not overwrite an input file")


def _atomic_copy(source: Path, target: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_file, temporary.open("wb") as output_file:
            shutil.copyfileobj(input_file, output_file, length=4 * 1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(value: dict[str, Any], target: Path) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def run_build(request: BuildRequest) -> dict[str, Any]:
    if not SAFE_DOMAIN_RE.fullmatch(request.domain):
        raise ValueError("domain must be a lowercase safe identifier")
    if not request.kenlm_version.strip() or len(request.kenlm_version) > 200:
        raise ValueError("kenlm_version must be a non-empty pinned version")
    if not re.fullmatch(r"[1-9][0-9]*%?", request.memory):
        raise ValueError("memory must be an integer byte count or percentage")
    if request.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    manifest = request.manifest.resolve()
    if not manifest.is_file():
        raise ValueError(f"manifest does not exist: {manifest}")
    lmplz = validate_executable(request.lmplz, "lmplz")
    build_binary = validate_executable(request.build_binary, "build_binary")
    _ensure_output_targets(request)

    documents = load_governed_documents(manifest, request.domain)
    defaults = load_template_defaults(request.template_defaults)
    _protect_inputs_from_output_overwrite(
        request,
        documents,
        manifest=manifest,
        lmplz=lmplz,
        build_binary=build_binary,
        defaults=defaults,
    )
    (
        corpus_bytes,
        document_inventory,
        stripped_line_count,
        sentence_count,
    ) = build_corpus_bytes(documents, defaults)
    corpus_sha = sha256_bytes(corpus_bytes)

    output_model = request.output_model.resolve()
    with tempfile.TemporaryDirectory(
        prefix=".radiology-kenlm-",
        dir=output_model.parent,
    ) as temporary_directory:
        work = Path(temporary_directory)
        staged_corpus = work / "corpus.txt"
        staged_arpa = work / "model.arpa"
        staged_model = work / "model.bin"
        staged_corpus.write_bytes(corpus_bytes)

        lmplz_command = [
            str(lmplz),
            "-o",
            str(KENLM_ORDER),
            "-S",
            request.memory,
            "-T",
            str(work),
        ]
        with staged_corpus.open("rb") as input_file, staged_arpa.open(
            "wb"
        ) as output_file:
            _run_tool(
                lmplz_command,
                cwd=work,
                timeout_seconds=request.timeout_seconds,
                stdin=input_file,
                stdout=output_file,
            )
        if not staged_arpa.is_file() or staged_arpa.stat().st_size == 0:
            raise RuntimeError("lmplz produced an empty ARPA model")

        build_binary_command = [
            str(build_binary),
            str(staged_arpa),
            str(staged_model),
        ]
        _run_tool(
            build_binary_command,
            cwd=work,
            timeout_seconds=request.timeout_seconds,
        )
        if not staged_model.is_file() or staged_model.stat().st_size == 0:
            raise RuntimeError("build_binary produced an empty model")

        arpa_sha = sha256_file(staged_arpa)
        model_sha = sha256_file(staged_model)
        model_size = staged_model.stat().st_size
        metadata: dict[str, Any] = {
            "schema_version": LOCK_SCHEMA_VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "domain": request.domain,
            "builder": {
                "version": BUILDER_VERSION,
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "python": sys.version.split()[0],
            },
            "inputs": {
                "manifest": {
                    "file": manifest.name,
                    "sha256": sha256_file(manifest),
                    "schema_columns": list(MANIFEST_COLUMNS),
                },
                "template_defaults": (
                    {
                        "file": defaults.path.name,
                        "sha256": defaults.file_sha256,
                        "schema_version": TEMPLATE_DEFAULTS_SCHEMA_VERSION,
                        "approved": True,
                        "deidentified": True,
                        "reviewed_by": defaults.reviewed_by,
                        "approved_at": defaults.approved_at,
                        "line_count": len(defaults.lines),
                    }
                    if defaults
                    else None
                ),
                "documents": document_inventory,
            },
            "corpus": {
                "file": request.output_corpus.name,
                "sha256": corpus_sha,
                "bytes": len(corpus_bytes),
                "document_count": len(documents),
                "sentence_count": sentence_count,
                "exact_template_default_lines_removed": stripped_line_count,
                "normalization_version": NORMALIZATION_VERSION,
                "encoding": "utf-8",
                "document_order": "id_ascending",
            },
            "model": {
                "file": request.output_model.name,
                "format": "kenlm_binary",
                "order": KENLM_ORDER,
                "sha256": model_sha,
                "bytes": model_size,
                "intermediate_arpa_sha256": arpa_sha,
            },
            "tools": {
                "lmplz": {
                    "file": lmplz.name,
                    "version": request.kenlm_version,
                    "binary_sha256": sha256_file(lmplz),
                },
                "build_binary": {
                    "file": build_binary.name,
                    "version": request.kenlm_version,
                    "binary_sha256": sha256_file(build_binary),
                },
            },
            "config": {
                "order": KENLM_ORDER,
                "memory": request.memory,
                "timeout_seconds": request.timeout_seconds,
                "environment": {"LC_ALL": "C", "LANG": "C"},
                "lmplz_argv": [
                    lmplz.name,
                    "-o",
                    str(KENLM_ORDER),
                    "-S",
                    request.memory,
                    "-T",
                    "<temporary_directory>",
                ],
                "build_binary_argv": [
                    build_binary.name,
                    "<model.arpa>",
                    "<model.bin>",
                ],
            },
        }

        # The lock is published last.  A model without its matching lock is
        # therefore never considered promotable after an interrupted build.
        _atomic_copy(staged_corpus, request.output_corpus.resolve())
        _atomic_copy(staged_model, output_model)
        _atomic_write_json(metadata, request.output_lock.resolve())
        return metadata


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Build a governed radiology 5-gram KenLM model from approved, "
            "deidentified train reports"
        )
    )
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--domain", required=True)
    result.add_argument("--lmplz", type=Path, required=True)
    result.add_argument("--build-binary", type=Path, required=True)
    result.add_argument(
        "--kenlm-version",
        required=True,
        help="pinned KenLM release or source commit recorded in the lock",
    )
    result.add_argument("--output-corpus", type=Path, required=True)
    result.add_argument("--output-model", type=Path, required=True)
    result.add_argument("--output-lock", type=Path, required=True)
    result.add_argument(
        "--approved-template-defaults",
        dest="template_defaults",
        type=Path,
        help=(
            "optional approved JSON; only its exact case-sensitive lines "
            "are stripped"
        ),
    )
    result.add_argument("--memory", default="50%")
    result.add_argument("--timeout-seconds", type=int, default=86_400)
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        metadata = run_build(
            BuildRequest(
                manifest=args.manifest,
                domain=args.domain,
                lmplz=args.lmplz,
                build_binary=args.build_binary,
                kenlm_version=args.kenlm_version,
                output_corpus=args.output_corpus,
                output_model=args.output_model,
                output_lock=args.output_lock,
                template_defaults=args.template_defaults,
                memory=args.memory,
                timeout_seconds=args.timeout_seconds,
                overwrite=args.overwrite,
            )
        )
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "lock": str(args.output_lock.resolve()),
                "corpus_sha256": metadata["corpus"]["sha256"],
                "model_sha256": metadata["model"]["sha256"],
                "document_count": metadata["corpus"]["document_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
