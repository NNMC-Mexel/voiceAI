#!/usr/bin/env python3
"""Persistent, reproducible GigaAM HTTP inference server.

The server keeps one model in memory and deliberately serializes access to it.
Audio conversion may happen concurrently, but a shared CUDA model is never called
from two request threads at once.

Environment:
  GIGAAM_MODEL                       model name or fine-tuned .ckpt path
  GIGAAM_DEVICE                      cuda | cpu
  GIGAAM_SERVER_HOST                 bind host (default: 127.0.0.1)
  GIGAAM_SERVER_PORT                 bind port
  GIGAAM_ALLOW_REMOTE_BIND           explicit opt-in for non-loopback bind
  GIGAAM_ALLOW_AUDIO_PATH            enable server-side paths (default: false)
  GIGAAM_AUDIO_PATH_ROOT             required allowlisted root when enabled
  GIGAAM_DOWNLOAD_ROOT               model cache directory
  GIGAAM_MODEL_CHECKSUM              optional md5:<hex> or sha256:<hex>
  GIGAAM_STRICT_RUNTIME_LOCK         fail when installed source commit is unknown/mismatched
  GIGAAM_HASH_CHECKPOINT             calculate SHA-256 at startup (default: true)
  GIGAAM_LONGFORM_MODE               auto | vad | overlap
  GIGAAM_CTC_DECODER                 greedy | beam
  GIGAAM_CTC_BEAM_PLUGIN             module:factory (required for beam)
  GIGAAM_CTC_LM_PATH                 optional on-prem n-gram LM file
  GIGAAM_CTC_LM_ALPHA/BETA           shallow-fusion parameters
  GIGAAM_CTC_BEAM_WIDTH              prefix beam width
  GIGAAM_CTC_CONTEXTS_PATH           approved scope -> hotword JSON
  GIGAAM_CTC_HOTWORD_WEIGHT          scoped phrase boost
  GIGAAM_VAD_MODEL_ID                local VAD model (default: pyannote/segmentation-3.0)
  GIGAAM_VAD_MODEL_REVISION          immutable local snapshot revision
  GIGAAM_VAD_TARGET_SECONDS          target speech window (default: 20)
  GIGAAM_VAD_HARD_MAX_SECONDS        hard speech window limit (default: 24)
  GIGAAM_VAD_PADDING_SECONDS         context around VAD windows (default: 0.25)
  GIGAAM_LONGFORM_STRICT             fail closed when VAD/CTC emissions are unavailable
  GIGAAM_FALLBACK_CHUNK_SECONDS      continuous-speech chunk size (default: 20)
  GIGAAM_FALLBACK_OVERLAP_SECONDS    overlap for fixed chunks (default: 2)
  GIGAAM_LOW_CONFIDENCE_LOGPROB      token-peak threshold (default: -1.0)
  GIGAAM_MAX_REQUEST_BYTES           JSON body limit (default: 64 MiB)
  GIGAAM_MAX_AUDIO_BYTES             decoded audio limit (default: 48 MiB)
  VOICEMED_SOURCE_COMMIT             immutable 40-char app commit for container builds
  VOICEMED_SOURCE_DIRTY              true | false paired with source commit

Endpoints:
  GET  /health
  GET  /metadata
  POST /transcribe  {"audio_base64":"..."} or {"audio_path":"/absolute/path"}

Confidence semantics:
  * CTC short/long-form inference exposes the mean of collapsed-token peak log
    probabilities and word-level geometric mean probabilities.
  * Long-form CTC windows are stitched in emission time and decoded once. Text
    overlap de-duplication is used only by an explicitly degraded non-CTC fallback.
  * GigaAM's public RNNT API does not expose comparable emissions. Those
    responses explicitly return confidence.available=false.
  * Scores are acoustic decoder scores, not calibrated clinical certainty.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import importlib.metadata
import ipaddress
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Callable, Iterable, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("gigaam-server")

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[2]

MODEL_SPEC = os.environ.get("GIGAAM_MODEL", "v3_ctc").strip()
DEVICE = os.environ.get("GIGAAM_DEVICE", "cuda").strip()
PORT = int(os.environ.get("GIGAAM_SERVER_PORT", "9001"))
HOST = os.environ.get("GIGAAM_SERVER_HOST", "127.0.0.1").strip()
ALLOW_REMOTE_BIND = os.environ.get(
    "GIGAAM_ALLOW_REMOTE_BIND", "false"
).strip().lower() in {"1", "true", "yes", "on"}
ALLOW_AUDIO_PATH = os.environ.get(
    "GIGAAM_ALLOW_AUDIO_PATH", "false"
).strip().lower() in {"1", "true", "yes", "on"}
AUDIO_PATH_ROOT = os.environ.get("GIGAAM_AUDIO_PATH_ROOT", "").strip()
DOWNLOAD_ROOT = Path(
    os.path.expanduser(
        os.environ.get("GIGAAM_DOWNLOAD_ROOT", "~/.cache/gigaam")
    )
).resolve()
LONGFORM_MODE = os.environ.get("GIGAAM_LONGFORM_MODE", "auto").strip().lower()
VAD_MODEL_ID = os.environ.get(
    "GIGAAM_VAD_MODEL_ID", "pyannote/segmentation-3.0"
).strip()
VAD_MODEL_REVISION = os.environ.get(
    "GIGAAM_VAD_MODEL_REVISION",
    "e66f3d3b9eb0873085418a7b813d3b369bf160bb",
).strip()
VAD_TARGET_SECONDS = float(os.environ.get("GIGAAM_VAD_TARGET_SECONDS", "20"))
VAD_HARD_MAX_SECONDS = float(
    os.environ.get("GIGAAM_VAD_HARD_MAX_SECONDS", "24")
)
VAD_PADDING_SECONDS = float(os.environ.get("GIGAAM_VAD_PADDING_SECONDS", "0.25"))
CTC_DECODER_MODE = os.environ.get("GIGAAM_CTC_DECODER", "greedy").strip().lower()
CTC_BEAM_PLUGIN = os.environ.get("GIGAAM_CTC_BEAM_PLUGIN", "").strip()
CTC_LM_PATH = os.environ.get("GIGAAM_CTC_LM_PATH", "").strip()
CTC_LM_ALPHA = float(os.environ.get("GIGAAM_CTC_LM_ALPHA", "0.5"))
CTC_LM_BETA = float(os.environ.get("GIGAAM_CTC_LM_BETA", "1.0"))
CTC_BEAM_WIDTH = int(os.environ.get("GIGAAM_CTC_BEAM_WIDTH", "32"))
CTC_CONTEXTS_PATH = os.environ.get("GIGAAM_CTC_CONTEXTS_PATH", "").strip()
CTC_HOTWORD_WEIGHT = float(
    os.environ.get("GIGAAM_CTC_HOTWORD_WEIGHT", "8.0")
)
LOW_CONFIDENCE_LOGPROB = float(
    os.environ.get("GIGAAM_LOW_CONFIDENCE_LOGPROB", "-1.0")
)
MAX_REQUEST_BYTES = int(
    os.environ.get("GIGAAM_MAX_REQUEST_BYTES", str(64 * 1024 * 1024))
)
MAX_AUDIO_BYTES = int(
    os.environ.get("GIGAAM_MAX_AUDIO_BYTES", str(48 * 1024 * 1024))
)
FALLBACK_CHUNK_SECONDS = max(
    5,
    min(int(os.environ.get("GIGAAM_FALLBACK_CHUNK_SECONDS", "20")), 24),
)
FALLBACK_OVERLAP_SECONDS = max(
    0.0,
    min(
        float(os.environ.get("GIGAAM_FALLBACK_OVERLAP_SECONDS", "2")),
        FALLBACK_CHUNK_SECONDS - 1.0,
    ),
)

MODEL: Any = None
GIGAAM_MODULE: Any = None
MODEL_LOCK = threading.Lock()
MODEL_METADATA: dict[str, Any] = {}
RUNTIME_METADATA: dict[str, Any] = {}
VAD_METADATA: dict[str, Any] = {}
VAD_PIPELINE: Any = None
CTC_DECODER_METADATA: dict[str, Any] = {}
CTC_CONTEXTS: dict[str, list[str]] = {}
CTC_BEAM_BACKEND: Any = None
SERVER_STARTED_AT = time.time()


@dataclass
class InferenceResult:
    text: str
    words: list[dict[str, Any]]
    chunks: int
    chunk_details: list[dict[str, Any]]
    confidence: dict[str, Any]
    method: str
    integrity: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioWindow:
    start: float
    end: float
    source: str


@dataclass
class CTCEmission:
    log_probs: Any
    frame_shift: float
    duration: float


@dataclass
class CTCEmissionWindow:
    window: AudioWindow
    emission: CTCEmission
    elapsed: float


@dataclass
class StitchedCTCEmission:
    log_probs: Any
    frame_times: list[tuple[float, float]]
    ownership: list[tuple[float, float]]
    seam_conflicts: list[dict[str, Any]]


class ConfidenceUnavailable(RuntimeError):
    """Raised when the installed model does not expose CTC emissions."""


class RuntimeDependencyUnavailable(RuntimeError):
    """Raised when production inference cannot satisfy its locked runtime contract."""


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def longform_strict_mode() -> bool:
    """Use the runtime lock as the production default, with an explicit override."""

    return env_bool(
        "GIGAAM_LONGFORM_STRICT",
        env_bool("GIGAAM_STRICT_RUNTIME_LOCK", False),
    )


def inference_integrity(
    *,
    degraded: bool = False,
    reasons: Optional[list[str]] = None,
    seam_conflicts: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    conflicts = list(seam_conflicts or [])
    critical_conflict = any(bool(item.get("critical")) for item in conflicts)
    degradation_reasons = sorted(set(reasons or []))
    return {
        "degraded": degraded,
        "degradation_reasons": degradation_reasons,
        "seam_conflicts": conflicts,
        "critical_seam_conflict": critical_conflict,
        "approval_blocked": degraded or critical_conflict,
        "benchmark_eligible": not degraded,
        "training_eligible": not degraded and not critical_conflict,
    }


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_security_configuration() -> None:
    if not is_loopback_host(HOST) and not ALLOW_REMOTE_BIND:
        raise RuntimeError(
            "Non-loopback GIGAAM_SERVER_HOST requires "
            "GIGAAM_ALLOW_REMOTE_BIND=true and an authenticated reverse proxy/ACL"
        )
    if MAX_REQUEST_BYTES <= 0 or MAX_REQUEST_BYTES > 512 * 1024 * 1024:
        raise RuntimeError("GIGAAM_MAX_REQUEST_BYTES must be between 1 and 512 MiB")
    if MAX_AUDIO_BYTES <= 0 or MAX_AUDIO_BYTES > MAX_REQUEST_BYTES:
        raise RuntimeError(
            "GIGAAM_MAX_AUDIO_BYTES must be positive and not exceed request limit"
        )
    if LONGFORM_MODE not in {"auto", "vad", "overlap"}:
        raise RuntimeError("GIGAAM_LONGFORM_MODE must be auto, vad, or overlap")
    if not VAD_MODEL_ID:
        raise RuntimeError("GIGAAM_VAD_MODEL_ID cannot be empty")
    if not 5.0 <= VAD_TARGET_SECONDS <= 20.0:
        raise RuntimeError("GIGAAM_VAD_TARGET_SECONDS must be in the range 5..20")
    if not VAD_TARGET_SECONDS <= VAD_HARD_MAX_SECONDS <= 24.0:
        raise RuntimeError(
            "GIGAAM_VAD_HARD_MAX_SECONDS must be between target and 24"
        )
    if not 0.0 <= VAD_PADDING_SECONDS <= 1.0:
        raise RuntimeError("GIGAAM_VAD_PADDING_SECONDS must be in the range 0..1")
    if FALLBACK_CHUNK_SECONDS > VAD_HARD_MAX_SECONDS:
        raise RuntimeError(
            "GIGAAM_FALLBACK_CHUNK_SECONDS cannot exceed the VAD hard limit"
        )
    if longform_strict_mode() and LONGFORM_MODE == "overlap":
        raise RuntimeError(
            "Strict long-form mode requires VAD; explicit overlap mode is dev-only"
        )
    if ALLOW_AUDIO_PATH:
        if not AUDIO_PATH_ROOT:
            raise RuntimeError(
                "GIGAAM_AUDIO_PATH_ROOT is required when audio_path is enabled"
            )
        root = Path(AUDIO_PATH_ROOT).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError("GIGAAM_AUDIO_PATH_ROOT must be an existing directory")


def resolve_allowed_audio_path(
    requested_path: str,
    *,
    enabled: Optional[bool] = None,
    root_value: Optional[str] = None,
) -> Path:
    """Resolve an audio path under the configured root without traversal."""
    use_enabled = ALLOW_AUDIO_PATH if enabled is None else enabled
    configured_root = AUDIO_PATH_ROOT if root_value is None else root_value
    if not use_enabled:
        raise ValueError("audio_path is disabled")
    if not configured_root:
        raise ValueError("audio_path root is not configured")

    root = Path(configured_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("audio_path root is unavailable")
    supplied = Path(requested_path).expanduser()
    candidate = (
        supplied.resolve()
        if supplied.is_absolute()
        else (root / supplied).resolve()
    )
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("audio_path is outside the allowed root") from exc
    if not candidate.is_file():
        raise ValueError("Audio file not found")
    return candidate


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def hash_file(path: Path, algorithms: Iterable[str]) -> dict[str, str]:
    requested = tuple(dict.fromkeys(algorithms))
    hashers = {name: hashlib.new(name) for name in requested}
    with path.open("rb") as source:
        while True:
            block = source.read(4 * 1024 * 1024)
            if not block:
                break
            for hasher in hashers.values():
                hasher.update(block)
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return sha256_text(text)


def load_runtime_lock() -> tuple[Path, dict[str, Any]]:
    configured = os.environ.get("GIGAAM_RUNTIME_LOCK", "").strip()
    path = (
        Path(configured).expanduser().resolve()
        if configured
        else SCRIPT_DIR / "gigaam-runtime.lock.json"
    )
    if not path.is_file():
        logger.warning("Runtime lock file not found: %s", path)
        return path, {}
    with path.open("r", encoding="utf-8") as source:
        return path, json.load(source)


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_runtime_lock_contract(
    lock_path: Path,
    runtime_lock: dict[str, Any],
    strict: bool,
) -> None:
    if not strict:
        return
    if not lock_path.is_file() or not runtime_lock:
        raise RuntimeError(
            "GIGAAM_STRICT_RUNTIME_LOCK=true requires a readable runtime lock"
        )
    if runtime_lock.get("schemaVersion") != 1:
        raise RuntimeError("Unsupported or missing runtime lock schemaVersion")
    expected_commit = str(runtime_lock.get("gigaamGitCommit") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise RuntimeError("Runtime lock must contain a full GigaAM Git commit")

    requirements = runtime_lock.get("requirements")
    if not isinstance(requirements, dict):
        raise RuntimeError("Runtime lock requirements object is missing")
    expected_python = str(requirements.get("python") or "").strip()
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if not expected_python or actual_python != expected_python:
        raise RuntimeError(
            f"Python runtime is {actual_python}, expected {expected_python or 'missing'}"
        )
    for file_key, hash_key in (
        ("runtime", "runtimeSha256"),
        ("optionalLongform", "optionalLongformSha256"),
    ):
        filename = str(requirements.get(file_key) or "").strip()
        expected_hash = str(requirements.get(hash_key) or "").strip().lower()
        if not filename or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise RuntimeError(
                f"Runtime lock must pin {file_key} filename and SHA-256"
            )
        requirements_path = (lock_path.parent / filename).resolve()
        if not requirements_path.is_file():
            raise RuntimeError(f"Locked requirements file not found: {filename}")
        actual_hash = canonical_text_sha256(requirements_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Locked requirements checksum mismatch for {filename}"
            )


def validate_locked_component_versions(
    runtime_lock: dict[str, Any],
    component_versions: dict[str, Optional[str]],
    strict: bool,
) -> None:
    if not strict:
        return
    expected = runtime_lock.get("components")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("Runtime lock must contain pinned component versions")
    mismatches = [
        f"{name}={component_versions.get(name) or 'missing'} "
        f"(expected {version})"
        for name, version in expected.items()
        if component_versions.get(name) != version
    ]
    optional = runtime_lock.get("optionalComponents") or {}
    if not isinstance(optional, dict):
        raise RuntimeError("Runtime lock optionalComponents must be an object")
    for name, version in optional.items():
        actual = component_versions.get(name)
        if actual is not None and actual != version:
            mismatches.append(f"{name}={actual} (expected {version})")
        if LONGFORM_MODE == "vad" and actual is None:
            mismatches.append(f"{name}=missing (required for longform mode vad)")
    if mismatches:
        raise RuntimeError(
            "Runtime component version mismatch: " + "; ".join(mismatches)
        )


def module_artifact_metadata(module_name: str, module: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"module": module_name}
    module_file = getattr(module, "__file__", None)
    if module_file:
        path = Path(module_file).resolve()
        if path.is_file():
            metadata.update(
                {
                    "file": path.name,
                    "sha256": hash_file(path, ("sha256",))["sha256"],
                }
            )
    top_level = module_name.split(".", 1)[0]
    distributions = importlib.metadata.packages_distributions().get(top_level, [])
    metadata["distributions"] = {
        distribution: package_version(distribution)
        for distribution in sorted(distributions)
    }
    return metadata


def installed_vcs_commit(distribution_name: str) -> Optional[str]:
    """Read the PEP 610 commit recorded for a direct-VCS installation."""
    try:
        distribution = importlib.metadata.distribution(distribution_name)
        raw = distribution.read_text("direct_url.json")
        if not raw:
            return None
        direct_url = json.loads(raw)
        commit = (direct_url.get("vcs_info") or {}).get("commit_id")
        return str(commit) if commit else None
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        return None


def command_first_line(command: list[str], timeout: int = 5) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else None


def project_git_metadata() -> dict[str, Any]:
    embedded_commit = os.environ.get("VOICEMED_SOURCE_COMMIT", "").strip().lower()
    embedded_dirty = os.environ.get("VOICEMED_SOURCE_DIRTY", "").strip().lower()
    if embedded_commit or embedded_dirty:
        if not re.fullmatch(r"[0-9a-f]{40}", embedded_commit):
            raise RuntimeError(
                "VOICEMED_SOURCE_COMMIT must be a full 40-character Git commit"
            )
        if embedded_dirty not in {"true", "false"}:
            raise RuntimeError(
                "VOICEMED_SOURCE_DIRTY must be true or false when source identity is embedded"
            )
        return {
            "commit": embedded_commit,
            "dirty": embedded_dirty == "true",
        }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return {"commit": commit, "dirty": bool(dirty_result.stdout.strip())}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def canonical_model_name(model_spec: str) -> str:
    aliases = {
        "ctc": "v3_ctc",
        "rnnt": "v3_rnnt",
        "e2e_ctc": "v3_e2e_ctc",
        "e2e_rnnt": "v3_e2e_rnnt",
        "ssl": "v3_ssl",
    }
    return aliases.get(model_spec, model_spec)


def checkpoint_path_for(model_spec: str) -> Optional[Path]:
    local = Path(os.path.expanduser(model_spec))
    if local.is_file():
        return local.resolve()
    canonical = canonical_model_name(model_spec)
    candidate = DOWNLOAD_ROOT / f"{canonical}.ckpt"
    return candidate if candidate.is_file() else None


def parse_checksum(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(
            "GIGAAM_MODEL_CHECKSUM must be prefixed with md5: or sha256:"
        )
    algorithm, expected = value.split(":", 1)
    algorithm = algorithm.strip().lower()
    expected = expected.strip().lower()
    if algorithm not in {"md5", "sha256"}:
        raise ValueError("Unsupported model checksum algorithm")
    expected_length = 32 if algorithm == "md5" else 64
    if len(expected) != expected_length or not re.fullmatch(r"[0-9a-f]+", expected):
        raise ValueError("Invalid model checksum")
    return algorithm, expected


def expected_model_checksum(
    runtime_lock: dict[str, Any], canonical_name: str
) -> Optional[str]:
    configured = os.environ.get("GIGAAM_MODEL_CHECKSUM", "").strip()
    if configured:
        return configured
    model_entry = (runtime_lock.get("models") or {}).get(canonical_name) or {}
    expected_md5 = model_entry.get("md5")
    return f"md5:{expected_md5}" if expected_md5 else None


def load_contexts(path_value: str) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if not path_value:
        return {}, {"configured": False, "sha256": None, "scopes": {}}
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"CTC contexts file not found: {path}")
    with path.open("r", encoding="utf-8") as source:
        raw = json.load(source)
    if not isinstance(raw, dict):
        raise RuntimeError("CTC contexts JSON must be an object of scope -> phrases")

    contexts: dict[str, list[str]] = {}
    for scope, phrases in raw.items():
        if not isinstance(scope, str) or not re.fullmatch(r"[a-z0-9_.-]{1,80}", scope):
            raise RuntimeError(f"Invalid CTC context scope: {scope!r}")
        if not isinstance(phrases, list):
            raise RuntimeError(f"CTC context {scope!r} must be a list")
        normalized: list[str] = []
        for phrase in phrases:
            if not isinstance(phrase, str):
                raise RuntimeError(f"CTC context {scope!r} contains a non-string phrase")
            clean = " ".join(phrase.split())
            if not clean or len(clean) > 200:
                raise RuntimeError(f"CTC context {scope!r} has an invalid phrase")
            normalized.append(clean)
        unique = list(dict.fromkeys(normalized))
        if len(unique) > 1000:
            raise RuntimeError(f"CTC context {scope!r} exceeds 1000 phrases")
        contexts[scope] = unique
    return contexts, {
        "configured": True,
        "file": path.name,
        "sha256": hash_file(path, ("sha256",))["sha256"],
        "scopes": {scope: len(phrases) for scope, phrases in contexts.items()},
    }


def initialize_ctc_decoder() -> None:
    """Initialize a fail-closed optional beam decoder plugin.

    Plugin factory contract:
      factory(labels, blank_id, lm_path, alpha, beta) -> backend

    Backend contract:
      backend.decode(log_probs, beam_width, hotwords, hotword_weight) -> dict

    The result dict must contain ``text`` and may contain word frame spans,
    acoustic/fused decoder scores, and score-derived confidence. A configured
    beam mode never silently falls back to greedy.
    """
    global CTC_BEAM_BACKEND, CTC_DECODER_METADATA, CTC_CONTEXTS

    if CTC_DECODER_MODE not in {"greedy", "beam"}:
        raise RuntimeError("GIGAAM_CTC_DECODER must be greedy or beam")
    if CTC_BEAM_WIDTH < 1 or CTC_BEAM_WIDTH > 1024:
        raise RuntimeError("GIGAAM_CTC_BEAM_WIDTH must be in the range 1..1024")
    for name, value in (
        ("GIGAAM_CTC_LM_ALPHA", CTC_LM_ALPHA),
        ("GIGAAM_CTC_LM_BETA", CTC_LM_BETA),
        ("GIGAAM_CTC_HOTWORD_WEIGHT", CTC_HOTWORD_WEIGHT),
        ("GIGAAM_LOW_CONFIDENCE_LOGPROB", LOW_CONFIDENCE_LOGPROB),
    ):
        if not math.isfinite(value):
            raise RuntimeError(f"{name} must be finite")

    CTC_CONTEXTS, contexts_metadata = load_contexts(CTC_CONTEXTS_PATH)
    lm_path: Optional[Path] = None
    if CTC_LM_PATH:
        lm_path = Path(CTC_LM_PATH).expanduser().resolve()
        if not lm_path.is_file():
            raise RuntimeError(f"CTC language model not found: {lm_path}")

    if CTC_DECODER_MODE == "greedy":
        if CTC_BEAM_PLUGIN or lm_path or CTC_CONTEXTS:
            raise RuntimeError(
                "Beam/LM/context configuration was supplied while "
                "GIGAAM_CTC_DECODER=greedy; refusing to pretend it is active"
            )
        CTC_DECODER_METADATA = {
            "mode": "greedy",
            "active": True,
            "implementation": "gigaam_ctc_greedy",
            "beamWidth": 1,
            "languageModel": {"active": False},
            "contexts": contexts_metadata,
        }
        return

    if not is_ctc_model():
        raise RuntimeError("GIGAAM_CTC_DECODER=beam requires a CTC acoustic model")
    if not CTC_BEAM_PLUGIN or ":" not in CTC_BEAM_PLUGIN:
        raise RuntimeError(
            "GIGAAM_CTC_BEAM_PLUGIN=module:factory is required for beam mode"
        )

    tokenizer = getattr(getattr(MODEL, "decoding", None), "tokenizer", None)
    blank_id = getattr(getattr(MODEL, "decoding", None), "blank_id", None)
    if tokenizer is None or blank_id is None:
        raise RuntimeError("CTC tokenizer/blank_id is unavailable")
    labels = [str(tokenizer.id_to_str(index)) for index in range(len(tokenizer))]
    if int(blank_id) != len(labels):
        raise RuntimeError(
            "Beam plugin currently requires the CTC blank to be the last class"
        )
    labels.append("")

    module_name, factory_name = CTC_BEAM_PLUGIN.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        CTC_BEAM_BACKEND = factory(
            labels=labels,
            blank_id=int(blank_id),
            lm_path=str(lm_path) if lm_path else None,
            alpha=CTC_LM_ALPHA,
            beta=CTC_LM_BETA,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize CTC beam plugin {CTC_BEAM_PLUGIN}: {exc}"
        ) from exc
    if not callable(getattr(CTC_BEAM_BACKEND, "decode", None)):
        raise RuntimeError("CTC beam plugin backend must expose decode(...)")

    backend_metadata: dict[str, Any] = {}
    metadata_method = getattr(CTC_BEAM_BACKEND, "metadata", None)
    if callable(metadata_method):
        reported = metadata_method()
        if not isinstance(reported, dict):
            raise RuntimeError("CTC beam plugin metadata() must return an object")
        backend_metadata = reported
    CTC_DECODER_METADATA = {
        "mode": "beam",
        "active": True,
        "implementation": CTC_BEAM_PLUGIN,
        "implementationArtifact": module_artifact_metadata(module_name, module),
        "implementationMetadata": backend_metadata,
        "beamWidth": CTC_BEAM_WIDTH,
        "languageModel": {
            "active": lm_path is not None,
            "file": lm_path.name if lm_path else None,
            "sha256": hash_file(lm_path, ("sha256",))["sha256"] if lm_path else None,
            "alpha": CTC_LM_ALPHA if lm_path else None,
            "beta": CTC_LM_BETA if lm_path else None,
        },
        "contexts": contexts_metadata,
        "hotwordWeight": CTC_HOTWORD_WEIGHT if CTC_CONTEXTS else None,
    }


def model_supports_ctc_emissions(model: Any = None) -> bool:
    selected = MODEL if model is None else model
    decoding = getattr(selected, "decoding", None)
    return bool(
        selected is not None
        and all(hasattr(selected, name) for name in ("prepare_wav", "forward", "head"))
        and getattr(decoding, "tokenizer", None) is not None
        and getattr(decoding, "blank_id", None) is not None
    )


def initialize_vad_runtime(*, strict: bool, load_pipeline: bool) -> None:
    """Resolve the local VAD snapshot and optionally warm its pipeline.

    Production never downloads a model implicitly. Development may discover and
    use the same local snapshot lazily; if it is absent, inference is marked as
    degraded and uses the deterministic continuous-speech fallback.
    """

    global VAD_METADATA, VAD_PIPELINE
    metadata: dict[str, Any] = {
        "modelId": VAD_MODEL_ID,
        "requestedRevision": VAD_MODEL_REVISION or None,
        "available": False,
        "loaded": False,
        "localOnly": True,
    }
    try:
        from huggingface_hub import snapshot_download

        local_path = Path(
            snapshot_download(
                repo_id=VAD_MODEL_ID,
                revision=VAD_MODEL_REVISION or None,
                local_files_only=True,
            )
        ).resolve()
        actual_revision = local_path.name
        metadata.update(
            {
                "available": True,
                "snapshotRevision": actual_revision,
            }
        )
        if VAD_MODEL_REVISION and actual_revision != VAD_MODEL_REVISION:
            raise RuntimeError(
                "Local VAD snapshot revision mismatch: "
                f"expected {VAD_MODEL_REVISION}, got {actual_revision}"
            )
        if load_pipeline:
            import torch
            from pyannote.audio import Model
            from pyannote.audio.core.task import Problem, Resolution, Specifications
            from pyannote.audio.pipelines import VoiceActivityDetection
            from torch.torch_version import TorchVersion

            with torch.serialization.safe_globals(
                [TorchVersion, Problem, Specifications, Resolution]
            ):
                segmentation = Model.from_pretrained(str(local_path))
            pipeline = VoiceActivityDetection(segmentation=segmentation)
            pipeline.instantiate(
                {"min_duration_on": 0.0, "min_duration_off": 0.0}
            )
            VAD_PIPELINE = pipeline.to(getattr(MODEL, "_device", DEVICE))
            metadata["loaded"] = True
    except Exception as exc:
        metadata["errorType"] = type(exc).__name__
        VAD_METADATA = metadata
        if strict:
            raise RuntimeError(
                "Strict long-form runtime requires a locally available "
                f"{VAD_MODEL_ID} VAD snapshot"
            ) from exc
        logger.warning(
            "Local VAD runtime is unavailable (%s); dev long-form requests "
            "will be degraded",
            type(exc).__name__,
        )
        return
    VAD_METADATA = metadata


def initialize_longform_runtime() -> None:
    strict = longform_strict_mode()
    if strict and not model_supports_ctc_emissions():
        raise RuntimeError(
            "Strict long-form runtime requires a CTC model exposing logits, "
            "tokenizer, and blank_id"
        )
    if LONGFORM_MODE == "overlap":
        # Configuration validation already rejects this combination in strict mode.
        return
    initialize_vad_runtime(strict=strict, load_pipeline=strict)


def initialize_model() -> None:
    """Load the configured model once and collect reproducibility metadata."""
    global MODEL, GIGAAM_MODULE, MODEL_METADATA, RUNTIME_METADATA

    lock_path, runtime_lock = load_runtime_lock()
    expected_commit = os.environ.get(
        "GIGAAM_EXPECTED_GIT_COMMIT",
        str(runtime_lock.get("gigaamGitCommit") or ""),
    ).strip()
    strict_lock = env_bool("GIGAAM_STRICT_RUNTIME_LOCK", False)
    validate_runtime_lock_contract(lock_path, runtime_lock, strict_lock)
    locked_commit = str(runtime_lock.get("gigaamGitCommit") or "").strip()
    if strict_lock and expected_commit != locked_commit:
        raise RuntimeError(
            "Strict mode does not allow GIGAAM_EXPECTED_GIT_COMMIT "
            "to override the runtime lock"
        )
    locked_vad = (
        (runtime_lock.get("models") or {}).get("pyannote_segmentation_3_0") or {}
    )
    locked_vad_id = str(locked_vad.get("repo") or "").strip()
    locked_vad_revision = str(locked_vad.get("revision") or "").strip()
    vad_lock_matches = (
        locked_vad_id == VAD_MODEL_ID
        and locked_vad_revision == VAD_MODEL_REVISION
    )
    if not vad_lock_matches:
        message = (
            "VAD model identity does not match the runtime lock: "
            f"{VAD_MODEL_ID}@{VAD_MODEL_REVISION}"
        )
        if strict_lock:
            raise RuntimeError(message)
        logger.warning(message)

    logger.info("Loading GigaAM model '%s' on %s ...", MODEL_SPEC, DEVICE)
    started = time.perf_counter()
    try:
        import gigaam
        import torch

        GIGAAM_MODULE = gigaam
        MODEL = gigaam.load_model(
            MODEL_SPEC,
            device=DEVICE,
            download_root=str(DOWNLOAD_ROOT),
        )
        MODEL.eval()
    except Exception:
        logger.exception("Failed to load GigaAM model")
        raise

    installed_commit = installed_vcs_commit("gigaam")
    if expected_commit and installed_commit != expected_commit:
        message = (
            "Installed GigaAM source commit is "
            f"{installed_commit or 'unknown'}, expected {expected_commit}"
        )
        if strict_lock:
            raise RuntimeError(message)
        logger.warning("%s; set GIGAAM_STRICT_RUNTIME_LOCK=true to fail closed", message)

    canonical_name = canonical_model_name(MODEL_SPEC)
    checkpoint_path = checkpoint_path_for(MODEL_SPEC)
    expected_checksum = expected_model_checksum(runtime_lock, canonical_name)
    if strict_lock and not expected_checksum:
        raise RuntimeError(
            "Strict mode requires a locked model checksum; for a fine-tuned "
            "checkpoint set GIGAAM_MODEL_CHECKSUM=sha256:<hex>"
        )
    algorithms: list[str] = []
    parsed_expected: Optional[tuple[str, str]] = None
    if expected_checksum:
        parsed_expected = parse_checksum(expected_checksum)
        algorithms.append(parsed_expected[0])
    if env_bool("GIGAAM_HASH_CHECKPOINT", True):
        algorithms.append("sha256")

    checkpoint_hashes: dict[str, str] = {}
    checkpoint_bytes: Optional[int] = None
    if checkpoint_path and algorithms:
        checkpoint_hashes = hash_file(checkpoint_path, algorithms)
        checkpoint_bytes = checkpoint_path.stat().st_size
    elif checkpoint_path:
        checkpoint_bytes = checkpoint_path.stat().st_size

    if parsed_expected:
        algorithm, expected = parsed_expected
        actual = checkpoint_hashes.get(algorithm)
        if actual is None:
            raise RuntimeError(
                f"Cannot validate {algorithm} checksum: checkpoint path unavailable"
            )
        if actual != expected:
            raise RuntimeError(
                f"GigaAM checkpoint checksum mismatch: expected {expected}, got {actual}"
            )

    initialize_ctc_decoder()
    initialize_longform_runtime()
    decoder_name = type(getattr(MODEL, "decoding", None)).__name__
    model_cfg_name = str(getattr(getattr(MODEL, "cfg", None), "model_name", ""))
    MODEL_METADATA = {
        "requested": (
            Path(MODEL_SPEC).name
            if Path(os.path.expanduser(MODEL_SPEC)).is_file()
            else MODEL_SPEC
        ),
        "name": model_cfg_name or canonical_name,
        "device": str(getattr(MODEL, "_device", DEVICE)),
        "acousticDecoder": decoder_name or None,
        "ctcDecoder": CTC_DECODER_METADATA,
        "checkpoint": {
            # Do not expose local filesystem paths through the API.
            "file": checkpoint_path.name if checkpoint_path else None,
            "bytes": checkpoint_bytes,
            "hashes": checkpoint_hashes,
            "expectedChecksum": expected_checksum,
            "verified": bool(parsed_expected),
        },
    }

    cuda: dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "runtime": getattr(torch.version, "cuda", None),
    }
    if torch.cuda.is_available():
        try:
            cuda["deviceName"] = torch.cuda.get_device_name(
                getattr(MODEL, "_device", DEVICE)
            )
        except Exception:
            cuda["deviceName"] = None

    project_git = project_git_metadata()
    component_versions = {
        name: package_version(name)
        for name in (
            "gigaam",
            "hydra-core",
            "numpy",
            "omegaconf",
            "onnx",
            "onnxruntime",
            "sentencepiece",
            "torch",
            "torchaudio",
            "soundfile",
            "tqdm",
            "numba",
            "pyannote.audio",
            "pyarrow",
            "torchcodec",
            "transformers",
        )
    }
    validate_locked_component_versions(
        runtime_lock,
        component_versions,
        strict_lock,
    )
    ffmpeg_version = command_first_line(["ffmpeg", "-version"])
    source = {
        "gigaamCommit": installed_commit,
        "expectedGigaamCommit": expected_commit or None,
        "runtimeLockSha256": (
            hash_file(lock_path, ("sha256",))["sha256"]
            if lock_path.is_file()
            else None
        ),
        "serverScriptSha256": hash_file(SCRIPT_PATH, ("sha256",))["sha256"],
        "projectCommit": project_git["commit"],
        "projectDirty": project_git["dirty"],
    }
    effective_configuration = {
        "longformMode": LONGFORM_MODE,
        "longformStrict": longform_strict_mode(),
        "vad": VAD_METADATA,
        "vadTargetSeconds": VAD_TARGET_SECONDS,
        "vadHardMaxSeconds": VAD_HARD_MAX_SECONDS,
        "vadPaddingSeconds": VAD_PADDING_SECONDS,
        "fallbackChunkSeconds": FALLBACK_CHUNK_SECONDS,
        "fallbackOverlapSeconds": FALLBACK_OVERLAP_SECONDS,
        "lowConfidenceLogprob": LOW_CONFIDENCE_LOGPROB,
        "maxRequestBytes": MAX_REQUEST_BYTES,
        "maxAudioBytes": MAX_AUDIO_BYTES,
        "strictRuntimeLock": strict_lock,
        "ctcDecoder": CTC_DECODER_METADATA,
    }
    runtime_identity = {
        "model": MODEL_METADATA,
        "components": component_versions,
        "source": source,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": cuda,
        "ffmpeg": ffmpeg_version,
        "configuration": effective_configuration,
    }
    RUNTIME_METADATA = {
        # Persist the complete SHA-256. Interfaces may display a short prefix,
        # but artifact and benchmark bindings must use the full collision
        # domain of the immutable runtime identity.
        "id": sha256_text(
            json.dumps(runtime_identity, ensure_ascii=False, sort_keys=True)
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "components": component_versions,
        "cuda": cuda,
        "ffmpeg": ffmpeg_version,
        "configuration": effective_configuration,
        "source": source,
        "runtimeLock": lock_path.name if lock_path.is_file() else None,
    }
    logger.info(
        "Model loaded in %.1fs | runtime_id=%s | checkpoint=%s",
        time.perf_counter() - started,
        RUNTIME_METADATA["id"],
        json.dumps(MODEL_METADATA["checkpoint"], sort_keys=True),
    )


def check_ffmpeg() -> bool:
    return command_first_line(["ffmpeg", "-version"]) is not None


def convert_to_wav(input_path: str) -> str:
    """Convert arbitrary audio to deterministic 16 kHz mono PCM16 WAV."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-vn",
                "-sn",
                "-dn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                wav_path,
            ],
            capture_output=True,
            timeout=120,
            check=True,
        )
        return wav_path
    except Exception:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        raise


def unavailable_confidence(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "calibrated": False,
    }


def serialize_public_words(words: Any, offset_seconds: float = 0.0) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for word in words or []:
        serialized.append(
            {
                "text": str(word.text),
                "start": round(float(word.start) + offset_seconds, 3),
                "end": round(float(word.end) + offset_seconds, 3),
            }
        )
    return serialized


def words_from_ctc_tokens(
    tokenizer: Any,
    token_ids: list[int],
    token_frames: list[int],
    token_logprobs: list[float],
    frame_shift: float,
    token_end_frames: Optional[list[int]] = None,
    frame_times: Optional[list[tuple[float, float]]] = None,
) -> list[dict[str, Any]]:
    """Group CTC emissions into words while preserving acoustic scores."""
    if token_end_frames is None:
        token_end_frames = [frame + 1 for frame in token_frames]
    if not (
        len(token_ids)
        == len(token_frames)
        == len(token_end_frames)
        == len(token_logprobs)
    ):
        raise ValueError("CTC token metadata lengths do not match")
    words: list[dict[str, Any]] = []
    chars: list[str] = []
    start_times: list[float] = []
    end_times: list[float] = []
    scores: list[float] = []

    def commit() -> None:
        if not chars:
            return
        text = "".join(chars).strip()
        if text and start_times:
            avg_logprob = sum(scores) / len(scores)
            words.append(
                {
                    "text": text,
                    "start": round(start_times[0], 3),
                    "end": round(end_times[-1], 3),
                    "confidence": round(math.exp(avg_logprob), 6),
                    "avg_logprob": round(avg_logprob, 6),
                    "score_type": "ctc_greedy_token_peak_geomean",
                }
            )
        chars.clear()
        start_times.clear()
        end_times.clear()
        scores.clear()

    for token_id, start_frame, end_frame, logprob in zip(
        token_ids,
        token_frames,
        token_end_frames,
        token_logprobs,
    ):
        piece = str(tokenizer.id_to_str(token_id))
        if piece.startswith("▁"):
            commit()
            piece = piece[1:]
        elif piece == " ":
            commit()
            continue
        if not piece:
            continue
        if frame_times is not None:
            if start_frame >= len(frame_times) or end_frame <= 0:
                raise ValueError("CTC frame span is outside the stitched timeline")
            start_time = frame_times[start_frame][0]
            end_time = frame_times[min(end_frame, len(frame_times)) - 1][1]
        else:
            start_time = start_frame * frame_shift
            end_time = end_frame * frame_shift
        chars.append(piece)
        start_times.append(start_time)
        end_times.append(end_time)
        scores.append(logprob)
    commit()
    return words


def ctc_token_runs(
    labels: list[int],
    blank_id: int,
) -> list[tuple[int, int, int]]:
    """Collapse greedy CTC labels into (token, start_frame, end_frame) runs."""

    runs: list[tuple[int, int, int]] = []
    start = 0
    while start < len(labels):
        token_id = int(labels[start])
        end = start + 1
        while end < len(labels) and int(labels[end]) == token_id:
            end += 1
        if token_id != blank_id:
            runs.append((token_id, start, end))
        start = end
    return runs


def decode_with_beam_backend(
    log_probs: Any,
    frame_shift: float,
    hotwords: list[str],
    backend: Any = None,
    frame_times: Optional[list[tuple[float, float]]] = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Invoke the configured beam plugin and validate its auditable result."""
    selected_backend = backend if backend is not None else CTC_BEAM_BACKEND
    if selected_backend is None:
        raise RuntimeError("CTC beam backend is not initialized")
    raw = selected_backend.decode(
        log_probs,
        beam_width=CTC_BEAM_WIDTH,
        hotwords=hotwords,
        hotword_weight=CTC_HOTWORD_WEIGHT,
    )
    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
        raise RuntimeError("CTC beam backend must return an object with string text")

    words: list[dict[str, Any]] = []
    for item in raw.get("words") or []:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise RuntimeError("CTC beam word must contain text")
        try:
            start_frame = int(item["start_frame"])
            end_frame = int(item["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "CTC beam word must contain integer start_frame/end_frame"
            ) from exc
        if start_frame < 0 or end_frame < start_frame:
            raise RuntimeError("CTC beam word has an invalid frame span")
        if frame_times is not None:
            if start_frame >= len(frame_times) or end_frame > len(frame_times):
                raise RuntimeError("CTC beam word is outside the stitched timeline")
            start_seconds = frame_times[start_frame][0]
            end_seconds = (
                frame_times[end_frame - 1][1]
                if end_frame > start_frame
                else frame_times[start_frame][1]
            )
        else:
            start_seconds = start_frame * frame_shift
            end_seconds = end_frame * frame_shift
        word: dict[str, Any] = {
            "text": item["text"],
            "start": round(start_seconds, 3),
            "end": round(end_seconds, 3),
        }
        if "avg_logprob" in item and "confidence" in item:
            word["avg_logprob"] = float(item["avg_logprob"])
            word["confidence"] = float(item["confidence"])
            word["score_type"] = str(
                item.get("score_type") or "beam_backend_emission"
            )
        words.append(word)

    decoder_scores: dict[str, Any] = {}
    for source, target in (
        ("acoustic_log_score", "acoustic_log_score"),
        ("fused_log_score", "fused_log_score"),
    ):
        if raw.get(source) is not None:
            decoder_scores[target] = float(raw[source])
    reported_confidence = raw.get("confidence")
    if reported_confidence is None:
        confidence = unavailable_confidence(
            "beam_backend_did_not_provide_aligned_acoustic_confidence"
        )
    elif not isinstance(reported_confidence, dict):
        raise RuntimeError("CTC beam confidence must be an object")
    else:
        confidence = dict(reported_confidence)
        if confidence.get("available"):
            for required in ("avg_logprob", "low_confidence", "method"):
                if required not in confidence:
                    raise RuntimeError(
                        f"CTC beam confidence is missing {required}"
                    )
            confidence["avg_logprob"] = float(confidence["avg_logprob"])
            confidence["low_confidence"] = bool(confidence["low_confidence"])
            confidence["calibrated"] = bool(confidence.get("calibrated", False))
        else:
            confidence = unavailable_confidence(
                str(confidence.get("reason") or "beam_confidence_unavailable")
            )
    if decoder_scores:
        confidence["decoder_scores"] = decoder_scores
    confidence["hotwords_applied"] = len(hotwords)
    return raw["text"].strip(), words, confidence


def is_ctc_model() -> bool:
    decoder_name = type(getattr(MODEL, "decoding", None)).__name__.lower()
    cfg_name = str(getattr(getattr(MODEL, "cfg", None), "model_name", "")).lower()
    return "ctc" in decoder_name or "ctc" in cfg_name


def extract_ctc_emission(wav_path: str) -> CTCEmission:
    """Run one CTC forward pass without decoding chunk text."""

    if not model_supports_ctc_emissions():
        raise ConfidenceUnavailable("installed_model_does_not_expose_ctc_head")

    import torch
    from gigaam.model import SAMPLE_RATE

    with torch.inference_mode():
        wav, wav_len = MODEL.prepare_wav(wav_path)
        encoded, encoded_len = MODEL.forward(wav, wav_len)
        log_probs = MODEL.head(encoder_output=encoded)
        sequence_length = int(encoded_len[0].item())
        selected = log_probs[0, :sequence_length].float().cpu().numpy()
        duration = int(wav_len[0].item()) / SAMPLE_RATE
    return CTCEmission(
        log_probs=selected,
        frame_shift=(duration / sequence_length if sequence_length > 0 else 0.0),
        duration=duration,
    )


def decode_ctc_emission(
    emission: CTCEmission,
    hotwords: Optional[list[str]] = None,
    *,
    frame_times: Optional[list[tuple[float, float]]] = None,
    method: str = "ctc_short",
) -> InferenceResult:
    """Decode a complete short or stitched emission sequence exactly once."""

    decoding = getattr(MODEL, "decoding", None)
    tokenizer = getattr(decoding, "tokenizer", None)
    blank_id = getattr(decoding, "blank_id", None)
    if tokenizer is None or blank_id is None:
        raise ConfidenceUnavailable("installed_decoder_does_not_expose_ctc_tokens")

    log_probs = emission.log_probs
    sequence_length = int(log_probs.shape[0])
    labels = log_probs.argmax(axis=-1).tolist() if sequence_length else []
    token_runs = ctc_token_runs(labels, int(blank_id))
    token_ids = [token_id for token_id, _, _ in token_runs]
    token_frames = [start for _, start, _ in token_runs]
    token_end_frames = [end for _, _, end in token_runs]
    token_logprobs = [
        float(log_probs[start:end, token_id].max())
        for token_id, start, end in token_runs
    ]

    if CTC_DECODER_MODE == "beam":
        beam_text, beam_words, beam_confidence = decode_with_beam_backend(
            log_probs,
            emission.frame_shift,
            hotwords or [],
            frame_times=frame_times,
        )
        return InferenceResult(
            text=beam_text,
            words=beam_words,
            chunks=1,
            chunk_details=[],
            confidence=beam_confidence,
            method=(
                "ctc_prefix_beam"
                if method == "ctc_short"
                else f"{method}_prefix_beam"
            ),
            integrity=inference_integrity(),
        )

    text = str(tokenizer.decode(token_ids)).strip()
    words = words_from_ctc_tokens(
        tokenizer,
        token_ids,
        token_frames,
        token_logprobs,
        emission.frame_shift,
        token_end_frames,
        frame_times,
    )
    if token_logprobs:
        avg_logprob = sum(token_logprobs) / len(token_logprobs)
        confidence = {
            "available": True,
            "method": (
                "ctc_greedy_token_peaks"
                if frame_times is None
                else "ctc_greedy_token_peaks_stitched_emissions"
            ),
            "avg_logprob": round(avg_logprob, 6),
            "mean_probability": round(math.exp(avg_logprob), 6),
            "low_confidence": avg_logprob < LOW_CONFIDENCE_LOGPROB,
            "threshold_logprob": LOW_CONFIDENCE_LOGPROB,
            "emitted_tokens": len(token_logprobs),
            "calibrated": False,
        }
    else:
        confidence = unavailable_confidence("no_non_blank_ctc_emissions")

    return InferenceResult(
        text=text,
        words=words,
        chunks=1,
        chunk_details=[],
        confidence=confidence,
        method=method,
        integrity=inference_integrity(),
    )


def transcribe_short_ctc_scored(
    wav_path: str, hotwords: Optional[list[str]] = None
) -> InferenceResult:
    """Run one scored CTC forward pass and decode it."""

    return decode_ctc_emission(extract_ctc_emission(wav_path), hotwords)


def call_public_transcribe(wav_path: str) -> tuple[Any, bool]:
    try:
        return MODEL.transcribe(wav_path, word_timestamps=True), True
    except TypeError:
        return MODEL.transcribe(wav_path), False


def transcribe_short(
    wav_path: str, hotwords: Optional[list[str]] = None
) -> InferenceResult:
    if is_ctc_model():
        try:
            return transcribe_short_ctc_scored(wav_path, hotwords)
        except ConfidenceUnavailable as exc:
            logger.warning("CTC confidence unavailable: %s", exc)
            reason = str(exc)
    else:
        reason = "rnnt_public_api_does_not_expose_comparable_emission_scores"

    result, has_timestamps = call_public_transcribe(wav_path)
    return InferenceResult(
        text=str(result.text).strip(),
        words=serialize_public_words(result.words) if has_timestamps else [],
        chunks=1,
        chunk_details=[],
        confidence=unavailable_confidence(reason),
        method="public_short",
        integrity=inference_integrity(),
    )


def fixed_overlap_windows(
    start: float,
    end: float,
    *,
    chunk_seconds: float = 20.0,
    overlap_seconds: float = 2.0,
    source: str = "continuous_fallback",
) -> list[AudioWindow]:
    """Create deterministic fixed windows for speech with no usable pause."""

    if end <= start:
        return []
    if chunk_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("Invalid fixed-overlap window configuration")
    windows: list[AudioWindow] = []
    cursor = start
    step = chunk_seconds - overlap_seconds
    while cursor < end - 1e-9:
        window_end = min(cursor + chunk_seconds, end)
        windows.append(AudioWindow(cursor, window_end, source))
        if window_end >= end - 1e-9:
            break
        cursor += step
    return windows


def _padded_vad_window(
    start: float,
    end: float,
    total_duration: float,
    *,
    padding_seconds: float,
    hard_max_seconds: float,
) -> AudioWindow:
    core_duration = end - start
    if core_duration > hard_max_seconds + 1e-6:
        raise ValueError("VAD core exceeds the hard window limit")
    left_available = min(padding_seconds, max(0.0, start))
    right_available = min(
        padding_seconds,
        max(0.0, total_duration - end),
    )
    padding_budget = max(0.0, hard_max_seconds - core_duration)
    left_padding = min(left_available, padding_budget / 2.0)
    right_padding = min(right_available, padding_budget - left_padding)
    remaining = padding_budget - left_padding - right_padding
    if remaining > 0:
        extra_left = min(left_available - left_padding, remaining)
        left_padding += extra_left
        remaining -= extra_left
        right_padding += min(right_available - right_padding, remaining)
    return AudioWindow(
        max(0.0, start - left_padding),
        min(total_duration, end + right_padding),
        "vad",
    )


def group_vad_regions(
    regions: Iterable[tuple[float, float]],
    total_duration: float,
    *,
    target_seconds: float = 20.0,
    hard_max_seconds: float = 24.0,
    padding_seconds: float = 0.25,
    fallback_chunk_seconds: float = 20.0,
    fallback_overlap_seconds: float = 2.0,
) -> list[AudioWindow]:
    """Group speech regions by pauses and split uninterrupted speech safely."""

    if total_duration <= 0:
        return []
    if not 0 < target_seconds <= hard_max_seconds:
        raise ValueError("VAD target must be positive and no larger than hard max")
    if hard_max_seconds > 24.0:
        raise ValueError("VAD hard max cannot exceed 24 seconds")

    clipped: list[tuple[float, float]] = []
    for raw_start, raw_end in regions:
        start = max(0.0, min(float(raw_start), total_duration))
        end = max(start, min(float(raw_end), total_duration))
        if end - start <= 1e-6:
            continue
        clipped.append((start, end))
    cleaned: list[tuple[float, float]] = []
    for start, end in sorted(clipped):
        if cleaned and start <= cleaned[-1][1] + 1e-6:
            cleaned[-1] = (cleaned[-1][0], max(cleaned[-1][1], end))
        else:
            cleaned.append((start, end))

    windows: list[AudioWindow] = []
    grouped_start: Optional[float] = None
    grouped_end: Optional[float] = None

    def flush_group() -> None:
        nonlocal grouped_start, grouped_end
        if grouped_start is None or grouped_end is None:
            return
        windows.append(
            _padded_vad_window(
                grouped_start,
                grouped_end,
                total_duration,
                padding_seconds=padding_seconds,
                hard_max_seconds=hard_max_seconds,
            )
        )
        grouped_start = None
        grouped_end = None

    for start, end in cleaned:
        if end - start > hard_max_seconds:
            flush_group()
            padded_start = max(0.0, start - padding_seconds)
            padded_end = min(total_duration, end + padding_seconds)
            windows.extend(
                fixed_overlap_windows(
                    padded_start,
                    padded_end,
                    chunk_seconds=fallback_chunk_seconds,
                    overlap_seconds=fallback_overlap_seconds,
                    source="continuous_fallback",
                )
            )
            continue
        if grouped_start is None:
            grouped_start, grouped_end = start, end
            continue
        assert grouped_end is not None
        if end - grouped_start <= target_seconds:
            grouped_end = end
            continue
        flush_group()
        grouped_start, grouped_end = start, end
    flush_group()
    return windows


def detect_vad_regions(wav_path: str) -> list[tuple[float, float]]:
    """Run the locked local pyannote VAD and return supported speech regions."""

    global VAD_METADATA
    if VAD_PIPELINE is None:
        if VAD_METADATA and not VAD_METADATA.get("available"):
            raise RuntimeDependencyUnavailable(
                "Local VAD snapshot was unavailable at runtime initialization"
            )
        initialize_vad_runtime(strict=longform_strict_mode(), load_pipeline=True)
    if VAD_PIPELINE is None:
        raise RuntimeDependencyUnavailable("Local VAD pipeline is unavailable")
    annotation = VAD_PIPELINE(wav_path)
    regions = [
        (float(segment.start), float(segment.end))
        for segment in annotation.get_timeline().support()
    ]
    VAD_METADATA = {**VAD_METADATA, "loaded": True}
    return regions


def _emission_frame_times(item: CTCEmissionWindow) -> list[tuple[float, float]]:
    frame_count = int(item.emission.log_probs.shape[0])
    if frame_count <= 0:
        return []
    shift = item.emission.frame_shift
    return [
        (
            item.window.start + index * shift,
            min(item.window.end, item.window.start + (index + 1) * shift),
        )
        for index in range(frame_count)
    ]


def _greedy_emission_text(log_probs: Any, tokenizer: Any, blank_id: int) -> str:
    if int(log_probs.shape[0]) == 0:
        return ""
    labels = log_probs.argmax(axis=-1).tolist()
    token_ids = [
        token_id
        for token_id, _, _ in ctc_token_runs(labels, blank_id)
    ]
    return str(tokenizer.decode(token_ids)).strip()


SEAM_CRITICAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "number",
        re.compile(
            r"(?:\d|нол|один|одн|дв[ае]|три|четыр|пят|шест|сем|вос|дев|десят|"
            r"сорок|сто|сот|тысяч)",
            re.IGNORECASE,
        ),
    ),
    ("unit", re.compile(r"\b(?:мм|см|мл|ху|единиц\w*)\b", re.IGNORECASE)),
    ("negation", re.compile(r"\b(?:не|нет|без)\b", re.IGNORECASE)),
    (
        "laterality",
        re.compile(r"\b(?:справа|слева|прав\w*|лев\w*)\b", re.IGNORECASE),
    ),
    (
        "contrast",
        re.compile(r"\b(?:контраст\w*|болюс\w*)\b", re.IGNORECASE),
    ),
)


def classify_seam_criticality(*texts: str) -> list[str]:
    joined = " ".join(texts)
    return [
        category
        for category, pattern in SEAM_CRITICAL_PATTERNS
        if pattern.search(joined)
    ]


def _overlap_text(
    item: CTCEmissionWindow,
    overlap_start: float,
    overlap_end: float,
    tokenizer: Any,
    blank_id: int,
) -> str:
    times = _emission_frame_times(item)
    indices = [
        index
        for index, (start, end) in enumerate(times)
        if end > overlap_start + 1e-9 and start < overlap_end - 1e-9
    ]
    if not indices:
        return ""
    return _greedy_emission_text(
        item.emission.log_probs[indices[0] : indices[-1] + 1],
        tokenizer,
        blank_id,
    )


def stitch_ctc_emissions(
    items: list[CTCEmissionWindow],
    tokenizer: Any,
    blank_id: int,
) -> StitchedCTCEmission:
    """Select center-owned overlap frames and create one global CTC sequence."""

    import numpy as np

    if not items:
        return StitchedCTCEmission(
            log_probs=np.empty((0, 0), dtype="float32"),
            frame_times=[],
            ownership=[],
            seam_conflicts=[],
        )
    ordered = sorted(items, key=lambda item: (item.window.start, item.window.end))
    vocabulary_size = int(ordered[0].emission.log_probs.shape[1])
    for item in ordered:
        shape = item.emission.log_probs.shape
        if len(shape) != 2 or int(shape[1]) != vocabulary_size:
            raise ValueError("CTC window emissions have incompatible vocabularies")

    lower_bounds = [item.window.start for item in ordered]
    upper_bounds = [item.window.end for item in ordered]
    conflicts: list[dict[str, Any]] = []
    for index in range(len(ordered) - 1):
        left = ordered[index]
        right = ordered[index + 1]
        overlap_start = max(left.window.start, right.window.start)
        overlap_end = min(left.window.end, right.window.end)
        if overlap_end <= overlap_start:
            continue
        boundary = (overlap_start + overlap_end) / 2.0
        upper_bounds[index] = min(upper_bounds[index], boundary)
        lower_bounds[index + 1] = max(lower_bounds[index + 1], boundary)
        left_text = _overlap_text(
            left, overlap_start, overlap_end, tokenizer, blank_id
        )
        right_text = _overlap_text(
            right, overlap_start, overlap_end, tokenizer, blank_id
        )
        normalized_left = re.sub(r"\W+", "", left_text.casefold())
        normalized_right = re.sub(r"\W+", "", right_text.casefold())
        if normalized_left == normalized_right:
            continue
        critical_classes = classify_seam_criticality(left_text, right_text)
        conflicts.append(
            {
                "seam": index + 1,
                "start": round(overlap_start, 3),
                "end": round(overlap_end, 3),
                "left_text": left_text,
                "right_text": right_text,
                "critical": bool(critical_classes),
                "critical_classes": critical_classes,
            }
        )

    selected_arrays: list[Any] = []
    stitched_times: list[tuple[float, float]] = []
    ownership: list[tuple[float, float]] = []
    previous_time: Optional[float] = None
    for index, item in enumerate(ordered):
        times = _emission_frame_times(item)
        lower = lower_bounds[index]
        upper = upper_bounds[index]
        ownership.append((lower, upper))
        selected = [
            frame
            for frame, (start, end) in enumerate(times)
            if (start + end) / 2.0 >= lower - 1e-9
            and (
                (start + end) / 2.0 < upper - 1e-9
                or index == len(ordered) - 1
            )
        ]
        if not selected and times:
            target = (lower + upper) / 2.0
            selected = [
                min(
                    range(len(times)),
                    key=lambda frame: abs(
                        (times[frame][0] + times[frame][1]) / 2.0 - target
                    ),
                )
            ]
        if not selected:
            continue
        owned_times = [
            (
                max(times[frame][0], lower),
                min(times[frame][1], upper),
            )
            for frame in range(selected[0], selected[-1] + 1)
        ]
        owned_times = [
            (start, max(start, end)) for start, end in owned_times
        ]
        first_time = owned_times[0][0]
        if previous_time is not None:
            gap_threshold = max(item.emission.frame_shift * 1.5, 0.001)
            if first_time - previous_time > gap_threshold:
                separator = np.full(
                    (1, vocabulary_size),
                    -30.0,
                    dtype=item.emission.log_probs.dtype,
                )
                separator[0, blank_id] = 0.0
                selected_arrays.append(separator)
                stitched_times.append((previous_time, first_time))
        selected_arrays.append(
            item.emission.log_probs[selected[0] : selected[-1] + 1]
        )
        stitched_times.extend(owned_times)
        previous_time = owned_times[-1][1]

    stitched = (
        np.concatenate(selected_arrays, axis=0)
        if selected_arrays
        else np.empty((0, vocabulary_size), dtype="float32")
    )
    return StitchedCTCEmission(
        log_probs=stitched,
        frame_times=stitched_times,
        ownership=ownership,
        seam_conflicts=conflicts,
    )


def transcribe_ctc_windows(
    wav_path: str,
    windows: list[AudioWindow],
    hotwords: Optional[list[str]],
    *,
    method: str,
    degraded_reason: Optional[str] = None,
) -> InferenceResult:
    """Extract all window emissions, stitch by time, and decode once."""

    import soundfile as sf

    if not windows:
        return InferenceResult(
            text="",
            words=[],
            chunks=0,
            chunk_details=[],
            confidence=unavailable_confidence("vad_detected_no_speech"),
            method=method,
            integrity=inference_integrity(
                degraded=degraded_reason is not None,
                reasons=[degraded_reason] if degraded_reason else [],
            ),
        )

    emission_windows: list[CTCEmissionWindow] = []
    details: list[dict[str, Any]] = []
    with sf.SoundFile(wav_path) as source, tempfile.TemporaryDirectory(
        prefix="gigaam_emissions_"
    ) as chunk_dir:
        sample_rate = int(source.samplerate)
        total_frames = int(source.frames)
        for index, requested in enumerate(windows):
            start_frame = max(0, min(total_frames, round(requested.start * sample_rate)))
            end_frame = max(
                start_frame,
                min(total_frames, round(requested.end * sample_rate)),
            )
            actual = AudioWindow(
                start_frame / sample_rate,
                end_frame / sample_rate,
                requested.source,
            )
            source.seek(start_frame)
            audio = source.read(end_frame - start_frame, dtype="float32")
            chunk_path = os.path.join(chunk_dir, f"chunk_{index:04d}.wav")
            sf.write(chunk_path, audio, sample_rate, subtype="PCM_16")
            started = time.perf_counter()
            emission = extract_ctc_emission(chunk_path)
            elapsed = time.perf_counter() - started
            emission_windows.append(
                CTCEmissionWindow(actual, emission, elapsed)
            )
            details.append(
                {
                    "chunk": index + 1,
                    "start": round(actual.start, 3),
                    "end": round(actual.end, 3),
                    "duration": round(actual.end - actual.start, 3),
                    "window_source": actual.source,
                    "emission_frames": int(emission.log_probs.shape[0]),
                    "frame_shift": round(emission.frame_shift, 6),
                    "elapsed": round(elapsed, 6),
                    "confidence": unavailable_confidence(
                        "window_not_independently_decoded"
                    ),
                }
            )

    decoding = getattr(MODEL, "decoding", None)
    tokenizer = getattr(decoding, "tokenizer", None)
    blank_id = getattr(decoding, "blank_id", None)
    if tokenizer is None or blank_id is None:
        raise ConfidenceUnavailable("installed_decoder_does_not_expose_ctc_tokens")
    stitched = stitch_ctc_emissions(
        emission_windows,
        tokenizer,
        int(blank_id),
    )
    for detail, (start, end) in zip(details, stitched.ownership):
        detail["ownership_start"] = round(start, 3)
        detail["ownership_end"] = round(end, 3)
    average_shift = (
        sum(item.emission.frame_shift for item in emission_windows)
        / len(emission_windows)
    )
    decoded = decode_ctc_emission(
        CTCEmission(
            log_probs=stitched.log_probs,
            frame_shift=average_shift,
            duration=sum(
                max(0.0, end - start) for start, end in stitched.frame_times
            ),
        ),
        hotwords,
        frame_times=stitched.frame_times,
        method=method,
    )
    decoded.chunks = len(details)
    decoded.chunk_details = details
    decoded.integrity = inference_integrity(
        degraded=degraded_reason is not None,
        reasons=[degraded_reason] if degraded_reason else [],
        seam_conflicts=stitched.seam_conflicts,
    )
    return decoded


def concatenate_chunk_text(parts: list[str]) -> str:
    """Join degraded fallback chunks without deleting any recognized token."""

    return " ".join(part.strip() for part in parts if part.strip()).strip()


def combine_chunk_confidence(
    chunk_confidences: list[dict[str, Any]],
) -> dict[str, Any]:
    if not chunk_confidences:
        return unavailable_confidence("no_non_empty_chunks")
    if any(not item.get("available") for item in chunk_confidences):
        reasons = sorted(
            {
                str(item.get("reason") or "unknown")
                for item in chunk_confidences
                if not item.get("available")
            }
        )
        return unavailable_confidence("chunk_confidence_unavailable:" + ",".join(reasons))

    total_tokens = sum(int(item.get("emitted_tokens") or 0) for item in chunk_confidences)
    if total_tokens <= 0:
        return unavailable_confidence("no_non_blank_ctc_emissions")
    weighted_logprob = sum(
        float(item["avg_logprob"]) * int(item["emitted_tokens"])
        for item in chunk_confidences
    ) / total_tokens
    return {
        "available": True,
        "method": "ctc_greedy_token_peaks_weighted_chunks",
        "avg_logprob": round(weighted_logprob, 6),
        "mean_probability": round(math.exp(weighted_logprob), 6),
        "low_confidence": weighted_logprob < LOW_CONFIDENCE_LOGPROB,
        "threshold_logprob": LOW_CONFIDENCE_LOGPROB,
        "emitted_tokens": total_tokens,
        "calibrated": False,
    }


def _transcribe_audio_with_text_overlap(
    wav_path: str, hotwords: Optional[list[str]] = None
) -> InferenceResult:
    """Degraded non-CTC fallback retained for development compatibility only.

    Chunks are concatenated losslessly. Overlap duplicates are deliberately kept:
    text-level deletion cannot establish which acoustic occurrence is correct.
    """
    import soundfile as sf

    chunk_seconds = FALLBACK_CHUNK_SECONDS
    overlap_seconds = FALLBACK_OVERLAP_SECONDS
    combined_text = ""
    combined_words: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    confidences: list[dict[str, Any]] = []

    with sf.SoundFile(wav_path) as source, tempfile.TemporaryDirectory(
        prefix="gigaam_chunks_"
    ) as chunk_dir:
        sample_rate = int(source.samplerate)
        total_frames = int(source.frames)
        chunk_frames = int(chunk_seconds * sample_rate)
        overlap_frames = int(overlap_seconds * sample_rate)
        step_frames = max(1, chunk_frames - overlap_frames)
        start_frame = 0
        index = 0

        while start_frame < total_frames:
            end_frame = min(start_frame + chunk_frames, total_frames)
            source.seek(start_frame)
            audio = source.read(end_frame - start_frame, dtype="float32")
            chunk_path = os.path.join(chunk_dir, f"chunk_{index:04d}.wav")
            sf.write(chunk_path, audio, sample_rate, subtype="PCM_16")

            chunk_started = time.perf_counter()
            result = transcribe_short(chunk_path, hotwords)
            chunk_elapsed = time.perf_counter() - chunk_started
            start_seconds = start_frame / sample_rate
            end_seconds = end_frame / sample_rate

            combined_text = concatenate_chunk_text([combined_text, result.text])
            offset_words = [
                {
                    **word,
                    "start": round(float(word["start"]) + start_seconds, 3),
                    "end": round(float(word["end"]) + start_seconds, 3),
                }
                for word in result.words
            ]
            combined_words.extend(offset_words)
            if result.text:
                confidences.append(result.confidence)

            details.append(
                {
                    "chunk": index + 1,
                    "start": round(start_seconds, 3),
                    "end": round(end_seconds, 3),
                    "duration": round(end_seconds - start_seconds, 3),
                    "overlap_seconds": (
                        round(overlap_seconds, 3) if index > 0 else 0.0
                    ),
                    "text_join": "lossless_no_deduplication",
                    "chars": len(result.text),
                    "elapsed": round(chunk_elapsed, 6),
                    "confidence": result.confidence,
                }
            )
            logger.info(
                "chunk %d: %.1f-%.1fs | %d chars | lossless text fallback",
                index + 1,
                start_seconds,
                end_seconds,
                len(result.text),
            )

            index += 1
            if end_frame >= total_frames:
                break
            start_frame += step_frames

    return InferenceResult(
        text=combined_text,
        words=combined_words,
        chunks=len(details),
        chunk_details=details,
        confidence=combine_chunk_confidence(confidences),
        method="fixed_overlap_text_degraded",
        integrity=inference_integrity(
            degraded=True,
            reasons=["ctc_emissions_unavailable_text_overlap_fallback"],
        ),
    )


def transcribe_audio_in_overlapping_chunks(
    wav_path: str,
    hotwords: Optional[list[str]] = None,
    *,
    degraded_reason: Optional[str] = "vad_bypassed_fixed_overlap",
) -> InferenceResult:
    """Continuous-speech fallback using one decode over stitched CTC emissions."""

    import soundfile as sf

    info = sf.info(wav_path)
    if model_supports_ctc_emissions():
        windows = fixed_overlap_windows(
            0.0,
            float(info.duration),
            chunk_seconds=float(FALLBACK_CHUNK_SECONDS),
            overlap_seconds=FALLBACK_OVERLAP_SECONDS,
        )
        return transcribe_ctc_windows(
            wav_path,
            windows,
            hotwords,
            method="fixed_overlap_ctc_stitched",
            degraded_reason=degraded_reason,
        )
    if longform_strict_mode():
        raise RuntimeDependencyUnavailable(
            "Strict long-form runtime requires CTC emissions"
        )
    return _transcribe_audio_with_text_overlap(wav_path, hotwords)


def transcribe_native_longform(wav_path: str) -> InferenceResult:
    try:
        result = MODEL.transcribe_longform(wav_path, word_timestamps=True)
        has_timestamps = True
    except TypeError:
        result = MODEL.transcribe_longform(wav_path)
        has_timestamps = False

    text_parts: list[str] = []
    words: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for index, segment in enumerate(result.segments, start=1):
        text = str(segment.text).strip()
        if text:
            text_parts.append(text)
        if has_timestamps:
            words.extend(serialize_public_words(segment.words))
        details.append(
            {
                "chunk": index,
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "duration": round(float(segment.end) - float(segment.start), 3),
                "chars": len(text),
                "confidence": unavailable_confidence(
                    "native_longform_api_does_not_expose_segment_log_probs"
                ),
            }
        )
    return InferenceResult(
        text=" ".join(text_parts).strip(),
        words=words,
        chunks=len(details),
        chunk_details=details,
        confidence=unavailable_confidence(
            "native_longform_api_does_not_expose_segment_log_probs"
        ),
        method="native_vad_longform",
        integrity=inference_integrity(
            degraded=True,
            reasons=["native_longform_does_not_expose_ctc_emissions"],
        ),
    )


def transcribe_audio_with_vad(
    wav_path: str,
    hotwords: Optional[list[str]] = None,
) -> InferenceResult:
    import soundfile as sf

    if not model_supports_ctc_emissions():
        raise ConfidenceUnavailable("longform_model_does_not_expose_ctc_emissions")
    info = sf.info(wav_path)
    regions = detect_vad_regions(wav_path)
    windows = group_vad_regions(
        regions,
        float(info.duration),
        target_seconds=VAD_TARGET_SECONDS,
        hard_max_seconds=VAD_HARD_MAX_SECONDS,
        padding_seconds=VAD_PADDING_SECONDS,
        fallback_chunk_seconds=float(FALLBACK_CHUNK_SECONDS),
        fallback_overlap_seconds=FALLBACK_OVERLAP_SECONDS,
    )
    return transcribe_ctc_windows(
        wav_path,
        windows,
        hotwords,
        method="vad_ctc_stitched",
    )


def transcribe_audio(
    wav_path: str, hotwords: Optional[list[str]] = None
) -> InferenceResult:
    from gigaam.model import LONGFORM_THRESHOLD, SAMPLE_RATE
    import soundfile as sf

    info = sf.info(wav_path)
    sample_count = int(info.duration * info.samplerate)
    if sample_count <= LONGFORM_THRESHOLD:
        return transcribe_short(wav_path, hotwords)

    logger.info(
        "Audio %.1fs > %.0fs; longform mode=%s",
        info.duration,
        LONGFORM_THRESHOLD / SAMPLE_RATE,
        LONGFORM_MODE,
    )
    if LONGFORM_MODE == "overlap":
        return transcribe_audio_in_overlapping_chunks(wav_path, hotwords)
    try:
        return transcribe_audio_with_vad(wav_path, hotwords)
    except (
        ConfidenceUnavailable,
        ImportError,
        ModuleNotFoundError,
        RuntimeDependencyUnavailable,
    ) as exc:
        if longform_strict_mode():
            raise RuntimeDependencyUnavailable(
                "Strict long-form inference requires the locked VAD and CTC "
                "emission decoder"
            ) from exc
        logger.warning(
            "VAD/scored long-form unavailable (%s); using degraded stitched "
            "fixed-overlap fallback",
            type(exc).__name__,
        )
        return transcribe_audio_in_overlapping_chunks(
            wav_path,
            hotwords,
            degraded_reason=(
                "vad_or_scored_decoder_unavailable:"
                + type(exc).__name__
            ),
        )


def run_serialized_inference(
    wav_path: str,
    transcriber: Callable[[str], InferenceResult] = transcribe_audio,
) -> tuple[InferenceResult, float, float]:
    """Queue one request on the single shared model and return timing data."""
    queued_at = time.perf_counter()
    with MODEL_LOCK:
        acquired_at = time.perf_counter()
        result = transcriber(wav_path)
        completed_at = time.perf_counter()
    return result, acquired_at - queued_at, completed_at - acquired_at


def resolve_context_hotwords(scope: Any) -> tuple[Optional[str], list[str]]:
    if scope is None or scope == "":
        return None, []
    if not isinstance(scope, str):
        raise ValueError("context_scope must be a string")
    if scope not in CTC_CONTEXTS:
        raise ValueError(f"Unknown or unapproved context_scope: {scope}")
    if CTC_DECODER_MODE != "beam":
        raise ValueError("context_scope requires GIGAAM_CTC_DECODER=beam")
    return scope, list(CTC_CONTEXTS[scope])


def public_runtime_metadata() -> dict[str, Any]:
    return {
        "schema_version": "gigaam.runtime.v1",
        "runtime_id": RUNTIME_METADATA.get("id"),
        "model": MODEL_METADATA,
        "runtime": RUNTIME_METADATA,
        "configuration": {
            "longform_mode": LONGFORM_MODE,
            "longform_strict": longform_strict_mode(),
            "vad": VAD_METADATA,
            "vad_target_seconds": VAD_TARGET_SECONDS,
            "vad_hard_max_seconds": VAD_HARD_MAX_SECONDS,
            "vad_padding_seconds": VAD_PADDING_SECONDS,
            "fallback_chunk_seconds": FALLBACK_CHUNK_SECONDS,
            "fallback_overlap_seconds": FALLBACK_OVERLAP_SECONDS,
            "low_confidence_logprob": LOW_CONFIDENCE_LOGPROB,
            "inference_concurrency": 1,
            "ctc_decoder": CTC_DECODER_METADATA,
            "strict_runtime_lock": env_bool("GIGAAM_STRICT_RUNTIME_LOCK", False),
            "audio_path_enabled": ALLOW_AUDIO_PATH,
            "max_request_bytes": MAX_REQUEST_BYTES,
            "max_audio_bytes": MAX_AUDIO_BYTES,
        },
    }


class GigaAMHandler(BaseHTTPRequestHandler):
    server_version = "GigaAMRuntime/2"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model": MODEL_METADATA.get("name", MODEL_SPEC),
                    "device": DEVICE,
                    "runtime_id": RUNTIME_METADATA.get("id"),
                    "uptime_seconds": round(time.time() - SERVER_STARTED_AT, 3),
                },
            )
        elif self.path == "/metadata":
            self._send_json(200, public_runtime_metadata())
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path == "/transcribe":
            self._handle_transcribe()
        else:
            self._send_json(404, {"error": "Not found"})

    def _read_json_body(self) -> Optional[dict[str, Any]]:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return None
        if content_length <= 0:
            self._send_json(400, {"error": "Empty request body"})
            return None
        if content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Request body too large"})
            return None
        try:
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError
            return data
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._send_json(400, {"error": "Invalid JSON body"})
            return None

    def _handle_transcribe(self) -> None:
        data = self._read_json_body()
        if data is None:
            return
        try:
            context_scope, context_hotwords = resolve_context_hotwords(
                data.get("context_scope")
            )
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        tmp_input: Optional[str] = None
        tmp_wav: Optional[str] = None
        request_started = time.perf_counter()
        try:
            if "audio_base64" in data:
                encoded = data["audio_base64"]
                if not isinstance(encoded, str):
                    self._send_json(400, {"error": "audio_base64 must be a string"})
                    return
                try:
                    audio_bytes = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError):
                    self._send_json(400, {"error": "Invalid base64 in audio_base64"})
                    return
                if not audio_bytes:
                    self._send_json(400, {"error": "Decoded audio is empty"})
                    return
                if len(audio_bytes) > MAX_AUDIO_BYTES:
                    self._send_json(413, {"error": "Decoded audio is too large"})
                    return
                with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as temp:
                    temp.write(audio_bytes)
                    tmp_input = temp.name
                audio_path = tmp_input
                audio_sha256 = sha256_bytes(audio_bytes)
            elif "audio_path" in data:
                requested_path = data["audio_path"]
                if not isinstance(requested_path, str):
                    self._send_json(400, {"error": "audio_path must be a string"})
                    return
                try:
                    audio_path = str(resolve_allowed_audio_path(requested_path))
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                audio_sha256 = hash_file(Path(audio_path), ("sha256",))["sha256"]
            else:
                self._send_json(400, {"error": "Provide audio_base64 or audio_path"})
                return

            conversion_started = time.perf_counter()
            try:
                tmp_wav = convert_to_wav(audio_path)
            except Exception as exc:
                self._send_json(
                    500,
                    {"error": f"Audio conversion failed: {type(exc).__name__}"},
                )
                return
            conversion_elapsed = time.perf_counter() - conversion_started
            normalized_audio_sha256 = hash_file(
                Path(tmp_wav), ("sha256",)
            )["sha256"]

            result, queue_wait, inference_elapsed = run_serialized_inference(
                tmp_wav,
                lambda path: transcribe_audio(path, context_hotwords),
            )
            total_elapsed = time.perf_counter() - request_started
            response: dict[str, Any] = {
                "schema_version": "gigaam.transcription.v2",
                "source": "gigaam",
                "text": result.text,
                "raw_text": result.text,
                "language": "ru",
                "words": result.words,
                "elapsed": round(total_elapsed, 6),
                "chunks": result.chunks,
                "chunk_details": result.chunk_details,
                "method": result.method,
                "context_bias": {
                    "scope": context_scope,
                    "active": bool(context_hotwords),
                    "terms": len(context_hotwords),
                },
                "confidence": result.confidence,
                "integrity": (
                    result.integrity
                    if result.integrity
                    else inference_integrity()
                ),
                "hashes": {
                    "audio_sha256": audio_sha256,
                    "normalized_audio_sha256": normalized_audio_sha256,
                    "raw_text_sha256": sha256_text(result.text),
                },
                "timings": {
                    "conversion_seconds": round(conversion_elapsed, 6),
                    "queue_wait_seconds": round(queue_wait, 6),
                    "inference_seconds": round(inference_elapsed, 6),
                    "total_seconds": round(total_elapsed, 6),
                },
                "model": MODEL_METADATA,
                "runtime_id": RUNTIME_METADATA.get("id"),
                "configuration": public_runtime_metadata()["configuration"],
            }
            # Backward-compatible fields are included only when they are real.
            if result.confidence.get("available"):
                response["avg_logprob"] = result.confidence["avg_logprob"]
                response["low_confidence"] = result.confidence["low_confidence"]

            logger.info(
                "OK %.2fs | queue=%.3fs | inference=%.2fs | %d chars | "
                "model=%s | audio=%s | text=%s",
                total_elapsed,
                queue_wait,
                inference_elapsed,
                len(result.text),
                MODEL_METADATA.get("name", MODEL_SPEC),
                audio_sha256[:12],
                response["hashes"]["raw_text_sha256"][:12],
            )
            self._send_json(200, response)
        except RuntimeDependencyUnavailable as exc:
            logger.error("Long-form runtime unavailable: %s", exc)
            self._send_json(
                503,
                {
                    "error": str(exc),
                    "code": "longform_runtime_unavailable",
                    "runtime_id": RUNTIME_METADATA.get("id"),
                },
            )
        except Exception as exc:
            logger.error("Transcription failed: %s", exc, exc_info=True)
            self._send_json(
                500,
                {
                    "error": str(exc),
                    "runtime_id": RUNTIME_METADATA.get("id"),
                },
            )
        finally:
            for path in (tmp_input, tmp_wav):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        logger.warning("Failed to remove temporary audio: %s", path)

    def _send_json(self, code: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    try:
        validate_security_configuration()
    except Exception as exc:
        logger.error("Unsafe GigaAM server configuration: %s", exc)
        return 1
    if not check_ffmpeg():
        logger.error("ffmpeg is required but was not found")
        return 1
    try:
        initialize_model()
    except Exception:
        logger.exception("GigaAM runtime initialization failed")
        return 1

    server = ThreadedHTTPServer((HOST, PORT), GigaAMHandler)
    logger.info("GigaAM HTTP server -> http://%s:%d", HOST, PORT)
    logger.info("Endpoints: GET /health   GET /metadata   POST /transcribe")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
