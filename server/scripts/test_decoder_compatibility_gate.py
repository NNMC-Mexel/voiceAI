from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asr_benchmark
import decoder_compatibility_gate as gate


def complete_evidence(word_count: int = 2) -> dict[str, Any]:
    return {
        "available": True,
        "word_count": word_count,
        "timestamped_words": word_count,
        "timestamps_valid": True,
        "timestamps_complete": True,
        "acoustic_confidence_words": word_count,
        "acoustic_confidence_valid": True,
        "acoustic_confidence_complete": True,
        "utterance_acoustic_confidence": True,
        "score_types": ["ctc_emission_geomean"],
        "timing_signature_sha256": "1" * 64,
        "acoustic_signature_sha256": "2" * 64,
        "all_repeats_complete": True,
        "repeat_deterministic": True,
    }


def case(case_id: str = "case-1") -> dict[str, Any]:
    return {
        "id": case_id,
        "audio_sha256": "a" * 64,
        "reference_sha256": "b" * 64,
        "prediction_sha256": "c" * 64,
        "deterministic": True,
        "word_evidence": complete_evidence(),
    }


def report(purpose: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "voicemed.asr-benchmark.v2",
        "purpose": purpose,
        "context_scope": None,
        "repeat": 10,
        "concurrency": 1,
        "limited": False,
        "manifest": {
            "sha256": "d" * 64,
            "dataset_kind": "real_frozen_test",
        },
        "server": {"runtime_id": f"{purpose}-runtime"},
        "metrics": {
            "cases": 1,
            "deterministic_cases": 1,
            "rtf": {"p95": 0.49},
        },
        "cases": [case()],
    }
    if purpose == "candidate":
        value["server"]["configuration"] = {
            "ctc_decoder": {
                "mode": "beam",
                "active": True,
                "implementation": "hospital_asr.ngpu_lm:create_decoder",
                "implementationMetadata": {
                    "backend": "nvidia-ngpu-lm",
                },
                "languageModel": {
                    "active": True,
                    "alpha": 0.0,
                },
            }
        }
    return value


def fallback() -> dict[str, Any]:
    return {
        "backend": "flashlight+kenlm",
        "runtime": "linux-cuda-container",
        "container_image": (
            "registry.local/voicemed/flashlight-kenlm@sha256:" + "e" * 64
        ),
        "decoder_artifact_sha256": "f" * 64,
        "kenlm_runtime_sha256": "1" * 64,
        "lock_file": "fallback.lock.json",
        "lock_sha256": "2" * 64,
    }


class WordEvidenceSummaryTests(unittest.TestCase):
    def test_summary_contains_no_dictated_text_and_accepts_acoustic_scores(
        self,
    ) -> None:
        summary = asr_benchmark.summarize_word_evidence(
            {
                "words": [
                    {
                        "text": "секрет",
                        "start": 0.0,
                        "end": 0.4,
                        "avg_logprob": -0.2,
                        "confidence": 0.81,
                        "score_type": "ctc_emission_geomean",
                    },
                    {
                        "text": "пациента",
                        "start": 0.4,
                        "end": 0.9,
                        "avg_logprob": -0.3,
                        "confidence": 0.74,
                        "score_type": "ctc_emission_geomean",
                    },
                ],
                "confidence": {
                    "available": True,
                    "avg_logprob": -0.25,
                    "method": "ctc_acoustic_token_peak_geomean",
                },
            }
        )
        self.assertTrue(summary["timestamps_complete"])
        self.assertTrue(summary["acoustic_confidence_complete"])
        serialized = json.dumps(summary, ensure_ascii=False)
        self.assertNotIn("секрет", serialized)
        self.assertNotIn("пациента", serialized)

    def test_fused_score_is_not_accepted_as_acoustic_confidence(self) -> None:
        summary = asr_benchmark.summarize_word_evidence(
            {
                "words": [
                    {
                        "text": "слово",
                        "start": 0.0,
                        "end": 0.4,
                        "avg_logprob": -0.2,
                        "confidence": 0.81,
                        "score_type": "fused_lm_score",
                    }
                ],
                "confidence": {
                    "available": True,
                    "avg_logprob": -0.2,
                    "method": "fused_language_model_confidence",
                },
            }
        )
        self.assertFalse(summary["acoustic_confidence_complete"])
        self.assertFalse(summary["utterance_acoustic_confidence"])


class DecoderCompatibilityGateTests(unittest.TestCase):
    def test_eligible_candidate_selects_ngpu_lm(self) -> None:
        decision = gate.evaluate_decoder_compatibility(
            report("baseline"),
            report("candidate"),
            fallback(),
            greedy_report_sha256="3" * 64,
            zero_lm_report_sha256="4" * 64,
        )
        self.assertTrue(decision["eligible"], decision["failed"])
        self.assertEqual(decision["selected"]["backend"], "ngpu-lm")

    def test_each_acceptance_condition_fails_closed_to_fallback(self) -> None:
        mutations = {
            "same_case_inventory": lambda value: value["cases"][0].__setitem__(
                "audio_sha256", "9" * 64
            ),
            "ngpu_lm_zero_weight_identity": lambda value: value["server"][
                "configuration"
            ]["ctc_decoder"]["languageModel"].__setitem__("alpha", 0.1),
            "zero_weight_predictions_equal_greedy": lambda value: value[
                "cases"
            ][0].__setitem__("prediction_sha256", "8" * 64),
            "timestamps_and_acoustic_confidence_preserved": lambda value: value[
                "cases"
            ][0]["word_evidence"].__setitem__(
                "acoustic_confidence_complete", False
            ),
            "candidate_deterministic": lambda value: value.__setitem__(
                "repeat", 1
            ),
            "candidate_p95_rtf": lambda value: value["metrics"]["rtf"].__setitem__(
                "p95", 0.5001
            ),
        }
        for expected_failure, mutate in mutations.items():
            with self.subTest(expected_failure=expected_failure):
                candidate = copy.deepcopy(report("candidate"))
                mutate(candidate)
                decision = gate.evaluate_decoder_compatibility(
                    report("baseline"),
                    candidate,
                    fallback(),
                )
                self.assertFalse(decision["eligible"])
                self.assertIn(expected_failure, decision["failed"])
                self.assertEqual(
                    decision["selected"]["backend"],
                    "flashlight+kenlm",
                )

    def test_old_report_without_word_evidence_is_rejected(self) -> None:
        candidate = report("candidate")
        del candidate["cases"][0]["word_evidence"]
        decision = gate.evaluate_decoder_compatibility(
            report("baseline"),
            candidate,
            fallback(),
        )
        self.assertIn(
            "timestamps_and_acoustic_confidence_preserved",
            decision["failed"],
        )

    def test_fallback_lock_requires_content_addressed_container(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fallback.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                gate.validate_fallback_lock({}, path)

            lock = {
                "schema_version": gate.FALLBACK_LOCK_SCHEMA_VERSION,
                "backend": "flashlight+kenlm",
                "container_image": "registry.local/image:latest",
                "decoder_artifact_sha256": "a" * 64,
                "kenlm_runtime_sha256": "b" * 64,
            }
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "container_image"):
                gate.validate_fallback_lock(lock, path)

    def test_cli_writes_fallback_decision_and_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            greedy_path = root / "greedy.json"
            candidate_path = root / "candidate.json"
            fallback_path = root / "fallback.json"
            output_path = root / "decision.json"
            candidate = report("candidate")
            candidate["metrics"]["rtf"]["p95"] = 0.8
            greedy_path.write_text(
                json.dumps(report("baseline")),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(candidate),
                encoding="utf-8",
            )
            fallback_path.write_text(
                json.dumps(
                    {
                        "schema_version": gate.FALLBACK_LOCK_SCHEMA_VERSION,
                        "backend": "flashlight+kenlm",
                        "container_image": (
                            "registry.local/voicemed/fallback@sha256:"
                            + "a" * 64
                        ),
                        "decoder_artifact_sha256": "b" * 64,
                        "kenlm_runtime_sha256": "c" * 64,
                    }
                ),
                encoding="utf-8",
            )
            exit_code = gate.main(
                [
                    "--greedy-report",
                    str(greedy_path),
                    "--zero-lm-report",
                    str(candidate_path),
                    "--fallback-lock",
                    str(fallback_path),
                    "--output",
                    str(output_path),
                ]
            )
            self.assertEqual(exit_code, 2)
            decision = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                decision["selected"]["backend"],
                "flashlight+kenlm",
            )
            self.assertIn("candidate_p95_rtf", decision["failed"])


if __name__ == "__main__":
    unittest.main()
