from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asr_benchmark as benchmark


class ASRBenchmarkTests(unittest.TestCase):
    @staticmethod
    def entity(
        entity_id: str,
        entity_type: str,
        text: str,
        start_word: int,
        end_word: int,
    ) -> dict[str, Any]:
        return {
            "id": entity_id,
            "type": entity_type,
            "text": text,
            "start_word": start_word,
            "end_word": end_word,
        }

    def test_normalization_is_case_punctuation_and_yo_insensitive(self) -> None:
        self.assertEqual(
            benchmark.normalized_words("Печёнка, 15 мм."),
            ["печенка", "15", "мм"],
        )

    def test_edit_distance(self) -> None:
        self.assertEqual(
            benchmark.edit_distance(
                ["печень", "не", "увеличена"],
                ["печень", "увеличена"],
            ),
            1,
        )

    def test_scores_medical_terms_and_number_units(self) -> None:
        score = benchmark.score_case(
            "Печень не увеличена, образование 15 мм справа",
            "печень не увеличена образование 15 мм справа",
            ["образование", "печень"],
            [
                self.entity("term-1", "medical_term", "печень", 0, 1),
                self.entity("neg-1", "negation", "не", 1, 2),
                self.entity("num-1", "number_unit", "15 мм", 4, 6),
                self.entity("side-1", "laterality", "справа", 6, 7),
            ],
        )
        self.assertEqual(score["word_edits"], 0)
        self.assertEqual(score["entities"]["medical_term"]["correct"], 1)
        self.assertEqual(score["entities"]["number_unit"]["correct"], 1)
        self.assertEqual(score["entities"]["negation"]["correct"], 1)
        self.assertEqual(score["entities"]["laterality"]["correct"], 1)

    def test_missing_unit_is_not_exact(self) -> None:
        score = benchmark.score_case(
            "образование 15 мм",
            "образование 15 см",
            [],
            [self.entity("size", "number_unit", "15 мм", 1, 3)],
        )
        self.assertEqual(score["entities"]["number_unit"]["correct"], 0)
        self.assertEqual(score["entities"]["number_unit"]["total"], 1)

    def test_insertion_inside_multiword_entity_is_not_exact(self) -> None:
        score = benchmark.score_case(
            "образование 15 мм",
            "образование 15 примерно мм",
            [],
            [self.entity("size", "number_unit", "15 мм", 1, 3)],
        )
        self.assertEqual(score["entities"]["number_unit"]["correct"], 0)
        self.assertEqual(score["entities"]["number_unit"]["word_edits"], 1)

    def test_medical_wer_counts_local_term_substitution(self) -> None:
        score = benchmark.score_case(
            "Вирсунгов проток не расширен",
            "Вирсунов поток не расширен",
            ["вирсунгов проток"],
            [
                self.entity(
                    "duct",
                    "medical_term",
                    "Вирсунгов проток",
                    0,
                    2,
                )
            ],
        )
        self.assertEqual(score["entities"]["medical_term"]["word_edits"], 2)
        self.assertEqual(score["entities"]["medical_term"]["word_count"], 2)

    def test_percentage_and_compound_unit_are_preserved(self) -> None:
        self.assertEqual(
            benchmark.extract_number_units("Плотность 62 HU, объём 15,5 мл; 20%"),
            {"62hu": 1, "15.5мл": 1, "20%": 1},
        )

    def test_spelled_number_substitution_is_not_counted_as_exact(self) -> None:
        score = benchmark.score_case(
            "образование пятнадцать миллиметров",
            "образование пятьдесят миллиметров",
            [],
            [
                self.entity(
                    "size",
                    "number_unit",
                    "пятнадцать миллиметров",
                    1,
                    3,
                )
            ],
        )
        self.assertEqual(score["entities"]["number_unit"]["correct"], 0)
        self.assertFalse(score["diagnostics"]["promotion_eligible"])

    def test_swapped_laterality_association_fails_span_accuracy(self) -> None:
        score = benchmark.score_case(
            "печень справа почка слева",
            "печень слева почка справа",
            [],
            [
                self.entity("liver-side", "laterality", "справа", 1, 2),
                self.entity("kidney-side", "laterality", "слева", 3, 4),
            ],
        )
        self.assertEqual(score["entities"]["laterality"]["correct"], 0)
        self.assertEqual(score["entities"]["laterality"]["total"], 2)

    def test_empty_annotation_category_is_unavailable_not_perfect(self) -> None:
        score = benchmark.score_case(
            "печень не увеличена",
            "печень не увеличена",
            [],
            [],
        )
        metrics = benchmark.aggregate(
            [
                {
                    "scores": score,
                    "inference_seconds": 0.1,
                    "duration_seconds": 1.0,
                    "wall_seconds": [0.2],
                    "deterministic": True,
                }
            ],
            "real_frozen_test",
        )
        self.assertFalse(metrics["entities"]["number_unit"]["available"])
        self.assertIsNone(
            metrics["entities"]["number_unit"]["exact_accuracy"]
        )

    def test_release_gate_fails_when_safety_denominator_is_missing(self) -> None:
        metrics = {
            "cases": 1,
            "deterministic_cases": 1,
            "medical_wer": 0.1,
            "normalized_wer": 0.1,
            "critical_entity_recall": {
                "available": False,
                "eligible_annotations": 0,
                "recall": None,
            },
            "entities": {
                entity_type: {
                    "available": False,
                    "eligible_annotations": 0,
                    "exact_accuracy": None,
                }
                for entity_type in benchmark.SAFETY_ENTITY_TYPES
            },
            "rtf": {"p95": 0.1},
        }
        current = {
            "schema_version": "voicemed.asr-benchmark.v2",
            "purpose": "release",
            "context_scope": None,
            "repeat": 10,
            "concurrency": 10,
            "limited": False,
            "manifest": {"sha256": "manifest"},
            "server": {"runtime_id": "runtime"},
            "metrics": metrics,
            "cases": [{"id": "case", "prediction_sha256": "text"}],
        }
        current["prediction_inventory_sha256"] = (
            benchmark.prediction_inventory_sha256(current["cases"])
        )
        baseline = {
            "schema_version": "voicemed.asr-benchmark.v2",
            "purpose": "baseline",
            "repeat": 10,
            "concurrency": 1,
            "limited": False,
            "manifest": {"sha256": "manifest"},
            "metrics": {
                "cases": 1,
                "deterministic_cases": 1,
                "medical_wer": 0.2,
                "normalized_wer": 0.1,
            },
            "cases": [{"id": "case", "prediction_sha256": "baseline"}],
        }
        sequential = {
            "schema_version": "voicemed.asr-benchmark.v2",
            "purpose": "candidate",
            "context_scope": None,
            "repeat": 10,
            "concurrency": 1,
            "limited": False,
            "manifest": {"sha256": "manifest"},
            "server": {"runtime_id": "runtime"},
            "metrics": {"cases": 1, "deterministic_cases": 1},
            "cases": [{"id": "case", "prediction_sha256": "text"}],
        }
        safety = {
            "schema_version": "voicemed.structuring-safety.v1",
            "manifest_sha256": "manifest",
            "runtime_id": "runtime",
            "context_scope": None,
            "cases_total": 1,
            "prediction_inventory_sha256": current[
                "prediction_inventory_sha256"
            ],
            "critical_facts_total": 1,
            "critical_facts_with_provenance": 1,
            "unsupported_critical_facts": 0,
        }
        snapshot = {
            "schema_version": "voicemed.asr-dataset-snapshot.v1",
            "audio_hashes_verified": True,
            "promotion": {"eligible": True, "blockers": []},
            "outputs": {
                "real_frozen_benchmark": {"sha256": "manifest"}
            },
        }
        gate = benchmark.evaluate_release_gates(
            current,
            baseline,
            sequential,
            safety,
            snapshot,
        )
        self.assertFalse(gate["eligible"])
        self.assertIn("number_unit_exact_accuracy", gate["failed"])

    def test_release_mode_rejects_limit_before_running(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            benchmark.main(
                [
                    "--manifest",
                    "unused.tsv",
                    "--output",
                    "unused.json",
                    "--purpose",
                    "release",
                    "--repeat",
                    "10",
                    "--concurrency",
                    "10",
                    "--limit",
                    "1",
                ]
            )
        self.assertEqual(raised.exception.code, 2)

    def test_release_gate_passes_only_with_bound_complete_evidence(self) -> None:
        entity_metrics = {
            entity_type: {
                "available": True,
                "eligible_annotations": 1,
                "exact_accuracy": 1.0,
            }
            for entity_type in benchmark.SAFETY_ENTITY_TYPES
        }
        current = {
            "schema_version": "voicemed.asr-benchmark.v2",
            "purpose": "release",
            "context_scope": None,
            "repeat": 10,
            "concurrency": 10,
            "limited": False,
            "manifest": {"sha256": "manifest"},
            "server": {
                "runtime_id": "runtime",
                "configuration": {"strict_runtime_lock": True},
                "model": {"checkpoint": {"verified": True}},
                "runtime": {"source": {"projectDirty": False}},
            },
            "metrics": {
                "cases": 1,
                "deterministic_cases": 1,
                "medical_wer": 0.1,
                "normalized_wer": 0.105,
                "critical_entity_recall": {
                    "available": True,
                    "eligible_annotations": 4,
                    "recall": 1.0,
                },
                "entities": entity_metrics,
                "rtf": {"p95": 0.1},
            },
            "cases": [{"id": "case", "prediction_sha256": "text"}],
        }
        current["prediction_inventory_sha256"] = (
            benchmark.prediction_inventory_sha256(current["cases"])
        )
        baseline = {
            "schema_version": "voicemed.asr-benchmark.v2",
            "purpose": "baseline",
            "repeat": 10,
            "concurrency": 1,
            "limited": False,
            "manifest": {"sha256": "manifest"},
            "metrics": {
                "cases": 1,
                "deterministic_cases": 1,
                "medical_wer": 0.2,
                "normalized_wer": 0.1,
            },
            "cases": [{"id": "case", "prediction_sha256": "baseline"}],
        }
        sequential = {
            "schema_version": "voicemed.asr-benchmark.v2",
            "purpose": "candidate",
            "context_scope": None,
            "repeat": 10,
            "concurrency": 1,
            "limited": False,
            "manifest": {"sha256": "manifest"},
            "server": {"runtime_id": "runtime"},
            "metrics": {"cases": 1, "deterministic_cases": 1},
            "cases": [{"id": "case", "prediction_sha256": "text"}],
        }
        safety = {
            "schema_version": "voicemed.structuring-safety.v1",
            "manifest_sha256": "manifest",
            "runtime_id": "runtime",
            "context_scope": None,
            "cases_total": 1,
            "prediction_inventory_sha256": current[
                "prediction_inventory_sha256"
            ],
            "critical_facts_total": 2,
            "critical_facts_with_provenance": 2,
            "unsupported_critical_facts": 0,
        }
        snapshot = {
            "schema_version": "voicemed.asr-dataset-snapshot.v1",
            "audio_hashes_verified": True,
            "promotion": {"eligible": True, "blockers": []},
            "outputs": {
                "real_frozen_benchmark": {"sha256": "manifest"}
            },
        }
        gate = benchmark.evaluate_release_gates(
            current,
            baseline,
            sequential,
            safety,
            snapshot,
        )
        self.assertTrue(gate["eligible"], gate["failed"])
        safety["prediction_inventory_sha256"] = "different-predictions"
        rebound = benchmark.evaluate_release_gates(
            current,
            baseline,
            sequential,
            safety,
            snapshot,
        )
        self.assertFalse(rebound["eligible"])
        self.assertIn(
            "structuring_safety_prediction_binding",
            rebound["failed"],
        )


if __name__ == "__main__":
    unittest.main()
