from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_asr_manifest as validator
import asr_benchmark as benchmark


class ManifestValidationTests(unittest.TestCase):
    def make_rows(self, root: Path) -> list[dict[str, str]]:
        definitions = [
            ("train", "real_train", "train", "doctor-a", "patient-a", "study-a"),
            ("validation", "real_validation", "validation", "doctor-a", "patient-b", "study-b"),
            ("test", "real_frozen_test", "test", "doctor-b", "patient-c", "study-c"),
        ]
        rows: list[dict[str, str]] = []
        for index, (case_id, kind, split, speaker, patient, study) in enumerate(
            definitions, start=1
        ):
            audio = root / f"{case_id}.wav"
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(
                    int(index).to_bytes(2, "little", signed=True) * 16_000 * 3
                )
            audio_bytes = audio.read_bytes()
            rows.append(
                {
                    "id": case_id,
                    "path": audio.name,
                    "duration": "3.0",
                    "transcription": "печень не увеличена",
                    "dataset_kind": kind,
                    "split": split,
                    "speaker_id": speaker,
                    "patient_id": patient,
                    "study_id": study,
                    "recorded_at": f"2026-01-0{index}",
                    "modality": "ct",
                    "anatomy": "abdomen",
                    "reviewed_by": "reviewer",
                    "frozen_at": "2026-07-01" if split == "test" else "",
                    "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
                    "source_recording_id": f"recording-{index}",
                    "transcript_source": "human_verbatim",
                    "entities_json": json.dumps(
                        [
                            {
                                "id": f"term-{index}",
                                "type": "medical_term",
                                "text": "печень",
                                "start_word": 0,
                                "end_word": 1,
                            },
                            {
                                "id": f"negation-{index}",
                                "type": "negation",
                                "text": "не",
                                "start_word": 1,
                                "end_word": 2,
                            },
                        ],
                        ensure_ascii=False,
                    ),
                    "terms": "печень",
                }
            )
        return rows

    def write_manifest(
        self,
        path: Path,
        rows: list[dict[str, str]],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=validator.MANIFEST_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_snapshot_has_speaker_held_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows_by_split, warnings = validator.validate(
                root / "manifest.tsv",
                self.make_rows(root),
                verify_audio=False,
            )
            self.assertEqual(len(rows_by_split["test"]), 1)
            self.assertTrue(any("general_replay" in warning for warning in warnings))

    def test_patient_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = copy.deepcopy(self.make_rows(root))
            rows[-1]["patient_id"] = rows[0]["patient_id"]
            with self.assertRaisesRegex(ValueError, "patient_id"):
                validator.validate(
                    root / "manifest.tsv",
                    rows,
                    verify_audio=False,
                )

    def test_synthetic_test_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_rows(root)
            rows[-1]["dataset_kind"] = "synthetic_train"
            with self.assertRaisesRegex(ValueError, "must use split=train"):
                validator.validate(
                    root / "manifest.tsv",
                    rows,
                    verify_audio=False,
                )

    def test_browser_transcript_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_rows(root)
            rows[0]["transcript_source"] = "browser_asr"
            with self.assertRaisesRegex(ValueError, "browser_asr"):
                validator.validate(
                    root / "manifest.tsv",
                    rows,
                    verify_audio=False,
                )

    def test_entity_text_must_match_annotated_word_span(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_rows(root)
            entities = json.loads(rows[-1]["entities_json"])
            entities[0]["text"] = "селезёнка"
            rows[-1]["entities_json"] = json.dumps(
                entities,
                ensure_ascii=False,
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                validator.validate(
                    root / "manifest.tsv",
                    rows,
                    verify_audio=False,
                )

    def test_promotion_assessment_is_explicitly_ineligible_for_tiny_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows_by_split, _ = validator.validate(
                root / "manifest.tsv",
                self.make_rows(root),
                verify_audio=True,
            )
            assessment = validator.promotion_assessment(
                rows_by_split,
                audio_hashes_verified=True,
            )
            self.assertFalse(assessment["eligible"])
            self.assertTrue(
                any("10h" in blocker for blocker in assessment["blockers"])
            )

    def test_emitted_frozen_manifest_is_accepted_by_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows_by_split, _ = validator.validate(
                root / "manifest.tsv",
                self.make_rows(root),
                verify_audio=True,
            )
            emitted = root / "real-frozen-benchmark.tsv"
            validator.write_governed_benchmark_manifest(
                emitted,
                rows_by_split["test"],
            )
            selected = benchmark.validate_benchmark_manifest(
                emitted,
                benchmark.read_manifest(emitted),
                purpose="release",
            )
            self.assertEqual([row["id"] for row in selected], ["test"])
            validation = root / "real-validation-benchmark.tsv"
            validator.write_governed_benchmark_manifest(
                validation,
                rows_by_split["validation"],
            )
            selected_validation = benchmark.validate_benchmark_manifest(
                validation,
                benchmark.read_manifest(validation),
                purpose="validation",
            )
            self.assertEqual(
                [row["id"] for row in selected_validation],
                ["validation"],
            )

    def test_verified_duration_must_match_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_rows(root)
            rows[0]["duration"] = "10.0"
            with self.assertRaisesRegex(ValueError, "duration mismatch"):
                validator.validate(
                    root / "manifest.tsv",
                    rows,
                    verify_audio=True,
                )

    def test_snapshot_is_immutable_without_explicit_draft_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.tsv"
            output = root / "snapshot"
            self.write_manifest(manifest, self.make_rows(root))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    validator.main(
                        [
                            "--manifest",
                            str(manifest),
                            "--output-dir",
                            str(output),
                        ]
                    ),
                    0,
                )
            with self.assertRaisesRegex(ValueError, "immutable"):
                with redirect_stdout(io.StringIO()):
                    validator.main(
                        [
                            "--manifest",
                            str(manifest),
                            "--output-dir",
                            str(output),
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
