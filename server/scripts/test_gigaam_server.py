from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gigaam_server as runtime


class FakeTokenizer:
    pieces = {0: "п", 1: "е", 2: "ч", 3: "н", 4: "ь", 5: " ", 6: "н", 7: "о", 8: "р", 9: "м", 10: "а"}

    def id_to_str(self, token_id: int) -> str:
        return self.pieces[token_id]


class FakeBeamBackend:
    def decode(
        self,
        log_probs: object,
        beam_width: int,
        hotwords: list[str],
        hotword_weight: float,
    ) -> dict[str, object]:
        return {
            "text": "печень не увеличена",
            "words": [
                {"text": "печень", "start_frame": 2, "end_frame": 7},
                {"text": "не", "start_frame": 8, "end_frame": 10},
                {"text": "увеличена", "start_frame": 11, "end_frame": 20},
            ],
            "acoustic_log_score": -12.5,
            "fused_log_score": -9.25,
        }


class TinyCTCModel:
    class Tokenizer:
        labels = ["а", " "]

        def __len__(self) -> int:
            return len(self.labels)

        def id_to_str(self, token_id: int) -> str:
            return self.labels[token_id]

        def decode(self, token_ids: list[int]) -> str:
            return "".join(self.labels[token_id] for token_id in token_ids)

    def __init__(self) -> None:
        self.decoding = types.SimpleNamespace(
            tokenizer=self.Tokenizer(),
            blank_id=2,
        )
        self.cfg = types.SimpleNamespace(model_name="v3_ctc")

    def prepare_wav(self, _: str) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, 1600), torch.tensor([1600])

    def forward(
        self, wav: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, 1, 4), torch.tensor([5])

    def head(self, encoder_output: torch.Tensor) -> torch.Tensor:
        probabilities = torch.tensor(
            [
                [
                    [0.8, 0.1, 0.1],
                    [0.6, 0.2, 0.2],
                    [0.05, 0.05, 0.9],
                    [0.7, 0.1, 0.2],
                    [0.1, 0.8, 0.1],
                ]
            ]
        )
        return probabilities.log()


class StitchTokenizer:
    pieces = {0: "a", 1: "b", 2: "c", 3: " ", 4: "1", 5: "5"}

    def id_to_str(self, token_id: int) -> str:
        return self.pieces[token_id]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.pieces[token_id] for token_id in token_ids)


def test_emission(labels: list[int], vocabulary_size: int = 7) -> np.ndarray:
    values = np.full((len(labels), vocabulary_size), -12.0, dtype=np.float32)
    for frame, token_id in enumerate(labels):
        values[frame, token_id] = 0.0
    return values


class GigaAMRuntimeTests(unittest.TestCase):
    def test_container_source_identity_uses_validated_build_metadata(self) -> None:
        commit = "a" * 40
        with patch.dict(
            os.environ,
            {
                "VOICEMED_SOURCE_COMMIT": commit,
                "VOICEMED_SOURCE_DIRTY": "false",
            },
        ):
            self.assertEqual(
                runtime.project_git_metadata(),
                {"commit": commit, "dirty": False},
            )

    def test_container_source_identity_rejects_partial_metadata(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOICEMED_SOURCE_COMMIT": "not-a-commit",
                "VOICEMED_SOURCE_DIRTY": "false",
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "40-character Git commit"):
                runtime.project_git_metadata()

    def test_network_bind_classification_is_fail_closed(self) -> None:
        self.assertTrue(runtime.is_loopback_host("127.0.0.1"))
        self.assertTrue(runtime.is_loopback_host("::1"))
        self.assertTrue(runtime.is_loopback_host("localhost"))
        self.assertFalse(runtime.is_loopback_host("0.0.0.0"))
        self.assertFalse(runtime.is_loopback_host("192.168.1.20"))
        self.assertFalse(runtime.is_loopback_host("gigaam.internal"))

    def test_non_loopback_bind_requires_explicit_opt_in(self) -> None:
        old_host = runtime.HOST
        old_remote = runtime.ALLOW_REMOTE_BIND
        old_audio_path = runtime.ALLOW_AUDIO_PATH
        try:
            runtime.HOST = "0.0.0.0"
            runtime.ALLOW_REMOTE_BIND = False
            runtime.ALLOW_AUDIO_PATH = False
            with self.assertRaisesRegex(RuntimeError, "ALLOW_REMOTE_BIND"):
                runtime.validate_security_configuration()
            runtime.ALLOW_REMOTE_BIND = True
            runtime.validate_security_configuration()
        finally:
            runtime.HOST = old_host
            runtime.ALLOW_REMOTE_BIND = old_remote
            runtime.ALLOW_AUDIO_PATH = old_audio_path

    def test_strict_longform_rejects_dev_overlap_mode(self) -> None:
        old_mode = runtime.LONGFORM_MODE
        old_value = os.environ.get("GIGAAM_LONGFORM_STRICT")
        try:
            runtime.LONGFORM_MODE = "overlap"
            os.environ["GIGAAM_LONGFORM_STRICT"] = "true"
            with self.assertRaisesRegex(RuntimeError, "requires VAD"):
                runtime.validate_security_configuration()
        finally:
            runtime.LONGFORM_MODE = old_mode
            if old_value is None:
                os.environ.pop("GIGAAM_LONGFORM_STRICT", None)
            else:
                os.environ["GIGAAM_LONGFORM_STRICT"] = old_value

    def test_vad_identity_matches_runtime_lock(self) -> None:
        lock = json.loads(
            (Path(runtime.__file__).parent / "gigaam-runtime.lock.json").read_text(
                encoding="utf-8"
            )
        )
        vad = lock["models"]["pyannote_segmentation_3_0"]
        self.assertEqual(vad["repo"], runtime.VAD_MODEL_ID)
        self.assertEqual(vad["revision"], runtime.VAD_MODEL_REVISION)

    def test_audio_path_is_disabled_unless_explicitly_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "disabled"):
            runtime.resolve_allowed_audio_path(
                "anything.wav",
                enabled=False,
                root_value="",
            )

    def test_audio_path_must_resolve_inside_allowlisted_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            allowed = base / "allowed"
            allowed.mkdir()
            inside = allowed / "inside.wav"
            inside.write_bytes(b"wav")
            outside = base / "outside.wav"
            outside.write_bytes(b"wav")

            resolved = runtime.resolve_allowed_audio_path(
                "inside.wav",
                enabled=True,
                root_value=str(allowed),
            )
            self.assertEqual(resolved, inside.resolve())
            with self.assertRaisesRegex(ValueError, "outside"):
                runtime.resolve_allowed_audio_path(
                    "../outside.wav",
                    enabled=True,
                    root_value=str(allowed),
                )
            with self.assertRaisesRegex(ValueError, "outside"):
                runtime.resolve_allowed_audio_path(
                    str(outside),
                    enabled=True,
                    root_value=str(allowed),
                )

    def test_audio_path_rejects_symlink_escape_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            allowed = base / "allowed"
            allowed.mkdir()
            outside = base / "outside.wav"
            outside.write_bytes(b"wav")
            link = allowed / "link.wav"
            try:
                os.symlink(outside, link)
            except OSError:
                self.skipTest("symlink creation is not permitted on this host")
            with self.assertRaisesRegex(ValueError, "outside"):
                runtime.resolve_allowed_audio_path(
                    "link.wav",
                    enabled=True,
                    root_value=str(allowed),
                )

    def test_degraded_text_fallback_never_deletes_overlap_tokens(self) -> None:
        self.assertEqual(
            runtime.concatenate_chunk_text(
                [
                    "Плотность пятьдесят",
                    "пятьдесят три единицы",
                ]
            ),
            "Плотность пятьдесят пятьдесят три единицы",
        )

    def test_ctc_words_have_real_geometric_mean_score(self) -> None:
        token_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        frames = list(range(len(token_ids)))
        scores = [-0.1] * len(token_ids)
        words = runtime.words_from_ctc_tokens(
            FakeTokenizer(),
            token_ids,
            frames,
            scores,
            frame_shift=0.02,
        )
        self.assertEqual([word["text"] for word in words], ["печнь", "норма"])
        self.assertAlmostEqual(words[0]["confidence"], math.exp(-0.1), places=6)
        self.assertEqual(words[0]["start"], 0.0)
        self.assertEqual(words[1]["end"], 0.22)

    def test_ctc_runs_preserve_repeated_label_frame_spans(self) -> None:
        self.assertEqual(
            runtime.ctc_token_runs(
                [0, 0, 2, 0, 0, 0, 1, 1, 2],
                blank_id=2,
            ),
            [(0, 0, 2), (0, 3, 6), (1, 6, 8)],
        )

    def test_vad_grouping_uses_pause_windows_and_20_by_2_continuous_fallback(
        self,
    ) -> None:
        windows = runtime.group_vad_regions(
            [(1.0, 4.0), (5.0, 10.0), (30.0, 55.0)],
            60.0,
        )
        self.assertEqual(windows[0], runtime.AudioWindow(0.75, 10.25, "vad"))
        self.assertEqual(
            windows[1:],
            [
                runtime.AudioWindow(29.75, 49.75, "continuous_fallback"),
                runtime.AudioWindow(47.75, 55.25, "continuous_fallback"),
            ],
        )
        self.assertTrue(
            all(window.end - window.start <= 24.0 for window in windows)
        )

    def test_vad_padding_never_exceeds_hard_limit(self) -> None:
        windows = runtime.group_vad_regions(
            [(0.1, 24.0)],
            30.0,
            target_seconds=20.0,
            hard_max_seconds=24.0,
            padding_seconds=0.25,
        )
        self.assertEqual(len(windows), 1)
        self.assertAlmostEqual(windows[0].end - windows[0].start, 24.0)

    def test_ctc_overlap_is_stitched_before_one_decode(self) -> None:
        blank_id = 6
        tokenizer = StitchTokenizer()
        left = runtime.CTCEmissionWindow(
            runtime.AudioWindow(0.0, 4.0, "continuous_fallback"),
            runtime.CTCEmission(
                test_emission([0, 0, 1, 6]),
                frame_shift=1.0,
                duration=4.0,
            ),
            elapsed=0.1,
        )
        right = runtime.CTCEmissionWindow(
            runtime.AudioWindow(2.0, 6.0, "continuous_fallback"),
            runtime.CTCEmission(
                test_emission([1, 6, 2, 2]),
                frame_shift=1.0,
                duration=4.0,
            ),
            elapsed=0.1,
        )
        stitched = runtime.stitch_ctc_emissions(
            [left, right],
            tokenizer,
            blank_id,
        )
        labels = stitched.log_probs.argmax(axis=-1).tolist()
        token_ids = [
            token_id
            for token_id, _, _ in runtime.ctc_token_runs(labels, blank_id)
        ]
        self.assertEqual(tokenizer.decode(token_ids), "abc")
        self.assertEqual(len(stitched.frame_times), 6)
        self.assertEqual(stitched.seam_conflicts, [])

    def test_stitch_records_critical_numeric_seam_conflict(self) -> None:
        blank_id = 6
        tokenizer = StitchTokenizer()
        left = runtime.CTCEmissionWindow(
            runtime.AudioWindow(0.0, 4.0, "continuous_fallback"),
            runtime.CTCEmission(
                test_emission([0, 6, 4, 6]),
                frame_shift=1.0,
                duration=4.0,
            ),
            elapsed=0.1,
        )
        right = runtime.CTCEmissionWindow(
            runtime.AudioWindow(2.0, 6.0, "continuous_fallback"),
            runtime.CTCEmission(
                test_emission([5, 6, 2, 6]),
                frame_shift=1.0,
                duration=4.0,
            ),
            elapsed=0.1,
        )
        stitched = runtime.stitch_ctc_emissions(
            [left, right],
            tokenizer,
            blank_id,
        )
        self.assertEqual(len(stitched.seam_conflicts), 1)
        self.assertTrue(stitched.seam_conflicts[0]["critical"])
        self.assertIn(
            "number",
            stitched.seam_conflicts[0]["critical_classes"],
        )
        integrity = runtime.inference_integrity(
            seam_conflicts=stitched.seam_conflicts
        )
        self.assertTrue(integrity["approval_blocked"])
        self.assertFalse(integrity["training_eligible"])

    def test_stitched_frame_timeline_drives_word_timestamps(self) -> None:
        words = runtime.words_from_ctc_tokens(
            StitchTokenizer(),
            [0, 1],
            [0, 1],
            [-0.1, -0.1],
            frame_shift=1.0,
            token_end_frames=[1, 2],
            frame_times=[(10.0, 10.1), (20.0, 20.1)],
        )
        self.assertEqual(words[0]["start"], 10.0)
        self.assertEqual(words[0]["end"], 20.1)

    def test_longform_ctc_extracts_each_window_but_decodes_once(self) -> None:
        old_model = runtime.MODEL
        runtime.MODEL = types.SimpleNamespace(
            decoding=types.SimpleNamespace(
                tokenizer=StitchTokenizer(),
                blank_id=6,
            )
        )
        extracted = 0
        decoded = 0

        def fake_extract(_: str) -> runtime.CTCEmission:
            nonlocal extracted
            labels = (
                [0, 0, 1, 6]
                if extracted == 0
                else [1, 6, 2, 2]
            )
            extracted += 1
            return runtime.CTCEmission(
                test_emission(labels),
                frame_shift=1.0,
                duration=4.0,
            )

        def fake_decode(*_: object, **__: object) -> runtime.InferenceResult:
            nonlocal decoded
            decoded += 1
            return runtime.InferenceResult(
                text="abc",
                words=[],
                chunks=1,
                chunk_details=[],
                confidence=runtime.unavailable_confidence("test"),
                method="test",
            )

        try:
            with tempfile.TemporaryDirectory() as directory:
                wav_path = Path(directory) / "audio.wav"
                sf.write(wav_path, np.zeros(6 * 16000), 16000)
                with (
                    patch.object(
                        runtime,
                        "extract_ctc_emission",
                        side_effect=fake_extract,
                    ),
                    patch.object(
                        runtime,
                        "decode_ctc_emission",
                        side_effect=fake_decode,
                    ),
                ):
                    result = runtime.transcribe_ctc_windows(
                        str(wav_path),
                        [
                            runtime.AudioWindow(
                                0.0, 4.0, "continuous_fallback"
                            ),
                            runtime.AudioWindow(
                                2.0, 6.0, "continuous_fallback"
                            ),
                        ],
                        None,
                        method="fixed_overlap_ctc_stitched",
                    )
        finally:
            runtime.MODEL = old_model
        self.assertEqual(extracted, 2)
        self.assertEqual(decoded, 1)
        self.assertEqual(result.chunks, 2)
        self.assertFalse(result.integrity["degraded"])

    def test_unavailable_confidence_has_no_fake_numeric_fields(self) -> None:
        confidence = runtime.unavailable_confidence("not_exposed")
        self.assertFalse(confidence["available"])
        self.assertNotIn("avg_logprob", confidence)
        self.assertNotIn("low_confidence", confidence)

    def test_chunk_confidence_is_token_weighted(self) -> None:
        confidence = runtime.combine_chunk_confidence(
            [
                {"available": True, "avg_logprob": -0.5, "emitted_tokens": 10},
                {"available": True, "avg_logprob": -1.0, "emitted_tokens": 30},
            ]
        )
        self.assertTrue(confidence["available"])
        self.assertAlmostEqual(confidence["avg_logprob"], -0.875)
        self.assertEqual(confidence["emitted_tokens"], 40)

    def test_beam_hook_validates_frames_and_does_not_invent_confidence(self) -> None:
        text, words, confidence = runtime.decode_with_beam_backend(
            log_probs=object(),
            frame_shift=0.02,
            hotwords=["печень"],
            backend=FakeBeamBackend(),
        )
        self.assertEqual(text, "печень не увеличена")
        self.assertEqual(words[0]["start"], 0.04)
        self.assertEqual(words[-1]["end"], 0.4)
        self.assertFalse(confidence["available"])
        self.assertEqual(confidence["hotwords_applied"], 1)
        self.assertEqual(
            confidence["decoder_scores"]["fused_log_score"],
            -9.25,
        )

    def test_ctc_forward_uses_selected_emission_logprob(self) -> None:
        old_model = runtime.MODEL
        old_mode = runtime.CTC_DECODER_MODE
        old_gigaam = sys.modules.get("gigaam")
        old_gigaam_model = sys.modules.get("gigaam.model")
        fake_package = types.ModuleType("gigaam")
        fake_model_module = types.ModuleType("gigaam.model")
        fake_model_module.SAMPLE_RATE = 16000
        sys.modules["gigaam"] = fake_package
        sys.modules["gigaam.model"] = fake_model_module
        try:
            runtime.MODEL = TinyCTCModel()
            runtime.CTC_DECODER_MODE = "greedy"
            result = runtime.transcribe_short_ctc_scored("unused.wav")
        finally:
            runtime.MODEL = old_model
            runtime.CTC_DECODER_MODE = old_mode
            if old_gigaam is None:
                sys.modules.pop("gigaam", None)
            else:
                sys.modules["gigaam"] = old_gigaam
            if old_gigaam_model is None:
                sys.modules.pop("gigaam.model", None)
            else:
                sys.modules["gigaam.model"] = old_gigaam_model
        self.assertEqual(result.text, "аа")
        self.assertEqual(result.confidence["emitted_tokens"], 3)
        expected = (math.log(0.8) + math.log(0.7) + math.log(0.8)) / 3
        self.assertAlmostEqual(result.confidence["avg_logprob"], expected, places=6)
        self.assertEqual(result.words[0]["end"], 0.08)

    def test_shared_inference_lock_serializes_threads(self) -> None:
        active = 0
        maximum_active = 0
        guard = threading.Lock()
        start = threading.Barrier(5)

        def fake_transcriber(_: str) -> runtime.InferenceResult:
            nonlocal active, maximum_active
            start.wait(timeout=2)
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return runtime.InferenceResult(
                text="ok",
                words=[],
                chunks=1,
                chunk_details=[],
                confidence=runtime.unavailable_confidence("test"),
                method="test",
            )

        # The barrier must be reached before the model lock, so wrap only the
        # critical part passed to run_serialized_inference.
        def worker() -> None:
            start.wait(timeout=2)

            def critical(_: str) -> runtime.InferenceResult:
                nonlocal active, maximum_active
                with guard:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.02)
                with guard:
                    active -= 1
                return runtime.InferenceResult(
                    text="ok",
                    words=[],
                    chunks=1,
                    chunk_details=[],
                    confidence=runtime.unavailable_confidence("test"),
                    method="test",
                )

            runtime.run_serialized_inference("unused", critical)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(maximum_active, 1)

    def test_strict_runtime_lock_rejects_missing_lock(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "readable runtime lock"):
            runtime.validate_runtime_lock_contract(
                Path("missing-runtime-lock.json"),
                {},
                strict=True,
            )

    def test_strict_runtime_lock_checks_python_and_requirement_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_requirements = root / "runtime.txt"
            longform_requirements = root / "longform.txt"
            runtime_requirements.write_text("runtime\n", encoding="utf-8")
            longform_requirements.write_text("longform\n", encoding="utf-8")
            lock_path = root / "runtime-lock.json"
            lock = {
                "schemaVersion": 1,
                "gigaamGitCommit": "a" * 40,
                "requirements": {
                    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "runtime": runtime_requirements.name,
                    "runtimeSha256": runtime.canonical_text_sha256(
                        runtime_requirements
                    ),
                    "optionalLongform": longform_requirements.name,
                    "optionalLongformSha256": runtime.canonical_text_sha256(
                        longform_requirements
                    ),
                },
            }
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            runtime.validate_runtime_lock_contract(
                lock_path,
                lock,
                strict=True,
            )
            lock["requirements"]["runtimeSha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                runtime.validate_runtime_lock_contract(
                    lock_path,
                    lock,
                    strict=True,
                )

    def test_strict_component_versions_fail_on_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "version mismatch"):
            runtime.validate_locked_component_versions(
                {
                    "components": {"torch": "expected"},
                    "optionalComponents": {},
                },
                {"torch": "actual"},
                strict=True,
            )


if __name__ == "__main__":
    unittest.main()
