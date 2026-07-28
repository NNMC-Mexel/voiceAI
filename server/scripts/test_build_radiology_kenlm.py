from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_radiology_kenlm as builder


class RadiologyKenlmBuilderTests(unittest.TestCase):
    domain = "ct_abdomen_contrast"

    def make_report(self, root: Path, name: str, text: str) -> dict[str, str]:
        path = root / f"{name}.txt"
        raw = text.encode("utf-8")
        path.write_bytes(raw)
        normalized = builder.conservative_whitespace(text)
        return {
            "id": name,
            "path": path.name,
            "dataset_kind": builder.EXPECTED_DATASET_KIND,
            "split": "train",
            "domain": self.domain,
            "reference_status": builder.EXPECTED_REFERENCE_STATUS,
            "approved": "true",
            "deidentified": "true",
            "reviewed_by": "reviewer-01",
            "approved_at": "2026-07-28T10:00:00+05:00",
            "document_sha256": hashlib.sha256(raw).hexdigest(),
            "text_sha256": builder.sha256_text(normalized),
        }

    def write_manifest(
        self,
        root: Path,
        rows: list[dict[str, str]],
    ) -> Path:
        path = root / "lm-manifest.tsv"
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output,
                delimiter="\t",
                fieldnames=builder.MANIFEST_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def make_defaults(self, root: Path, lines: list[str]) -> Path:
        path = root / "approved-defaults.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": builder.TEMPLATE_DEFAULTS_SCHEMA_VERSION,
                    "approved": True,
                    "deidentified": True,
                    "reviewed_by": "reviewer-02",
                    "approved_at": "2026-07-28",
                    "lines": lines,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    def make_request(
        self,
        root: Path,
        manifest: Path,
        *,
        defaults: Path | None = None,
    ) -> builder.BuildRequest:
        lmplz = root / "lmplz"
        build_binary = root / "build_binary"
        lmplz.write_bytes(b"pinned lmplz executable")
        build_binary.write_bytes(b"pinned build_binary executable")
        return builder.BuildRequest(
            manifest=manifest,
            domain=self.domain,
            lmplz=lmplz,
            build_binary=build_binary,
            kenlm_version="kenlm-2026-01-01-commit-deadbeef",
            output_corpus=root / "ct-abdomen.corpus.txt",
            output_model=root / "ct-abdomen-5gram.bin",
            output_lock=root / "ct-abdomen-5gram.lock.json",
            template_defaults=defaults,
            memory="65%",
        )

    def successful_tool_runner(
        self,
        calls: list[tuple[list[str], dict[str, object]]],
    ):
        def run(command, **kwargs):
            calls.append((list(command), dict(kwargs)))
            if Path(command[0]).name == "lmplz":
                kwargs["stdout"].write(b"\\data\\\nngram 1=1\n")
            else:
                Path(command[-1]).write_bytes(b"binary kenlm model")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"",
                stderr=b"",
            )

        return run

    def test_build_is_content_addressed_and_invokes_tools_without_shell(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.make_report(
                    root,
                    "report-02",
                    "Печень не увеличена.\nСелезёнка: киста 9 мм.",
                ),
                self.make_report(
                    root,
                    "report-01",
                    "Печень не увеличена.\nЧревный ствол: стеноз 50%.",
                ),
            ]
            manifest = self.write_manifest(root, rows)
            defaults = self.make_defaults(root, ["Печень не увеличена."])
            request = self.make_request(root, manifest, defaults=defaults)
            calls: list[tuple[list[str], dict[str, object]]] = []

            with mock.patch.object(
                builder.subprocess,
                "run",
                side_effect=self.successful_tool_runner(calls),
            ):
                metadata = builder.run_build(request)

            self.assertEqual(
                request.output_corpus.read_text(encoding="utf-8"),
                (
                    "Чревный ствол: стеноз 50%.\n"
                    "Селезёнка: киста 9 мм.\n"
                ),
            )
            self.assertEqual(request.output_model.read_bytes(), b"binary kenlm model")
            lock = json.loads(request.output_lock.read_text(encoding="utf-8"))
            self.assertEqual(lock, metadata)
            self.assertEqual(lock["model"]["order"], 5)
            self.assertEqual(
                lock["corpus"]["sha256"],
                hashlib.sha256(request.output_corpus.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                lock["model"]["sha256"],
                hashlib.sha256(request.output_model.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                lock["corpus"]["exact_template_default_lines_removed"],
                2,
            )
            self.assertEqual(
                [item["id"] for item in lock["inputs"]["documents"]],
                ["report-01", "report-02"],
            )
            self.assertEqual(
                lock["tools"]["lmplz"]["binary_sha256"],
                builder.sha256_file(request.lmplz),
            )
            self.assertEqual(
                lock["tools"]["build_binary"]["version"],
                request.kenlm_version,
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0][1:3], ["-o", "5"])
            self.assertEqual(Path(calls[1][0][0]).name, "build_binary")
            for _, kwargs in calls:
                self.assertIs(kwargs["shell"], False)
                self.assertEqual(kwargs["env"]["LC_ALL"], "C")

    def test_template_default_matching_is_exact_after_whitespace_normalization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.make_report(
                    root,
                    "report-01",
                    (
                        "  Печень\u00a0не увеличена.  \n"
                        "печень не увеличена.\n"
                        "Печень не увеличена!\n"
                    ),
                )
            ]
            documents = builder.load_governed_documents(
                self.write_manifest(root, rows),
                self.domain,
            )
            defaults = builder.load_template_defaults(
                self.make_defaults(root, ["Печень не увеличена."])
            )
            corpus, _, removed, _ = builder.build_corpus_bytes(
                documents,
                defaults,
            )
            self.assertEqual(removed, 1)
            self.assertEqual(
                corpus.decode("utf-8"),
                "печень не увеличена.\nПечень не увеличена!\n",
            )

    def test_validation_and_test_rows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.make_report(root, "report-01", "Печень без особенностей.")
            row["split"] = "validation"
            with self.assertRaisesRegex(ValueError, "validation/test/non-train"):
                builder.load_governed_documents(
                    self.write_manifest(root, [row]),
                    self.domain,
                )

    def test_unapproved_or_non_deidentified_rows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.make_report(root, "report-01", "Почки без особенностей.")
            row["approved"] = "false"
            with self.assertRaisesRegex(ValueError, "approved"):
                builder.load_governed_documents(
                    self.write_manifest(root, [row]),
                    self.domain,
                )
            row["approved"] = "true"
            row["deidentified"] = "false"
            with self.assertRaisesRegex(ValueError, "deidentified"):
                builder.load_governed_documents(
                    self.write_manifest(root, [row]),
                    self.domain,
                )

    def test_duplicate_document_and_text_hashes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.make_report(root, "report-01", "Селезёнка не увеличена.")
            second = dict(first)
            second["id"] = "report-02"
            with self.assertRaisesRegex(ValueError, "duplicate document_sha256"):
                builder.load_governed_documents(
                    self.write_manifest(root, [first, second]),
                    self.domain,
                )

            second = self.make_report(
                root,
                "report-02",
                "  Селезёнка  не увеличена.\r\n",
            )
            self.assertNotEqual(
                first["document_sha256"],
                second["document_sha256"],
            )
            self.assertEqual(first["text_sha256"], second["text_sha256"])
            with self.assertRaisesRegex(ValueError, "duplicate text_sha256"):
                builder.load_governed_documents(
                    self.write_manifest(root, [first, second]),
                    self.domain,
                )

    def test_obvious_phi_is_rejected_despite_manifest_assertion(self) -> None:
        phi_examples = (
            "ФИО: Иванов Иван Иванович\nПечень не увеличена.",
            "ИИН: 990101123456\nПочки без особенностей.",
            "Телефон: +7 701 123 45 67\nСелезёнка не увеличена.",
        )
        for index, text in enumerate(phi_examples):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                row = self.make_report(root, "report-01", text)
                with self.assertRaisesRegex(ValueError, "obvious PHI"):
                    builder.load_governed_documents(
                        self.write_manifest(root, [row]),
                        self.domain,
                    )

    def test_declared_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.make_report(root, "report-01", "Желчный пузырь не изменён.")
            row["document_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "document_sha256 mismatch"):
                builder.load_governed_documents(
                    self.write_manifest(root, [row]),
                    self.domain,
                )

    def test_unapproved_template_defaults_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_defaults(root, ["Печень не увеличена."])
            value = json.loads(path.read_text(encoding="utf-8"))
            value["approved"] = False
            path.write_text(
                json.dumps(value, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "explicitly approved"):
                builder.load_template_defaults(path)

    def test_overwrite_cannot_target_a_governed_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.make_report(root, "report-01", "Печень не увеличена.")
            manifest = self.write_manifest(root, [row])
            request = self.make_request(root, manifest)
            request = builder.BuildRequest(
                **{
                    **request.__dict__,
                    "output_corpus": root / row["path"],
                    "overwrite": True,
                }
            )
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                builder.run_build(request)

    def test_tool_failure_does_not_publish_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.make_report(root, "report-01", "Печень не увеличена.")
            request = self.make_request(
                root,
                self.write_manifest(root, [row]),
            )

            with mock.patch.object(
                builder.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    ["lmplz"],
                    1,
                    stdout=b"",
                    stderr=b"failure",
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "exit code 1"):
                    builder.run_build(request)

            self.assertFalse(request.output_corpus.exists())
            self.assertFalse(request.output_model.exists())
            self.assertFalse(request.output_lock.exists())


if __name__ == "__main__":
    unittest.main()
