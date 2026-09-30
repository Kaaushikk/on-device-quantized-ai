import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from quantbench.common import sha256, verify_files


class ProvenanceTests(unittest.TestCase):
    def test_modified_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "fixture.bin"
            artifact.write_bytes(b"original")
            manifest = {"files": {"fixture.bin": sha256(artifact)}}
            with patch("quantbench.common.ARTIFACTS", root):
                verify_files(manifest)
                artifact.write_bytes(b"modified")
                with self.assertRaises(ValueError):
                    verify_files(manifest)

    def test_directory_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_files({"files": {"../outside.bin": "unused"}})
