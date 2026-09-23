#!/usr/bin/env python3
"""HW-hotspot offline weight staging: copy local files, never hit a hub."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pi_stage_weights_offline.sh"
MAC_SCRIPT = ROOT / "scripts" / "mac_download_edge_weights.sh"


class OfflineStageTests(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT.is_file())
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("192.168.149.1", text)
        self.assertIn("qwen2.5-1.5b-instruct-q4_k_m.gguf", text)
        self.assertNotIn("snapshot_download", text)

    def test_copies_local_gguf_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "usb"
            dest_llm = Path(tmp) / "edge_llm"
            dest_ser = Path(tmp) / "audio_ser"
            src.mkdir()
            dest_llm.mkdir()
            dest_ser.mkdir()
            blob = src / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
            blob.write_bytes(b"fake-gguf")
            env = os.environ.copy()
            env["WEIGHTS_LLM_DIR"] = str(dest_llm)
            env["WEIGHTS_SER_DIR"] = str(dest_ser)
            env["OFFLINE"] = "1"
            subprocess.check_call(["bash", str(SCRIPT), str(src)], env=env)
            out = dest_llm / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
            self.assertTrue(out.is_file())
            self.assertEqual(out.read_bytes(), b"fake-gguf")

    def test_mac_download_script_is_local_first(self):
        self.assertTrue(MAC_SCRIPT.is_file())
        text = MAC_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("qwen2.5-1.5b-instruct-q4_k_m.gguf", text)
        self.assertIn("VNC", text)
        self.assertIn("192.168.149.1", text)
        self.assertIn("hf-mirror.com", text)

    def test_missing_source_is_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest_llm = Path(tmp) / "edge_llm"
            dest_ser = Path(tmp) / "audio_ser"
            dest_llm.mkdir()
            dest_ser.mkdir()
            env = os.environ.copy()
            env["WEIGHTS_LLM_DIR"] = str(dest_llm)
            env["WEIGHTS_SER_DIR"] = str(dest_ser)
            subprocess.check_call(["bash", str(SCRIPT), str(Path(tmp) / "empty")], env=env)
            self.assertFalse(any(dest_llm.iterdir()))


if __name__ == "__main__":
    unittest.main()
