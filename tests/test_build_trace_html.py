import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))


class BuildTraceHtmlTests(unittest.TestCase):
    def _make_run_full(self, root: Path) -> Path:
        from utils.trace_artifacts import TraceRunWriter

        run_dir = root / "run_full"
        (run_dir / "00_input").mkdir(parents=True)
        (run_dir / "00_input" / "full.wav").write_bytes(b"wav")

        segments = [
            {
                "index": "00000",
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_00",
                "text": "xin chao",
                "text_whisper": "xin chao",
                "enhanced_audio": [0.0, 0.2, -0.2, 0.0],
            },
            {
                "index": "00001",
                "start": 1.0,
                "end": 2.0,
                "speaker": "SPEAKER_01",
                "text": "tam biet",
                "text_whisper": "tam biet",
            },
        ]
        audio = {"waveform": [0.0, 0.2, -0.2, 0.0], "sample_rate": 16000}
        writer = TraceRunWriter(run_dir, source_audio_path="input.wav")
        writer.write_diarization(segments, metadata={"audio_duration_seconds": 2.0})
        writer.write_music_clean(segments, [True, False], audio)
        writer.write_overlap(segments, audio)
        writer.write_asr(segments)

        source_segments = root / "source_segments"
        source_segments.mkdir()
        (source_segments / "00000_SPEAKER_00.mp3").write_bytes(b"mp3")
        (source_segments / "00001_SPEAKER_01.mp3").write_bytes(b"mp3")
        writer.write_export({"metadata": {}, "segments": segments}, source_segments_dir=source_segments)
        return run_dir

    def test_builds_factual_report_from_run_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            run_dir = self._make_run_full(tmp_root)
            out = tmp_root / "trace_report.html"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "build_trace_html.py"),
                    str(run_dir),
                    "--out",
                    str(out),
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            html = out.read_text(encoding="utf-8")
            self.assertIn("Sommelier True Trace Report", html)
            self.assertIn("01_diarization/diarization.json", html)
            self.assertIn("02_music_clean/segment_flags.json", html)
            self.assertIn("05_export/final/data_audio/00000_SPEAKER_00.mp3", html)
            self.assertIn("&quot;proxy&quot;: false", html)
            self.assertNotIn("có thể", html.lower())
            self.assertNotIn("nghi ", html.lower())
            self.assertNotIn("score", html.lower())

    def test_builds_factual_report_from_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            self._make_run_full(tmp_root)
            zip_path = Path(shutil.make_archive(str(tmp_root / "run_full_download"), "zip", tmp_root, "run_full"))
            out = tmp_root / "from_zip.html"
            extract_dir = tmp_root / "assets"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "build_trace_html.py"),
                    str(zip_path),
                    "--out",
                    str(out),
                    "--extract-dir",
                    str(extract_dir),
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            html = out.read_text(encoding="utf-8")
            self.assertIn("Sommelier True Trace Report", html)
            self.assertTrue((extract_dir / "run_full" / "01_diarization" / "diarization.json").exists())
            self.assertIn("assets/run_full/05_export/final/data_audio/00001_SPEAKER_01.mp3", html)

    def test_rejects_proxy_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            run_dir = self._make_run_full(tmp_root)
            diar_path = run_dir / "01_diarization" / "diarization.json"
            data = json.loads(diar_path.read_text(encoding="utf-8"))
            data["metadata"]["proxy"] = True
            diar_path.write_text(json.dumps(data), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "build_trace_html.py"),
                    str(run_dir),
                    "--out",
                    str(tmp_root / "trace_report.html"),
                ],
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("proxy artifact", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
