import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))


class BuildTraceHtmlTests(unittest.TestCase):
    def _make_run_full(self, root: Path, name: str = "run_full") -> Path:
        from utils.trace_artifacts import TraceRunWriter

        run_dir = root / name
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
                "text_phowhisper": "xin chao pho",
                "text_chunkformer": "xin chao chunk",
                "enhanced_audio": [0.0, 0.2, -0.2, 0.0],
            },
            {
                "index": "00001",
                "start": 1.0,
                "end": 2.0,
                "speaker": "SPEAKER_01",
                "text": "tam biet",
                "text_whisper": "tam biet",
                "text_phowhisper": "tam biet pho",
                "text_chunkformer": "tam biet chunk",
            },
        ]
        audio = {"waveform": [0.0, 0.2, -0.2, 0.0], "sample_rate": 16000}
        writer = TraceRunWriter(run_dir, source_audio_path="input.wav")
        writer.write_diarization(segments, metadata={"audio_duration_seconds": 2.0})
        writer.write_music_clean(segments, [True, False], audio)
        writer.write_overlap(segments, audio)
        writer.write_asr(segments)

        source_segments = root / f"source_segments_{name}"
        source_segments.mkdir()
        (source_segments / "00000_SPEAKER_00.mp3").write_bytes(b"mp3")
        (source_segments / "00001_SPEAKER_01.mp3").write_bytes(b"mp3")
        writer.write_export({"metadata": {}, "segments": segments}, source_segments_dir=source_segments)
        return run_dir

    def test_injects_correct_script_column_into_beautiful_run_viewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            run_dir = self._make_run_full(tmp_root)
            asr_path = run_dir / "04_asr" / "transcript.json"
            asr_data = json.loads(asr_path.read_text(encoding="utf-8"))
            for segment in asr_data["segments"]:
                segment.pop("index", None)
            asr_path.write_text(json.dumps(asr_data), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "build_trace_html.py"),
                    str(run_dir),
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            html = (run_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Sommelier VI Pipeline", html)
            self.assertIn('id="save-annotated"', html)
            self.assertIn("Save annotated HTML", html)
            self.assertIn("<th>Correct Script</th>", html)
            self.assertIn("<th>Note</th>", html)
            self.assertIn("min-width: 1700px", html)
            self.assertIn(".seg-text { min-width: 640px; max-width: 860px;", html)
            self.assertNotIn("<th>Flags</th>", html)
            self.assertIn('class="correct-script"', html)
            self.assertIn('class="note-script"', html)
            self.assertIn("saveAnnotatedHtml", html)
            self.assertIn("updateNote", html)
            self.assertIn("'<td>' + (idx + 1) + '</td>'", html)
            self.assertIn('"text_phowhisper": "xin chao pho"', html)
            self.assertIn('"text_chunkformer": "xin chao chunk"', html)
            self.assertIn('"audio_file": "05_export/final/data_audio/00000_SPEAKER_00.mp3"', html)
            self.assertNotIn("Stage Artifacts", html)
            self.assertNotIn("Factual Checks", html)
            self.assertNotIn("Raw Metadata", html)
            self.assertNotIn("có thể", html.lower())
            self.assertNotIn("nghi ", html.lower())
            self.assertNotIn("score", html.lower())

    def test_builds_even_when_proxy_metadata_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            run_dir = self._make_run_full(tmp_root)
            diar_path = run_dir / "01_diarization" / "diarization.json"
            data = json.loads(diar_path.read_text(encoding="utf-8"))
            data["metadata"]["proxy"] = True
            diar_path.write_text(json.dumps(data), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "build_trace_html.py"),
                    str(run_dir),
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            html = (run_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Sommelier VI Pipeline", html)
            self.assertIn("Correct Script", html)

    def test_builds_html_for_every_run_full_under_parent_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            runs_root = tmp_root / "runs"
            first = self._make_run_full(runs_root, "run_full_001_alpha_intro")
            second = self._make_run_full(runs_root, "run_full_002_beta_outro")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "build_trace_html.py"),
                    str(runs_root),
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("2 run_full", result.stdout)
            for run_dir in [first, second]:
                html = (run_dir / "index.html").read_text(encoding="utf-8")
                self.assertIn(run_dir.name, html)
                self.assertIn("<th>Correct Script</th>", html)
                self.assertIn("<th>Note</th>", html)


if __name__ == "__main__":
    unittest.main()
