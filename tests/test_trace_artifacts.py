import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))


class TraceArtifactTests(unittest.TestCase):
    def test_trace_writer_creates_real_stage_artifacts(self):
        from utils.trace_artifacts import TraceRunWriter

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_full"
            writer = TraceRunWriter(run_dir)

            segments = [
                {
                    "index": "00000",
                    "start": 0.0,
                    "end": 0.5,
                    "speaker": "SPEAKER_00",
                    "text": "xin chao",
                    "enhanced_audio": [0.0, 0.1, -0.1],
                }
            ]
            audio = {"waveform": [0.0, 0.1, -0.1, 0.0], "sample_rate": 16000}

            writer.write_diarization(segments, metadata={"audio_duration_seconds": 0.5})
            writer.write_music_clean(segments, [True], audio)
            writer.write_overlap(segments, audio)
            writer.write_asr(segments)

            diarization = json.loads((run_dir / "01_diarization" / "diarization.json").read_text())
            self.assertEqual(diarization["metadata"]["stage"], "diarization")
            self.assertNotIn("enhanced_audio", diarization["segments"][0])

            music = json.loads((run_dir / "02_music_clean" / "segment_flags.json").read_text())
            self.assertEqual(music["segment_demucs_flags"], [True])
            self.assertTrue((run_dir / "02_music_clean" / "cleaned_audio.wav").exists())

            overlap = json.loads((run_dir / "03_overlap" / "segments.json").read_text())
            enhanced_path = overlap["segments"][0]["enhanced_audio_path"]
            self.assertTrue((run_dir / enhanced_path).exists())
            self.assertNotIn("enhanced_audio", overlap["segments"][0])

            asr = json.loads((run_dir / "04_asr" / "transcript.json").read_text())
            self.assertEqual(asr["segments"][0]["text"], "xin chao")

            source_segments = Path(tmp) / "source_segments"
            source_segments.mkdir()
            (source_segments / "00000_SPEAKER_00.mp3").write_bytes(b"mp3")
            writer.write_export({"metadata": {}, "segments": segments}, source_segments_dir=source_segments)

            final = json.loads((run_dir / "05_export" / "final" / "data_audio.json").read_text())
            self.assertEqual(final["segments"][0]["audio_file"], "data_audio/00000_SPEAKER_00.mp3")
            self.assertTrue((run_dir / "05_export" / "final" / "data_audio" / "00000_SPEAKER_00.mp3").exists())


if __name__ == "__main__":
    unittest.main()
