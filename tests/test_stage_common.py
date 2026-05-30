import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))


class FakeNumpyScalar:
    def item(self):
        return 3.5


class FakeNumpyArray:
    def tolist(self):
        return [1, 2, 3]


class StageCommonTests(unittest.TestCase):
    def test_dump_json_converts_numpy_like_values(self):
        import stage_common

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "nested" / "data.json"
            stage_common.dump_json(
                {
                    "scalar": FakeNumpyScalar(),
                    "array": FakeNumpyArray(),
                    "path": Path("/tmp/audio.wav"),
                },
                out_path,
            )

            data = json.loads(out_path.read_text())

        self.assertEqual(data["scalar"], 3.5)
        self.assertEqual(data["array"], [1, 2, 3])
        self.assertEqual(data["path"], "/tmp/audio.wav")

    def test_clean_segments_for_json_removes_audio_payloads(self):
        import stage_common

        cleaned = stage_common.clean_segments_for_json(
            [
                {
                    "index": "00000",
                    "start": 0.0,
                    "end": 1.0,
                    "enhanced_audio": FakeNumpyArray(),
                    "_private": "skip",
                    "speaker": "SPEAKER_00",
                }
            ]
        )

        self.assertEqual(cleaned, [{"index": "00000", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}])

    def test_default_stage_dir_uses_audio_stem(self):
        import stage_common

        out_dir = stage_common.default_stage_dir("/data/podcast/sample.wav")

        self.assertEqual(out_dir, Path("/data/podcast/_staged/sample"))

    def test_postprocess_merges_only_same_speaker_micro_gap_and_marks_backchannel(self):
        import stage_common

        segments = [
            {"index": "00000", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"index": "00001", "start": 1.2, "end": 2.0, "speaker": "SPEAKER_00"},
            {"index": "00002", "start": 2.1, "end": 2.45, "speaker": "SPEAKER_01"},
            {"index": "00003", "start": 2.55, "end": 3.0, "speaker": "SPEAKER_00"},
        ]

        postprocessed, stats = stage_common.postprocess_diarization_segments(
            segments,
            same_speaker_merge_gap=0.3,
            short_backchannel_seconds=1.0,
        )

        self.assertEqual(len(postprocessed), 3)
        self.assertEqual(postprocessed[0]["speaker"], "SPEAKER_00")
        self.assertEqual(postprocessed[0]["start"], 0.0)
        self.assertEqual(postprocessed[0]["end"], 2.0)
        self.assertFalse(postprocessed[0]["is_short_backchannel"])
        self.assertTrue(postprocessed[1]["is_short_backchannel"])
        self.assertEqual(postprocessed[1]["train_quality_label"], "short_backchannel_review")
        self.assertEqual(postprocessed[2]["speaker"], "SPEAKER_00")
        self.assertEqual(stats["merged_same_speaker_gap_count"], 1)
        self.assertEqual(stats["short_backchannel_count"], 2)


if __name__ == "__main__":
    unittest.main()
