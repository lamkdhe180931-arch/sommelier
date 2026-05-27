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


if __name__ == "__main__":
    unittest.main()
