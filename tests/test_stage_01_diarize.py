import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
sys.modules.setdefault("pandas", MagicMock())


class Stage01DiarizeTests(unittest.TestCase):
    def test_stage1_defaults_do_not_crop_short_backchannels(self):
        import stage_01_diarize

        args = stage_01_diarize.parse_args(["--input_audio", "/tmp/full.wav"])

        self.assertEqual(args.sortformer_pad_onset, -0.05)
        self.assertEqual(args.sortformer_pad_offset, 0.15)
        self.assertEqual(args.short_backchannel_seconds, 1.0)

    def test_apply_sortformer_backchannel_tuning_sets_model_threshold(self):
        import stage_01_diarize

        model = SimpleNamespace(cfg=SimpleNamespace(soft_label_thres=0.5))

        changed = stage_01_diarize.apply_sortformer_backchannel_tuning(
            model,
            soft_label_threshold=0.15,
            logger=None,
        )

        self.assertTrue(changed)
        self.assertEqual(model.cfg.soft_label_thres, 0.15)


if __name__ == "__main__":
    unittest.main()
