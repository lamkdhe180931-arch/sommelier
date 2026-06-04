import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ASRGPUConfigTests(unittest.TestCase):
    def test_notebook_exposes_separate_gpu_controls_for_three_asr_models(self):
        generator = (ROOT / "tools" / "build_kaggle_main_html_notebook.py").read_text(encoding="utf-8")

        self.assertIn("WHISPER_DEVICE_INDEX", generator)
        self.assertIn("PHOWHISPER_DEVICE_INDEX", generator)
        self.assertIn("CTC_DEVICE_INDEX", generator)
        self.assertIn("--whisper_device_index", generator)
        self.assertIn("--phowhisper_device_index", generator)
        self.assertIn("--ctc_device_index", generator)

    def test_main_accepts_separate_gpu_controls_for_three_asr_models(self):
        main_script = (ROOT / "podcast-pipeline" / "main_original_ASR_MoE.py").read_text(encoding="utf-8")

        self.assertIn("--whisper_device_index", main_script)
        self.assertIn("--phowhisper_device_index", main_script)
        self.assertIn("--ctc_device_index", main_script)
        self.assertIn("phowhisper_device", main_script)
        self.assertIn("ctc_device", main_script)


if __name__ == "__main__":
    unittest.main()
