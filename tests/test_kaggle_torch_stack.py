import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class KaggleTorchStackTests(unittest.TestCase):
    def test_notebook_installs_matching_torchvision_for_torch_271(self):
        generator = (ROOT / "tools" / "build_kaggle_main_html_notebook.py").read_text(encoding="utf-8")

        self.assertIn("torch==2.7.1", generator)
        self.assertIn("torchaudio==2.7.1", generator)
        self.assertIn("torchvision==0.22.1", generator)
        self.assertIn("09_pip_torch_stack.log", generator)

    def test_environment_check_imports_torchvision_before_pipeline(self):
        generator = (ROOT / "tools" / "build_kaggle_main_html_notebook.py").read_text(encoding="utf-8")

        self.assertIn("import numpy, numba, torch, torchvision, torchaudio", generator)
        self.assertIn("torchvision.__version__", generator)
        self.assertIn("torchaudio.__version__", generator)


if __name__ == "__main__":
    unittest.main()
