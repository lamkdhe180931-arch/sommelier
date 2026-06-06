import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "kaggle_notebooks" / "06_v06_main_html_viewer.ipynb"
GENERATOR_PATH = ROOT / "tools" / "build_kaggle_main_html_notebook.py"


def _notebook_code_cells():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


class KaggleMainHtmlNotebookSetupTests(unittest.TestCase):
    def test_notebook_dependency_cell_installs_runtime_libraries(self):
        install_cell = next(cell for cell in _notebook_code_cells() if "05_pip_requirements.log" in cell)

        self.assertIn('"libaio-dev"', install_cell)
        self.assertIn('"chunkformer==1.2.2"', install_cell)
        self.assertIn('"--no-deps"', install_cell)
        self.assertIn('"colorama==0.4.6"', install_cell)
        self.assertIn('"nvidia-cudnn-cu12==8.9.7.29"', install_cell)
        self.assertIn("libcudnn_ops_infer.so.8", install_cell)
        self.assertIn('os.environ["LD_LIBRARY_PATH"]', install_cell)
        self.assertIn('os.environ["LD_PRELOAD"]', install_cell)
        self.assertIn("ctypes.CDLL", install_cell)
        self.assertIn("17a_cudnn_smoke.log", install_cell)
        self.assertIn("from chunkformer import ChunkFormerModel", install_cell)

    def test_notebook_main_pipeline_passes_runtime_environment(self):
        pipeline_cell = next(cell for cell in _notebook_code_cells() if "18_main_pipeline_batch.log" in cell and "main_original_ASR_MoE.py" in cell)

        self.assertIn("env = os.environ.copy()", pipeline_cell)
        self.assertIn("env=env", pipeline_cell)

    def test_generator_contains_same_kaggle_runtime_fixes(self):
        generator = GENERATOR_PATH.read_text(encoding="utf-8")

        for expected in [
            '"libaio-dev"',
            '"chunkformer==1.2.2"',
            '"--no-deps"',
            '"colorama==0.4.6"',
            '"nvidia-cudnn-cu12==8.9.7.29"',
            "libcudnn_ops_infer.so.8",
            'os.environ["LD_LIBRARY_PATH"]',
            'os.environ["LD_PRELOAD"]',
            "ctypes.CDLL",
            "17a_cudnn_smoke.log",
            "from chunkformer import ChunkFormerModel",
            "env = os.environ.copy()",
            "env=env",
        ]:
            self.assertIn(expected, generator)


if __name__ == "__main__":
    unittest.main()
