import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "kaggle_notebooks" / "06_v06_main_html_viewer.ipynb"
GENERATOR_PATH = ROOT / "tools" / "build_kaggle_main_html_notebook.py"
REQUIREMENTS_PATH = ROOT / "podcast-pipeline" / "requirements.txt"


def _notebook_code_cells():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


class KaggleMainHtmlNotebookSetupTests(unittest.TestCase):
    def test_ctranslate2_is_pinned_to_cuda_runtime_safe_version(self):
        requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8")

        self.assertIn("ctranslate2==4.6.3", requirements)
        self.assertNotIn("ctranslate2==4.4.0", requirements)

    def test_notebook_dependency_cell_installs_runtime_libraries(self):
        install_cell = next(cell for cell in _notebook_code_cells() if "05_pip_requirements.log" in cell)

        self.assertIn('"libaio-dev"', install_cell)
        self.assertIn('"chunkformer==1.2.2"', install_cell)
        self.assertIn('"--no-deps"', install_cell)
        self.assertIn('"colorama==0.4.6"', install_cell)
        self.assertIn('"ctranslate2==4.6.3"', install_cell)
        self.assertIn("13b_pip_ctranslate2.log", install_cell)
        self.assertIn("13c_ctranslate2_cuda_smoke.log", install_cell)
        self.assertIn("configure_cuda_library_paths()", install_cell)
        self.assertIn('os.environ["LD_LIBRARY_PATH"]', install_cell)
        self.assertIn('os.environ.pop("LD_PRELOAD", None)', install_cell)
        self.assertIn("ctranslate2.get_supported_compute_types('cuda')", install_cell)
        self.assertIn("from chunkformer import ChunkFormerModel", install_cell)

    def test_notebook_main_pipeline_passes_runtime_environment(self):
        pipeline_cell = next(cell for cell in _notebook_code_cells() if "18_main_pipeline_batch.log" in cell and "main_original_ASR_MoE.py" in cell)

        self.assertIn("ensure_ctranslate2_runtime()", pipeline_cell)
        self.assertIn("17b_ctranslate2_before_main.log", pipeline_cell)
        self.assertIn("ctranslate2.__version__", pipeline_cell)
        self.assertIn("ctranslate2.get_supported_compute_types('cuda')", pipeline_cell)
        self.assertIn("env = os.environ.copy()", pipeline_cell)
        self.assertIn("env=env", pipeline_cell)

    def test_generator_contains_same_kaggle_runtime_fixes(self):
        generator = GENERATOR_PATH.read_text(encoding="utf-8")

        for expected in [
            '"libaio-dev"',
            '"chunkformer==1.2.2"',
            '"--no-deps"',
            '"colorama==0.4.6"',
            '"ctranslate2==4.6.3"',
            "13b_pip_ctranslate2.log",
            "13c_ctranslate2_cuda_smoke.log",
            "configure_cuda_library_paths()",
            'os.environ["LD_LIBRARY_PATH"]',
            'os.environ.pop("LD_PRELOAD", None)',
            "ensure_ctranslate2_runtime()",
            "17b_ctranslate2_before_main.log",
            "ctranslate2.get_supported_compute_types('cuda')",
            "from chunkformer import ChunkFormerModel",
            "env = os.environ.copy()",
            "env=env",
        ]:
            self.assertIn(expected, generator)


if __name__ == "__main__":
    unittest.main()
