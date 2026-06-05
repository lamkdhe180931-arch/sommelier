import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class KaggleMainHtmlBatchTests(unittest.TestCase):
    def test_generator_uses_batch_run_dirs_named_after_input_audio(self):
        generator = (ROOT / "tools" / "build_kaggle_main_html_notebook.py").read_text(encoding="utf-8")

        self.assertIn('BATCH_ROOT = "/kaggle/working/sommelier_batch_outputs"', generator)
        self.assertIn('RUNS_ROOT = f"{BATCH_ROOT}/runs"', generator)
        self.assertIn('PIPELINE_INPUT_DIR = f"{BATCH_ROOT}/pipeline_input"', generator)
        self.assertIn("AUDIO_INPUT_PATHS = []", generator)
        self.assertIn('run_name = f"run_full_{index:03d}_{safe_stem(source_path)}"', generator)
        self.assertIn("AUDIO_JOBS.append", generator)
        self.assertNotIn("AUDIO_IN = str(audio_candidates[0])", generator)

    def test_generated_notebook_builds_html_for_all_batch_runs(self):
        notebook = json.loads((ROOT / "kaggle_notebooks" / "06_v06_main_html_viewer.ipynb").read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("/kaggle/working/sommelier_batch_outputs", joined)
        self.assertIn('"--input_folder_path", PIPELINE_INPUT_DIR', joined)
        self.assertIn("for job in AUDIO_JOBS:", joined)
        self.assertIn("tools/build_trace_html.py", joined)
        self.assertIn("RUNS_ROOT", joined)
        self.assertIn("sommelier_batch_outputs.zip", joined)
        self.assertNotIn("tools/inject_beautiful_html.py", joined)
        self.assertNotIn("AUDIO_IN = str(audio_candidates[0])", joined)


if __name__ == "__main__":
    unittest.main()
