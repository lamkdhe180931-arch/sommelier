import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class KaggleTraceNotebookGPUClampTests(unittest.TestCase):
    def test_pipeline_command_clamps_device_indices_to_visible_gpu_count(self):
        notebook = json.loads((ROOT / "kaggle_trace_run_full.ipynb").read_text(encoding="utf-8"))
        code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        pipeline_cell = next(cell for cell in code_cells if "main_original_ASR_MoE.py" in cell and "--whisper_device_index" in cell)

        self.assertIn("VISIBLE_GPU_COUNT", pipeline_cell)
        self.assertIn("_clamp_device_index", pipeline_cell)
        self.assertIn("EFFECTIVE_WHISPER_DEVICE_INDEX", pipeline_cell)
        self.assertIn('"--whisper_device_index", str(EFFECTIVE_WHISPER_DEVICE_INDEX)', pipeline_cell)
        self.assertIn('"--panns_device_index", str(EFFECTIVE_PANNS_DEVICE_INDEX)', pipeline_cell)
        self.assertNotIn('"--whisper_device_index", str(WHISPER_DEVICE_INDEX)', pipeline_cell)
        self.assertNotIn('"--sepreformer_device_index", str(SEPREFORMER_DEVICE_INDEX)', pipeline_cell)

    def test_notebook_does_not_build_html_outputs(self):
        notebook = json.loads((ROOT / "kaggle_trace_run_full.ipynb").read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertNotIn("tools/build_trace_html.py", joined)
        self.assertNotIn("tools/build_run_viewer.py", joined)
        self.assertNotIn("TRACE_REPORT_HTML", joined)
        self.assertNotIn("viewer.html", joined)
        self.assertNotIn("trace_report.html", joined)

    def test_pipeline_command_uses_pyannote_diarization_31(self):
        notebook = json.loads((ROOT / "kaggle_trace_run_full.ipynb").read_text(encoding="utf-8"))
        code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        pipeline_cell = next(cell for cell in code_cells if "main_original_ASR_MoE.py" in cell and "--whisper_device_index" in cell)

        self.assertIn('"--dia3"', pipeline_cell)

    def test_pipeline_command_falls_back_from_float16_when_gpu_backend_rejects_it(self):
        notebook = json.loads((ROOT / "kaggle_trace_run_full.ipynb").read_text(encoding="utf-8"))
        code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        pipeline_cell = next(cell for cell in code_cells if "main_original_ASR_MoE.py" in cell and "--compute_type" in cell)

        self.assertIn("EFFECTIVE_COMPUTE_TYPE", pipeline_cell)
        self.assertIn("get_device_capability", pipeline_cell)
        self.assertIn('"--compute_type", EFFECTIVE_COMPUTE_TYPE', pipeline_cell)
        self.assertNotIn('"--compute_type", COMPUTE_TYPE', pipeline_cell)

    def test_notebook_installs_and_import_checks_chunkformer(self):
        notebook = json.loads((ROOT / "kaggle_trace_run_full.ipynb").read_text(encoding="utf-8"))
        code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        install_cell = next(cell for cell in code_cells if "13_pip_chunkformer.log" in cell)
        check_cell = next(cell for cell in code_cells if "chunkformer import ok" in cell)

        self.assertIn('"chunkformer==1.2.2"', install_cell)
        self.assertIn('"colorama==0.4.6"', install_cell)
        self.assertIn('"--no-deps"', install_cell)
        self.assertIn("from chunkformer import ChunkFormerModel", check_cell)

    def test_zip_cell_requires_real_stage_outputs_before_download(self):
        notebook = json.loads((ROOT / "kaggle_trace_run_full.ipynb").read_text(encoding="utf-8"))
        code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        zip_cell = next(cell for cell in code_cells if "run_full_download" in cell)

        for expected in [
            '"01_diarization" / "diarization.json"',
            '"02_music_clean" / "segment_flags.json"',
            '"02_music_clean" / "cleaned_audio.wav"',
            '"03_overlap" / "segments.json"',
            '"04_asr" / "transcript.json"',
            "Path(FINAL_DIR) / \"data_audio.json\"",
        ]:
            self.assertIn(expected, zip_cell)
        self.assertIn("raise FileNotFoundError", zip_cell)

    def test_notebook_controls_vietnamese_asr_models_explicitly(self):
        notebook = json.loads((ROOT / "kaggle_trace_run_full.ipynb").read_text(encoding="utf-8"))
        code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        controls_cell = next(cell for cell in code_cells if "WHISPER_ARCH" in cell and "COMPUTE_TYPE" in cell)
        pipeline_cell = next(cell for cell in code_cells if "main_original_ASR_MoE.py" in cell and "--whisper_arch" in cell)

        self.assertIn('ASR_LANGUAGE = "vi"', controls_cell)
        self.assertIn('PHOWHISPER_MODEL_NAME = "vinai/PhoWhisper-large"', controls_cell)
        self.assertIn('CTC_MODEL_NAME = "khanhld/chunkformer-ctc-large-vie"', controls_cell)
        self.assertIn('"--asr_language", ASR_LANGUAGE', pipeline_cell)
        self.assertIn('"--phowhisper_model_name", PHOWHISPER_MODEL_NAME', pipeline_cell)
        self.assertIn('"--ctc_model_name", CTC_MODEL_NAME', pipeline_cell)


if __name__ == "__main__":
    unittest.main()
