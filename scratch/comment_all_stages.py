import json
from pathlib import Path

# Xác định đường dẫn gốc
base_dir = Path("/Users/lam/Library/CloudStorage/GoogleDrive-he180931@gmail.com/Drive của tôi/AI thực chiến/tts/sommelier")
if not base_dir.exists():
    base_dir = Path("/Users/lam/Library/CloudStorage/GoogleDrive-he180931@gmail.com/Drive của tôi/AI thực chiến/tts/sommelier")

input_nb = base_dir / "kaggle_notebooks/04_v04_full_mossformer_phowhisper_local_speaker_safe.ipynb"
output_nb = base_dir / "kaggle_notebooks/kaggle incremental/01_stage_01_diarize.ipynb"

print("Reading original notebook:", input_nb)
with open(input_nb, 'r', encoding='utf-8') as f:
    data = json.load(f)

unrelated_configs = [
    "RUN_DEMUCS",
    "RUN_OVERLAP_MARK_ONLY",
    "RUN_MOSSFORMER_SEPARATION",
    "MOSSFORMER_MODEL_NAME",
    "MOSSFORMER_TASK",
    "MOSSFORMER_FAIL_OPEN",
    "MOSSFORMER_MIN_OVERLAP_SECONDS",
    "MOSSFORMER_MIN_SEGMENT_SECONDS",
    "MOSSFORMER_CONTEXT_SECONDS",
    "MOSSFORMER_MAX_WINDOW_SECONDS",
    "MOSSFORMER_SAVE_WINDOW_STEMS",
    "MOSSFORMER_USE_SPEAKER_EMBEDDING_ASSIGNMENT",
    "MOSSFORMER_EMBEDDING_DEVICE",
    "MOSSFORMER_EMBEDDING_FAIL_OPEN",
    "MOSSFORMER_REFERENCE_MIN_SECONDS",
    "MOSSFORMER_REFERENCE_MAX_SEGMENTS_PER_SPEAKER",
    "MOSSFORMER_ASSIGNMENT_MIN_CONFIDENCE",
    "MOSSFORMER_ASSIGNMENT_MIN_MARGIN",
    "MOSSFORMER_REQUIRE_EMBEDDING_ASSIGNMENT",
    "MOSSFORMER_DISABLE_LOW_CONFIDENCE_ASR",
    "OVERLAP_MARK_THRESHOLD_SECONDS",
    "OVERLAP_REVIEW_THRESHOLD_SECONDS",
    "OVERLAP_REQUIRE_DIFFERENT_SPEAKER",
    "RUN_PYANNOTE_OSD",
    "PYANNOTE_OSD_MODEL",
    "PYANNOTE_OSD_MIN_DURATION_SECONDS",
    "PYANNOTE_OSD_FAIL_OPEN",
    "MIN_SEPREFORMER_OVERLAP_SECONDS",
    "MIN_SEPREFORMER_SEGMENT_SECONDS",
    "ASR_QUALITY_GUARD",
    "ASR_MICRO_SEGMENT_SECONDS",
    "ASR_SHORT_SEGMENT_SECONDS",
    "ASR_VI_AGREEMENT_THRESHOLD",
    "ASR_CONTEXT_PAD_BEFORE_SECONDS",
    "ASR_CONTEXT_PAD_AFTER_SECONDS",
    "ASR_MOE",
    "WHISPER_DEVICE_INDEX",
    "VI_ASR_DEVICE_INDEX",
    "WHISPER_ARCH",
    "COMPUTE_TYPE",
    "ASR_THREADS",
    "USE_WHISPER_INITIAL_PROMPT",
    "WHISPER_INITIAL_PROMPT",
    "WHISPER_HOTWORDS",
    "PHOWHISPER_USE_HF_API",
    "PHOWHISPER_API_MODEL",
    "PHOWHISPER_API_PROVIDER",
    "PHOWHISPER_API_TIMEOUT_SECONDS"
]

is_past_stage_1 = False

for idx, cell in enumerate(data['cells']):
    source = cell.get('source', [])
    if isinstance(source, str):
        source = source.splitlines(keepends=True)
    source_str = "".join(source)
    
    # Phát hiện khi bắt đầu chuyển sang Stage 2 (từ Phần 9 trở đi)
    if "## 9. Stage 02" in source_str:
        is_past_stage_1 = True
        print(f"Cell index {idx}: Detected transition to Stage 2.")
        
    if cell['cell_type'] == 'code':
        # A. Nếu nằm từ Stage 2 trở đi -> comment out toàn bộ code trong cell
        if is_past_stage_1:
            new_source = []
            for line in source:
                stripped = line.strip()
                if stripped:
                    new_source.append("# " + line)
                else:
                    new_source.append("#\n")
            cell['source'] = new_source
            continue
            
        # B. Cell cấu hình Phần 0
        if "REPO_URL =" in source_str:
            new_source = []
            in_multiline_prompt = False
            for line in source:
                stripped = line.strip()
                if "WHISPER_INITIAL_PROMPT = (" in line:
                    in_multiline_prompt = True
                    new_source.append("# " + line)
                    continue
                if in_multiline_prompt:
                    new_source.append("# " + line)
                    if ")" in line:
                        in_multiline_prompt = False
                    continue
                
                is_unrelated = False
                for cfg in unrelated_configs:
                    if stripped.startswith(cfg + " =") or stripped.startswith(cfg + "="):
                        is_unrelated = True
                        break
                
                if is_unrelated:
                    new_source.append("# " + line)
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("Commented configs in Part 0.")
            continue
            
        # C. Cell dependencies Phần 2
        if "requirements-kaggle.txt" in source_str:
            new_source = []
            for line in source:
                stripped = line.strip()
                # Comment out clearvoice, pillow, torchmetrics, numpy/numba
                if "clearvoice==0.1.2" in line or "pillow<12.0" in line or "torchmetrics==1.7.4" in line or "numpy==2.2.6" in line or "numba==0.61.2" in line:
                    new_source.append("# " + line.lstrip())
                elif "13_pip_numpy_numba.log" in line:
                    new_source.append("# " + line.lstrip())
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("Commented dependencies in Part 2.")
            continue
            
        # D. Cell kiểm tra môi trường Phần 3
        if "importlib_metadata" in source_str:
            new_source = []
            for line in source:
                stripped = line.strip()
                if "chunkformer" in line or "whisperx" in line:
                    new_source.append("# " + line.lstrip())
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("Commented environment checks in Part 3.")
            continue

        # E. Các cell chuẩn bị Phần 6 (tải PANNs và cấu hình MossFormer)
        if "thelou1s/panns-inference" in source_str:
            new_source = []
            for line in source:
                stripped = line.strip()
                if stripped:
                    new_source.append("# " + line)
                else:
                    new_source.append("#\n")
            cell['source'] = new_source
            print("Commented PANNs cell.")
            continue
            
        if "Stage 03 mode: MossFormer2" in source_str:
            new_source = []
            for line in source:
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from ") or "os.chdir(" in line or not stripped:
                    new_source.append(line)
                else:
                    new_source.append("# " + line)
            cell['source'] = new_source
            print("Commented MossFormer setup cell.")
            continue

# 4. Thêm các cell đóng gói và tải về ở cuối cùng
cell_markdown = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "## 15. Đóng gói và tải dữ liệu Stage 1 về máy\n",
  "\n",
  "Chạy cell dưới đây để nén thư mục kết quả diarization và logs thành file `.zip`, sau đó hiển thị link tải trực tiếp từ Kaggle."
 ]
}

cell_code = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {
  "trusted": True
 },
 "outputs": [],
 "source": [
  "import os\n",
  "import shutil\n",
  "from IPython.display import FileLink, display\n",
  "\n",
  "# Tạo file zip chứa thư mục diarization kết quả và logs\n",
  "output_zip_path = \"/kaggle/working/diarization_stage_01_results\"\n",
  "source_dir = \"/kaggle/working/run_full/01_diarization\"\n",
  "log_dir = \"/kaggle/working/run_full/logs\"\n",
  "\n",
  "# Tạo thư mục tạm để gom cả kết quả và logs trước khi nén\n",
  "temp_export_dir = \"/kaggle/working/temp_stage_01_export\"\n",
  "os.makedirs(temp_export_dir, exist_ok=True)\n",
  "\n",
  "if os.path.exists(source_dir):\n",
  "    shutil.copytree(source_dir, f\"{temp_export_dir}/01_diarization\", dirs_exist_ok=True)\n",
  "if os.path.exists(log_dir):\n",
  "    shutil.copytree(log_dir, f\"{temp_export_dir}/logs\", dirs_exist_ok=True)\n",
  "\n",
  "shutil.make_archive(output_zip_path, 'zip', temp_export_dir)\n",
  "shutil.rmtree(temp_export_dir, ignore_errors=True)\n",
  "\n",
  "zip_file = f\"{output_zip_path}.zip\"\n",
  "print(f\"\\nĐóng gói thành công! File zip lưu tại: {zip_file}\")\n",
  "print(\"Click vào đường link bên dưới để tải trực tiếp về máy:\")\n",
  "display(FileLink(zip_file))\n"
 ]
}

data['cells'].append(cell_markdown)
data['cells'].append(cell_code)

# Ghi đè vào file notebook
with open(output_nb, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)

print("Saved output notebook:", output_nb)
