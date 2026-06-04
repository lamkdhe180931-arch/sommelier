from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT_PATH = Path("kaggle_notebooks/06_v06_main_html_viewer.ipynb")


def _source(text: str) -> str:
    return textwrap.dedent(text).strip("\n") + "\n"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": _source(text),
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        md(
            """
            # Sommelier Main Branch Kaggle HTML Viewer

            Notebook này chạy nhánh mới `codex/kaggle-main-html-viewer` trên Kaggle, sau đó tạo đủ artifact `run_full` để mở bằng HTML viewer giống nhánh `test-divide-stage`.

            Điều kiện trước khi chạy:
            - Kaggle Internet: bật.
            - Kaggle Accelerator: nên bật GPU.
            - Kaggle Secret có `HF_TOKEN`.
            - Add dataset audio vào notebook. Notebook tự chọn file audio đầu tiên trong `/kaggle/input`.

            Output chính:
            - `/kaggle/working/run_full/index.html`: viewer template `index.html` được inject data giống `tools/inject_beautiful_html.py`.
            - `/kaggle/working/run_viewer/index.html`: viewer tổng hợp từ `tools/build_run_viewer.py`.
            - `/kaggle/working/sommelier_main_html_outputs.zip`: zip đủ `run_full` + `run_viewer` để tải về và mở HTML local.
            """
        ),
        md(
            """
            ## 0. Cấu hình run

            Chỉnh các biến toàn cục này trước khi chạy. Notebook sẽ dùng các biến này để clone branch, chọn audio, chọn GPU hiển thị cho pipeline, bật/tắt model nặng và truyền tham số vào `main_original_ASR_MoE.py`.
            """
        ),
        code(
            """
            # =========================
            # 0. Repo/output controls
            # =========================
            REPO_URL = "https://github.com/lamkdhe180931-arch/sommelier.git"
            BRANCH = "codex/kaggle-main-html-viewer"

            RUN_DIR = "/kaggle/working/run_full"
            INPUT_DIR = f"{RUN_DIR}/00_input"
            DIAR_DIR = f"{RUN_DIR}/01_diarization"
            MUSIC_DIR = f"{RUN_DIR}/02_music_clean"
            OVERLAP_DIR = f"{RUN_DIR}/03_overlap"
            ASR_DIR = f"{RUN_DIR}/04_asr"
            EXPORT_DIR = f"{RUN_DIR}/05_export"
            FINAL_DIR = f"{EXPORT_DIR}/final"
            EVAL_DIR = f"{RUN_DIR}/06_eval"
            PREVIEW_DIR = f"{RUN_DIR}/preview"
            LOG_DIR_PATH = f"{RUN_DIR}/logs"
            AUDIO_WAV = f"{INPUT_DIR}/full.wav"

            # =========================
            # 1. Audio input controls
            # =========================
            # Để "" nếu muốn notebook tự tìm audio đầu tiên trong /kaggle/input.
            # Nếu muốn chỉ định rõ file, ví dụ:
            # AUDIO_INPUT_PATH = "/kaggle/input/my-dataset/audio.mp3"
            AUDIO_INPUT_PATH = ""
            AUDIO_INPUT_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus")

            # None để chạy full audio. Ví dụ 300 để test nhanh 5 phút.
            AUDIO_LIMIT_SECONDS = 300

            # =========================
            # 2. GPU/runtime controls
            # =========================
            # Kaggle 2x T4: giữ cả 2 GPU visible rồi chọn GPU cho từng model bằng *_DEVICE_INDEX.
            # DEVICE_INDEX là index trong danh sách visible của Kaggle: 0 hoặc 1. Dùng -1 để ép CPU.
            CUDA_VISIBLE_DEVICES = "0,1"
            PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
            PRINT_NVIDIA_SMI = True
            REQUIRE_GPU = True

            # Gợi ý mặc định cho 2x T4:
            # - GPU 0: diarization + Sortformer + Whisper large-v3
            # - GPU 1: Demucs + PANNs + SepReformer + PhoWhisper/ChunkFormer CTC nếu bật ASR-MoE
            DIAR_DEVICE_INDEX = 0
            SORTFORMER_DEVICE_INDEX = 0
            WHISPER_DEVICE_INDEX = 0
            ASR_MOE_DEVICE_INDEX = 1
            PANNS_DEVICE_INDEX = 1
            DEMUCS_DEVICE_INDEX = 1
            SEPREFORMER_DEVICE_INDEX = 1

            # =========================
            # 3. Install/model switches
            # =========================
            INSTALL_DEPENDENCIES = True
            RUN_DEMUCS = True
            RUN_SEPREFORMER = True
            ASR_MOE = True
            WHISPERX_WORD_TIMESTAMPS = False
            QWEN3OMNI = False

            # =========================
            # 4. Main pipeline params
            # =========================
            LLM_CASE = "case_2"
            WHISPER_ARCH = "large-v3"
            ASR_LANGUAGE = "vi"
            PHOWHISPER_MODEL_NAME = "vinai/PhoWhisper-large"
            CTC_MODEL_NAME = "khanhld/chunkformer-ctc-large-vie"
            COMPUTE_TYPE = "float16"
            ASR_THREADS = 4
            BATCH_SIZE = 64
            INIT_PROMPT = True
            DIA3 = False
            KOREAN_G2P = False

            # Diarization / clustering.
            MERGE_GAP = 2.0
            SPEAKER_LINK_THRESHOLD = 0.75
            DIAR_SEGMENTATION_THRESHOLD = 0.15
            DIAR_MIN_CLUSTER_SIZE = 10
            DIAR_CLUSTER_THRESHOLD = 0.5

            # Overlap / Sortformer boundary tuning.
            OVERLAP_THRESHOLD = 1.0
            SORTFORMER_PARAM = True
            SORTFORMER_PAD_ONSET = 0.05
            SORTFORMER_PAD_OFFSET = 0.05

            # ASR ensemble quality guard.
            ASR_QUALITY_GUARD = True
            ASR_MICRO_SEGMENT_SECONDS = 0.5
            ASR_SHORT_SEGMENT_SECONDS = 1.0
            ASR_VI_AGREEMENT_THRESHOLD = 0.75
            ASR_CONTEXT_PAD_BEFORE = 0.25
            ASR_CONTEXT_PAD_AFTER = 0.35

            # Opus/Ogg pre-decode controls.
            OPUS_DECODE_WORKERS = 8
            FFMPEG_THREADS_PER_DECODE = 1

            # =========================
            # 5. Secrets
            # =========================
            HF_SECRET_NAME = "HF_TOKEN"
            """
        ),
        code(
            """
            from pathlib import Path
            import datetime
            import os
            import shlex
            import subprocess
            import time

            LOG_DIR = Path(LOG_DIR_PATH)
            for _dir in [
                INPUT_DIR, DIAR_DIR, MUSIC_DIR, OVERLAP_DIR, ASR_DIR,
                EXPORT_DIR, FINAL_DIR, EVAL_DIR, PREVIEW_DIR, LOG_DIR_PATH,
            ]:
                Path(_dir).mkdir(parents=True, exist_ok=True)

            if CUDA_VISIBLE_DEVICES is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
            if PYTORCH_CUDA_ALLOC_CONF:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(PYTORCH_CUDA_ALLOC_CONF)

            print("Runtime GPU controls:")
            print("  CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
            print("  PYTORCH_CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<not set>"))
            DEVICE_ASSIGNMENTS = {
                "diar/vad/speaker-link": DIAR_DEVICE_INDEX,
                "sortformer": SORTFORMER_DEVICE_INDEX,
                "whisper": WHISPER_DEVICE_INDEX,
                "phowhisper/chunkformer_ctc": ASR_MOE_DEVICE_INDEX,
                "panns": PANNS_DEVICE_INDEX,
                "demucs": DEMUCS_DEVICE_INDEX,
                "sepreformer": SEPREFORMER_DEVICE_INDEX,
            }
            print("  GPU assignment (-1 = CPU, otherwise visible CUDA index):")
            for name, idx in DEVICE_ASSIGNMENTS.items():
                print(f"    - {name}: {idx}")
            print("Feature switches:")
            print("  INSTALL_DEPENDENCIES =", INSTALL_DEPENDENCIES)
            print("  RUN_DEMUCS =", RUN_DEMUCS)
            print("  RUN_SEPREFORMER =", RUN_SEPREFORMER)
            print("  ASR_MOE =", ASR_MOE)
            print("  WHISPERX_WORD_TIMESTAMPS =", WHISPERX_WORD_TIMESTAMPS)
            print("  QWEN3OMNI =", QWEN3OMNI)
            print("Main pipeline params:")
            print("  AUDIO_INPUT_PATH =", AUDIO_INPUT_PATH or "<auto /kaggle/input>")
            print("  AUDIO_LIMIT_SECONDS =", AUDIO_LIMIT_SECONDS)
            print("  WHISPER_ARCH =", WHISPER_ARCH)
            print("  ASR_LANGUAGE =", ASR_LANGUAGE)
            print("  PHOWHISPER_MODEL_NAME =", PHOWHISPER_MODEL_NAME)
            print("  CTC_MODEL_NAME =", CTC_MODEL_NAME)
            print("  COMPUTE_TYPE =", COMPUTE_TYPE)
            print("  BATCH_SIZE =", BATCH_SIZE)
            print("  ASR_THREADS =", ASR_THREADS)
            print("  MERGE_GAP =", MERGE_GAP)
            print("  SPEAKER_LINK_THRESHOLD =", SPEAKER_LINK_THRESHOLD)
            print("  ASR_QUALITY_GUARD =", ASR_QUALITY_GUARD)

            AUDIO_TIMING = {}

            def fmt_duration(seconds):
                seconds = float(seconds or 0.0)
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                if hours:
                    return f"{hours}h {minutes:02d}m {secs:05.2f}s"
                if minutes:
                    return f"{minutes}m {secs:05.2f}s"
                return f"{secs:.2f}s"

            def now_label():
                return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            def _format_cmd(cmd):
                if isinstance(cmd, (list, tuple)):
                    return " ".join(shlex.quote(str(part)) for part in cmd)
                return str(cmd)

            def tail_file(path, n=30):
                path = Path(path)
                if not path.exists():
                    return ""
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                return "\\n".join(lines[-n:])

            def run_logged(cmd, log_name, cwd=None, env=None, shell=False, tail=20):
                log_path = LOG_DIR / log_name
                cwd = cwd or os.getcwd()
                print("Running:", _format_cmd(cmd))
                print("Log:", log_path)
                with open(log_path, "w", encoding="utf-8", errors="replace") as log:
                    proc = subprocess.run(
                        cmd,
                        cwd=cwd,
                        env=env,
                        shell=shell,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                print("Exit code:", proc.returncode)
                if tail:
                    log_tail = tail_file(log_path, n=tail)
                    if log_tail:
                        print(f"--- last {tail} log lines ---")
                        print(log_tail)
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, cmd)
                return log_path

            def copy_log(src_name, dst_name, header):
                src = LOG_DIR / src_name
                dst = LOG_DIR / dst_name
                body = tail_file(src, n=400) if src.exists() else ""
                dst.write_text(header + "\\n\\n" + body + "\\n", encoding="utf-8")
                return dst
            """
        ),
        md("## 1. Clone nhánh mới"),
        code(
            """
            import os
            import shutil
            import subprocess
            from pathlib import Path

            os.chdir("/kaggle/working")
            repo_dir = Path("/kaggle/working/sommelier")
            if repo_dir.exists():
                shutil.rmtree(repo_dir)

            run_logged(["git", "clone", "-b", BRANCH, REPO_URL, str(repo_dir)], "01_clone_repo.log", cwd="/kaggle/working", tail=30)
            os.chdir(repo_dir / "podcast-pipeline")
            print("cwd:", os.getcwd())
            print("branch:", subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip())
            print("commit:", subprocess.check_output(["git", "log", "-1", "--oneline"], text=True).strip())
            """
        ),
        md("## 2. Cài dependencies"),
        code(
            """
            import os
            from pathlib import Path

            os.chdir("/kaggle/working/sommelier/podcast-pipeline")

            if INSTALL_DEPENDENCIES:
                run_logged(["apt-get", "update", "-y"], "02_apt_update.log", tail=10)
                run_logged(["apt-get", "install", "-y", "ffmpeg", "git", "git-lfs"], "03_apt_install.log", tail=10)
                run_logged(["python", "-m", "pip", "install", "-U", "pip", "setuptools", "wheel", "packaging", "ninja"], "04_pip_base.log", tail=12)

                req = Path("requirements.txt").read_text(encoding="utf-8")
                filtered = [line for line in req.splitlines() if "nemo-toolkit[all]" not in line]
                Path("requirements-kaggle.txt").write_text("\\n".join(filtered) + "\\n", encoding="utf-8")
                run_logged(["python", "-m", "pip", "install", "-r", "requirements-kaggle.txt"], "05_pip_requirements.log", tail=20)

                run_logged(["python", "-m", "pip", "uninstall", "-y", "nemo-toolkit", "lightning", "pytorch-lightning"], "06_pip_uninstall_nemo.log", tail=8)
                run_logged(["python", "-m", "pip", "install", "lightning==2.4.0", "pytorch-lightning==2.5.2"], "07_pip_lightning.log", tail=12)
                run_logged(["python", "-m", "pip", "install", "nemo-toolkit[asr]==2.4.0"], "08_pip_nemo_asr.log", tail=20)

                run_logged(["python", "-m", "pip", "install", "pillow<12.0"], "09_pip_pillow.log", tail=8)
                run_logged(["python", "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", "--no-deps", "torchmetrics==1.7.4"], "10_pip_torchmetrics.log", tail=8)
                run_logged([
                    "python", "-m", "pip", "install", "--no-cache-dir", "--force-reinstall",
                    "numpy==2.2.6", "numba==0.61.2", "llvmlite==0.44.0",
                ], "11_pip_numpy_numba.log", tail=12)
            else:
                print("INSTALL_DEPENDENCIES=False, bỏ qua cài dependencies.")
            """
        ),
        md("## 3. Kiểm tra môi trường"),
        code(
            """
            import importlib.metadata as importlib_metadata
            import numpy, numba, torch

            for pkg in ["nemo-toolkit", "chunkformer", "transformers", "numpy", "numba", "torch"]:
                try:
                    version = importlib_metadata.version(pkg) if pkg != "numpy" else numpy.__version__
                    print(pkg + ":", version)
                except Exception as exc:
                    print(pkg + ":", "missing", exc)

            print("CUDA:", torch.cuda.is_available())
            visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print("Visible CUDA device count:", visible_gpu_count)
            if torch.cuda.is_available():
                for i in range(visible_gpu_count):
                    print(f"GPU {i}:", torch.cuda.get_device_name(i))
                invalid_assignments = {
                    name: idx for name, idx in DEVICE_ASSIGNMENTS.items()
                    if idx is not None and idx >= visible_gpu_count
                }
                if invalid_assignments:
                    raise ValueError(
                        "GPU index không hợp lệ. Kaggle chỉ thấy "
                        f"{visible_gpu_count} GPU: {invalid_assignments}"
                    )
            if REQUIRE_GPU and not torch.cuda.is_available():
                raise RuntimeError("REQUIRE_GPU=True nhưng torch.cuda.is_available() = False. Hãy bật Kaggle GPU hoặc đặt REQUIRE_GPU=False.")
            if PRINT_NVIDIA_SMI:
                subprocess.run(["nvidia-smi"], check=False)
            """
        ),
        md("## 4. Gắn Hugging Face token vào config"),
        code(
            """
            import json
            from kaggle_secrets import UserSecretsClient
            from huggingface_hub import whoami

            token = UserSecretsClient().get_secret(HF_SECRET_NAME)
            print("HF token:", token[:8] + "..." if token else "missing")
            print(whoami(token=token))

            cfg_path = Path("/kaggle/working/sommelier/podcast-pipeline/config.json")
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg["huggingface_token"] = token
            cfg.setdefault("entrypoint", {})["input_folder_path"] = INPUT_DIR
            cfg["entrypoint"]["SAMPLE_RATE"] = 16000
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            print("config.json updated:", cfg_path)
            """
        ),
        md("## 5. Tìm audio input và chuẩn hóa về mono 16 kHz"),
        code(
            """
            from pathlib import Path

            AUDIO_TIMING["audio_job_start_perf"] = time.perf_counter()
            AUDIO_TIMING["audio_job_start_clock"] = now_label()
            AUDIO_TIMING["audio_job_start_label"] = "start: tìm audio input + chuẩn hóa về 16 kHz"
            print("Audio timing start:", AUDIO_TIMING["audio_job_start_clock"])
            print("Measured from:", AUDIO_TIMING["audio_job_start_label"])

            if AUDIO_INPUT_PATH:
                audio_path = Path(AUDIO_INPUT_PATH)
                if not audio_path.exists():
                    raise FileNotFoundError(f"AUDIO_INPUT_PATH không tồn tại: {audio_path}")
                if audio_path.suffix.lower() not in AUDIO_INPUT_EXTENSIONS:
                    raise ValueError(f"AUDIO_INPUT_PATH không phải định dạng audio được hỗ trợ: {audio_path}")
                AUDIO_IN = str(audio_path)
            else:
                audio_candidates = sorted(
                    p for p in Path("/kaggle/input").rglob("*")
                    if p.is_file() and p.suffix.lower() in AUDIO_INPUT_EXTENSIONS
                )
                if not audio_candidates:
                    raise FileNotFoundError("Không tìm thấy audio trong /kaggle/input. Hãy Add Input hoặc Upload audio trước.")
                AUDIO_IN = str(audio_candidates[0])

            print("AUDIO_IN:", AUDIO_IN)
            print("AUDIO_WAV:", AUDIO_WAV)
            print("RUN_DIR:", RUN_DIR)

            cmd = ["ffmpeg", "-hide_banner", "-y", "-i", AUDIO_IN]
            if AUDIO_LIMIT_SECONDS:
                cmd += ["-t", str(AUDIO_LIMIT_SECONDS)]
            cmd += ["-ac", "1", "-ar", "16000", AUDIO_WAV]
            run_logged(cmd, "00_prepare_audio_ffmpeg.log", cwd="/kaggle/working", tail=20)
            AUDIO_TIMING["audio_prepare_done_perf"] = time.perf_counter()
            AUDIO_TIMING["audio_prepare_seconds"] = (
                AUDIO_TIMING["audio_prepare_done_perf"] - AUDIO_TIMING["audio_job_start_perf"]
            )
            print("Audio prepare runtime:", fmt_duration(AUDIO_TIMING["audio_prepare_seconds"]))
            """
        ),
        code(
            """
            from pydub import AudioSegment
            from IPython.display import Audio, display

            audio = AudioSegment.from_file(AUDIO_WAV)
            AUDIO_DURATION_SECONDS = len(audio) / 1000
            AUDIO_TIMING["audio_duration_seconds"] = AUDIO_DURATION_SECONDS
            print("Audio:", AUDIO_WAV)
            print("Duration seconds:", AUDIO_DURATION_SECONDS)
            print(f"Audio timeline: 0.00s -> {AUDIO_DURATION_SECONDS:.2f}s ({fmt_duration(AUDIO_DURATION_SECONDS)})")
            print("Frame rate:", audio.frame_rate)
            print("Channels:", audio.channels)

            preview_path = Path(INPUT_DIR) / "preview_input_30s.wav"
            audio[:30000].export(preview_path, format="wav")
            print("Preview first 30s:", preview_path)
            display(Audio(str(preview_path)))
            """
        ),
        md("## 6. Tải model phụ nếu bật Demucs/SepReformer"),
        code(
            """
            from huggingface_hub import hf_hub_download

            if RUN_DEMUCS:
                panns_path = hf_hub_download(
                    repo_id="thelou1s/panns-inference",
                    filename="Cnn14_mAP=0.431.pth",
                    local_dir="/kaggle/working/sommelier/panns_data",
                )
                print("PANNs checkpoint:", panns_path)
            else:
                print("RUN_DEMUCS=False, bỏ qua tải PANNs")
            """
        ),
        code(
            """
            if RUN_SEPREFORMER:
                run_logged(["git", "lfs", "install"], "12_git_lfs_install.log", cwd="/kaggle/working/sommelier", tail=10)
                os.chdir("/kaggle/working/sommelier")
                if not Path("SepReformer").exists():
                    run_logged(["git", "clone", "https://github.com/dmlguq456/SepReformer.git", "SepReformer"], "13_clone_sepreformer.log", cwd="/kaggle/working/sommelier", tail=20)
                run_logged(["git", "lfs", "pull"], "14_sepreformer_lfs_pull.log", cwd="/kaggle/working/sommelier/SepReformer", tail=20)
                run_logged([
                    "python", "-m", "pip", "install", "--no-deps",
                    "mir-eval==0.7", "ptflops==0.7.4", "thop==0.1.1.post2209072238", "torchinfo==1.8.0",
                ], "15_sepreformer_extra_deps.log", cwd="/kaggle/working/sommelier/SepReformer", tail=12)

                log = Path("/kaggle/working/sommelier/SepReformer/models/SepReformer_Base_WSJ0/log")
                src = log / "scratch_weight"
                dst = log / "scratch_weights"
                if src.exists() and not dst.exists():
                    os.symlink(src, dst)

                ckpts = list(log.rglob("*.pt")) + list(log.rglob("*.pth"))
                print("SepReformer checkpoints:", len(ckpts))
                for p in ckpts[:10]:
                    print(p)
            else:
                print("RUN_SEPREFORMER=False, bỏ qua SepReformer")

            os.chdir("/kaggle/working/sommelier/podcast-pipeline")
            """
        ),
        md("## 7. Chạy pipeline gốc của main"),
        code(
            """
            import os

            os.chdir("/kaggle/working/sommelier/podcast-pipeline")

            cmd = [
                "python", "main_original_ASR_MoE.py",
                "--input_folder_path", INPUT_DIR,
                "--config_path", "config.json",
                "--batch_size", str(BATCH_SIZE),
                "--LLM", LLM_CASE,
                "--merge_gap", str(MERGE_GAP),
                "--speaker-link-threshold", str(SPEAKER_LINK_THRESHOLD),
                "--seg_th", str(DIAR_SEGMENTATION_THRESHOLD),
                "--min_cluster_size", str(DIAR_MIN_CLUSTER_SIZE),
                "--clust_th", str(DIAR_CLUSTER_THRESHOLD),
                "--whisper_arch", WHISPER_ARCH,
                "--asr_language", ASR_LANGUAGE,
                "--phowhisper_model_name", PHOWHISPER_MODEL_NAME,
                "--ctc_model_name", CTC_MODEL_NAME,
                "--compute_type", COMPUTE_TYPE,
                "--threads", str(ASR_THREADS),
                "--overlap_threshold", str(OVERLAP_THRESHOLD),
                "--opus_decode_workers", str(OPUS_DECODE_WORKERS),
                "--ffmpeg_threads_per_decode", str(FFMPEG_THREADS_PER_DECODE),
                "--diar_device_index", str(DIAR_DEVICE_INDEX),
                "--sortformer_device_index", str(SORTFORMER_DEVICE_INDEX),
                "--whisper_device_index", str(WHISPER_DEVICE_INDEX),
                "--asr_moe_device_index", str(ASR_MOE_DEVICE_INDEX),
                "--panns_device_index", str(PANNS_DEVICE_INDEX),
                "--demucs_device_index", str(DEMUCS_DEVICE_INDEX),
                "--sepreformer_device_index", str(SEPREFORMER_DEVICE_INDEX),
                "--initprompt" if INIT_PROMPT else "--no-initprompt",
                "--dia3" if DIA3 else "--no-dia3",
                "--korean" if KOREAN_G2P else "--no-korean",
                "--demucs" if RUN_DEMUCS else "--no-demucs",
                "--sepreformer" if RUN_SEPREFORMER else "--no-sepreformer",
                "--ASRMoE" if ASR_MOE else "--no-ASRMoE",
                "--asr_quality_guard" if ASR_QUALITY_GUARD else "--no-asr_quality_guard",
                "--asr_micro_segment_seconds", str(ASR_MICRO_SEGMENT_SECONDS),
                "--asr_short_segment_seconds", str(ASR_SHORT_SEGMENT_SECONDS),
                "--asr_vi_agreement_threshold", str(ASR_VI_AGREEMENT_THRESHOLD),
                "--asr_context_pad_before", str(ASR_CONTEXT_PAD_BEFORE),
                "--asr_context_pad_after", str(ASR_CONTEXT_PAD_AFTER),
                "--whisperx_word_timestamps" if WHISPERX_WORD_TIMESTAMPS else "--no-whisperx_word_timestamps",
                "--qwen3omni" if QWEN3OMNI else "--no-qwen3omni",
                "--sortformer-param" if SORTFORMER_PARAM else "--no-sortformer-param",
                "--sortformer-pad-onset", str(SORTFORMER_PAD_ONSET),
                "--sortformer-pad-offset", str(SORTFORMER_PAD_OFFSET),
            ]
            MAIN_PIPELINE_START_PERF = time.perf_counter()
            MAIN_PIPELINE_START_CLOCK = now_label()
            print("Main pipeline timing start:", MAIN_PIPELINE_START_CLOCK)
            try:
                run_logged(cmd, "18_main_pipeline.log", cwd="/kaggle/working/sommelier/podcast-pipeline", tail=80)
            finally:
                MAIN_PIPELINE_END_PERF = time.perf_counter()
                MAIN_PIPELINE_END_CLOCK = now_label()
                MAIN_PIPELINE_RUNTIME_SECONDS = MAIN_PIPELINE_END_PERF - MAIN_PIPELINE_START_PERF
                print("Main pipeline timing end:", MAIN_PIPELINE_END_CLOCK)
                print("Main pipeline runtime:", fmt_duration(MAIN_PIPELINE_RUNTIME_SECONDS), f"({MAIN_PIPELINE_RUNTIME_SECONDS:.2f}s)")
            """
        ),
        md("## 8. Dựng artifact `run_full` cho HTML viewer"),
        code(
            """
            import json
            import math
            import shutil
            from pathlib import Path
            from pydub import AudioSegment

            def load_json(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)

            def dump_json(data, path):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

            result_candidates = sorted(Path(INPUT_DIR).glob("_final/**/*.json"), key=lambda p: p.stat().st_mtime)
            if not result_candidates:
                result_candidates = sorted(Path("/kaggle/working").glob("**/full.json"), key=lambda p: p.stat().st_mtime)
            if not result_candidates:
                raise FileNotFoundError("Không tìm thấy JSON output từ main pipeline.")

            RESULT_JSON = result_candidates[-1]
            RESULT_DIR = RESULT_JSON.parent
            RESULT_AUDIO_DIR = RESULT_DIR / RESULT_JSON.stem
            print("RESULT_JSON:", RESULT_JSON)
            print("RESULT_AUDIO_DIR:", RESULT_AUDIO_DIR)

            result_data = load_json(RESULT_JSON)
            segments = result_data.get("segments", [])
            audio = AudioSegment.from_file(AUDIO_WAV)
            duration_seconds = len(audio) / 1000.0

            # Stage 00/02 audio files expected by both HTML generators.
            shutil.copy2(AUDIO_WAV, Path(MUSIC_DIR) / "cleaned_audio.wav")
            audio[:30000].export(Path(MUSIC_DIR) / "preview_cleaned_30s.wav", format="wav")

            # Normalize segment indexes and copy exported MP3 files into 05_export/final/data_audio.
            final_audio_dir = Path(FINAL_DIR) / "data_audio"
            final_audio_dir.mkdir(parents=True, exist_ok=True)
            exported_files = sorted(RESULT_AUDIO_DIR.glob("*.mp3")) if RESULT_AUDIO_DIR.exists() else []

            normalized = []
            for i, seg in enumerate(segments):
                item = dict(seg)
                idx = str(item.get("index", f"{i:05d}"))
                if idx.isdigit():
                    idx = idx.zfill(5)
                item["index"] = idx
                item.setdefault("speaker", "UNKNOWN")
                item.setdefault("start", 0.0)
                item.setdefault("end", item["start"])
                item["duration"] = max(0.0, float(item["end"]) - float(item["start"]))

                preferred_name = f"{idx}_{item['speaker']}.mp3"
                src = RESULT_AUDIO_DIR / preferred_name
                if not src.exists() and i < len(exported_files):
                    src = exported_files[i]
                if src.exists():
                    dst = final_audio_dir / src.name
                    shutil.copy2(src, dst)
                    item["audio_file"] = f"data_audio/{dst.name}"
                normalized.append(item)

            speakers = sorted({s.get("speaker", "UNKNOWN") for s in normalized})
            demucs_flags = [bool(s.get("demucs", False)) for s in normalized]

            diar_data = {
                "audio_path": AUDIO_WAV,
                "source_audio_path": AUDIO_IN,
                "audio_name": "full",
                "sample_rate": 16000,
                "audio_duration_seconds": duration_seconds,
                "segments": [{k: v for k, v in s.items() if k not in {"text", "text_whisper", "text_phowhisper", "text_chunkformer", "words"}} for s in normalized],
                "metadata": {
                    "stage": "diarization_proxy_from_main",
                    "note": "main branch does not persist intermediate diarization JSON; this proxy keeps timeline/speaker data for HTML review.",
                    "speaker_count": len(speakers),
                    "speakers": speakers,
                },
            }
            dump_json(diar_data, Path(DIAR_DIR) / "diarization.json")

            vad_chunks = {
                "audio_path": AUDIO_WAV,
                "vad_chunks": [{"index": "000", "offset": 0.0, "duration": duration_seconds, "path": AUDIO_WAV}],
                "chunks": [{"index": "000", "offset": 0.0, "duration": duration_seconds, "path": AUDIO_WAV}],
                "metadata": {"stage": "vad_chunk_proxy_from_main"},
            }
            dump_json(vad_chunks, Path(DIAR_DIR) / "vad_chunks.json")
            dump_json(vad_chunks, Path(DIAR_DIR) / "trace_vad_chunks.json")

            music_data = {
                "audio_path": str(Path(MUSIC_DIR) / "cleaned_audio.wav"),
                "source_audio_path": AUDIO_WAV,
                "audio_name": "full",
                "sample_rate": 16000,
                "audio_duration_seconds": duration_seconds,
                "segments": diar_data["segments"],
                "segment_demucs_flags": demucs_flags,
                "metadata": {
                    "stage": "music_clean_proxy_from_main",
                    "demucs_enabled": RUN_DEMUCS,
                    "demucs_segment_count": sum(demucs_flags),
                    "note": "main branch does not save cleaned full waveform separately; cleaned_audio.wav is the normalized timeline audio used for HTML playback.",
                },
            }
            dump_json(music_data, Path(MUSIC_DIR) / "segment_flags.json")

            overlap_data = {
                "audio_path": str(Path(MUSIC_DIR) / "cleaned_audio.wav"),
                "source_audio_path": AUDIO_WAV,
                "audio_name": "full",
                "sample_rate": 16000,
                "audio_duration_seconds": duration_seconds,
                "segments": normalized,
                "segment_demucs_flags": demucs_flags,
                "overlap_pairs": [],
                "metadata": {
                    "stage": "overlap_proxy_from_main",
                    "sepreformer_enabled": RUN_SEPREFORMER,
                    "separated_segments_count": sum(bool(s.get("is_separated", False)) for s in normalized),
                },
            }
            dump_json(overlap_data, Path(OVERLAP_DIR) / "segments.json")

            asr_data = {
                "audio_path": str(Path(MUSIC_DIR) / "cleaned_audio.wav"),
                "source_audio_path": AUDIO_WAV,
                "audio_name": "full",
                "sample_rate": 16000,
                "audio_duration_seconds": duration_seconds,
                "segments": normalized,
                "segment_demucs_flags": demucs_flags,
                "metadata": {
                    "stage": "asr_from_main",
                    **result_data.get("metadata", {}),
                },
            }
            dump_json(asr_data, Path(ASR_DIR) / "transcript.json")

            final_data = {
                "audio_path": str(Path(MUSIC_DIR) / "cleaned_audio.wav"),
                "source_audio_path": AUDIO_WAV,
                "audio_name": "data_audio",
                "sample_rate": 16000,
                "audio_duration_seconds": duration_seconds,
                "segments": normalized,
                "metadata": {
                    "stage": "export_from_main",
                    "exported_mp3_count": len(list(final_audio_dir.glob("*.mp3"))),
                    **result_data.get("metadata", {}),
                },
            }
            dump_json(final_data, Path(FINAL_DIR) / "data_audio.json")

            short_count = sum(1 for s in normalized if float(s.get("end", 0)) - float(s.get("start", 0)) < 1.0)
            qa_count = sum(1 for s in normalized if s.get("asr_quality_actions"))
            eval_report = {
                "audio_duration_seconds": duration_seconds,
                "segment_count": len(normalized),
                "speaker_count": len(speakers),
                "speakers": speakers,
                "demucs_segment_count": sum(demucs_flags),
                "separated_segment_count": sum(bool(s.get("is_separated", False)) for s in normalized),
                "short_segment_count": short_count,
                "quality_guard_touched_count": qa_count,
                "source_result_json": str(RESULT_JSON),
            }
            dump_json(eval_report, Path(EVAL_DIR) / "eval_report.json")
            report_md = "\\n".join([
                "# Sommelier main HTML eval",
                "",
                f"- Audio duration: {duration_seconds:.2f}s",
                f"- Segments: {len(normalized)}",
                f"- Speakers: {len(speakers)} ({', '.join(speakers) if speakers else '-'})",
                f"- Demucs flagged segments: {sum(demucs_flags)}",
                f"- SepReformer separated segments: {eval_report['separated_segment_count']}",
                f"- Short segments < 1s: {short_count}",
                f"- Quality guard touched: {qa_count}",
                f"- Source JSON: `{RESULT_JSON}`",
                "",
                "Note: This notebook runs the original main-branch pipeline, then materializes the stage-shaped artifacts required by the test-divide-stage HTML viewers.",
            ])
            Path(EVAL_DIR, "eval_report.md").write_text(report_md, encoding="utf-8")

            copy_log("18_main_pipeline.log", "18_stage_01_diarize.log", "Stage 01 proxy log from main pipeline")
            copy_log("18_main_pipeline.log", "19_stage_02_music_clean.log", "Stage 02 proxy log from main pipeline")
            copy_log("18_main_pipeline.log", "20_stage_03_overlap_separate.log", "Stage 03 proxy log from main pipeline")
            copy_log("18_main_pipeline.log", "22_stage_04_asr.log", "Stage 04 proxy log from main pipeline")
            copy_log("18_main_pipeline.log", "23_stage_05_export.log", "Stage 05 proxy log from main pipeline")
            Path(LOG_DIR, "24_stage_06_eval.log").write_text(report_md, encoding="utf-8")

            print("run_full artifacts created under:", RUN_DIR)
            print("MP3 files:", len(list(final_audio_dir.glob("*.mp3"))))
            """
        ),
        md("## 9. Review nhanh output theo stage"),
        code(
            """
            import json
            import pandas as pd
            from IPython.display import Audio, Markdown, display

            pd.set_option("display.max_colwidth", None)
            pd.set_option("display.max_columns", None)

            STAGE_JSON = {
                "1": Path(DIAR_DIR) / "diarization.json",
                "2": Path(MUSIC_DIR) / "segment_flags.json",
                "3": Path(OVERLAP_DIR) / "segments.json",
                "4": Path(ASR_DIR) / "transcript.json",
                "5": Path(FINAL_DIR) / "data_audio.json",
                "6": Path(EVAL_DIR) / "eval_report.json",
            }

            def review_stage(stage, n=10):
                path = STAGE_JSON[str(stage)]
                data = json.loads(path.read_text(encoding="utf-8"))
                print("Stage", stage, "->", path)
                print("Metadata keys:", sorted(data.get("metadata", data).keys()) if isinstance(data, dict) else [])
                segments = data.get("segments", []) if isinstance(data, dict) else []
                print("Segments:", len(segments))
                if segments:
                    df = pd.DataFrame(segments[:n])
                    cols = [c for c in ["index", "start", "end", "duration", "speaker", "text", "audio_file", "demucs", "is_separated"] if c in df.columns]
                    display(df[cols])
                if str(stage) == "6":
                    display(Markdown((Path(EVAL_DIR) / "eval_report.md").read_text(encoding="utf-8")))

            for s in ["1", "2", "3", "4", "5", "6"]:
                review_stage(s, n=5)
            """
        ),
        md("## 10. Tạo HTML giống nhánh `test-divide-stage`"),
        code(
            """
            os.chdir("/kaggle/working/sommelier")

            run_logged([
                "python", "tools/build_run_viewer.py",
                "--root", "/kaggle/working",
                "--out", "/kaggle/working/run_viewer/index.html",
            ], "25_build_run_viewer.log", cwd="/kaggle/working/sommelier", tail=30)

            run_logged([
                "python", "tools/inject_beautiful_html.py",
                RUN_DIR,
            ], "26_inject_beautiful_html.log", cwd="/kaggle/working/sommelier", tail=30)

            print("Per-run injected HTML:", Path(RUN_DIR) / "index.html")
            print("All-run viewer HTML:", Path("/kaggle/working/run_viewer/index.html"))
            """
        ),
        md("## 11. Zip output để tải về"),
        code(
            """
            import zipfile
            from IPython.display import FileLink, display

            ZIP_START_PERF = time.perf_counter()
            zip_path = Path("/kaggle/working/sommelier_main_html_outputs.zip")
            if zip_path.exists():
                zip_path.unlink()

            roots = [Path(RUN_DIR), Path("/kaggle/working/run_viewer")]
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for root in roots:
                    for path in root.rglob("*"):
                        if path.is_file():
                            zf.write(path, arcname=path.relative_to("/kaggle/working"))

            ZIP_END_PERF = time.perf_counter()
            ZIP_RUNTIME_SECONDS = ZIP_END_PERF - ZIP_START_PERF
            AUDIO_TIMING["audio_job_end_perf"] = time.perf_counter()
            AUDIO_TIMING["audio_job_end_clock"] = now_label()
            AUDIO_TIMING["audio_job_end_label"] = "end: HTML + zip output ready"
            AUDIO_JOB_RUNTIME_SECONDS = AUDIO_TIMING["audio_job_end_perf"] - AUDIO_TIMING["audio_job_start_perf"]

            print("Created:", zip_path)
            print("Zip includes:")
            print("- run_full/index.html")
            print("- run_viewer/index.html")
            print("- run_full/00_input, 01_diarization, 02_music_clean, 03_overlap, 04_asr, 05_export, 06_eval, logs")
            print()
            print("=== AUDIO RUNTIME SUMMARY ===")
            print("Input audio:", AUDIO_IN)
            print("Measured from:", AUDIO_TIMING["audio_job_start_label"])
            print("Measured to:", AUDIO_TIMING["audio_job_end_label"])
            print("Start clock:", AUDIO_TIMING["audio_job_start_clock"])
            print("End clock:", AUDIO_TIMING["audio_job_end_clock"])
            print(f"Audio timeline: 0.00s -> {AUDIO_DURATION_SECONDS:.2f}s")
            print("Audio length:", fmt_duration(AUDIO_DURATION_SECONDS), f"({AUDIO_DURATION_SECONDS:.2f}s)")
            print("Audio job wall time:", fmt_duration(AUDIO_JOB_RUNTIME_SECONDS), f"({AUDIO_JOB_RUNTIME_SECONDS:.2f}s)")
            print("Audio prepare time:", fmt_duration(AUDIO_TIMING.get("audio_prepare_seconds", 0.0)))
            if "MAIN_PIPELINE_RUNTIME_SECONDS" in globals():
                print("Main pipeline time:", fmt_duration(MAIN_PIPELINE_RUNTIME_SECONDS), f"({MAIN_PIPELINE_RUNTIME_SECONDS:.2f}s)")
            print("HTML + zip packaging time:", fmt_duration(ZIP_RUNTIME_SECONDS), f"({ZIP_RUNTIME_SECONDS:.2f}s)")
            if AUDIO_DURATION_SECONDS:
                print("Wall-time / audio-duration ratio:", f"{AUDIO_JOB_RUNTIME_SECONDS / AUDIO_DURATION_SECONDS:.3f}x")
            print("Timing is printed in the notebook only; it is not written into run_full JSON/data files.")
            display(FileLink(str(zip_path)))
            display(FileLink(str(Path(RUN_DIR) / "index.html")))
            display(FileLink("/kaggle/working/run_viewer/index.html"))
            """
        ),
    ]

    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
            "kaggle": {
                "accelerator": "gpu",
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells,
    }

    OUT_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
