import json
import os

out_dir = "stages/notebooks"
os.makedirs(out_dir, exist_ok=True)

def create_nb(filename, title, install_cmd, run_cmd, zip_input, zip_output, eval_cmd=None):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n\nĐảm bảo bạn đã add Dataset audio gốc (nếu là Stage 01), hoặc Output của Stage trước vào Kaggle Dataset."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone project từ nhánh test-divide-stage\n",
                "import os\n",
                "if not os.path.exists('/kaggle/working/sommelier'):\n",
                "    !git clone -b test-divide-stage https://github.com/lamkdhe180931-arch/sommelier.git\n",
                "else:\n",
                "    !cd /kaggle/working/sommelier && git pull origin test-divide-stage\n",
                "%cd /kaggle/working/sommelier/podcast-pipeline/stages"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Đăng nhập HuggingFace (Cần thiết cho Pyannote)\n",
                "# BẠN CẦN TẠO SECRET CÓ TÊN LÀ HF_TOKEN TRONG KAGGLE SECRETS TRƯỚC NHÉ\n",
                "import os\n",
                "try:\n",
                "    from kaggle_secrets import UserSecretsClient\n",
                "    user_secrets = UserSecretsClient()\n",
                "    hf_token = user_secrets.get_secret('HF_TOKEN')\n",
                "    os.environ['HF_TOKEN'] = hf_token\n",
                "    import json\n",
                "    if os.path.exists('config.json'):\n",
                "        with open('config.json', 'r') as f: cfg = json.load(f)\n",
                "        cfg['huggingface_token'] = hf_token\n",
                "        with open('config.json', 'w') as f: json.dump(cfg, f, indent=4)\n",
                "    print('Đã load HF_TOKEN thành công!')\n",
                "except Exception as e:\n",
                "    print('Chưa cấu hình HF_TOKEN trong Kaggle Secrets. Nếu chạy lỗi, vui lòng cấu hình HF_TOKEN.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": install_cmd
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": run_cmd
        }
    ]
    
    if eval_cmd:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": eval_cmd
        })
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import subprocess\n",
            "from IPython.display import FileLink\n\n",
            f"subprocess.run(['zip', '-r', '{zip_output}', '{zip_input}'])\n",
            f"FileLink('{zip_output}')"
        ]
    })
    
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {path}")


# Stage 01
create_nb(
    "nb_stage_01_diarize.ipynb",
    "Stage 01: Diarization",
    [
        "!pip install -q nemo_toolkit[asr] pyannote.audio\n",
        "!pip install -q soundfile librosa pandas pydub onnxruntime-gpu"
    ],
    [
        "AUDIO_INPUT = '/kaggle/input/your-dataset/audio.wav' # THAY ĐỔI ĐƯỜNG DẪN NÀY\n",
        "OUT_JSON = '/kaggle/working/diarization.json'\n\n",
        "# ==========================================\n",
        "# THAM SỐ TINH CHỈNH MODEL\n",
        "# ==========================================\n",
        "# 1. Model Sortformer (Thuật toán Binarize & Padding)\n",
        "ONSET = 0.53         # Ngưỡng bắt đầu giọng nói (Tăng -> Cắt gắt hơn)\n",
        "OFFSET = 0.49        # Ngưỡng kết thúc giọng nói (Giảm -> Kéo dài đuôi câu hơn)\n",
        "MIN_DUR_ON = 0.42    # Đoạn nói tối thiểu (giây). Ngắn hơn mức này bị xoá bỏ\n",
        "MIN_DUR_OFF = 0.34   # Khoảng lặng tối thiểu (giây). Ngắn hơn mức này bị gộp làm một\n",
        "PAD_ONSET = 0.23     # Kéo mốc thời gian bắt đầu ra trước (giây) tránh mất chữ cái đầu\n",
        "PAD_OFFSET = 0.01    # Kéo mốc thời gian kết thúc ra sau (giây)\n",
        "\n",
        "# 2. Model Pyannote (Nối Speaker các Chunk)\n",
        "SPK_LINK_TH = 0.6          # Ngưỡng giống nhau để nối người nói giữa các đoạn âm thanh. (Tăng -> Khó gộp hơn)\n",
        "SPK_RECLUSTER_TH = 0.75    # Ngưỡng dọn dẹp ID vụn ở bước cuối cùng. (Tăng -> Khó gộp nhóm hơn)\n",
        "\n",
        "!python stage_01_diarize.py \\\n",
        "  --input_audio \"{AUDIO_INPUT}\" \\\n",
        "  --out \"{OUT_JSON}\" \\\n",
        "  --onset {ONSET} \\\n",
        "  --offset {OFFSET} \\\n",
        "  --min-duration-on {MIN_DUR_ON} \\\n",
        "  --min-duration-off {MIN_DUR_OFF} \\\n",
        "  --sortformer-pad-onset {PAD_ONSET} \\\n",
        "  --sortformer-pad-offset {PAD_OFFSET} \\\n",
        "  --speaker-link-threshold {SPK_LINK_TH} \\\n",
        "  --speaker-recluster-threshold {SPK_RECLUSTER_TH}"
    ],
    "/kaggle/working/diarization.json",
    "/kaggle/working/diarization_output.zip",
    eval_cmd=[
        "# ==========================================\n",
        "# CELL ĐÁNH GIÁ NHANH KẾT QUẢ DIARIZATION\n",
        "# ==========================================\n",
        "import json\n",
        "import librosa\n",
        "import IPython.display as ipd\n",
        "from IPython.core.display import display, HTML\n\n",
        "try:\n",
        "    with open(OUT_JSON, 'r') as f:\n",
        "        data = json.load(f)\n",
        "    segments = data.get('segments', [])\n",
        "    if not segments:\n",
        "        print('Không tìm thấy phân đoạn nào trong JSON!')\n",
        "    else:\n",
        "        print(f'Tổng cộng {len(segments)} đoạn. Đang tải audio để hiển thị 30 đoạn đầu tiên...')\n",
        "        waveform, sr = librosa.load(AUDIO_INPUT, sr=16000)\n",
        "        html_out = \"<table border='1' style='width:100%; text-align:center;'>\"\n",
        "        html_out += \"<tr><th>Speaker ID</th><th>Thời gian</th><th>Nghe thử</th></tr>\"\n",
        "        \n",
        "        for seg in segments[:30]:\n",
        "            spk = seg['speaker']\n",
        "            start, end = seg['start'], seg['end']\n",
        "            start_sample = int(start * sr)\n",
        "            end_sample = int(end * sr)\n",
        "            clip = waveform[start_sample:end_sample]\n",
        "            \n",
        "            # Tạo thẻ <audio> bằng IPython.display\n",
        "            audio_widget = ipd.Audio(data=clip, rate=sr)\n",
        "            audio_html = audio_widget._repr_html_()\n",
        "            \n",
        "            html_out += f\"<tr><td><b>{spk}</b></td><td>{start:.2f}s - {end:.2f}s</td><td>{audio_html}</td></tr>\"\n",
        "        html_out += \"</table>\"\n",
        "        display(HTML(html_out))\n",
        "except Exception as e:\n",
        "    print('Có lỗi khi tạo bảng nghe thử:', e)\n"
    ]
)

# Stage 02
create_nb(
    "nb_stage_02_music_clean.ipynb",
    "Stage 02: Music Clean",
    [
        "!pip install -q demucs panns_inference\n",
        "!pip install -q soundfile librosa pandas pydub"
    ],
    [
        "DIAR_JSON = '/kaggle/input/diarization-output/diarization.json' # THAY ĐỔI ĐƯỜNG DẪN NÀY\n",
        "OUT_DIR = '/kaggle/working/stage_02_out'\n",
        "!mkdir -p $OUT_DIR\n\n",
        "!python stage_02_music_clean.py \\\n",
        "  --diarization_json \"$DIAR_JSON\" \\\n",
        "  --out \"$OUT_DIR\""
    ],
    "/kaggle/working/stage_02_out",
    "/kaggle/working/stage_02_output.zip"
)

# Stage 03
create_nb(
    "nb_stage_03_overlap.ipynb",
    "Stage 03: Overlap Separation",
    [
        "!pip install -q pyannote.audio\n",
        "!pip install -q soundfile librosa pandas pydub pyyaml\n\n",
        "# Clone SepReformer_Base_WSJ0\n",
        "%cd /kaggle/working\n",
        "!git clone https://github.com/hnam/SepReformer_Base_WSJ0.git\n",
        "%cd /kaggle/working/sommelier/podcast-pipeline/stages"
    ],
    [
        "CLEANED_AUDIO = '/kaggle/input/stage-02-output/stage_02_out/cleaned_audio.wav' # THAY ĐỔI ĐƯỜNG DẪN NÀY\n",
        "FLAGS_JSON = '/kaggle/input/stage-02-output/stage_02_out/segment_flags.json' # THAY ĐỔI ĐƯỜNG DẪN NÀY\n",
        "OUT_DIR = '/kaggle/working/stage_03_out'\n",
        "!mkdir -p $OUT_DIR\n\n",
        "!python stage_03_overlap.py \\\n",
        "  --cleaned_audio \"$CLEANED_AUDIO\" \\\n",
        "  --segment_flags_json \"$FLAGS_JSON\" \\\n",
        "  --out \"$OUT_DIR\""
    ],
    "/kaggle/working/stage_03_out",
    "/kaggle/working/stage_03_output.zip"
)

# Stage 04
create_nb(
    "nb_stage_04_asr.ipynb",
    "Stage 04: ASR",
    [
        "!pip install -q faster-whisper transformers\n",
        "!pip install -q soundfile librosa pandas pydub"
    ],
    [
        "OVERLAP_JSON = '/kaggle/input/stage-03-output/stage_03_out/segments.json' # THAY ĐỔI ĐƯỜNG DẪN NÀY\n",
        "OUT_DIR = '/kaggle/working/stage_04_out'\n",
        "!mkdir -p $OUT_DIR\n\n",
        "!python stage_04_asr.py \\\n",
        "  --overlap_json \"$OVERLAP_JSON\" \\\n",
        "  --out \"$OUT_DIR\""
    ],
    "/kaggle/working/stage_04_out",
    "/kaggle/working/stage_04_output.zip"
)
