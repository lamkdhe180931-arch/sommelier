import json
import os

out_dir = "stages/notebooks"
os.makedirs(out_dir, exist_ok=True)

def create_nb(filename, title, install_cmd, run_cmd, zip_input, zip_output):
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
                "# Clone project (tạm thời clone nhánh main, khi nào code được push lên github thì sửa link/nhánh)\n",
                "!git clone https://github.com/hnam/sommelier.git\n",
                "%cd sommelier/podcast-pipeline/stages"
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
        },
        {
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
        }
    ]
    
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
        "!pip install -q soundfile librosa pandas pydub"
    ],
    [
        "AUDIO_INPUT = '/kaggle/input/your-dataset/audio.wav' # THAY ĐỔI ĐƯỜNG DẪN NÀY\n",
        "OUT_JSON = '/kaggle/working/diarization.json'\n\n",
        "!python stage_01_diarize.py \\\n",
        "  --input_audio \"$AUDIO_INPUT\" \\\n",
        "  --out \"$OUT_JSON\""
    ],
    "/kaggle/working/diarization.json",
    "/kaggle/working/diarization_output.zip"
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
