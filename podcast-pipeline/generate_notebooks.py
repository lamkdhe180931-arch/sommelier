import json
import os
from pathlib import Path

def create_notebook(filename, title, pip_installs, run_command, zip_folder=None):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n", "This notebook runs a single stage of the pipeline to save setup time and resources."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import subprocess\n",
                "def run_logged(cmd, log_file, cwd='/kaggle/working/sommelier'):\n",
                "    print(f'Running: {\" \".join(cmd)}')\n",
                "    with open(f'/kaggle/working/{log_file}', 'w') as f:\n",
                "        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd)\n",
                "    print(f'Done. Check {log_file} for output.')\n\n",
                "if not os.path.exists('/kaggle/working/sommelier'):\n",
                "    !git clone https://github.com/dmlguq456/sommelier.git /kaggle/working/sommelier\n",
                "os.chdir('/kaggle/working/sommelier')\n",
                "print('Repo cloned.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": pip_installs
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": run_command
        }
    ]

    if zip_folder:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import shutil\n",
                "from IPython.display import FileLink, display\n",
                f"shutil.make_archive('/kaggle/working/{zip_folder}_download', 'zip', '/kaggle/working/sommelier/{zip_folder}')\n",
                f"display(FileLink('{zip_folder}_download.zip'))"
            ]
        })

    notebook = {
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

    out_path = Path("../kaggle_notebooks") / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Generated {out_path}")


def main():
    os.makedirs("../kaggle_notebooks", exist_ok=True)

    # Stage 1: Diarize
    create_notebook(
        "nb_stage_01_diarize.ipynb",
        "Stage 01: Diarization",
        ["!pip install Cython nemo_toolkit[asr] pyannote.audio"],
        [
            "os.chdir('/kaggle/working/sommelier/podcast-pipeline')\n",
            "run_logged([\n",
            "    'python', 'stage_01_diarize.py',\n",
            "    '--input_audio', '/kaggle/input/your-audio/audio.wav',\n",
            "], 'stage_01.log')"
        ],
        "podcast-pipeline/data/diarization"
    )

    # Stage 2: Music Clean
    create_notebook(
        "nb_stage_02_music_clean.ipynb",
        "Stage 02: Music Clean",
        ["!pip install demucs panns-inference soundfile librosa numpy scipy"],
        [
            "os.chdir('/kaggle/working/sommelier/podcast-pipeline')\n",
            "run_logged([\n",
            "    'python', 'stage_02_music_clean.py',\n",
            "    '--diarization_json', '/kaggle/input/your-diarization/diarization.json',\n",
            "], 'stage_02.log')"
        ],
        "podcast-pipeline/data/music_clean"
    )

    # Stage 3: Overlap Separate
    create_notebook(
        "nb_stage_03_overlap.ipynb",
        "Stage 03: Overlap Separation",
        ["!pip install clearvoice torch torchaudio pydub"],
        [
            "os.chdir('/kaggle/working/sommelier/podcast-pipeline')\n",
            "run_logged([\n",
            "    'python', 'stage_03_overlap_separate.py',\n",
            "    '--cleaned_audio', '/kaggle/input/your-music-clean/cleaned_audio.wav',\n",
            "    '--segment_flags_json', '/kaggle/input/your-music-clean/segment_flags.json',\n",
            "], 'stage_03.log')"
        ],
        "podcast-pipeline/data/overlap"
    )

    # Stage 4: ASR
    create_notebook(
        "nb_stage_04_asr.ipynb",
        "Stage 04: ASR",
        ["!pip install transformers openai-whisper torch torchaudio pydub pandas"],
        [
            "os.chdir('/kaggle/working/sommelier/podcast-pipeline')\n",
            "run_logged([\n",
            "    'python', 'stage_04_asr.py',\n",
            "    '--overlap_json', '/kaggle/input/your-overlap/segments.json',\n",
            "], 'stage_04.log')"
        ],
        "podcast-pipeline/data/asr"
    )

    # Stage 5: Export
    create_notebook(
        "nb_stage_05_export.ipynb",
        "Stage 05: Export MP3",
        ["!pip install pydub pandas"],
        [
            "os.chdir('/kaggle/working/sommelier/podcast-pipeline')\n",
            "run_logged([\n",
            "    'python', 'stage_05_export.py',\n",
            "    '--transcript_json', '/kaggle/input/your-asr/transcript.json',\n",
            "], 'stage_05.log')"
        ],
        "podcast-pipeline/data/export"
    )

if __name__ == "__main__":
    main()
