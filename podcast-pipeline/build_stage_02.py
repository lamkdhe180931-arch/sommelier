import os

with open("main_original_ASR_MoE.py", "r") as f:
    lines = f.readlines()

def get(start, end):
    return "".join(lines[start-1:end])

out = "stages/stage_02_music_clean.py"

imports = """from __future__ import annotations
import argparse
import shutil
import time
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np
import librosa
import torch
from pydub import AudioSegment
import soundfile as sf
import tempfile
import warnings
import subprocess
import copy

try:
    from panns_inference import AudioTagging
except ImportError:
    AudioTagging = None

import stage_common
from utils.tool import load_cfg
try:
    from utils.logger import Logger
except ImportError:
    import logging
    class Logger:
        @staticmethod
        def get_logger():
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger("stage_02")

warnings.filterwarnings("ignore")
"""

content = get(699, 741) + "\n"
content += get(744, 794) + "\n"
content += get(797, 881) + "\n"
content += get(885, 982) + "\n"

wrapper = """
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 02: detect and remove background music.")
    parser.add_argument("--diarization_json", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--config_path", default="config.json")
    parser.add_argument("--skip_music_removal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logger = Logger.get_logger()
    cfg = load_cfg(args.config_path)

    device_name = args.device
    if not device_name:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    logger.info(f"Stage 02 device: {device_name}")

    if AudioTagging is None:
        raise ImportError("panns_inference is required for music detection")

    start_time = time.time()
    diar_data = stage_common.load_json(args.diarization_json)
    audio_path = Path(diar_data["audio_path"])
    out_dir = audio_path.parent / "_staged" / stage_common.audio_name_from_path(audio_path)
    if args.out:
        out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_json_path = out_dir / "segment_flags.json"
    cleaned_audio_path = out_dir / "cleaned_audio.wav"

    segments = diar_data.get("segments", [])

    if getattr(args, "skip_music_removal", False):
        logger.info("Skipping music removal due to --skip_music_removal")
        cleaned_audio_path = audio_path
        flags = [False] * len(segments)
    else:
        sample_rate = diar_data.get("sample_rate", int(cfg["entrypoint"]["SAMPLE_RATE"]))
        audio_info = stage_common.load_audio_info(audio_path, sample_rate)
        
        flags, cleaned_wav = preprocess_segments_with_demucs(
            segments,
            audio_info,
            demucs_model=cfg["model"]["demucs"]["model"],
            device=device_name,
            logger=logger
        )
        
        if cleaned_wav is not None:
            stage_common.write_wav(cleaned_audio_path, cleaned_wav, sample_rate)
            logger.info(f"Saved cleaned full audio to {cleaned_audio_path}")
        else:
            cleaned_audio_path = audio_path

    elapsed = time.time() - start_time
    logger.info(f"Music clean finished in {elapsed:.2f}s.")

    out_data = {
        "audio_path": str(cleaned_audio_path),
        "source_audio_path": str(audio_path),
        "audio_name": diar_data.get("audio_name", ""),
        "sample_rate": diar_data.get("sample_rate", 16000),
        "audio_duration_seconds": diar_data.get("audio_duration_seconds", 0.0),
        "segments": segments,
        "segment_demucs_flags": flags,
        "metadata": {
            "stage": "music_clean",
            "enabled": not getattr(args, "skip_music_removal", False),
            "processing_time_seconds": elapsed,
        },
    }
    stage_common.dump_json(out_data, out_json_path)
    logger.info(f"Saved music clean flags to {out_json_path}")

if __name__ == "__main__":
    main()
"""

with open(out, "w") as f:
    f.write(imports + content + wrapper)

print(f"Generated {out}")
