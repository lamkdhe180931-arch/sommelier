import os

with open("main_original_ASR_MoE.py", "r") as f:
    lines = f.readlines()

def get(start, end):
    return "".join(lines[start-1:end])

out = "stages/stage_04_asr.py"

imports = """from __future__ import annotations
import argparse
import shutil
import time
import json
import collections
import difflib
from itertools import zip_longest
from typing import List, Tuple, Dict
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np
import librosa
import torch
import concurrent.futures
import warnings

import stage_common
from utils.tool import load_cfg
from utils.asr_quality import choose_asr_text
try:
    from utils.logger import Logger
except ImportError:
    import logging
    class Logger:
        @staticmethod
        def get_logger():
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger("stage_04")

from models import whisper_asr, vietnamese_asr

warnings.filterwarnings("ignore")
"""

content = get(158, 400) + "\n"
content += get(403, 457) + "\n"
content += get(1563, 1678) + "\n"
content += get(1683, 1888) + "\n"

wrapper = """
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 04: run ASR.")
    parser.add_argument("--overlap_json", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--config_path", default="config.json")
    parser.add_argument("--whisper_arch", default="large-v3", help="Model arch for WhisperX")
    parser.add_argument("--device", default="")
    parser.add_argument("--asr-moe", action=argparse.BooleanOptionalAction, default=True, help="Use multiple ASR models (Whisper, PhoWhisper, Chunkformer) and ROVER voting")
    parser.add_argument("--word-timestamps", action=argparse.BooleanOptionalAction, default=False, help="Enable word-level timestamps (slower)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logger = Logger.get_logger()
    cfg = load_cfg(args.config_path)

    device_name = args.device
    if not device_name:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    logger.info(f"Stage 04 device: {device_name}")

    start_time = time.time()
    seg_data = stage_common.load_json(args.overlap_json)
    audio_path = Path(seg_data["audio_path"])
    out_dir = audio_path.parent
    if args.out:
        out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_json_path = out_dir / "transcript.json"
    segments = seg_data.get("segments", [])
    sample_rate = seg_data.get("sample_rate", int(cfg["entrypoint"]["SAMPLE_RATE"]))
    
    pipe_args = SimpleNamespace(
        asr_moe=args.asr_moe,
        whisper_arch=args.whisper_arch,
        word_timestamps=args.word_timestamps,
    )

    if args.asr_moe:
        asr_model = whisper_asr.load_asr_model(
            args.whisper_arch,
            device_name,
            compute_type=cfg["model"]["whisper"]["compute_type"]
        )
        phowhisper_model = vietnamese_asr.load_phowhisper_model(
            "vinai/PhoWhisper-large", device=device_name
        )
        chunkformer_model = vietnamese_asr.load_chunkformer_model(
            "trungdt21/chunkformer-v2", device=device_name
        )
        
        segments, stats = asr_MoE(
            segments,
            str(audio_path),
            asr_model,
            phowhisper_model,
            chunkformer_model,
            pipe_args,
            logger
        )
    else:
        asr_model = whisper_asr.load_asr_model(
            args.whisper_arch,
            device_name,
            compute_type=cfg["model"]["whisper"]["compute_type"]
        )
        segments, stats = asr(
            segments,
            str(audio_path),
            asr_model,
            pipe_args,
            logger
        )

    elapsed = time.time() - start_time
    logger.info(f"ASR finished in {elapsed:.2f}s.")

    out_data = {
        "audio_path": str(audio_path),
        "source_audio_path": seg_data.get("source_audio_path", ""),
        "audio_name": seg_data.get("audio_name", ""),
        "sample_rate": sample_rate,
        "audio_duration_seconds": seg_data.get("audio_duration_seconds", 0.0),
        "segments": segments,
        "segment_demucs_flags": seg_data.get("segment_demucs_flags", []),
        "metadata": {
            "stage": "asr",
            "asr_moe": args.asr_moe,
            "whisper_arch": args.whisper_arch,
            "processing_time_seconds": elapsed,
            **stats
        },
    }
    stage_common.dump_json(out_data, out_json_path)
    logger.info(f"Saved transcripts to {out_json_path}")

if __name__ == "__main__":
    main()
"""

with open(out, "w") as f:
    f.write(imports + content + wrapper)

print(f"Generated {out}")
