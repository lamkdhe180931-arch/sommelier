import os

with open("main_original_ASR_MoE.py", "r") as f:
    lines = f.readlines()

def get(start, end):
    return "".join(lines[start-1:end])

out = "stages/stage_03_overlap.py"

imports = """from __future__ import annotations
import argparse
import shutil
import time
import json
import yaml
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np
import librosa
import torch
from pydub import AudioSegment
import soundfile as sf
import warnings

try:
    from pyannote.audio import Model as PyannoteModel, Inference
except ImportError:
    PyannoteModel = Inference = None

import stage_common
from utils.tool import load_cfg
from utils.asr_quality import should_skip_sepreformer_pair
try:
    from utils.logger import Logger
except ImportError:
    import logging
    class Logger:
        @staticmethod
        def get_logger():
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger("stage_03")

warnings.filterwarnings("ignore")
"""

content = get(1097, 1141) + "\n"
content += get(1144, 1283) + "\n"
content += get(1286, 1348) + "\n"

# Patch process_overlapping_segments_with_separation to use should_skip_sepreformer_pair locally
process_overlap = get(1352, 1559)
content += process_overlap + "\n"

wrapper = """
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 03: detect and separate overlapping speech.")
    parser.add_argument("--cleaned_audio", required=True)
    parser.add_argument("--segment_flags_json", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--config_path", default="config.json")
    parser.add_argument("--overlap_threshold", type=float, default=0.2)
    parser.add_argument("--min_sepreformer_overlap", type=float, default=0.5)
    parser.add_argument("--min_sepreformer_segment", type=float, default=2.0)
    parser.add_argument("--skip_overlap_separation", action=argparse.BooleanOptionalAction, default=False)
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
    logger.info(f"Stage 03 device: {device_name}")

    start_time = time.time()
    seg_data = stage_common.load_json(args.segment_flags_json)
    audio_path = Path(args.cleaned_audio)
    out_dir = audio_path.parent
    if args.out:
        out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_json_path = out_dir / "segments.json"
    
    segments = seg_data.get("segments", [])

    if getattr(args, "skip_overlap_separation", False):
        logger.info("Skipping overlap separation due to --skip_overlap_separation")
    else:
        sample_rate = seg_data.get("sample_rate", int(cfg["entrypoint"]["SAMPLE_RATE"]))
        audio_info = stage_common.load_audio_info(audio_path, sample_rate)
        
        separator = SepReformerSeparator(device=device_name)
        
        speaker_embedder = None
        if cfg.get("huggingface_token", "").startswith("hf"):
            try:
                if Inference is None:
                    raise ImportError("pyannote.audio is not installed")
                from pyannote.audio import Model as PyannoteModel
                emb_model = PyannoteModel.from_pretrained("pyannote/embedding", use_auth_token=cfg["huggingface_token"])
                speaker_embedder = Inference(emb_model, device=device, window="whole")
            except Exception as exc:
                logger.warning(f"Speaker embedder unavailable: {exc}")
                
        segments = process_overlapping_segments_with_separation(
            segments,
            audio_info,
            separator,
            out_dir,
            speaker_embedder=speaker_embedder,
            overlap_threshold=args.overlap_threshold,
            min_overlap_duration=args.min_sepreformer_overlap,
            min_segment_duration=args.min_sepreformer_segment,
            logger=logger
        )

    elapsed = time.time() - start_time
    logger.info(f"Overlap separation finished in {elapsed:.2f}s.")

    out_data = {
        "audio_path": str(audio_path),
        "source_audio_path": seg_data.get("source_audio_path", ""),
        "audio_name": seg_data.get("audio_name", ""),
        "sample_rate": seg_data.get("sample_rate", 16000),
        "audio_duration_seconds": seg_data.get("audio_duration_seconds", 0.0),
        "segments": segments,
        "segment_demucs_flags": seg_data.get("segment_demucs_flags", []),
        "metadata": {
            "stage": "overlap_separate",
            "enabled": not getattr(args, "skip_overlap_separation", False),
            "processing_time_seconds": elapsed,
            "overlap_threshold_seconds": args.overlap_threshold,
            "min_sepreformer_overlap_seconds": args.min_sepreformer_overlap,
            "min_sepreformer_segment_seconds": args.min_sepreformer_segment,
        },
    }
    stage_common.dump_json(out_data, out_json_path)
    logger.info(f"Saved segments to {out_json_path}")

if __name__ == "__main__":
    main()
"""

with open(out, "w") as f:
    f.write(imports + content + wrapper)

print(f"Generated {out}")
