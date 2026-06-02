import os
with open("main_original_ASR_MoE.py", "r") as f:
    lines = f.readlines()

def get(start, end):
    return "".join(lines[start-1:end])

out = "stages/stage_01_diarize.py"

imports = """from __future__ import annotations
import argparse
import shutil
import time
import json
import datetime
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

try:
    from nemo.collections.asr.models import SortformerEncLabelModel
except ImportError:
    SortformerEncLabelModel = None

try:
    from pyannote.audio import Inference
except ImportError:
    Inference = None

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
            return logging.getLogger("stage_01")
            
from models.silero_vad import SileroVAD

warnings.filterwarnings("ignore")
"""

content = get(148, 156) + "\n"
content += get(108, 144) + "\n"
content += get(2133, 2179) + "\n"
content += get(2181, 2221) + "\n"
content += get(2223, 2240) + "\n"
content += get(2260, 2306) + "\n"

# Patch silence_intervals to accept vad
silence_int = get(2309, 2355)
silence_int = silence_int.replace("def _build_silence_intervals(audio_info):", "def _build_silence_intervals(audio_info, vad):")
silence_int = silence_int.replace("pipeline.vad.", "vad.")
content += silence_int + "\n"

content += get(2358, 2419) + "\n"
content += get(2422, 2483) + "\n"
content += get(2486, 2493) + "\n"
content += get(2496, 2519) + "\n"
content += get(2522, 2592) + "\n"
content += get(2595, 2718) + "\n"

# Patch prepare chunks to accept vad
prep = get(2721, 2797)
prep = prep.replace("def prepare_diarization_chunks(audio_path: str, audio_info):", "def prepare_diarization_chunks(audio_path: str, audio_info, vad):")
prep = prep.replace("_build_silence_intervals(audio_info)", "_build_silence_intervals(audio_info, vad)")
content += prep + "\n"

wrapper = """
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 01: run diarization and save segments JSON.")
    parser.add_argument("--input_audio", required=True, help="Input audio file path.")
    parser.add_argument("--out", default="", help="Output diarization JSON path.")
    parser.add_argument("--config_path", default="config.json")
    parser.add_argument("--seg_th", type=float, default=0.11)
    parser.add_argument("--min_cluster_size", type=int, default=11)
    parser.add_argument("--clust_th", type=float, default=0.5)
    parser.add_argument("--merge_gap", type=float, default=2.0)
    parser.add_argument("--same_speaker_merge_gap", type=float, default=0.3)
    parser.add_argument("--short_backchannel_seconds", type=float, default=1.0)
    parser.add_argument("--speaker-link-threshold", type=float, default=0.6)
    parser.add_argument("--max_segment_duration", type=float, default=30.0)
    parser.add_argument("--speaker-recluster-threshold", type=float, default=0.75)
    parser.add_argument("--sortformer-param", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sortformer-pad-offset", type=float, default=0.01)
    parser.add_argument("--sortformer-pad-onset", type=float, default=0.23)
    parser.add_argument("--onset", type=float, default=0.53)
    parser.add_argument("--offset", type=float, default=0.49)
    parser.add_argument("--min-duration-on", type=float, default=0.42)
    parser.add_argument("--min-duration-off", type=float, default=0.34)
    parser.add_argument("--use-custom-binarize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    audio_path = Path(args.input_audio).expanduser().resolve()
    out_path = Path(args.out) if args.out else stage_common.default_stage_dir(audio_path) / "diarization.json"

    cfg = load_cfg(args.config_path)
    logger = Logger.get_logger()
    
    pipe_args = SimpleNamespace(
        merge_gap=args.merge_gap,
        sortformer_param=args.sortformer_param,
        sortformer_pad_offset=args.sortformer_pad_offset,
        sortformer_pad_onset=args.sortformer_pad_onset,
        onset=args.onset,
        offset=args.offset,
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
        use_custom_binarize=args.use_custom_binarize,
    )

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    logger.info(f"Stage 01 device: {device_name}")

    vad = SileroVAD(device=device)

    speaker_embedder = None
    if cfg.get("huggingface_token", "").startswith("hf"):
        try:
            if Inference is None:
                raise ImportError("pyannote.audio is not installed")
            speaker_embedder = Inference("pyannote/embedding", device=device, token=cfg["huggingface_token"], window="whole")
        except Exception as exc:
            logger.warning(f"Speaker embedder unavailable: {exc}")

    if SortformerEncLabelModel is None:
        raise ImportError("nemo_toolkit[asr] is required")
        
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    try:
        diar_model = diar_model.to(device)
    except Exception:
        pass
    diar_model.eval()

    sample_rate = int(cfg["entrypoint"]["SAMPLE_RATE"])
    audio_info = stage_common.load_audio_info(audio_path, sample_rate)
    audio_duration = len(audio_info["waveform"]) / audio_info["sample_rate"]

    start_time = time.time()
    diar_chunks, temp_chunk_dir = prepare_diarization_chunks(str(audio_path), audio_info, vad)
    diarization_frames = []

    try:
        for chunk in diar_chunks:
            predicted_segments, tensor_outputs = diar_model.diarize(
                audio=chunk["path"], batch_size=1, include_tensor_outputs=True
            )
            
            chunk_df = None
            if getattr(pipe_args, "use_custom_binarize", False) and tensor_outputs is not None:
                probs = None
                if isinstance(tensor_outputs, torch.Tensor):
                    probs = tensor_outputs.squeeze(0).cpu().numpy()
                elif isinstance(tensor_outputs, list) and len(tensor_outputs) > 0 and isinstance(tensor_outputs[0], torch.Tensor):
                    probs = tensor_outputs[0].squeeze(0).cpu().numpy()
                elif isinstance(tensor_outputs, dict) and "preds" in tensor_outputs:
                    probs = tensor_outputs["preds"].squeeze(0).cpu().numpy()
                    
                if probs is not None:
                    if probs.min() < 0 or probs.max() > 1.0:
                        probs = 1.0 / (1.0 + np.exp(-probs))
                    try:
                        chunk_duration = float(librosa.get_duration(path=chunk["path"]))
                        frame_shift = chunk_duration / probs.shape[0]
                        onset = float(getattr(pipe_args, "onset", 0.53))
                        offset = float(getattr(pipe_args, "offset", 0.49))
                        min_duration_on = float(getattr(pipe_args, "min_duration_on", 0.42))
                        min_duration_off = float(getattr(pipe_args, "min_duration_off", 0.34))
                        
                        if getattr(pipe_args, "_logged_custom_binarize", None) != True:
                            logger.info(f"Applying custom binarize -> onset: {onset}, offset: {offset}")
                            setattr(pipe_args, "_logged_custom_binarize", True)
                            
                        custom_segments = custom_binarize(probs, frame_shift, onset, offset, min_duration_on, min_duration_off)
                        chunk_df = pd.DataFrame(custom_segments)
                        if not chunk_df.empty:
                            chunk_df = chunk_df.sort_values(by="start").reset_index(drop=True)
                            chunk_df["label"] = [chr(ord('A') + i) for i in range(len(chunk_df))]
                            def fmt(sec):
                                td = datetime.timedelta(seconds=sec)
                                hrs = td.seconds // 3600 + td.days * 24
                                mins = (td.seconds // 60) % 60
                                secs = td.seconds % 60
                                ms = int(td.microseconds / 1000)
                                return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
                            chunk_df["segment"] = chunk_df.apply(lambda row: f"[ {fmt(row['start'])} --> {fmt(row['end'])}]", axis=1)
                        else:
                            chunk_df = pd.DataFrame(columns=['segment','label','speaker','start','end'])
                    except Exception as e:
                        logger.warning(f"Custom binarize failed: {e}")
                        chunk_df = None
                        
            if chunk_df is None:
                chunk_df = sortformer_dia(predicted_segments)
                
            if not chunk_df.empty:
                chunk_df["start"] += chunk["offset"]
                chunk_df["end"] += chunk["offset"]
                chunk_df = _apply_sortformer_segment_padding_from_args(chunk_df, pipe_args, logger, audio_duration)
            diarization_frames.append(chunk_df)
    finally:
        if temp_chunk_dir:
            shutil.rmtree(temp_chunk_dir, ignore_errors=True)

    if diarization_frames:
        diarization_frames = align_speakers_across_chunks(
            diarization_frames, audio_info=audio_info, embedder=speaker_embedder, similarity_threshold=args.speaker_link_threshold
        )
        speakerdia = pd.concat(diarization_frames, ignore_index=True)
    else:
        speakerdia = pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])

    recluster_stats = {"skipped": True, "reason": "disabled_or_no_embedder"}
    if args.speaker_recluster_threshold > 0 and speaker_embedder is not None:
        speakerdia, recluster_stats = re_cluster_speakers(
            speakerdia, audio_info=audio_info, embedder=speaker_embedder, similarity_threshold=args.speaker_recluster_threshold
        )

    segments = df_to_list(speakerdia)
    segments = split_long_segments(segments, max_duration=args.max_segment_duration)
    segments, postproc_stats = stage_common.postprocess_diarization_segments(
        segments, same_speaker_merge_gap=args.same_speaker_merge_gap, short_backchannel_seconds=args.short_backchannel_seconds, max_segment_duration=args.max_segment_duration
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Diarization finished in {elapsed:.2f}s.")

    out_data = {
        "audio_path": str(audio_path),
        "audio_name": audio_info["name"],
        "sample_rate": audio_info["sample_rate"],
        "audio_duration_seconds": audio_duration,
        "segments": stage_common.clean_segments_for_json(segments),
        "metadata": {
            "stage": "diarize",
            "device": device_name,
            "processing_time_seconds": elapsed,
            "postprocessing": postproc_stats,
            "reclustering": recluster_stats,
        },
    }
    
    stage_common.dump_json(out_data, out_path)
    logger.info(f"Saved diarization to {out_path}")

if __name__ == "__main__":
    main()
"""

with open(out, "w") as f:
    f.write(imports + content + wrapper)

print(f"Generated {out}")
