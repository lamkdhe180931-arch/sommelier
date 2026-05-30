from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import stage_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 01: run diarization and save segments JSON.")
    parser.add_argument("--input_audio", required=True, help="Input audio file path.")
    parser.add_argument("--out", default="", help="Output diarization JSON path.")
    parser.add_argument("--config_path", default="config.json")
    parser.add_argument("--seg_th", type=float, default=0.11)
    parser.add_argument("--min_cluster_size", type=int, default=11)
    parser.add_argument("--clust_th", type=float, default=0.5)
    parser.add_argument("--merge_gap", type=float, default=2.0)
    parser.add_argument(
        "--same_speaker_merge_gap",
        type=float,
        default=0.3,
        help="Merge only adjacent same-speaker segments when the gap is at or below this many seconds.",
    )
    parser.add_argument(
        "--short_backchannel_seconds",
        type=float,
        default=1.0,
        help="Keep but label segments shorter than this as short_backchannel.",
    )
    parser.add_argument("--speaker-link-threshold", type=float, default=0.6)
    parser.add_argument("--max_segment_duration", type=float, default=30.0)
    parser.add_argument("--sortformer-param", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sortformer-pad-offset", type=float, default=-0.24)
    parser.add_argument("--sortformer-pad-onset", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.input_audio).expanduser().resolve()
    out_path = Path(args.out) if args.out else stage_common.default_stage_dir(audio_path) / "diarization.json"

    import main_original_ASR_MoE as pipeline

    cfg = pipeline.load_cfg(args.config_path)
    logger = pipeline.Logger.get_logger()
    pipeline.cfg = cfg
    pipeline.logger = logger
    pipeline.args = SimpleNamespace(
        merge_gap=args.merge_gap,
        sortformer_param=args.sortformer_param,
        sortformer_pad_offset=args.sortformer_pad_offset,
        sortformer_pad_onset=args.sortformer_pad_onset,
    )

    device_name = "cuda" if pipeline.torch.cuda.is_available() else "cpu"
    device = pipeline.torch.device(device_name)
    pipeline.device_name = device_name
    pipeline.device = device
    logger.info(f"Stage 01 device: {device_name}")

    pipeline.vad = pipeline.silero_vad.SileroVAD(device=device)

    speaker_embedder = None
    if cfg.get("huggingface_token", "").startswith("hf"):
        try:
            speaker_embedder = pipeline.Inference(
                "pyannote/embedding",
                device=device,
                use_auth_token=cfg["huggingface_token"],
                window="whole",
            )
        except Exception as exc:
            logger.warning(f"Speaker embedder unavailable; cross-chunk linking skipped: {exc}")
    else:
        logger.warning("Hugging Face token is not configured; cross-chunk linking skipped.")

    diar_model = pipeline.SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    try:
        diar_model = diar_model.to(device)
    except Exception:
        pass
    diar_model.eval()

    sample_rate = int(cfg["entrypoint"]["SAMPLE_RATE"])
    audio_info = stage_common.load_audio_info(audio_path, sample_rate)
    audio_duration = len(audio_info["waveform"]) / audio_info["sample_rate"]

    start_time = time.time()
    diar_chunks, temp_chunk_dir = pipeline.prepare_diarization_chunks(str(audio_path), audio_info)
    diarization_frames = []

    try:
        for chunk in diar_chunks:
            predicted_segments, _ = diar_model.diarize(
                audio=chunk["path"],
                batch_size=1,
                include_tensor_outputs=True,
            )
            chunk_df = pipeline.sortformer_dia(predicted_segments)
            if not chunk_df.empty:
                chunk_df["start"] += chunk["offset"]
                chunk_df["end"] += chunk["offset"]
                chunk_df = pipeline._apply_sortformer_segment_padding_from_args(
                    chunk_df,
                    args=pipeline.args,
                    logger=logger,
                    audio_duration=audio_duration,
                )
            diarization_frames.append(chunk_df)
    finally:
        if temp_chunk_dir:
            shutil.rmtree(temp_chunk_dir, ignore_errors=True)

    if diarization_frames:
        diarization_frames = pipeline.align_speakers_across_chunks(
            diarization_frames,
            audio_info=audio_info,
            embedder=speaker_embedder,
            similarity_threshold=args.speaker_link_threshold,
        )
        speakerdia = pd.concat(diarization_frames, ignore_index=True)
    else:
        speakerdia = pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])

    segments = pipeline.split_long_segments(
        pipeline.df_to_list(speakerdia),
        max_duration=args.max_segment_duration,
    )
    segments, postprocess_stats = stage_common.postprocess_diarization_segments(
        segments,
        same_speaker_merge_gap=args.same_speaker_merge_gap,
        short_backchannel_seconds=args.short_backchannel_seconds,
        max_segment_duration=args.max_segment_duration,
    )
    processing_time = time.time() - start_time

    stage_common.dump_json(
        {
            "audio_path": str(audio_path),
            "audio_name": stage_common.audio_name_from_path(audio_path),
            "sample_rate": sample_rate,
            "audio_duration_seconds": audio_duration,
            "segments": stage_common.clean_segments_for_json(segments),
            "metadata": {
                "stage": "diarize",
                "processing_time_seconds": processing_time,
                "rt_factor": processing_time / audio_duration if audio_duration > 0 else 0,
                "seg_th": args.seg_th,
                "min_cluster_size": args.min_cluster_size,
                "clust_th": args.clust_th,
                "speaker_link_threshold": args.speaker_link_threshold,
                "same_speaker_merge_gap_seconds": args.same_speaker_merge_gap,
                "short_backchannel_seconds": args.short_backchannel_seconds,
                "postprocess": postprocess_stats,
            },
        },
        out_path,
    )
    print(f"Stage 01 complete: {out_path}")


if __name__ == "__main__":
    main()
