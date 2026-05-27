from __future__ import annotations

import argparse
import time
from pathlib import Path

import stage_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 02: optionally clean music with PANNs + Demucs.")
    parser.add_argument("--input_audio", required=True, help="Original audio path.")
    parser.add_argument("--diarization_json", required=True, help="Output from stage_01_diarize.py.")
    parser.add_argument("--out_audio", default="", help="Cleaned WAV output path.")
    parser.add_argument("--out_flags", default="", help="Segment flags JSON output path.")
    parser.add_argument("--config_path", default="config.json")
    parser.add_argument("--demucs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--padding", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.input_audio).expanduser().resolve()
    stage_dir = stage_common.default_stage_dir(audio_path)
    out_audio = Path(args.out_audio) if args.out_audio else stage_dir / "cleaned_audio.wav"
    out_flags = Path(args.out_flags) if args.out_flags else stage_dir / "segment_flags.json"

    import main_original_ASR_MoE as pipeline

    cfg = pipeline.load_cfg(args.config_path)
    logger = pipeline.Logger.get_logger()
    pipeline.cfg = cfg
    pipeline.logger = logger

    diarization = stage_common.load_json(args.diarization_json)
    segments = diarization["segments"]
    sample_rate = int(diarization.get("sample_rate") or cfg["entrypoint"]["SAMPLE_RATE"])
    audio_info = stage_common.load_audio_info(audio_path, sample_rate)

    panns_model = None
    if args.demucs:
        try:
            panns_data_dir = Path(__file__).resolve().parent.parent / "panns_data"
            panns_data_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = panns_data_dir / "Cnn14_mAP=0.431.pth"
            panns_model = pipeline.AudioTagging(
                checkpoint_path=str(checkpoint_path),
                device="cuda" if pipeline.torch.cuda.is_available() else "cpu",
            )
        except Exception as exc:
            logger.warning(f"PANNs unavailable; Demucs cleaning will be skipped: {exc}")
            panns_model = None

    start_time = time.time()
    cleaned_audio, segment_demucs_flags = pipeline.preprocess_segments_with_demucs(
        segments,
        audio_info,
        panns_model=panns_model,
        use_demucs=args.demucs and panns_model is not None,
        padding=args.padding,
    )
    processing_time = time.time() - start_time

    stage_common.write_wav(out_audio, cleaned_audio["waveform"], cleaned_audio["sample_rate"])
    stage_common.dump_json(
        {
            "audio_path": str(out_audio.resolve()),
            "source_audio_path": str(audio_path),
            "audio_name": diarization.get("audio_name") or stage_common.audio_name_from_path(audio_path),
            "sample_rate": cleaned_audio["sample_rate"],
            "audio_duration_seconds": len(cleaned_audio["waveform"]) / cleaned_audio["sample_rate"],
            "segments": stage_common.clean_segments_for_json(segments),
            "segment_demucs_flags": segment_demucs_flags,
            "metadata": {
                "stage": "music_clean",
                "enabled": bool(args.demucs and panns_model is not None),
                "processing_time_seconds": processing_time,
                "padding_seconds": args.padding,
            },
        },
        out_flags,
    )
    print(f"Stage 02 complete: {out_audio} / {out_flags}")


if __name__ == "__main__":
    main()
