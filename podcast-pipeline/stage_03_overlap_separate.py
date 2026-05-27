from __future__ import annotations

import argparse
import time
from pathlib import Path

import stage_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 03: optionally separate overlapping speech with SepReformer.")
    parser.add_argument("--cleaned_audio", required=True, help="Cleaned WAV from stage_02_music_clean.py.")
    parser.add_argument("--segment_flags_json", required=True, help="Segment flags JSON from stage_02_music_clean.py.")
    parser.add_argument("--out_segments", default="", help="Segments JSON output path.")
    parser.add_argument("--separated_dir", default="", help="Directory for separated segment WAV files.")
    parser.add_argument("--config_path", default="config.json")
    parser.add_argument("--sepreformer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap_threshold", type=float, default=0.2)
    parser.add_argument("--sepreformer_path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.cleaned_audio).expanduser().resolve()
    stage_dir = audio_path.parent
    out_segments = Path(args.out_segments) if args.out_segments else stage_dir / "segments.json"
    separated_dir = Path(args.separated_dir) if args.separated_dir else stage_dir / "separated_segments"

    import main_original_ASR_MoE as pipeline

    cfg = pipeline.load_cfg(args.config_path)
    logger = pipeline.Logger.get_logger()
    pipeline.cfg = cfg
    pipeline.logger = logger

    segment_data = stage_common.load_json(args.segment_flags_json)
    segments = segment_data["segments"]
    sample_rate = int(segment_data.get("sample_rate") or cfg["entrypoint"]["SAMPLE_RATE"])
    audio_info = stage_common.load_audio_info(audio_path, sample_rate)

    device_name = "cuda" if pipeline.torch.cuda.is_available() else "cpu"
    device = pipeline.torch.device(device_name)
    pipeline.device_name = device_name
    pipeline.device = device

    processing_time = 0.0
    if args.sepreformer:
        sepreformer_path = (
            Path(args.sepreformer_path).expanduser().resolve()
            if args.sepreformer_path
            else Path(__file__).resolve().parent.parent / "SepReformer"
        )

        start_time = time.time()
        try:
            from pyannote.audio import Model as PyannoteModel

            embedding_model = PyannoteModel.from_pretrained(
                "pyannote/embedding",
                use_auth_token=cfg.get("huggingface_token"),
            ).to(device)
            separator = pipeline.SepReformerSeparator(
                sepreformer_path=str(sepreformer_path),
                device=device,
            )
            audio_info, segments = pipeline.process_overlapping_segments_with_separation(
                segments,
                audio_info,
                overlap_threshold=args.overlap_threshold,
                separator=separator,
                embedding_model=embedding_model,
            )
        except Exception as exc:
            logger.warning(f"SepReformer stage skipped after load/process failure: {exc}")
        processing_time = time.time() - start_time

    separated_dir.mkdir(parents=True, exist_ok=True)
    clean_segments = []
    for idx, segment in enumerate(segments):
        clean_segment = stage_common.clean_segment_for_json(segment)
        if segment.get("sepreformer", False) and "enhanced_audio" in segment:
            seg_index = clean_segment.get("index") or stage_common.normalized_index(idx)
            speaker = clean_segment.get("speaker", "Unknown")
            enhanced_path = separated_dir / f"{seg_index}_{speaker}.wav"
            stage_common.write_wav(enhanced_path, segment["enhanced_audio"], sample_rate)
            clean_segment["enhanced_audio_path"] = str(enhanced_path.resolve())
            clean_segment["is_separated"] = True
        else:
            clean_segment["is_separated"] = False
        clean_segments.append(clean_segment)

    stage_common.dump_json(
        {
            "audio_path": str(audio_path),
            "source_audio_path": segment_data.get("source_audio_path") or str(audio_path),
            "audio_name": segment_data.get("audio_name") or stage_common.audio_name_from_path(audio_path),
            "sample_rate": sample_rate,
            "audio_duration_seconds": len(audio_info["waveform"]) / sample_rate,
            "segments": clean_segments,
            "segment_demucs_flags": segment_data.get("segment_demucs_flags", [False] * len(clean_segments)),
            "metadata": {
                "stage": "overlap_separate",
                "enabled": bool(args.sepreformer),
                "processing_time_seconds": processing_time,
                "overlap_threshold_seconds": args.overlap_threshold,
            },
        },
        out_segments,
    )
    print(f"Stage 03 complete: {out_segments}")


if __name__ == "__main__":
    main()
