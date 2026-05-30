from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import stage_common


DEFAULT_INITIAL_PROMPT = (
    "Cuộc trò chuyện podcast tiếng Việt tự nhiên. "
    "Có thể có các từ đệm như ờ, ừ, à, dạ, vâng, rồi, thì, là."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 04: run ASR on staged segments.")
    parser.add_argument("--segments_json", required=True, help="Segments JSON from stage_03_overlap_separate.py.")
    parser.add_argument("--audio", default="", help="Override audio path. Defaults to segments_json audio_path.")
    parser.add_argument("--out", default="", help="Transcript JSON output path.")
    parser.add_argument("--config_path", default="config.json")
    parser.add_argument("--ASRMoE", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--whisperx_word_timestamps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--initprompt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--whisper_arch", default="large-v3")
    parser.add_argument("--compute_type", default="float16")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--asr_quality_guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable conservative ASR post-vote guard for short-segment hallucinations.",
    )
    parser.add_argument(
        "--asr_micro_segment_seconds",
        type=float,
        default=0.5,
        help="Segments shorter than this are treated as micro segments by the ASR quality guard.",
    )
    parser.add_argument(
        "--asr_short_segment_seconds",
        type=float,
        default=1.0,
        help="Segments shorter than this are treated as short segments by the ASR quality guard.",
    )
    parser.add_argument(
        "--asr_vi_agreement_threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for PhoWhisper/ChunkFormer agreement in the ASR quality guard.",
    )
    parser.add_argument(
        "--asr_context_pad_before",
        type=float,
        default=0.0,
        help="Seconds of audio context to prepend during ASR while keeping original output timestamps.",
    )
    parser.add_argument(
        "--asr_context_pad_after",
        type=float,
        default=0.0,
        help="Seconds of audio context to append during ASR while keeping original output timestamps.",
    )
    parser.add_argument(
        "--whisper_hotwords",
        default="",
        help="Optional comma-separated or free-text hotwords passed to faster-whisper.",
    )
    parser.add_argument(
        "--whisper_device_index",
        type=int,
        default=0,
        help="CUDA device index for Whisper/faster-whisper. Ignored on CPU.",
    )
    parser.add_argument(
        "--nemo_device_index",
        type=int,
        default=1,
        help="Deprecated alias for --vi_asr_device_index.",
    )
    parser.add_argument(
        "--vi_asr_device_index",
        type=int,
        default=None,
        help="CUDA device index for PhoWhisper and ChunkFormer when --ASRMoE is enabled. Ignored on CPU.",
    )
    return parser.parse_args()


def find_source_segment(result: dict, segments: list[dict], fallback_index: int) -> dict | None:
    start = float(result.get("start", -1))
    end = float(result.get("end", -1))
    for segment in segments:
        seg_start = float(segment.get("start", -1))
        seg_end = float(segment.get("end", -1))
        if start >= seg_start - 0.01 and end <= seg_end + 0.01:
            return segment
    if fallback_index < len(segments):
        return segments[fallback_index]
    return None


def main() -> None:
    args = parse_args()
    segment_data = stage_common.load_json(args.segments_json)
    audio_path = Path(args.audio or segment_data["audio_path"]).expanduser().resolve()
    out_path = Path(args.out) if args.out else Path(args.segments_json).resolve().parent / "transcript.json"

    import main_original_ASR_MoE as pipeline

    cfg = pipeline.load_cfg(args.config_path)
    logger = pipeline.Logger.get_logger()
    pipeline.cfg = cfg
    pipeline.logger = logger
    pipeline.args = SimpleNamespace(
        ASRMoE=args.ASRMoE,
        whisperx_word_timestamps=args.whisperx_word_timestamps,
    )

    device_name = "cuda" if pipeline.torch.cuda.is_available() else "cpu"
    device = pipeline.torch.device(device_name)
    pipeline.device_name = device_name
    pipeline.device = device
    compute_type = args.compute_type if device_name == "cuda" else "int8"
    cuda_device_count = pipeline.torch.cuda.device_count() if device_name == "cuda" else 0
    whisper_device_index = 0
    vi_asr_device = device
    vi_asr_device_index = args.vi_asr_device_index if args.vi_asr_device_index is not None else args.nemo_device_index

    if device_name == "cuda":
        if 0 <= args.whisper_device_index < cuda_device_count:
            whisper_device_index = args.whisper_device_index

        if 0 <= vi_asr_device_index < cuda_device_count:
            selected_vi_asr_device_index = vi_asr_device_index
        else:
            selected_vi_asr_device_index = whisper_device_index

        vi_asr_device = pipeline.torch.device(f"cuda:{selected_vi_asr_device_index}")
        logger.info(
            "ASR device plan: Whisper on cuda:%s, Vietnamese ASRMoE models on %s, CUDA devices=%s",
            whisper_device_index,
            vi_asr_device,
            cuda_device_count,
        )

    sample_rate = int(segment_data.get("sample_rate") or cfg["entrypoint"]["SAMPLE_RATE"])
    audio_info = stage_common.load_audio_info(audio_path, sample_rate)
    segments = segment_data["segments"]

    for segment in segments:
        enhanced_path = segment.get("enhanced_audio_path")
        if enhanced_path:
            enhanced_audio, _ = stage_common.load_wav_mono(enhanced_path, target_sample_rate=sample_rate)
            segment["enhanced_audio"] = enhanced_audio

    asr_options = {}
    if args.initprompt:
        asr_options["initial_prompt"] = DEFAULT_INITIAL_PROMPT
    if args.whisper_hotwords.strip():
        asr_options["hotwords"] = args.whisper_hotwords.strip()
    if args.whisperx_word_timestamps:
        asr_options["word_timestamps"] = True

    pipeline.asr_model = pipeline.whisper_asr.load_asr_model(
        args.whisper_arch,
        device_name,
        device_index=whisper_device_index,
        compute_type=compute_type,
        threads=args.threads,
        language="vi",
        asr_options=asr_options if asr_options else None,
    )

    if args.ASRMoE:
        pipeline.phowhisper_model = pipeline.vietnamese_asr.load_phowhisper_model(device=vi_asr_device)
        pipeline.chunkformer_model = pipeline.vietnamese_asr.load_chunkformer_model(device=vi_asr_device)

    segment_demucs_flags = segment_data.get("segment_demucs_flags") or [False] * len(segments)
    start_time = time.time()
    if args.ASRMoE:
        asr_result, whisper_time, alignment_time = pipeline.asr_MoE(
            segments,
            audio_info,
            segment_demucs_flags=segment_demucs_flags,
            enable_word_timestamps=args.whisperx_word_timestamps,
            device=device_name,
            asr_quality_guard=args.asr_quality_guard,
            asr_micro_segment_seconds=args.asr_micro_segment_seconds,
            asr_short_segment_seconds=args.asr_short_segment_seconds,
            asr_vi_agreement_threshold=args.asr_vi_agreement_threshold,
            asr_context_pad_before=args.asr_context_pad_before,
            asr_context_pad_after=args.asr_context_pad_after,
        )
    else:
        asr_result = pipeline.asr(segments, audio_info)
        whisper_time = time.time() - start_time
        alignment_time = 0.0
    total_time = time.time() - start_time

    clean_results = []
    for idx, result in enumerate(asr_result):
        clean_result = stage_common.clean_segment_for_json(result)
        clean_result.setdefault("index", stage_common.normalized_index(idx))

        source = find_source_segment(result, segments, idx)
        if source is not None:
            clean_result["source_index"] = source.get("index")
            clean_result["source_start"] = source.get("start")
            clean_result["source_end"] = source.get("end")
            if source.get("enhanced_audio_path"):
                clean_result["enhanced_audio_path"] = source["enhanced_audio_path"]
                clean_result["is_separated"] = bool(source.get("is_separated"))
            if "demucs" not in clean_result and idx < len(segment_demucs_flags):
                clean_result["demucs"] = bool(segment_demucs_flags[idx])

        clean_results.append(clean_result)

    audio_duration = len(audio_info["waveform"]) / audio_info["sample_rate"]
    stage_common.dump_json(
        {
            "audio_path": str(audio_path),
            "source_audio_path": segment_data.get("source_audio_path") or str(audio_path),
            "audio_name": segment_data.get("audio_name") or stage_common.audio_name_from_path(audio_path),
            "sample_rate": audio_info["sample_rate"],
            "audio_duration_seconds": audio_duration,
            "segments": clean_results,
            "metadata": {
                "stage": "asr",
                "asr_moe": bool(args.ASRMoE),
                "whisper_arch": args.whisper_arch,
                "processing_time_seconds": total_time,
                "whisper_processing_time_seconds": whisper_time,
                "whisper_rt_factor": whisper_time / audio_duration if audio_duration > 0 else 0,
                "alignment_processing_time_seconds": alignment_time,
                "word_timestamps_enabled": bool(args.whisperx_word_timestamps),
                "device": device_name,
                "cuda_device_count": cuda_device_count,
                "whisper_device_index": whisper_device_index if device_name == "cuda" else None,
                "vi_asr_device": str(vi_asr_device) if args.ASRMoE else None,
                "asr_quality_guard": bool(args.asr_quality_guard),
                "asr_micro_segment_seconds": args.asr_micro_segment_seconds,
                "asr_short_segment_seconds": args.asr_short_segment_seconds,
                "asr_vi_agreement_threshold": args.asr_vi_agreement_threshold,
                "asr_context_pad_before": args.asr_context_pad_before,
                "asr_context_pad_after": args.asr_context_pad_after,
                "whisper_hotwords": args.whisper_hotwords.strip(),
            },
        },
        out_path,
    )
    print(f"Stage 04 complete: {out_path}")


if __name__ == "__main__":
    main()
