# Staged Pipeline for Kaggle

This mode splits the original all-in-one pipeline into separate Python
processes. Each stage writes intermediate files, so Kaggle can release GPU
memory when the previous process exits.

## Stage Outputs

```text
run_full/
  00_input/
    full.wav
    preview_input_30s.wav
  01_diarization/
    diarization.json
    trace_vad_chunks.json
    vad_chunks/
  02_music_clean/
    cleaned_audio.wav
    segment_flags.json
    preview_cleaned_30s.wav
  03_overlap/
    segments.json
    separated_segments/
  04_asr/
    transcript.json
  05_export/
    final/
      data_audio.json
      data_audio/
        00000_SPEAKER_00.mp3
  06_eval/
    eval_report.json
    eval_report.md
  logs/
  preview/
```

## Kaggle Setup

Run from `podcast-pipeline` after installing dependencies and setting
`huggingface_token` in `config.json`.

```bash
python stage_01_diarize.py \
  --input_audio /kaggle/working/run_full/00_input/full.wav
```

For a low-VRAM first run, skip Demucs:

```bash
python stage_02_music_clean.py \
  --input_audio /kaggle/working/run_full/00_input/full.wav \
  --diarization_json /kaggle/working/run_full/01_diarization/diarization.json \
  --no-demucs
```

For a low-VRAM first run, skip SepReformer:

```bash
python stage_03_overlap_separate.py \
  --cleaned_audio /kaggle/working/run_full/02_music_clean/cleaned_audio.wav \
  --segment_flags_json /kaggle/working/run_full/02_music_clean/segment_flags.json \
  --no-sepreformer
```

Run Whisper-only ASR first:

```bash
python stage_04_asr.py \
  --segments_json /kaggle/working/run_full/03_overlap/segments.json \
  --no-ASRMoE \
  --no-whisperx_word_timestamps
```

Export final JSON and MP3 segment files:

```bash
python stage_05_export.py \
  --transcript_json /kaggle/working/run_full/04_asr/transcript.json
```

Evaluate the run outputs:

```bash
python stage_06_eval.py \
  --run_dir /kaggle/working/run_full \
  --print_markdown
```

## Enabling Heavier Parts

Enable one heavy option at a time:

```bash
# Stage 02: background music removal
python stage_02_music_clean.py \
  --input_audio /kaggle/working/run_full/00_input/full.wav \
  --diarization_json /kaggle/working/run_full/01_diarization/diarization.json \
  --demucs

# Stage 03: overlapping speech separation
python stage_03_overlap_separate.py \
  --cleaned_audio /kaggle/working/run_full/02_music_clean/cleaned_audio.wav \
  --segment_flags_json /kaggle/working/run_full/02_music_clean/segment_flags.json \
  --sepreformer \
  --min_sepreformer_overlap 1.0 \
  --min_sepreformer_segment 1.0

# Stage 04: multi-model ASR ensemble
python stage_04_asr.py \
  --segments_json /kaggle/working/run_full/03_overlap/segments.json \
  --ASRMoE \
  --asr_quality_guard \
  --asr_micro_segment_seconds 0.5 \
  --asr_short_segment_seconds 1.0 \
  --asr_vi_agreement_threshold 0.75
```

Quality guard knobs:

- `--min_sepreformer_overlap`: avoid separating tiny overlaps that are too short
  for stable speaker assignment.
- `--min_sepreformer_segment`: avoid separating pairs where one segment is too
  short for reliable embedding/ASR.
- `--asr_quality_guard`: after ROVER voting, replace obvious short-segment
  Whisper hallucinations with Vietnamese-model consensus/candidates, or drop
  micro boilerplate such as `Thank you`.

## Notes

- The staged mode is for inference only.
- Use short audio first, for example 30 to 60 seconds.
- Kaggle T4/P100 may not handle `--ASRMoE`, `--demucs`, and `--sepreformer`
  together. Run those parts separately and only for files that need them.
- `stage_04_asr.py` does not run Qwen3-Omni captioning. The original main script
  calls that through a local API server, which is not a good fit for Kaggle.
