# Staged Pipeline for Kaggle

This mode splits the original all-in-one pipeline into separate Python
processes. Each stage writes intermediate files, so Kaggle can release GPU
memory when the previous process exits.

## Stage Outputs

```text
_staged/<audio_name>/
  diarization.json
  cleaned_audio.wav
  segment_flags.json
  segments.json
  separated_segments/
  transcript.json
  final/
    <audio_name>.json
    <audio_name>/
      00000_SPEAKER_00.mp3
```

## Kaggle Setup

Run from `podcast-pipeline` after installing dependencies and setting
`huggingface_token` in `config.json`.

```bash
python stage_01_diarize.py \
  --input_audio /kaggle/working/audio/test.wav
```

For a low-VRAM first run, skip Demucs:

```bash
python stage_02_music_clean.py \
  --input_audio /kaggle/working/audio/test.wav \
  --diarization_json /kaggle/working/audio/_staged/test/diarization.json \
  --no-demucs
```

For a low-VRAM first run, skip SepReformer:

```bash
python stage_03_overlap_separate.py \
  --cleaned_audio /kaggle/working/audio/_staged/test/cleaned_audio.wav \
  --segment_flags_json /kaggle/working/audio/_staged/test/segment_flags.json \
  --no-sepreformer
```

Run Whisper-only ASR first:

```bash
python stage_04_asr.py \
  --segments_json /kaggle/working/audio/_staged/test/segments.json \
  --no-ASRMoE \
  --no-whisperx_word_timestamps
```

Export final JSON and MP3 segment files:

```bash
python stage_05_export.py \
  --transcript_json /kaggle/working/audio/_staged/test/transcript.json
```

## Enabling Heavier Parts

Enable one heavy option at a time:

```bash
# Stage 02: background music removal
python stage_02_music_clean.py \
  --input_audio /kaggle/working/audio/test.wav \
  --diarization_json /kaggle/working/audio/_staged/test/diarization.json \
  --demucs

# Stage 03: overlapping speech separation
python stage_03_overlap_separate.py \
  --cleaned_audio /kaggle/working/audio/_staged/test/cleaned_audio.wav \
  --segment_flags_json /kaggle/working/audio/_staged/test/segment_flags.json \
  --sepreformer

# Stage 04: multi-model ASR ensemble
python stage_04_asr.py \
  --segments_json /kaggle/working/audio/_staged/test/segments.json \
  --ASRMoE
```

## Notes

- The staged mode is for inference only.
- Use short audio first, for example 30 to 60 seconds.
- Kaggle T4/P100 may not handle `--ASRMoE`, `--demucs`, and `--sepreformer`
  together. Run those parts separately and only for files that need them.
- `stage_04_asr.py` does not run Qwen3-Omni captioning. The original main script
  calls that through a local API server, which is not a good fit for Kaggle.

