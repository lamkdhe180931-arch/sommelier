from __future__ import annotations
import os
import argparse
import shutil
import time
import json
import datetime
import copy
from collections import Counter
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
import models.silero_vad as silero_vad

warnings.filterwarnings("ignore")
audio_count = 0

# Giới hạn mỗi chunk diarization để tránh model xử lý audio quá dài một lần.
# Ưu tiên cắt tại khoảng lặng do VAD tìm được để hạn chế cắt ngang câu nói.
MAX_DIA_CHUNK_DURATION = 3 * 60  # giây; giữ dưới ngưỡng dài để Sortformer ổn định hơn
MIN_SPLIT_SILENCE = 1  # giây im lặng tối thiểu để được chọn làm điểm cắt
SILERO_MIN_SILENCE_DURATION_MS = 100
MIN_EMBED_DURATION = 0.5  # giây; segment ngắn hơn mức này sẽ bỏ qua embedding
QWEN_3_OMNI_PORT = "11500"

def _apply_sortformer_segment_padding_from_args(
    df: pd.DataFrame, args, logger, audio_duration: float | None = None
) -> pd.DataFrame:
    """
    Dịch nhẹ mốc bắt đầu/kết thúc của segment Sortformer sau khi model trả kết quả.

    Mục đích:
    - Một số cấu hình nội bộ của NeMo có thể không nhận override như mong muốn.
    - Khi bật --sortformer-param, ta chỉnh trực tiếp DataFrame đầu ra để thay đổi
      timing có hiệu lực rõ ràng.
    - start/end luôn được clamp để không âm và không vượt quá thời lượng audio.
    """
    if df is None or df.empty:
        return df
    if not getattr(args, "sortformer_param", False):
        return df

    pad_onset = float(getattr(args, "sortformer_pad_onset", 0.0))
    pad_offset = float(getattr(args, "sortformer_pad_offset", 0.0))

    if pad_onset == 0.0 and pad_offset == 0.0:
        return df

    if getattr(args, "_logged_sortformer_pad", None) != True:
        logger.info(f"Applying Sortformer padding -> pad_onset: {pad_onset}, pad_offset: {pad_offset}")
        setattr(args, "_logged_sortformer_pad", True)

    df = df.copy()
    df["start"] = (df["start"].astype(float) + pad_onset).clip(lower=0.0)
    df["end"] = df["end"].astype(float) + pad_offset
    if audio_duration is not None and audio_duration > 0:
        df["end"] = df["end"].clip(lower=0.0, upper=float(audio_duration))
    else:
        df["end"] = df["end"].clip(lower=0.0)
    df["end"] = df[["start", "end"]].max(axis=1)

    return df

def custom_binarize(probs: np.ndarray, frame_shift: float, onset: float, offset: float, min_duration_on: float, min_duration_off: float):
    """
    Áp dụng Binarize theo phong cách Pyannote cho tensor prob của Sortformer.
    probs: numpy array có shape (num_frames, num_speakers) chứa xác suất [0, 1]
    """
    if probs.ndim != 2:
        return []
    
    num_frames, num_speakers = probs.shape
    segments = []
    for spk_idx in range(num_speakers):
        spk_probs = probs[:, spk_idx]
        active = False
        start_frame = 0
        spk_segments = []
        for i in range(num_frames):
            if not active and spk_probs[i] >= onset:
                active = True
                start_frame = i
            elif active and spk_probs[i] < offset:
                active = False
                end_frame = i
                spk_segments.append([start_frame * frame_shift, end_frame * frame_shift])
        if active:
            spk_segments.append([start_frame * frame_shift, num_frames * frame_shift])
        
        # Merge các đoạn gần nhau (min_duration_off)
        merged_segments = []
        for seg in spk_segments:
            if not merged_segments:
                merged_segments.append(seg)
            else:
                if seg[0] - merged_segments[-1][1] <= min_duration_off:
                    merged_segments[-1][1] = seg[1]
                else:
                    merged_segments.append(seg)
        
        # Xoá các đoạn quá ngắn (min_duration_on)
        for seg in merged_segments:
            if seg[1] - seg[0] >= min_duration_on:
                segments.append({
                    'speaker': f"SPEAKER_{spk_idx:02d}", 
                    'start': float(seg[0]), 
                    'end': float(seg[1])
                })
                
    return segments

def sortformer_dia(predicted_segments):
    """
    Chuyển output thô của NeMo Sortformer thành DataFrame diarization chuẩn.

    Sortformer trả segment dạng chuỗi "start end SPEAKER_xx"; hàm này parse về
    cột start/end/speaker để các bước sau dùng chung format với pyannote.
    """
    lists = [x for x in predicted_segments if isinstance(x, (list, tuple))]
    if not lists:
        lists = predicted_segments
    segs = [s for sub in lists for s in sub]

    rows = []
    for idx, seg in enumerate(segs):
        start_s, end_s, sp = seg.split()
        start, end = float(start_s), float(end_s)
        # Chuyển SPEAKER format về dạng SPEAKER_00, SPEAKER_01...
        num = int(sp.split('_')[1])
        speaker = f"SPEAKER_{num:02d}"
        # Label chỉ dùng để tương thích format DataFrame cũ.
        label = chr(ord('A') + idx)
        # Format timestamp giống pyannote segment string.
        def fmt(sec):
            td = datetime.timedelta(seconds=sec)
            hrs = td.seconds // 3600 + td.days * 24
            mins = (td.seconds // 60) % 60
            secs = td.seconds % 60
            ms = int(td.microseconds / 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
        segment_str = f"[ {fmt(start)} --> {fmt(end)}]"
        rows.append({
            'segment': segment_str,
            'label': label,
            'speaker': speaker,
            'start': start,
            'end': end
        })

    df = pd.DataFrame(rows, columns=['segment','label','speaker','start','end'])
    df = df.sort_values(by='start').reset_index(drop=True)
    return df

def df_to_list(df: pd.DataFrame) -> list[dict]:
    """
    Chuyển DataFrame diarization thành list dict dùng cho ASR/export.

    Mỗi item gồm:
      - index: chuỗi 5 chữ số.
      - start/end: giây trên timeline file gốc.
      - speaker: speaker label.
      - các metadata public khác như identity_low_confidence nếu có.
    """
    records = []
    for i, row in df.iterrows():
        record = {
            'index': f"{i:05d}",
            'start': float(row['start']),
            'end': float(row['end']),
            'speaker': row['speaker']
        }
        for column in df.columns:
            if column in {"segment", "label", "start", "end", "speaker"}:
                continue
            if str(column).startswith("_"):
                continue
            value = row[column]
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            record[column] = value
        records.append(record)
    return records


def _as_trace_dict(value) -> dict:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return {}


def _segment_duration_seconds(start, end) -> float:
    try:
        return max(0.0, float(end) - float(start))
    except (TypeError, ValueError):
        return 0.0


def _round_similarity(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0:
        return None
    return round(numeric, 6)


def _similarity_gap(best_similarity, second_best_similarity):
    best = _round_similarity(best_similarity)
    second = _round_similarity(second_best_similarity)
    if best is None or second is None:
        return None
    return round(best - second, 6)


def split_long_segments(segment_list, max_duration=30.0):
    """
    Cắt các segment dài hơn max_duration theo thời gian, không dùng VAD.

    Mục đích là tránh đưa một đoạn quá dài vào ASR, vì vừa chậm vừa dễ làm model
    mất tập trung hoặc lệch timestamp.

    Tham số:
        segment_list: danh sách segment cần xử lý.
        max_duration: độ dài tối đa mỗi segment, tính bằng giây.

    Trả về:
        Danh sách segment mới đã đánh lại index.
    """
    new_segments = []
    new_index = 0

    for segment in segment_list:
        start_time = segment['start']
        end_time = segment['end']
        duration = end_time - start_time

        # Segment đủ ngắn thì giữ nguyên và chỉ đánh lại index.
        if duration <= max_duration:
            copied_segment = copy.deepcopy(segment)
            copied_segment['index'] = str(new_index).zfill(5)
            new_segments.append(copied_segment)
            new_index += 1
        # Segment quá dài thì cắt thành nhiều chunk nối tiếp nhau.
        else:
            split_ranges = []
            current_start = start_time
            while current_start < end_time:
                # Điểm kết thúc chunk tiếp theo không được vượt end gốc.
                chunk_end = min(current_start + max_duration, end_time)
                split_ranges.append((current_start, chunk_end))
                # Chunk kế tiếp bắt đầu ngay tại end của chunk hiện tại.
                current_start = chunk_end

            split_part_count = len(split_ranges)
            split_from_index = str(segment.get("index", ""))
            for split_part, (current_start, chunk_end) in enumerate(split_ranges):
                split_segment = copy.deepcopy(segment)
                split_segment['index'] = str(new_index).zfill(5)
                split_segment['start'] = round(current_start, 3)
                split_segment['end'] = round(chunk_end, 3)
                trace = _as_trace_dict(split_segment.get("stage1_trace"))
                trace["split"] = {
                    "split_from_index": split_from_index,
                    "split_part": split_part,
                    "split_part_count": split_part_count,
                    "original_start": round(float(start_time), 6),
                    "original_end": round(float(end_time), 6),
                    "max_duration": round(float(max_duration), 6),
                }
                split_segment["stage1_trace"] = trace
                new_segments.append(split_segment)
                new_index += 1
                
    return new_segments

def _build_silence_intervals(waveform, sample_rate, min_silence):
    """
    Dùng VAD để tìm các khoảng im lặng có thể dùng làm điểm cắt chunk.

    Trả về:
        (total_duration, silence_intervals)
        silence_intervals là list các khoảng (start, end) tính bằng giây.
    """
    vad_model = globals().get("vad")
    if vad_model is None:
        return len(waveform) / sample_rate, []

    if len(waveform) == 0:
        return 0.0, []

    resampled = librosa.resample(
        waveform, orig_sr=sample_rate, target_sr=silero_vad.SAMPLING_RATE
    )
    if resampled.size == 0:
        return len(waveform) / sample_rate, []

    speech_ts = vad_model.get_speech_timestamps(
        resampled,
        vad_model.vad_model,
        sampling_rate=silero_vad.SAMPLING_RATE,
        min_silence_duration_ms=SILERO_MIN_SILENCE_DURATION_MS,
    )
    total_duration = len(waveform) / sample_rate
    if not speech_ts:
        return total_duration, [(0.0, total_duration)]

    silence = []
    first_start = speech_ts[0]["start"] / silero_vad.SAMPLING_RATE
    if first_start >= min_silence:
        silence.append((0.0, first_start))

    for prev_seg, next_seg in zip(speech_ts[:-1], speech_ts[1:]):
        sil_start = prev_seg["end"] / silero_vad.SAMPLING_RATE
        sil_end = next_seg["start"] / silero_vad.SAMPLING_RATE
        if sil_end - sil_start >= min_silence:
            silence.append((sil_start, sil_end))

    last_end = speech_ts[-1]["end"] / silero_vad.SAMPLING_RATE
    trailing = total_duration - last_end
    if trailing >= min_silence:
        silence.append((last_end, last_end + trailing))
    return total_duration, silence

def _build_chunk_ranges(total_duration, silence_intervals, max_duration):
    """
    Tạo danh sách khoảng chunk cho diarization.

    Ưu tiên:
    - Nếu audio ngắn hơn max_duration: dùng một chunk.
    - Nếu có khoảng lặng: cắt tại midpoint của khoảng lặng gần giới hạn nhất.
    - Nếu không có khoảng lặng phù hợp: hard cut tại max_duration.

    Mục tiêu là giữ mỗi chunk đủ ngắn cho Sortformer nhưng giảm khả năng cắt
    giữa câu nói.
    """
    epsilon = 1e-3
    if total_duration <= max_duration + epsilon:
        return [(0.0, total_duration)]

    # Dùng midpoint của mỗi khoảng lặng làm ứng viên điểm cắt.
    silence_points = sorted([(start + end) / 2.0 for start, end in silence_intervals])

    if not silence_points:
        # Không có khoảng lặng: dự phòng bằng cách cắt cứng theo max_duration.
        chunk_ranges = []
        chunk_start = 0.0
        while chunk_start < total_duration - epsilon:
            chunk_end = min(chunk_start + max_duration, total_duration)
            chunk_ranges.append((chunk_start, chunk_end))
            chunk_start = chunk_end
        return chunk_ranges if chunk_ranges else [(0.0, total_duration)]

    chunk_ranges = []
    chunk_start = 0.0

    while chunk_start < total_duration - epsilon:
        # Tìm điểm lặng xa nhất nhưng vẫn nằm trong giới hạn max_duration.
        limit = chunk_start + max_duration

        # Chỉ xét điểm cắt nằm sau chunk_start và không vượt limit.
        candidates = [p for p in silence_points if chunk_start + epsilon < p <= limit]

        if candidates:
            # Dùng điểm lặng xa nhất để chunk dài nhất có thể nhưng vẫn an toàn.
            chunk_end = candidates[-1]
        else:
            # Không có điểm lặng trong giới hạn. Nếu có điểm lặng ở xa hơn thì
            # vẫn phải hard cut tại limit để không vượt max_duration.
            future_candidates = [p for p in silence_points if p > limit]
            if future_candidates:
                chunk_end = min(limit, total_duration)
            else:
                # Không còn điểm lặng nào phía sau, lấy tới cuối audio.
                chunk_end = total_duration

        # Bảo đảm vòng lặp luôn tiến, tránh chunk có độ dài gần 0.
        if chunk_end - chunk_start < epsilon:
            chunk_end = min(chunk_start + max_duration, total_duration)
            if chunk_end - chunk_start < epsilon:
                break

        chunk_ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end

    return chunk_ranges if chunk_ranges else [(0.0, total_duration)]

def _extract_speaker_embedding(
    audio_info,
    start: float,
    end: float,
    embedder: Inference | None,
    sample_window: float = 2.0,
    min_duration: float = 0.5,
):
    """
    Trích một speaker embedding từ một đoạn audio gốc.

    Hàm dùng cho việc nối speaker label giữa các chunk diarization. Với segment
    dài, chỉ lấy cửa sổ giữa đoạn để giảm nhiễu và tiết kiệm thời gian.
    """
    if embedder is None:
        return None

    waveform = audio_info.get("waveform")
    sample_rate = audio_info.get("sample_rate")
    if waveform is None or sample_rate is None:
        return None

    total_duration = len(waveform) / sample_rate
    start = max(0.0, min(start, total_duration))
    end = max(start, min(end, total_duration))
    duration = end - start
    if duration < min_duration:
        return None

    # Với segment dài, lấy cửa sổ quanh giữa đoạn để embedding ổn định hơn.
    if duration > sample_window:
        center = (start + end) / 2.0
        start = center - sample_window / 2.0
        end = center + sample_window / 2.0

    start_idx = int(start * sample_rate)
    end_idx = int(end * sample_rate)
    segment = waveform[start_idx:end_idx]
    if segment.size == 0:
        return None

    target_sr = getattr(embedder, "sample_rate", 16000)
    try:
        if sample_rate != target_sr:
            segment = librosa.resample(segment, orig_sr=sample_rate, target_sr=target_sr)
    except Exception:
        # Nếu resample lỗi, dùng lại segment gốc.
        target_sr = sample_rate

    # pyannote Inference nhận input dạng dict có waveform và sample_rate.
    try:
        torch_seg = torch.as_tensor(segment, dtype=torch.float32).unsqueeze(0)
        emb = embedder({"waveform": torch_seg, "sample_rate": target_sr})
    except Exception:
        return None
    if emb is None:
        return None
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    if isinstance(emb, np.ndarray) and emb.ndim > 1:
        emb = emb.mean(axis=0)
    return emb

def _cosine_similarity(vec_a, vec_b):
    """Tính cosine similarity; trả -1.0 nếu vector không hợp lệ."""
    if vec_a is None or vec_b is None:
        return -1.0
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return -1.0
    return float(np.dot(vec_a, vec_b) / denom)

def _chunk_clean_identity_mask(chunk_df: pd.DataFrame, min_identity_duration: float) -> pd.Series:
    """
    Mark rows that are safe to use for speaker identity learning.

    A row must be long enough and must not overlap a different local speaker in
    the same chunk. Short/overlap rows can still be kept in the transcript, but
    they are not trusted as identity evidence.
    """
    if chunk_df is None or chunk_df.empty:
        return pd.Series(dtype=bool)

    clean_flags = []
    for _, row in chunk_df.iterrows():
        start = float(row["start"])
        end = float(row["end"])
        speaker = row["speaker"]
        duration = max(0.0, end - start)

        overlaps_other = False
        for _, other in chunk_df.iterrows():
            if other["speaker"] == speaker:
                continue
            overlap = min(end, float(other["end"])) - max(start, float(other["start"]))
            if overlap > 0:
                overlaps_other = True
                break

        clean_flags.append(duration >= min_identity_duration and not overlaps_other)

    return pd.Series(clean_flags, index=chunk_df.index)


def _collect_speaker_embeddings(
    rows: pd.DataFrame,
    audio_info,
    embedder: Inference | None,
    *,
    max_embeddings: int = 3,
):
    embeddings = []
    if rows is None or rows.empty:
        return embeddings

    rows = rows.copy()
    rows["_dur"] = rows["end"] - rows["start"]
    for _, row in rows.sort_values("_dur", ascending=False).iterrows():
        emb = _extract_speaker_embedding(
            audio_info, row["start"], row["end"], embedder=embedder
        )
        if emb is not None:
            embeddings.append(emb)
        if len(embeddings) >= max_embeddings:
            break
    return embeddings


def _compute_chunk_speaker_identity_profiles(
    chunk_df: pd.DataFrame,
    audio_info,
    embedder: Inference | None,
    identity_min_duration: float = 2.0,
):
    """
    Build per-local-speaker profiles for global speaker linking.

    `embedding` may come from weak rows so a short backchannel can still match a
    clear existing speaker. `has_clean_identity_evidence` is only true when the
    embedding came from long non-overlap rows, allowing centroid create/update.
    """
    if embedder is None or chunk_df is None or chunk_df.empty:
        return {}, pd.Series(dtype=bool)

    clean_mask = _chunk_clean_identity_mask(chunk_df, identity_min_duration)
    profiles = {}

    for speaker, rows in chunk_df.groupby("speaker"):
        clean_rows = rows.loc[clean_mask.reindex(rows.index, fill_value=False)]
        clean_embeddings = _collect_speaker_embeddings(clean_rows, audio_info, embedder)
        if clean_embeddings:
            profiles[speaker] = {
                "embedding": np.mean(clean_embeddings, axis=0),
                "has_clean_identity_evidence": True,
            }
            continue

        weak_embeddings = _collect_speaker_embeddings(rows, audio_info, embedder)
        profiles[speaker] = {
            "embedding": np.mean(weak_embeddings, axis=0) if weak_embeddings else None,
            "has_clean_identity_evidence": False,
        }

    return profiles, clean_mask


def _compute_chunk_speaker_centroids(chunk_df: pd.DataFrame, audio_info, embedder: Inference | None):
    """
    Tạo centroid embedding cho từng speaker trong một chunk diarization.

    Mỗi speaker lấy tối đa vài segment đầu đủ điều kiện, rồi trung bình embedding
    để tạo đại diện speaker trong chunk đó.
    """
    profiles, _ = _compute_chunk_speaker_identity_profiles(chunk_df, audio_info, embedder)
    return {
        speaker: profile["embedding"]
        for speaker, profile in profiles.items()
        if profile.get("embedding") is not None
    }


def align_speakers_across_chunks(
    chunk_frames: list[pd.DataFrame],
    audio_info,
    embedder: Inference | None,
    similarity_threshold: float = 0.75,
    similarity_margin: float = 0.08,
    weak_match_threshold: float = 0.55,
    centroid_update_threshold: float = 0.85,
    identity_min_duration: float = 2.0,
    review_speaker_label: str = "SPEAKER_REVIEW",
    return_stats: bool = False,
):
    """
    Nối speaker label cục bộ giữa các chunk thành speaker label toàn file.

    Vì Sortformer chạy từng chunk, SPEAKER_00 ở chunk A không chắc là cùng người
    với SPEAKER_00 ở chunk B. Hàm này dùng embedding similarity để map speaker
    local về speaker global nhất quán trên toàn recording.
    """
    if embedder is None or not chunk_frames:
        logger.warning("Speaker embedder unavailable; skipping cross-chunk speaker linking.")
        stats = {
            "skipped": True,
            "reason": "no_embedder_or_chunks",
        }
        return (chunk_frames, stats) if return_stats else chunk_frames

    global_centroids: dict[str, np.ndarray | None] = {}
    global_counts: dict[str, int] = {}
    next_global_idx = 0
    aligned_frames: list[pd.DataFrame] = []
    chunk_stats = []

    for chunk_idx, df in enumerate(chunk_frames):
        if df is None or df.empty:
            aligned_frames.append(df)
            continue

        profiles, clean_mask = _compute_chunk_speaker_identity_profiles(
            df, audio_info, embedder, identity_min_duration=identity_min_duration
        )
        mapping: dict[str, str] = {}
        decision_by_local: dict[str, dict] = {}
        decision_trace_by_local: dict[str, dict] = {}
        # Theo dõi global speaker đã dùng trong chunk để hai local speaker khác
        # nhau không bị map vào cùng một global ID.
        used_global_ids_in_chunk: set[str] = set()
        decisions = []

        local_speakers = list(df["speaker"].unique())
        local_speakers.sort(
            key=lambda sp: (
                not bool(profiles.get(sp, {}).get("has_clean_identity_evidence", False)),
                str(sp),
            )
        )

        for local_speaker in local_speakers:
            profile = profiles.get(local_speaker, {})
            emb = profile.get("embedding")
            has_clean_identity_evidence = bool(profile.get("has_clean_identity_evidence", False))

            best_id = None
            best_sim = -1.0
            second_best_id = None
            second_best_sim = -1.0
            candidates = []
            if emb is not None:
                for gid, centroid in global_centroids.items():
                    if centroid is None:
                        continue
                    # Không cho hai local speaker trong cùng chunk dùng chung global ID.
                    if gid in used_global_ids_in_chunk:
                        continue
                    sim = _cosine_similarity(emb, centroid)
                    candidates.append({"speaker": gid, "similarity": round(sim, 6)})
                    if sim > best_sim:
                        second_best_id = best_id
                        second_best_sim = best_sim
                        best_sim = sim
                        best_id = gid
                    elif sim > second_best_sim:
                        second_best_id = gid
                        second_best_sim = sim
            candidates.sort(key=lambda item: item["similarity"], reverse=True)

            next_global_id = f"SPEAKER_{next_global_idx:02d}"
            decision = stage_common.resolve_speaker_identity_decision(
                best_id=best_id,
                best_similarity=best_sim,
                second_best_similarity=second_best_sim,
                has_clean_identity_evidence=has_clean_identity_evidence,
                next_global_id=next_global_id,
                similarity_threshold=similarity_threshold,
                similarity_margin=similarity_margin,
                weak_match_threshold=weak_match_threshold,
                centroid_update_threshold=centroid_update_threshold,
                review_speaker_label=review_speaker_label,
            )

            mapped_speaker = decision["mapped_speaker"]
            mapping[local_speaker] = mapped_speaker
            decision_by_local[local_speaker] = decision

            centroid_updated = False
            if decision["should_create_new"]:
                next_global_idx += 1
                used_global_ids_in_chunk.add(mapped_speaker)
                global_centroids[mapped_speaker] = emb
                global_counts[mapped_speaker] = 1 if emb is not None else 0
                centroid_updated = emb is not None
            elif mapped_speaker != review_speaker_label:
                used_global_ids_in_chunk.add(mapped_speaker)
                if decision["should_update_centroid"] and emb is not None:
                    count = global_counts.get(mapped_speaker, 0)
                    if global_centroids.get(mapped_speaker) is None or count <= 0:
                        global_centroids[mapped_speaker] = emb
                        global_counts[mapped_speaker] = 1
                    else:
                        global_centroids[mapped_speaker] = (
                            global_centroids[mapped_speaker] * count + emb
                        ) / (count + 1)
                        global_counts[mapped_speaker] = count + 1
                    centroid_updated = True

            decision_record = {
                "local_speaker": local_speaker,
                "mapped_speaker": mapped_speaker,
                "action": decision["action"],
                "reason": decision["reason"],
                "embedding_available": emb is not None,
                "has_clean_identity_evidence": has_clean_identity_evidence,
                "best_global_speaker": best_id,
                "best_similarity": _round_similarity(best_sim),
                "second_best_global_speaker": second_best_id,
                "second_best_similarity": _round_similarity(second_best_sim),
                "similarity_margin": _similarity_gap(best_sim, second_best_sim),
                "identity_low_confidence": decision["identity_low_confidence"],
                "centroid_updated": centroid_updated,
                "candidates": candidates[:10],
                "thresholds": {
                    "similarity_threshold": similarity_threshold,
                    "similarity_margin": similarity_margin,
                    "weak_match_threshold": weak_match_threshold,
                    "centroid_update_threshold": centroid_update_threshold,
                    "identity_min_duration_seconds": identity_min_duration,
                },
            }
            decision_trace_by_local[local_speaker] = decision_record
            decisions.append(decision_record)

        remapped_df = df.copy()
        remapped_df["_speaker_link_local"] = remapped_df["speaker"]
        remapped_df["pre_link_speaker"] = remapped_df["speaker"]
        remapped_df["speaker_identity_clean_candidate"] = clean_mask.reindex(remapped_df.index, fill_value=False).astype(bool)
        remapped_df["speaker"] = remapped_df["speaker"].map(mapping)
        for local_speaker, decision in decision_by_local.items():
            mask = remapped_df["_speaker_link_local"] == local_speaker
            remapped_df.loc[mask, "speaker_link_action"] = decision["action"]
            remapped_df.loc[mask, "speaker_link_reason"] = decision["reason"]
            remapped_df.loc[mask, "identity_low_confidence"] = bool(decision["identity_low_confidence"])
            if decision["identity_low_confidence"]:
                remapped_df.loc[mask, "needs_manual_review"] = True
                remapped_df.loc[mask, "train_quality_label"] = "speaker_identity_review"
            if decision["mapped_speaker"] == review_speaker_label:
                remapped_df.loc[mask, "speaker_identity_status"] = "review"
            elif decision["identity_low_confidence"]:
                remapped_df.loc[mask, "speaker_identity_status"] = "weak_context_match"
            else:
                remapped_df.loc[mask, "speaker_identity_status"] = "clean"

        traces = []
        trace_ids = []
        for local_segment_index, (_row_index, row) in enumerate(remapped_df.iterrows()):
            local_speaker = str(row.get("_speaker_link_local", ""))
            decision_record = decision_trace_by_local.get(local_speaker, {})
            trace_id = f"chunk{chunk_idx:03d}:local:{local_speaker}:row{local_segment_index:05d}"
            trace_ids.append(trace_id)
            source_start = float(row.get("start", 0.0))
            source_end = float(row.get("end", source_start))
            traces.append(
                {
                    "trace_id": trace_id,
                    "chunk": {
                        "chunk_index": chunk_idx,
                        "chunk_offset": row.get("_stage1_chunk_offset"),
                        "chunk_duration": row.get("_stage1_chunk_duration"),
                        "local_segment_index": local_segment_index,
                        "local_speaker": local_speaker,
                        "source_start": round(source_start, 6),
                        "source_end": round(source_end, 6),
                        "duration": round(_segment_duration_seconds(source_start, source_end), 6),
                    },
                    "identity": {
                        "clean_candidate": bool(row.get("speaker_identity_clean_candidate", False)),
                        "has_clean_identity_evidence": bool(
                            decision_record.get("has_clean_identity_evidence", False)
                        ),
                        "embedding_available": bool(decision_record.get("embedding_available", False)),
                        "identity_low_confidence": bool(row.get("identity_low_confidence", False)),
                        "status": row.get("speaker_identity_status"),
                    },
                    "link": {
                        "pre_link_speaker": local_speaker,
                        "mapped_speaker": row.get("speaker"),
                        "action": row.get("speaker_link_action"),
                        "reason": row.get("speaker_link_reason"),
                        "best_candidate": decision_record.get("best_global_speaker"),
                        "best_similarity": decision_record.get("best_similarity"),
                        "second_candidate": decision_record.get("second_best_global_speaker"),
                        "second_similarity": decision_record.get("second_best_similarity"),
                        "similarity_margin": decision_record.get("similarity_margin"),
                        "centroid_updated": bool(decision_record.get("centroid_updated", False)),
                        "thresholds": decision_record.get("thresholds", {}),
                        "top_candidates": copy.deepcopy(decision_record.get("candidates", [])[:5]),
                    },
                }
            )
        remapped_df["stage1_trace_id"] = trace_ids
        remapped_df["stage1_trace"] = traces
        aligned_frames.append(remapped_df)
        chunk_stats.append(
            {
                "chunk_index": chunk_idx,
                "local_speaker_count": len(local_speakers),
                "decisions": decisions,
            }
        )

    stats = {
        "skipped": False,
        "similarity_threshold": similarity_threshold,
        "similarity_margin": similarity_margin,
        "weak_match_threshold": weak_match_threshold,
        "centroid_update_threshold": centroid_update_threshold,
        "identity_min_duration_seconds": identity_min_duration,
        "review_speaker_label": review_speaker_label,
        "global_speaker_count": next_global_idx,
        "chunks": chunk_stats,
    }
    return (aligned_frames, stats) if return_stats else aligned_frames


def _apply_recluster_trace(
    speakerdia: pd.DataFrame,
    *,
    mapping: dict[str, str],
    decisions: dict[str, dict],
) -> pd.DataFrame:
    result = speakerdia.copy()
    if result.empty or "speaker" not in result.columns:
        return result

    original_speakers = result["speaker"].astype(str)
    result["pre_recluster_speaker"] = original_speakers
    result["speaker"] = original_speakers.map(mapping).fillna(original_speakers)

    traces = []
    actions = []
    reasons = []
    best_candidates = []
    best_similarities = []

    for _, row in result.iterrows():
        source_speaker = str(row.get("pre_recluster_speaker", ""))
        final_speaker = str(row.get("speaker", source_speaker))
        decision = decisions.get(source_speaker, {})
        top_candidates = copy.deepcopy(decision.get("top_candidates", []))
        best_candidate = decision.get("best_recluster_candidate")
        best_similarity = _round_similarity(decision.get("best_recluster_similarity", -1.0))
        action = str(decision.get("action", "kept"))
        reason = str(decision.get("reason", "not_evaluated"))

        trace = _as_trace_dict(row.get("stage1_trace"))
        trace["recluster"] = {
            "pre_recluster_speaker": source_speaker,
            "final_speaker": final_speaker,
            "action": action,
            "reason": reason,
            "best_candidate": best_candidate,
            "best_similarity": best_similarity,
            "similarity_threshold": decision.get("similarity_threshold"),
            "top_candidates": top_candidates,
        }
        traces.append(trace)
        actions.append(action)
        reasons.append(reason)
        best_candidates.append(best_candidate)
        best_similarities.append(best_similarity)

    result["stage1_trace"] = traces
    result["recluster_action"] = actions
    result["recluster_reason"] = reasons
    result["recluster_best_candidate"] = best_candidates
    result["recluster_best_similarity"] = best_similarities
    return result


def _build_speaker_diagnostics(
    segments: list[dict],
    *,
    speaker_linking_stats: dict,
    recluster_stats: dict,
) -> dict:
    speaker_counts = Counter()
    speaker_durations = Counter()
    speaker_link_actions = Counter()
    speaker_link_reasons = Counter()
    speaker_identity_statuses = Counter()
    recluster_actions = Counter()
    recluster_reasons = Counter()

    low_confidence_segments = 0
    manual_review_segments = 0
    short_backchannel_segments = 0
    clean_identity_candidate_segments = 0
    missing_speaker_link_action_segments = 0
    split_segments = 0

    for segment in segments:
        speaker = str(segment.get("speaker", "")).strip() or "UNKNOWN"
        duration = _segment_duration_seconds(segment.get("start", 0.0), segment.get("end", 0.0))
        speaker_counts[speaker] += 1
        speaker_durations[speaker] += duration

        action = segment.get("speaker_link_action")
        if action is None:
            missing_speaker_link_action_segments += 1
        else:
            speaker_link_actions[str(action)] += 1

        reason = segment.get("speaker_link_reason")
        if reason is not None:
            speaker_link_reasons[str(reason)] += 1

        status = segment.get("speaker_identity_status")
        if status is not None:
            speaker_identity_statuses[str(status)] += 1

        trace = segment.get("stage1_trace")
        if isinstance(trace, dict):
            if isinstance(trace.get("split"), dict):
                split_segments += 1
            recluster = trace.get("recluster")
            if isinstance(recluster, dict):
                if recluster.get("action") is not None:
                    recluster_actions[str(recluster["action"])] += 1
                if recluster.get("reason") is not None:
                    recluster_reasons[str(recluster["reason"])] += 1

        if bool(segment.get("identity_low_confidence")):
            low_confidence_segments += 1
        if bool(segment.get("needs_manual_review")):
            manual_review_segments += 1
        if bool(segment.get("is_short_backchannel")):
            short_backchannel_segments += 1
        if bool(segment.get("speaker_identity_clean_candidate")):
            clean_identity_candidate_segments += 1

    bottleneck_summary = {
        "segments_total": len(segments),
        "speakers_final": dict(sorted(speaker_counts.items())),
        "speaker_duration_seconds": {
            speaker: round(duration, 6)
            for speaker, duration in sorted(speaker_durations.items(), key=lambda item: (-item[1], item[0]))
        },
        "speaker_link_actions": dict(sorted(speaker_link_actions.items())),
        "speaker_link_reasons": dict(sorted(speaker_link_reasons.items())),
        "speaker_identity_statuses": dict(sorted(speaker_identity_statuses.items())),
        "recluster_actions": dict(sorted(recluster_actions.items())),
        "recluster_reasons": dict(sorted(recluster_reasons.items())),
        "low_confidence_segments": low_confidence_segments,
        "manual_review_segments": manual_review_segments,
        "short_backchannel_segments": short_backchannel_segments,
        "clean_identity_candidate_segments": clean_identity_candidate_segments,
        "missing_speaker_link_action_segments": missing_speaker_link_action_segments,
        "split_segments": split_segments,
    }

    return {
        "schema_version": 1,
        "bottleneck_summary": bottleneck_summary,
        "speaker_linking": speaker_linking_stats,
        "reclustering": recluster_stats,
    }


def re_cluster_speakers(
    speakerdia,
    audio_info,
    embedder,
    similarity_threshold=0.75,
    identity_min_duration=2.0,
    review_speaker_label="SPEAKER_REVIEW",
):
    """
    Re-cluster fragmented speaker IDs after cross-chunk alignment.

    Diarization models may assign different IDs to the same person across
    different parts of the audio (e.g. before/after an ad break). This function
    compares speaker embeddings globally and merges speakers whose embedding
    cosine similarity exceeds the threshold.

    Speakers are processed in descending order of total duration so that the
    dominant speaker keeps their ID and smaller fragments merge into them.
    Speakers with very different voices (ads, intros) will have low similarity
    and remain separate — they get excluded later by assign_duplex_train_groups.

    Returns (remapped_df, recluster_stats).
    """
    if embedder is None or speakerdia is None or speakerdia.empty:
        return speakerdia, {"skipped": True, "reason": "no_embedder_or_empty"}

    speakers = speakerdia["speaker"].unique().tolist()
    if len(speakers) <= 1:
        return speakerdia, {"skipped": True, "reason": "single_speaker"}

    # 1. Per-speaker total duration, sorted descending
    speaker_durations = {}
    for sp in speakers:
        mask = speakerdia["speaker"] == sp
        speaker_durations[sp] = float(
            (speakerdia.loc[mask, "end"] - speakerdia.loc[mask, "start"]).sum()
        )
    ranked = sorted(speaker_durations.items(), key=lambda x: -x[1])

    # 2. Extract representative embedding per speaker (centroid of longest segments)
    speaker_embeddings = {}
    embedding_sources = {}
    for sp, _ in ranked:
        mask = speakerdia["speaker"] == sp
        rows = speakerdia[mask].copy()
        candidate_segment_count_before_filter = len(rows)
        rows = rows[rows["speaker"] != review_speaker_label]
        if "identity_low_confidence" in rows.columns:
            rows = rows[rows["identity_low_confidence"] != True]
        if "speaker_identity_clean_candidate" in rows.columns:
            rows = rows[rows["speaker_identity_clean_candidate"] == True]
        else:
            rows = rows[(rows["end"] - rows["start"]) >= identity_min_duration]
        if rows.empty:
            embedding_sources[sp] = {
                "candidate_segment_count_before_filter": candidate_segment_count_before_filter,
                "clean_segment_count": 0,
                "embedding_segment_count": 0,
                "reason": "no_clean_segments",
            }
            continue
        rows = rows.assign(_dur=rows["end"] - rows["start"])
        rows = rows.sort_values("_dur", ascending=False)

        embs = []
        for _, row in rows.head(5).iterrows():
            emb = _extract_speaker_embedding(
                audio_info, row["start"], row["end"], embedder=embedder
            )
            if emb is not None:
                embs.append(emb)
            if len(embs) >= 3:
                break
        if embs:
            speaker_embeddings[sp] = np.mean(embs, axis=0)
            embedding_sources[sp] = {
                "candidate_segment_count_before_filter": candidate_segment_count_before_filter,
                "clean_segment_count": len(rows),
                "embedding_segment_count": len(embs),
                "reason": "embedding_created",
            }
        else:
            embedding_sources[sp] = {
                "candidate_segment_count_before_filter": candidate_segment_count_before_filter,
                "clean_segment_count": len(rows),
                "embedding_segment_count": 0,
                "reason": "embedding_extraction_failed",
            }

    # 3. Greedy clustering: process speakers by descending duration
    #    The first (longest) speaker always creates a new cluster.
    #    Subsequent speakers either join an existing cluster if similarity
    #    is high enough, or create a new one.
    clusters = {}
    speaker_to_cluster = {}
    recluster_decisions = {}

    for sp, dur in ranked:
        emb = speaker_embeddings.get(sp)

        best_cid = None
        best_sim = -1.0
        candidates = []
        if emb is not None:
            for cid, cl in clusters.items():
                if cl["embedding"] is None:
                    continue
                sim = _cosine_similarity(emb, cl["embedding"])
                candidates.append({"speaker": cid, "similarity": round(sim, 6)})
                if sim > best_sim:
                    best_sim = sim
                    best_cid = cid
        candidates.sort(key=lambda item: item["similarity"], reverse=True)

        if best_sim >= similarity_threshold and best_cid is not None:
            # Merge into existing cluster — update centroid with duration-weighted average
            cl = clusters[best_cid]
            old_d = cl["duration"]
            new_d = old_d + dur
            if cl["embedding"] is not None and emb is not None:
                cl["embedding"] = (cl["embedding"] * old_d + emb * dur) / new_d
            cl["duration"] = new_d
            cl["members"].append(sp)
            speaker_to_cluster[sp] = best_cid
            recluster_decisions[sp] = {
                "speaker": sp,
                "mapped_speaker": best_cid,
                "action": "merged",
                "reason": "above_threshold",
                "duration": round(float(dur), 6),
                "best_recluster_candidate": best_cid,
                "best_recluster_similarity": _round_similarity(best_sim),
                "similarity_threshold": similarity_threshold,
                "top_candidates": candidates[:10],
                "embedding_source": embedding_sources.get(sp, {}),
            }
        else:
            # Create new cluster with this speaker as representative
            clusters[sp] = {
                "embedding": emb,
                "duration": dur,
                "members": [sp],
            }
            speaker_to_cluster[sp] = sp
            if sp == review_speaker_label:
                keep_reason = "review_speaker"
            elif emb is None:
                keep_reason = "no_clean_embedding"
            elif best_cid is None:
                keep_reason = "cluster_representative"
            else:
                keep_reason = "below_threshold"
            recluster_decisions[sp] = {
                "speaker": sp,
                "mapped_speaker": sp,
                "action": "kept",
                "reason": keep_reason,
                "duration": round(float(dur), 6),
                "best_recluster_candidate": best_cid,
                "best_recluster_similarity": _round_similarity(best_sim),
                "similarity_threshold": similarity_threshold,
                "top_candidates": candidates[:10],
                "embedding_source": embedding_sources.get(sp, {}),
            }

    # 4. Build mapping and apply
    mapping = {sp: cid for sp, cid in speaker_to_cluster.items()}
    merges = {}
    active_logger = globals().get("logger") or Logger.get_logger()
    for cid, cl in clusters.items():
        if len(cl["members"]) > 1:
            merged_from = [m for m in cl["members"] if m != cid]
            merges[cid] = merged_from
            active_logger.info(
                f"Speaker re-cluster: {', '.join(merged_from)} -> {cid} "
                f"(total {cl['duration']:.1f}s)"
            )

    stats = {
        "skipped": False,
        "input_speakers": len(speakers),
        "output_speakers": len(clusters),
        "merges": {k: v for k, v in merges.items()},
        "similarity_threshold": similarity_threshold,
        "speaker_durations_before": {sp: round(d, 2) for sp, d in ranked},
        "speaker_embeddings": embedding_sources,
        "decisions": recluster_decisions,
        "mapping": mapping,
    }

    result = _apply_recluster_trace(speakerdia, mapping=mapping, decisions=recluster_decisions)
    return result, stats


def prepare_diarization_chunks(
    audio_path,
    audio_info,
    max_duration=MAX_DIA_CHUNK_DURATION,
    min_silence=MIN_SPLIT_SILENCE,
):
    """
    Chuẩn bị chunk audio trước khi chạy diarization.

    Hàm này:
    - Dùng VAD tìm khoảng lặng.
    - Tạo các chunk dưới MAX_DIA_CHUNK_DURATION.
    - Export chunk thành WAV mono tạm.
    - Trả metadata path/offset/duration để sau khi Sortformer trả thời gian local
      có thể cộng offset về timeline file gốc.

    Trả về:
        (chunk_entries, temp_dir)
    """
    waveform = audio_info["waveform"]
    sample_rate = audio_info["sample_rate"]
    total_duration, silence_intervals = _build_silence_intervals(
        waveform, sample_rate, min_silence
    )
    chunk_ranges = _build_chunk_ranges(total_duration, silence_intervals, max_duration)

    epsilon = 1e-3
    normalized_audio = audio_info.get("audio_segment")

    if (
        len(chunk_ranges) == 1
        and chunk_ranges[0][0] <= epsilon
        and abs(chunk_ranges[0][1] - total_duration) <= epsilon
    ):
        # Trường hợp audio đủ ngắn: vẫn export ra WAV mono tạm để Sortformer đọc ổn định.
        if normalized_audio is not None:
            # Dùng audio đã chuẩn hoá nếu có.
            temp_dir = tempfile.mkdtemp(prefix="pre_diar_")
            temp_path = os.path.join(temp_dir, "full_audio.wav")
            normalized_audio.export(temp_path, format="wav", parameters=["-ac", "1"])
            return [{"path": temp_path, "offset": 0.0, "duration": total_duration}], temp_dir
        else:
            # Dự phòng: tự load lại từ file và ép mono.
            temp_audio = AudioSegment.from_file(audio_path).set_channels(1)
            temp_dir = tempfile.mkdtemp(prefix="pre_diar_")
            temp_path = os.path.join(temp_dir, "full_audio.wav")
            temp_audio.export(temp_path, format="wav", parameters=["-ac", "1"])
            return [{"path": temp_path, "offset": 0.0, "duration": total_duration}], temp_dir


    if normalized_audio is None:
        normalized_audio = AudioSegment.from_file(audio_path)
        # Ép mono để diarization model nhận input đúng format.
        normalized_audio = normalized_audio.set_channels(1)
    temp_dir = tempfile.mkdtemp(prefix="pre_diar_")
    chunk_entries = []

    for idx, (start_sec, end_sec) in enumerate(chunk_ranges):
        start_ms = max(0, int(round(start_sec * 1000)))
        end_ms = max(start_ms, int(round(end_sec * 1000)))
        chunk_audio = normalized_audio[start_ms:end_ms]
        chunk_path = os.path.join(temp_dir, f"chunk_{idx:03d}.wav")
        # Export WAV mono rõ ràng để tránh file nguồn stereo gây sai shape.
        chunk_audio.export(chunk_path, format="wav", parameters=["-ac", "1"])
        chunk_entries.append(
            {
                "path": chunk_path,
                "offset": start_sec,
                "duration": end_sec - start_sec,
            }
        )

    logger.info(
        f"Pre-diarization chunking created {len(chunk_entries)} chunks "
        f"(max {max_duration}s) from {os.path.basename(audio_path)}"
    )
    return chunk_entries, temp_dir


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
    parser.add_argument("--max_segment_duration", type=float, default=30.0)
    
    # Sortformer params
    parser.add_argument("--sortformer-model-name", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    parser.add_argument("--sortformer-param", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sortformer-pad-onset", type=float, default=0.0)
    parser.add_argument("--sortformer-pad-offset", type=float, default=0.0)
    parser.add_argument("--onset", type=float, default=0.53)
    parser.add_argument("--offset", type=float, default=0.49)
    parser.add_argument("--min-duration-on", type=float, default=0.42)
    parser.add_argument("--min-duration-off", type=float, default=0.34)
    parser.add_argument("--use-custom-binarize", action=argparse.BooleanOptionalAction, default=True)
    
    # Re-cluster
    parser.add_argument("--speaker-link-threshold", type=float, default=0.75)
    parser.add_argument("--speaker-link-margin", type=float, default=0.08)
    parser.add_argument("--speaker-weak-match-threshold", type=float, default=0.55)
    parser.add_argument("--speaker-centroid-update-threshold", type=float, default=0.85)
    parser.add_argument("--speaker-identity-min-duration", type=float, default=2.0)
    parser.add_argument("--speaker-review-label", default="SPEAKER_REVIEW")
    parser.add_argument("--speaker-recluster-threshold", type=float, default=0.7)
    
    return parser.parse_args()

def main() -> None:
    global logger, cfg, pipe_args, vad, device, device_name
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
            from pyannote.audio import Model as PyannoteModel
            emb_model = PyannoteModel.from_pretrained("pyannote/embedding", use_auth_token=cfg["huggingface_token"])
            speaker_embedder = Inference(emb_model, device=device, window="whole")
        except Exception as exc:
            logger.warning(f"Speaker embedder unavailable: {exc}")

    if SortformerEncLabelModel is None:
        raise ImportError("nemo_toolkit[asr] is required")
        
    logger.info(f"Loading Sortformer model: {args.sortformer_model_name}")
    diar_model = SortformerEncLabelModel.from_pretrained(args.sortformer_model_name)
    try:
        diar_model = diar_model.to(device)
    except Exception:
        pass
    diar_model.eval()

    sample_rate = int(cfg["entrypoint"]["SAMPLE_RATE"])
    audio_info = stage_common.load_audio_info(audio_path, sample_rate)
    audio_duration = len(audio_info["waveform"]) / audio_info["sample_rate"]

    start_time = time.time()
    diar_chunks, temp_chunk_dir = prepare_diarization_chunks(str(audio_path), audio_info)
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
                chunk_df["_stage1_chunk_offset"] = float(chunk["offset"])
                chunk_df["_stage1_chunk_duration"] = float(chunk["duration"])
                chunk_df = _apply_sortformer_segment_padding_from_args(chunk_df, pipe_args, logger, audio_duration)
            diarization_frames.append(chunk_df)
    finally:
        if temp_chunk_dir:
            shutil.rmtree(temp_chunk_dir, ignore_errors=True)

    speaker_linking_stats = {"skipped": True, "reason": "no_diarization_frames"}
    if diarization_frames:
        diarization_frames, speaker_linking_stats = align_speakers_across_chunks(
            diarization_frames,
            audio_info=audio_info,
            embedder=speaker_embedder,
            similarity_threshold=args.speaker_link_threshold,
            similarity_margin=args.speaker_link_margin,
            weak_match_threshold=args.speaker_weak_match_threshold,
            centroid_update_threshold=args.speaker_centroid_update_threshold,
            identity_min_duration=args.speaker_identity_min_duration,
            review_speaker_label=args.speaker_review_label,
            return_stats=True,
        )
        speakerdia = pd.concat(diarization_frames, ignore_index=True)
    else:
        speakerdia = pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])

    recluster_stats = {"skipped": True, "reason": "disabled_or_no_embedder"}
    if args.speaker_recluster_threshold > 0 and speaker_embedder is not None:
        speakerdia, recluster_stats = re_cluster_speakers(
            speakerdia,
            audio_info=audio_info,
            embedder=speaker_embedder,
            similarity_threshold=args.speaker_recluster_threshold,
            identity_min_duration=args.speaker_identity_min_duration,
            review_speaker_label=args.speaker_review_label,
        )

    segments = df_to_list(speakerdia)
    segments = split_long_segments(segments, max_duration=args.max_segment_duration)
    segments, postproc_stats = stage_common.postprocess_diarization_segments(
        segments, same_speaker_merge_gap=args.same_speaker_merge_gap, short_backchannel_seconds=args.short_backchannel_seconds, max_segment_duration=args.max_segment_duration
    )
    segments_for_json = stage_common.clean_segments_for_json(segments)
    speaker_diagnostics = _build_speaker_diagnostics(
        segments_for_json,
        speaker_linking_stats=speaker_linking_stats,
        recluster_stats=recluster_stats,
    )
    speaker_diag_out_path = Path(out_path).parent / "speaker_diagnostics.json"
    
    elapsed = time.time() - start_time
    logger.info(f"Diarization finished in {elapsed:.2f}s.")

    vad_chunks = [{"offset": c["offset"], "duration": c["duration"]} for c in diar_chunks]
    
    out_data = {
        "audio_path": str(audio_path),
        "audio_name": audio_info["name"],
        "sample_rate": audio_info["sample_rate"],
        "audio_duration_seconds": audio_duration,
        "segments": segments_for_json,
        "metadata": {
            "stage": "diarize",
            "device": device_name,
            "sortformer_model_name": args.sortformer_model_name,
            "processing_time_seconds": elapsed,
            "postprocessing": postproc_stats,
            "reclustering": recluster_stats,
            "speaker_linking": speaker_linking_stats,
            "speaker_diagnostics_path": str(speaker_diag_out_path),
        },
    }
    
    stage_common.dump_json(out_data, out_path)
    logger.info(f"Saved diarization to {out_path}")
    stage_common.dump_json(speaker_diagnostics, speaker_diag_out_path)
    logger.info(f"Saved speaker diagnostics to {speaker_diag_out_path}")
    
    # Save VAD chunks to a separate vad_chunks.json file in the same directory
    vad_out_path = Path(out_path).parent / "vad_chunks.json"
    stage_common.dump_json({"vad_chunks": vad_chunks}, vad_out_path)
    logger.info(f"Saved VAD chunks to {vad_out_path}")

if __name__ == "__main__":
    main()
