from __future__ import annotations
import os
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
    """
    records = []
    for i, row in df.iterrows():
        records.append({
            'index': f"{i:05d}",
            'start': float(row['start']),
            'end': float(row['end']),
            'speaker': row['speaker']
        })
    return records

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
        speaker = segment['speaker']
        duration = end_time - start_time

        # Segment đủ ngắn thì giữ nguyên và chỉ đánh lại index.
        if duration <= max_duration:
            segment['index'] = str(new_index).zfill(5)
            new_segments.append(segment)
            new_index += 1
        # Segment quá dài thì cắt thành nhiều chunk nối tiếp nhau.
        else:
            current_start = start_time
            # Lặp tới khi phủ hết khoảng thời gian gốc.
            while current_start < end_time:
                # Điểm kết thúc chunk tiếp theo không được vượt end gốc.
                chunk_end = min(current_start + max_duration, end_time)
                
                new_segments.append({
                    'index': str(new_index).zfill(5),
                    'start': round(current_start, 3),
                    'end': round(chunk_end, 3),
                    'speaker': speaker
                })
                new_index += 1
                # Chunk kế tiếp bắt đầu ngay tại end của chunk hiện tại.
                current_start = chunk_end
                
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

def _compute_chunk_speaker_centroids(chunk_df: pd.DataFrame, audio_info, embedder: Inference | None):
    """
    Tạo centroid embedding cho từng speaker trong một chunk diarization.

    Mỗi speaker lấy tối đa vài segment đầu đủ điều kiện, rồi trung bình embedding
    để tạo đại diện speaker trong chunk đó.
    """
    if embedder is None or chunk_df is None or chunk_df.empty:
        return {}

    centroids = {}
    for speaker, rows in chunk_df.groupby("speaker"):
        rows = rows.copy()
        rows["_dur"] = rows["end"] - rows["start"]
        
        embeddings = []
        for _, row in rows.sort_values("_dur", ascending=False).iterrows():
            emb = _extract_speaker_embedding(
                audio_info, row["start"], row["end"], embedder=embedder
            )
            if emb is not None:
                embeddings.append(emb)
            if len(embeddings) >= 3:
                break
        if embeddings:
            centroids[speaker] = np.mean(embeddings, axis=0)


def align_speakers_across_chunks(
    chunk_frames: list[pd.DataFrame],
    audio_info,
    embedder: Inference | None,
    similarity_threshold: float = 0.75,
):
    """
    Nối speaker label cục bộ giữa các chunk thành speaker label toàn file.

    Vì Sortformer chạy từng chunk, SPEAKER_00 ở chunk A không chắc là cùng người
    với SPEAKER_00 ở chunk B. Hàm này dùng embedding similarity để map speaker
    local về speaker global nhất quán trên toàn recording.
    """
    if embedder is None or not chunk_frames:
        logger.warning("Speaker embedder unavailable; skipping cross-chunk speaker linking.")
        return chunk_frames

    global_centroids: dict[str, np.ndarray | None] = {}
    global_counts: dict[str, int] = {}
    next_global_idx = 0
    aligned_frames: list[pd.DataFrame] = []

    for chunk_idx, df in enumerate(chunk_frames):
        if df is None or df.empty:
            aligned_frames.append(df)
            continue

        local_centroids = _compute_chunk_speaker_centroids(df, audio_info, embedder)
        mapping: dict[str, str] = {}
        # Theo dõi global speaker đã dùng trong chunk để hai local speaker khác
        # nhau không bị map vào cùng một global ID.
        used_global_ids_in_chunk: set[str] = set()

        for local_speaker in df["speaker"].unique():
            emb = local_centroids.get(local_speaker)

            best_id = None
            best_sim = -1.0
            if emb is not None:
                for gid, centroid in global_centroids.items():
                    if centroid is None:
                        continue
                    # Không cho hai local speaker trong cùng chunk dùng chung global ID.
                    if gid in used_global_ids_in_chunk:
                        continue
                    sim = _cosine_similarity(emb, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = gid

            if best_sim >= similarity_threshold and best_id is not None:
                mapping[local_speaker] = best_id
                used_global_ids_in_chunk.add(best_id)
                count = global_counts.get(best_id, 0)
                global_centroids[best_id] = (global_centroids[best_id] * count + emb) / (
                    count + 1
                )
                global_counts[best_id] = count + 1
            else:
                global_id = f"SPEAKER_{next_global_idx:02d}"
                next_global_idx += 1
                mapping[local_speaker] = global_id
                used_global_ids_in_chunk.add(global_id)
                global_centroids[global_id] = emb
                global_counts[global_id] = 1 if emb is not None else 0

        remapped_df = df.copy()
        remapped_df["speaker"] = remapped_df["speaker"].map(mapping)
        aligned_frames.append(remapped_df)



def re_cluster_speakers(
    speakerdia,
    audio_info,
    embedder,
    similarity_threshold=0.75,
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
    for sp, _ in ranked:
        mask = speakerdia["speaker"] == sp
        rows = speakerdia[mask].copy()
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

    # 3. Greedy clustering: process speakers by descending duration
    #    The first (longest) speaker always creates a new cluster.
    #    Subsequent speakers either join an existing cluster if similarity
    #    is high enough, or create a new one.
    clusters = {}
    speaker_to_cluster = {}

    for sp, dur in ranked:
        emb = speaker_embeddings.get(sp)

        best_cid = None
        best_sim = -1.0
        if emb is not None:
            for cid, cl in clusters.items():
                if cl["embedding"] is None:
                    continue
                sim = _cosine_similarity(emb, cl["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_cid = cid

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
        else:
            # Create new cluster with this speaker as representative
            clusters[sp] = {
                "embedding": emb,
                "duration": dur,
                "members": [sp],
            }
            speaker_to_cluster[sp] = sp

    # 4. Build mapping and apply
    mapping = {sp: cid for sp, cid in speaker_to_cluster.items()}
    merges = {}
    for cid, cl in clusters.items():
        if len(cl["members"]) > 1:
            merged_from = [m for m in cl["members"] if m != cid]
            merges[cid] = merged_from
            logger.info(
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
    }

    if merges:
        result = speakerdia.copy()
        result["speaker"] = result["speaker"].map(mapping)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 01: run diarization and save segments JSON.")
    parser.add_argument("--input_audio", required=True, help="Input audio file path.")
    parser.add_argument("--out", default="", help="Output diarization JSON path.")
    parser.add_argument("--config_path", default="config.json")
    parser.add_argument("--seg_th", type=float, default=0.11)
    parser.add_argument("--min_cluster_size", type=int, default=11)
    parser.add_argument("--same_speaker_merge_gap", type=float, default=1.0)
    parser.add_argument("--short_backchannel_seconds", type=float, default=1.0)
    parser.add_argument("--max_segment_duration", type=float, default=30.0)
    
    # Sortformer params
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

    vad_chunks = [{"offset": c["offset"], "duration": c["duration"]} for c in diar_chunks]
    
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
    
    # Save VAD chunks to a separate vad_chunks.json file in the same directory
    vad_out_path = Path(out_path).parent / "vad_chunks.json"
    stage_common.dump_json({"vad_chunks": vad_chunks}, vad_out_path)
    logger.info(f"Saved VAD chunks to {vad_out_path}")

if __name__ == "__main__":
    main()
