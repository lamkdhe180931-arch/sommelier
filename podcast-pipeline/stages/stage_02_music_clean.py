from __future__ import annotations
import argparse
import shutil
import time
import json
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
import subprocess
import copy

try:
    from panns_inference import AudioTagging
except ImportError:
    AudioTagging = None

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
            return logging.getLogger("stage_02")

warnings.filterwarnings("ignore")
def detect_segment_background_music(segment_audio, sample_rate, panns_model, threshold=0.3):
    """
    Phát hiện nhạc nền trong một segment audio.

    Tham số:
        segment_audio: waveform của segment.
        sample_rate: sample rate của segment.
        panns_model: model PANNs đã load.
        threshold: ngưỡng xác suất Music.

    Trả về:
        (has_music, music_prob)
    """
    if panns_model is None:
        logger.warning("PANNs model is not loaded, skipping music detection")
        return False, 0.0

    # PANNs cần 32 kHz.
    if sample_rate != 32000:
        waveform_32k = librosa.resample(segment_audio, orig_sr=sample_rate, target_sr=32000)
    else:
        waveform_32k = segment_audio

    # Segment quá ngắn sẽ không đi qua được các lớp pooling của Cnn14.
    min_length = 32000  # 1 second at 32kHz
    if len(waveform_32k) < min_length:
        logger.warning(f"Segment too short for music detection ({len(waveform_32k)/32000:.2f}s < 1.0s), skipping music detection")
        return False, 0.0

    # Chạy PANNs trên segment.
    (clipwise_output, embedding) = panns_model.inference(waveform_32k[None, :])

    # Lấy label để tìm index "Music".
    labels = panns_model.labels

    # Trích xác suất label Music.
    music_idx = labels.index('Music') if 'Music' in labels else None
    if music_idx is not None:
        music_prob = float(clipwise_output[0, music_idx])
        has_music = music_prob > threshold
        return has_music, music_prob
    else:
        return False, 0.0

def separate_full_vocals_demucs(full_audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """
    Chạy Demucs một lần trên toàn bộ audio để lấy stem giọng nói.

    Lý do chạy toàn file:
    - Nếu từng segment đều gọi Demucs thì rất chậm.
    - Stem vocals của toàn file có thể được cắt lại theo start/end của từng segment.

    Trả về:
        Waveform vocals ở sample_rate gốc, hoặc None nếu Demucs lỗi.
    """
    temp_dir = tempfile.mkdtemp(prefix="demucs_full_")

    try:
        temp_input = os.path.join(temp_dir, "full.wav")
        sf.write(temp_input, full_audio, sample_rate)

        import subprocess
        demucs_output_dir = os.path.join(temp_dir, "separated")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running single Demucs pass on device: {device}")

        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-d", device,
            "-o", demucs_output_dir,
            temp_input,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Demucs full-audio run failed: {result.stderr}")
            return None

        input_stem = Path(temp_input).stem
        vocal_path = os.path.join(demucs_output_dir, "htdemucs", input_stem, "vocals.wav")
        if not os.path.exists(vocal_path):
            logger.error(f"Vocal track not found at {vocal_path}")
            return None

        vocal_waveform, _ = librosa.load(vocal_path, sr=sample_rate, mono=True)
        return vocal_waveform.astype(np.float32)

    except Exception as e:
        logger.error(f"Error during full Demucs processing: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def remove_segment_background_music_demucs(segment_audio, sample_rate, full_vocals=None, start_frame=None, end_frame=None):
    """
    Loại nhạc nền khỏi một segment bằng Demucs, chỉ giữ phần vocals.

    Nếu đã có full_vocals từ `separate_full_vocals_demucs`, hàm chỉ cắt đúng
    vùng frame tương ứng. Nếu chưa có, hàm dự phòng sang chạy Demucs riêng cho
    segment đó.

    Tham số:
        segment_audio: waveform của segment cần xử lý.
        sample_rate: sample rate hiện tại.
        full_vocals: stem vocals đã tách từ toàn bộ audio, nếu có.
        start_frame/end_frame: vị trí segment trong waveform gốc.

    Trả về:
        Waveform chỉ còn vocals; nếu lỗi thì trả lại segment_audio gốc.
    """
    if full_vocals is not None and start_frame is not None and end_frame is not None:
        # Đường nhanh: cắt trực tiếp từ stem vocals đã tách sẵn.
        start = max(0, int(start_frame))
        end = min(int(end_frame), len(full_vocals))
        if end <= start:
            return segment_audio

        vocal_slice = full_vocals[start:end]
        if len(vocal_slice) == 0:
            return segment_audio

        target_length = len(segment_audio)
        if len(vocal_slice) >= target_length:
            return vocal_slice[:target_length].astype(np.float32)

        padded = np.zeros(target_length, dtype=np.float32)
        padded[: len(vocal_slice)] = vocal_slice.astype(np.float32)
        return padded

    # Đường chậm: tạo thư mục tạm để chạy Demucs riêng cho segment.
    temp_dir = tempfile.mkdtemp(prefix="demucs_seg_")

    try:
        # Lưu segment thành WAV tạm vì CLI Demucs nhận file path.
        temp_input = os.path.join(temp_dir, "segment.wav")
        sf.write(temp_input, segment_audio, sample_rate)

        # Gọi Demucs CLI để tách nguồn.
        import subprocess
        demucs_output_dir = os.path.join(temp_dir, "separated")

        # Dùng CUDA nếu có để tăng tốc.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Running Demucs on device: {device}")

        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-d", device,  # Chỉ định rõ cuda/cpu cho Demucs.
            "-o", demucs_output_dir,
            temp_input
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Demucs failed for segment: {result.stderr}")
            return segment_audio

        # Demucs ghi kết quả vocals.wav theo cấu trúc thư mục cố định.
        vocal_path = os.path.join(demucs_output_dir, "htdemucs", "segment", "vocals.wav")

        if not os.path.exists(vocal_path):
            logger.error(f"Vocal track not found at {vocal_path}")
            return segment_audio

        # Nạp lại phần giọng nói bằng sample_rate gốc để thay thế vào waveform pipeline.
        vocal_waveform, _ = librosa.load(vocal_path, sr=sample_rate, mono=True)
        return vocal_waveform.astype(np.float32)

    except Exception as e:
        logger.error(f"Error during segment Demucs processing: {e}")
        return segment_audio
    finally:
        # Dọn thư mục tạm để không tích tụ file WAV lớn.
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def preprocess_segments_with_demucs(segment_list, audio, panns_model=None, use_demucs=False, padding=0.5):
    """
    Phát hiện nhạc nền từng segment và áp dụng Demucs trước ASR.

    Chiến lược:
    - Nếu --demucs tắt: giữ nguyên audio và trả flag False cho mọi segment.
    - Nếu bật: dùng PANNs kiểm tra từng segment có padding.
    - Khi segment có nhạc nền, ưu tiên chạy Demucs một lần trên toàn audio rồi
      cắt stem vocals, tránh chạy Demucs lặp lại nhiều lần.
    - Trả về audio đã thay waveform và danh sách flag đánh dấu segment nào đã
      được xử lý bằng Demucs.
    """
    if not use_demucs:
        logger.info("Demucs preprocessing skipped (flag disabled)")
        return audio, [False] * len(segment_list)

    logger.info(f"Preprocessing {len(segment_list)} segments with background music detection and removal (padding={padding}s)")

    waveform = audio["waveform"].copy()
    sample_rate = audio["sample_rate"]
    total_samples = len(waveform)
    segment_demucs_flags = []
    vocal_full = None
    full_demucs_attempted = False

    for idx, segment in enumerate(segment_list):
        # Thêm padding quanh segment để tránh bỏ sót nhạc nền sát biên timestamp.
        start_time = max(0, segment["start"] - padding)
        end_time = segment["end"] + padding  # Phép cắt mảng bên dưới sẽ tự chặn vượt biên.
        
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        
        # Không cho end_frame vượt quá độ dài waveform.
        end_frame = min(end_frame, total_samples)

        segment_audio = waveform[start_frame:end_frame]
        
        # Segment quá ngắn không đủ ổn định để PANNs/Demucs xử lý.
        if len(segment_audio) < 16000:  # Ngắn hơn khoảng 0.5 giây.
            segment_demucs_flags.append(False)
            continue

        # Kiểm tra nhạc nền trên vùng đã padding.
        has_music, music_prob = detect_segment_background_music(segment_audio, sample_rate, panns_model, threshold=0.3)
        
        if has_music:
            logger.info(f"Segment {idx} (with padding): Background music detected (prob={music_prob:.3f}), applying Demucs...")
            # Ưu tiên tách vocals toàn file một lần, rồi tái sử dụng cho các segment.
            if vocal_full is None and not full_demucs_attempted:
                vocal_full = separate_full_vocals_demucs(waveform, sample_rate)
                full_demucs_attempted = True
                if vocal_full is None:
                    logger.warning("Full Demucs pass failed; falling back to per-segment separation.")

            if vocal_full is not None:
                vocal_audio = remove_segment_background_music_demucs(
                    segment_audio,
                    sample_rate,
                    full_vocals=vocal_full,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            else:
                vocal_audio = remove_segment_background_music_demucs(segment_audio, sample_rate)
            
            # Ghi vocals trở lại waveform đã chuẩn hoá. Cần khớp độ dài vì output
            # Demucs có thể lệch vài sample so với input.
            target_length = len(segment_audio)
            if len(vocal_audio) >= target_length:
                waveform[start_frame:end_frame] = vocal_audio[:target_length]
            else:
                # Trường hợp hiếm: vocals ngắn hơn, chỉ thay phần có dữ liệu.
                waveform[start_frame : start_frame + len(vocal_audio)] = vocal_audio

            segment_demucs_flags.append(True)
            logger.info(f"Segment {idx}: Demucs applied successfully")
        else:
            segment_demucs_flags.append(False)

    # Cập nhật dict audio để các bước sau dùng waveform đã xử lý nhạc nền.
    updated_audio = audio.copy()
    updated_audio["waveform"] = waveform

    # Đồng bộ AudioSegment để phần export MP3 dùng đúng audio đã xử lý.
    from pydub import AudioSegment as PydubAudioSegment
    waveform_clipped = np.clip(waveform, -1.0, 1.0)
    waveform_int16 = (waveform_clipped * 32767).astype(np.int16)
    updated_audio_segment = PydubAudioSegment(
        waveform_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    updated_audio["audio_segment"] = updated_audio_segment

    logger.info(f"Demucs preprocessing completed: {sum(segment_demucs_flags)}/{len(segment_list)} segments processed")
    return updated_audio, segment_demucs_flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 02: detect and remove background music.")
    parser.add_argument("--diarization_json", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--config_path", default="config.json")
    parser.add_argument("--skip_music_removal", action=argparse.BooleanOptionalAction, default=False)
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
    logger.info(f"Stage 02 device: {device_name}")

    if AudioTagging is None:
        raise ImportError("panns_inference is required for music detection")

    start_time = time.time()
    diar_data = stage_common.load_json(args.diarization_json)
    audio_path = Path(diar_data["audio_path"])
    out_dir = audio_path.parent / "_staged" / stage_common.audio_name_from_path(audio_path)
    if args.out:
        out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_json_path = out_dir / "segment_flags.json"
    cleaned_audio_path = out_dir / "cleaned_audio.wav"

    segments = diar_data.get("segments", [])

    if getattr(args, "skip_music_removal", False):
        logger.info("Skipping music removal due to --skip_music_removal")
        cleaned_audio_path = audio_path
        flags = [False] * len(segments)
    else:
        sample_rate = diar_data.get("sample_rate", int(cfg["entrypoint"]["SAMPLE_RATE"]))
        audio_info = stage_common.load_audio_info(audio_path, sample_rate)
        
        flags, cleaned_wav = preprocess_segments_with_demucs(
            segments,
            audio_info,
            demucs_model=cfg["model"]["demucs"]["model"],
            device=device_name,
            logger=logger
        )
        
        if cleaned_wav is not None:
            stage_common.write_wav(cleaned_audio_path, cleaned_wav, sample_rate)
            logger.info(f"Saved cleaned full audio to {cleaned_audio_path}")
        else:
            cleaned_audio_path = audio_path

    elapsed = time.time() - start_time
    logger.info(f"Music clean finished in {elapsed:.2f}s.")

    out_data = {
        "audio_path": str(cleaned_audio_path),
        "source_audio_path": str(audio_path),
        "audio_name": diar_data.get("audio_name", ""),
        "sample_rate": diar_data.get("sample_rate", 16000),
        "audio_duration_seconds": diar_data.get("audio_duration_seconds", 0.0),
        "segments": segments,
        "segment_demucs_flags": flags,
        "metadata": {
            "stage": "music_clean",
            "enabled": not getattr(args, "skip_music_removal", False),
            "processing_time_seconds": elapsed,
        },
    }
    stage_common.dump_json(out_data, out_json_path)
    logger.info(f"Saved music clean flags to {out_json_path}")

if __name__ == "__main__":
    main()
