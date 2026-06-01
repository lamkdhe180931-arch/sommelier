from __future__ import annotations
import argparse
import shutil
import time
import json
import yaml
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np
import librosa
import torch
from pydub import AudioSegment
import soundfile as sf
import warnings

try:
    from pyannote.audio import Model as PyannoteModel, Inference
except ImportError:
    PyannoteModel = Inference = None

import stage_common
from utils.tool import load_cfg
from utils.asr_quality import should_skip_sepreformer_pair
try:
    from utils.logger import Logger
except ImportError:
    import logging
    class Logger:
        @staticmethod
        def get_logger():
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger("stage_03")

warnings.filterwarnings("ignore")
def detect_overlapping_segments(segment_list, overlap_threshold=0.2):
    """
    Tìm các cặp segment bị chồng thời gian lớn hơn overlap_threshold.

    Tham số:
        segment_list: danh sách segment có start, end, speaker.
        overlap_threshold: số giây chồng tối thiểu để coi là overlap cần xử lý.

    Trả về:
        Danh sách cặp overlap kèm overlap_start, overlap_end, overlap_duration.
    """
    overlapping_pairs = []

    # Sort theo start để khi seg2 bắt đầu sau seg1.end thì có thể break sớm.
    sorted_segments = sorted(segment_list, key=lambda x: x['start'])

    for i in range(len(sorted_segments)):
        for j in range(i + 1, len(sorted_segments)):
            seg1 = sorted_segments[i]
            seg2 = sorted_segments[j]

            # Vì list đã sort, nếu seg2 bắt đầu sau seg1 kết thúc thì các seg sau
            # cũng không thể overlap với seg1.
            if seg2['start'] >= seg1['end']:
                break

            # Giao nhau của hai khoảng thời gian [start, end].
            overlap_start = max(seg1['start'], seg2['start'])
            overlap_end = min(seg1['end'], seg2['end'])
            overlap_duration = overlap_end - overlap_start

            # Chỉ giữ overlap đủ dài để đáng chạy SepReformer.
            if overlap_duration >= overlap_threshold:
                overlapping_pairs.append({
                    'seg1': seg1,
                    'seg2': seg2,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_duration': overlap_duration
                })
                logger.info(f"Overlap detected: {overlap_duration:.2f}s between "
                           f"[{seg1['start']:.2f}-{seg1['end']:.2f}] and "
                           f"[{seg2['start']:.2f}-{seg2['end']:.2f}]")

    return overlapping_pairs

class SepReformerSeparator:
    """
    Lớp bao nạp SepReformer một lần và tái sử dụng cho nhiều segment overlap.

    SepReformer có cấu trúc package riêng cũng tên `models`/`utils`, dễ đụng
    module với repo hiện tại. Vì vậy phần __init__ tạm chỉnh sys.path và khôi
    phục lại sau khi nạp model.
    """
    def __init__(self, sepreformer_path, device):
        """
        Khởi tạo model SepReformer và nạp checkpoint.

        Tham số:
            sepreformer_path: đường dẫn thư mục SepReformer.
            device: torch device (cuda/cpu).
        """
        import sys
        import yaml

        self.sepreformer_path = sepreformer_path
        self.device = device

        print(f"[SepReformer] Initializing on device: {self.device}")

        # Lưu sys.path gốc để sau khi import SepReformer sẽ trả môi trường về như cũ.
        original_sys_path = sys.path.copy()

        try:
            # Ghi nhớ module `models`/`utils` hiện có của podcast-pipeline.
            original_models = sys.modules.get('models', None)
            original_utils = sys.modules.get('utils', None)

            # Tạm bỏ path của podcast-pipeline để import đúng package của SepReformer.
            podcast_pipeline_path = os.path.dirname(os.path.abspath(__file__))
            paths_to_remove = [p for p in sys.path if podcast_pipeline_path in p]
            for path in paths_to_remove:
                sys.path.remove(path)

            # Đưa SepReformer lên đầu sys.path để import model nội bộ của nó.
            if sepreformer_path not in sys.path:
                sys.path.insert(0, sepreformer_path)

            # Xoá module conflict khỏi sys.modules để Python không reuse nhầm module cũ.
            modules_to_clear = [key for key in sys.modules.keys()
                              if key.startswith('models.') or key.startswith('utils.') or key in ['models', 'utils']]
            cleared_modules = {}
            for module_name in modules_to_clear:
                cleared_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]

            # Nạp class Model của SepReformer sau khi đã xử lý path/module conflict.
            from models.SepReformer_Base_WSJ0.model import Model

            # Khôi phục module cũ để các phần khác của pipeline không bị ảnh hưởng.
            for module_name, module_obj in cleared_modules.items():
                sys.modules[module_name] = module_obj

            # Đọc config YAML đi kèm checkpoint SepReformer.
            config_path = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/configs.yaml")
            with open(config_path, 'r') as f:
                yaml_dict = yaml.safe_load(f)
            self.config = yaml_dict["config"]

            # Tạo kiến trúc model từ config.
            print("[SepReformer] Loading model...")
            self.model = Model(**self.config["model"])

            # Nạp checkpoint: ưu tiên pretrain_weights, dự phòng sang scratch_weights.
            checkpoint_dir = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/log/pretrain_weights")
            if not os.path.exists(checkpoint_dir) or not os.listdir(checkpoint_dir):
                checkpoint_dir = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/log/scratch_weights")

            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pt', '.pth'))]
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(device)
            self.model.eval()

            print("[SepReformer] Model initialization complete!")

        finally:
            # Luôn trả sys.path về trạng thái ban đầu, kể cả khi load lỗi.
            sys.path = original_sys_path

    def separate(self, audio_segment, sample_rate):
        """
        Tách audio overlap thành hai nguồn giọng nói.

        Tham số:
            audio_segment: waveform vùng overlap.
            sample_rate: sample rate hiện tại của waveform.

        Trả về:
            (separated_audio_1, separated_audio_2)
        """
        try:
            # Model SepReformer này được train ở 8 kHz, nên phải resample trước.
            if sample_rate != 8000:
                audio_8k = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=8000)
            else:
                audio_8k = audio_segment

            # Chuẩn bị tensor batch size 1.
            mixture_tensor = torch.tensor(audio_8k, dtype=torch.float32).unsqueeze(0)

            # Pad để độ dài chia hết cho stride encoder, tránh lỗi shape khi inference.
            stride = self.config["model"]["module_audio_enc"]["stride"]
            remains = mixture_tensor.shape[-1] % stride
            if remains != 0:
                padding = stride - remains
                mixture_padded = torch.nn.functional.pad(mixture_tensor, (0, padding), "constant", 0)
            else:
                mixture_padded = mixture_tensor

            # Chạy inference tách 2 source.
            with torch.inference_mode():
                nnet_input = mixture_padded.to(self.device)
                estim_src, _ = self.model(nnet_input)

                # Cắt bỏ phần padding và đưa về numpy.
                src1 = estim_src[0][..., :mixture_tensor.shape[-1]].squeeze().cpu().numpy()
                src2 = estim_src[1][..., :mixture_tensor.shape[-1]].squeeze().cpu().numpy()

            # Đưa source về sample_rate gốc để có thể chèn lại vào waveform pipeline.
            if sample_rate != 8000:
                src1 = librosa.resample(src1, orig_sr=8000, target_sr=sample_rate)
                src2 = librosa.resample(src2, orig_sr=8000, target_sr=sample_rate)

            return src1, src2

        except Exception as e:
            logger.error(f"SepReformer separation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return audio_segment, audio_segment


def identify_speaker_with_embedding(audio_segment, sample_rate, reference_embeddings, speaker_labels, embedding_model):
    """
    Xác định source tách ra thuộc speaker nào bằng embedding.

    Tham số:
        audio_segment: source audio sau khi tách.
        sample_rate: sample rate của audio.
        reference_embeddings: embedding tham chiếu theo speaker label.
        speaker_labels: các speaker có thể khớp.
        embedding_model: pyannote embedding model đã load.

    Trả về:
        Speaker label khớp nhất.
    """

    # Pyannote embedding kỳ vọng 16 kHz.
    if sample_rate != 16000:
        audio_16k = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=16000)
    else:
        audio_16k = audio_segment

    # Segment quá ngắn không đủ receptive field cho TDNN, fallback về speaker đầu.
    if len(audio_16k) < int(MIN_EMBED_DURATION * 16000):
        logger.warning(
            f"Embedding skip: segment too short ({len(audio_16k)/16000:.2f}s < {MIN_EMBED_DURATION}s); "
            f"falling back to first candidate {speaker_labels[0] if speaker_labels else 'Unknown'}"
        )
        return speaker_labels[0] if speaker_labels else None

    # Chuyển sang tensor batch size 1 để đưa vào embedding model.
    audio_tensor = torch.tensor(audio_16k, dtype=torch.float32).unsqueeze(0).to(device)

    # Trích embedding; nếu lỗi thì fallback để pipeline không dừng.
    try:
        with torch.inference_mode():
            embedding = embedding_model(audio_tensor)
    except Exception as e:
        logger.warning(
            f"Embedding model failed on segment ({len(audio_16k)/16000:.2f}s): {e}. "
            f"Falling back to first candidate {speaker_labels[0] if speaker_labels else 'Unknown'}"
        )
        return speaker_labels[0] if speaker_labels else None

    # So khớp với embedding tham chiếu bằng cosine similarity.
    best_speaker = None
    best_similarity = -1.0

    for speaker_label in speaker_labels:
        if speaker_label in reference_embeddings:
            ref_embedding = reference_embeddings[speaker_label]
            # Cosine similarity càng cao thì càng giống speaker tham chiếu.
            similarity = torch.nn.functional.cosine_similarity(
                embedding.mean(dim=1),
                ref_embedding.mean(dim=1),
                dim=0
            ).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_label

    logger.debug(f"Speaker identification: {best_speaker} (similarity: {best_similarity:.3f})")
    return best_speaker

def process_overlapping_segments_with_separation(
    segment_list,
    audio,
    overlap_threshold=1.0,
    separator=None,
    embedding_model=None,
    min_sepreformer_overlap=1.0,
    min_sepreformer_segment=1.0,
):
    """
    Xử lý các đoạn nói chồng nhau bằng SepReformer.

    Luồng xử lý:
    1. Mỗi segment được gắn `enhanced_audio` ban đầu bằng audio gốc của chính nó.
    2. Tìm các cặp segment overlap đủ dài.
    3. Tách vùng overlap thành hai source.
    4. Dùng speaker embedding để quyết định source nào thuộc speaker nào.
    5. Ghi source đã tách vào đúng vị trí tương đối trong `enhanced_audio`.
    6. Match âm lượng để tránh output sau tách bị to/nhỏ đột ngột.

    Tham số:
        segment_list: danh sách segment diarization.
        audio: dict audio chuẩn hoá.
        overlap_threshold: ngưỡng overlap cần xử lý.
        separator: SepReformerSeparator đã load.
        embedding_model: pyannote embedding model đã load.
        min_sepreformer_overlap: bỏ qua SepReformer nếu vùng overlap ngắn hơn mức này.
        min_sepreformer_segment: bỏ qua SepReformer nếu một trong hai segment quá ngắn.
    """
    if separator is None:
        logger.warning("SepReformer separator not provided, skipping separation")
        return audio, segment_list

    if embedding_model is None:
        logger.warning("Embedding model not provided, skipping separation")
        return audio, segment_list

    logger.info(
        "Processing overlapping segments with SepReformer "
        f"(threshold: {overlap_threshold}s, min_overlap: {min_sepreformer_overlap}s, "
        f"min_segment: {min_sepreformer_segment}s)"
    )

    # Hàm phụ khớp âm lượng để source tách ra không tạo bước nhảy volume khi
    # ghép lại với phần audio gốc của segment.
    def match_target_amplitude(source_wav, target_wav):
        """
        Khớp RMS của source_wav theo target_wav.
        """
        # Epsilon tránh chia cho 0 khi audio gần như im lặng.
        epsilon = 1e-10

        # RMS đại diện năng lượng âm lượng trung bình.
        src_rms = np.sqrt(np.mean(source_wav**2))
        tgt_rms = np.sqrt(np.mean(target_wav**2))
        
        if src_rms < epsilon:
            return source_wav
        
        # Gain cần nhân để source có RMS gần target.
        gain = tgt_rms / (src_rms + epsilon)
        
        # Áp dụng gain.
        adjusted_wav = source_wav * gain
        
        # Chặn clipping ngoài khoảng [-1, 1].
        return np.clip(adjusted_wav, -1.0, 1.0)

    # Khởi tạo enhanced_audio bằng audio gốc của từng segment. Vùng overlap sẽ
    # được thay thế sau, còn vùng không overlap giữ nguyên.
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    
    for seg in segment_list:
        if 'enhanced_audio' not in seg:
            start_frame = int(seg['start'] * sample_rate)
            end_frame = int(seg['end'] * sample_rate)
            seg['enhanced_audio'] = waveform[start_frame:end_frame].copy()
        
        if 'sepreformer' not in seg:
            seg['sepreformer'] = False

    # Tìm các cặp overlap cần xử lý.
    overlapping_pairs = detect_overlapping_segments(segment_list, overlap_threshold)

    if not overlapping_pairs:
        logger.info("No overlapping segments found")
        return audio, segment_list

    logger.info(f"Found {len(overlapping_pairs)} overlapping segment pairs")

    # Tạo embedding tham chiếu cho từng speaker từ các đoạn không overlap.
    # Các đoạn sạch này giúp nhận diện source sau khi SepReformer tách.
    reference_embeddings = {}
    all_speakers = list(set([seg['speaker'] for seg in segment_list]))

    # Mỗi speaker chỉ cần một đoạn đủ dài và không overlap để làm tham chiếu.
    for speaker in all_speakers:
        speaker_segments = [seg for seg in segment_list if seg['speaker'] == speaker]
        for seg in speaker_segments:
            is_overlapping = any(pair['seg1'] == seg or pair['seg2'] == seg for pair in overlapping_pairs)
            if not is_overlapping and (seg['end'] - seg['start']) >= 2.0:
                start_frame = int(seg['start'] * sample_rate)
                end_frame = int(seg['end'] * sample_rate)
                seg_audio = waveform[start_frame:end_frame]
                if sample_rate != 16000:
                    seg_audio_16k = librosa.resample(seg_audio, orig_sr=sample_rate, target_sr=16000)
                else:
                    seg_audio_16k = seg_audio

                if len(seg_audio_16k) < int(MIN_EMBED_DURATION * 16000):
                    logger.debug(
                        f"Skipping reference embedding for {speaker}: segment too short "
                        f"({len(seg_audio_16k)/16000:.2f}s < {MIN_EMBED_DURATION}s)"
                    )
                    continue

                seg_tensor = torch.tensor(seg_audio_16k, dtype=torch.float32).unsqueeze(0).to(device)
                try:
                    with torch.inference_mode():
                        embedding = embedding_model(seg_tensor)
                    reference_embeddings[speaker] = embedding
                    break
                except Exception as e:
                    logger.warning(f"Failed to compute reference embedding for {speaker}: {e}")
                    continue

    # Xử lý từng cặp overlap.
    for pair_idx, pair in enumerate(overlapping_pairs):
        overlap_start = pair['overlap_start']
        overlap_end = pair['overlap_end']
        seg1 = pair['seg1']
        seg2 = pair['seg2']
        
        seg1_speaker = seg1['speaker']
        seg2_speaker = seg2['speaker']

        skip_pair, skip_reasons = should_skip_sepreformer_pair(
            pair,
            min_overlap_seconds=min_sepreformer_overlap,
            min_segment_seconds=min_sepreformer_segment,
        )
        if skip_pair:
            skip_payload = {
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "reasons": skip_reasons,
            }
            seg1.setdefault("sepreformer_skip_reasons", []).append(skip_payload)
            seg2.setdefault("sepreformer_skip_reasons", []).append(skip_payload)
            logger.info(
                f"Skipping SepReformer overlap {overlap_start:.2f}-{overlap_end:.2f}: "
                f"{', '.join(skip_reasons)}"
            )
            continue

        # Cắt vùng audio đang chứa hai speaker nói chồng nhau.
        start_frame = int(overlap_start * sample_rate)
        end_frame = int(overlap_end * sample_rate)
        overlap_audio = waveform[start_frame:end_frame]

        # Tách vùng overlap thành hai source.
        separated_src1, separated_src2 = separator.separate(
            overlap_audio, sample_rate
        )

        # Xác định source thứ nhất gần với speaker nào hơn.
        speaker1_identity = identify_speaker_with_embedding(
            separated_src1, sample_rate, reference_embeddings, [seg1_speaker, seg2_speaker], embedding_model
        )
        
        if speaker1_identity == seg1_speaker:
            seg1_part = separated_src1
            seg2_part = separated_src2
        else:
            seg1_part = separated_src2
            seg2_part = separated_src1

        # Match RMS của source tách theo vùng overlap gốc để tránh source mới bị
        # vọt âm lượng. Dù overlap gốc có hai giọng nên RMS cao hơn từng source,
        # mốc này vẫn tự nhiên hơn để chống output bị spike.
        
        logger.debug(f"   Adjusting volume for overlap {pair_idx+1}...")
        seg1_part = match_target_amplitude(seg1_part, overlap_audio)
        seg2_part = match_target_amplitude(seg2_part, overlap_audio)
        # ---------------------------------------------------------------------

        # Ghi source của seg1 vào đúng vị trí tương đối trong enhanced_audio.
        seg1_start_global = int(seg1['start'] * sample_rate)
        rel_start_1 = start_frame - seg1_start_global
        
        limit_len_1 = min(len(seg1_part), len(seg1['enhanced_audio'][rel_start_1:]))
        if limit_len_1 > 0:
            seg1['enhanced_audio'][rel_start_1 : rel_start_1 + limit_len_1] = seg1_part[:limit_len_1]
            seg1['sepreformer'] = True
            logger.info(f"  ✓ Updated Seg1 enhanced_audio with volume-adjusted separated audio") 

        # Ghi source của seg2 vào đúng vị trí tương đối trong enhanced_audio.
        seg2_start_global = int(seg2['start'] * sample_rate)
        rel_start_2 = start_frame - seg2_start_global

        limit_len_2 = min(len(seg2_part), len(seg2['enhanced_audio'][rel_start_2:]))
        if limit_len_2 > 0:
            seg2['enhanced_audio'][rel_start_2 : rel_start_2 + limit_len_2] = seg2_part[:limit_len_2]
            seg2['sepreformer'] = True
            logger.info(f"  ✓ Updated Seg2 enhanced_audio with volume-adjusted separated audio")

    return audio, segment_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 03: detect and separate overlapping speech.")
    parser.add_argument("--cleaned_audio", required=True)
    parser.add_argument("--segment_flags_json", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--config_path", default="config.json")
    parser.add_argument("--overlap_threshold", type=float, default=0.2)
    parser.add_argument("--min_sepreformer_overlap", type=float, default=0.5)
    parser.add_argument("--min_sepreformer_segment", type=float, default=2.0)
    parser.add_argument("--skip_overlap_separation", action=argparse.BooleanOptionalAction, default=False)
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
    logger.info(f"Stage 03 device: {device_name}")

    start_time = time.time()
    seg_data = stage_common.load_json(args.segment_flags_json)
    audio_path = Path(args.cleaned_audio)
    out_dir = audio_path.parent
    if args.out:
        out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_json_path = out_dir / "segments.json"
    
    segments = seg_data.get("segments", [])

    if getattr(args, "skip_overlap_separation", False):
        logger.info("Skipping overlap separation due to --skip_overlap_separation")
    else:
        sample_rate = seg_data.get("sample_rate", int(cfg["entrypoint"]["SAMPLE_RATE"]))
        audio_info = stage_common.load_audio_info(audio_path, sample_rate)
        
        separator = SepReformerSeparator(device=device_name)
        
        speaker_embedder = None
        if cfg.get("huggingface_token", "").startswith("hf"):
            try:
                if Inference is None:
                    raise ImportError("pyannote.audio is not installed")
                speaker_embedder = Inference("pyannote/embedding", device=device, use_auth_token=cfg["huggingface_token"], window="whole")
            except Exception as exc:
                logger.warning(f"Speaker embedder unavailable: {exc}")
                
        segments = process_overlapping_segments_with_separation(
            segments,
            audio_info,
            separator,
            out_dir,
            speaker_embedder=speaker_embedder,
            overlap_threshold=args.overlap_threshold,
            min_overlap_duration=args.min_sepreformer_overlap,
            min_segment_duration=args.min_sepreformer_segment,
            logger=logger
        )

    elapsed = time.time() - start_time
    logger.info(f"Overlap separation finished in {elapsed:.2f}s.")

    out_data = {
        "audio_path": str(audio_path),
        "source_audio_path": seg_data.get("source_audio_path", ""),
        "audio_name": seg_data.get("audio_name", ""),
        "sample_rate": seg_data.get("sample_rate", 16000),
        "audio_duration_seconds": seg_data.get("audio_duration_seconds", 0.0),
        "segments": segments,
        "segment_demucs_flags": seg_data.get("segment_demucs_flags", []),
        "metadata": {
            "stage": "overlap_separate",
            "enabled": not getattr(args, "skip_overlap_separation", False),
            "processing_time_seconds": elapsed,
            "overlap_threshold_seconds": args.overlap_threshold,
            "min_sepreformer_overlap_seconds": args.min_sepreformer_overlap,
            "min_sepreformer_segment_seconds": args.min_sepreformer_segment,
        },
    }
    stage_common.dump_json(out_data, out_json_path)
    logger.info(f"Saved segments to {out_json_path}")

if __name__ == "__main__":
    main()
