# Sommelier
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT
import torch

# =============================================================================
# Tổng quan file
# =============================================================================
# File này là entrypoint chính của pipeline xử lý podcast/audio:
# 1. Đọc cấu hình và tham số dòng lệnh.
# 2. Chuẩn hoá audio đầu vào về mono WAV 24 kHz, 16-bit và mức âm lượng ổn định.
# 3. Chạy diarization để tách đoạn nói theo người nói.
# 4. Tuỳ chọn xoá nhạc nền bằng PANNs + Demucs.
# 5. Tuỳ chọn tách giọng nói chồng nhau bằng SepReformer.
# 6. Chạy ASR thường hoặc ASR MoE, sau đó xuất MP3 từng đoạn và JSON kết quả.
#
# Ghi chú quan trọng: các comment trong file giải thích "vì sao" từng bước tồn
# tại và dữ liệu được chuyển qua pipeline như thế nào. Logic xử lý không nên đổi
# khi chỉ chỉnh comment.

# PyTorch 2.6+ đổi mặc định weights_only=True, có thể làm pyannote load model lỗi.
# Đoạn patch này ép lightning_fabric dùng weights_only=False để giữ tương thích
# với checkpoint pyannote cũ.
import lightning_fabric.utilities.cloud_io as cloud_io
from pathlib import Path
from typing import Union, IO, Any

_original_load = cloud_io._load

def _patched_load(path_or_url: Union[IO, str, Path], map_location=None) -> Any:
    """Phiên bản _load đã vá để pyannote đọc được checkpoint cần weights_only=False."""
    if not isinstance(path_or_url, (str, Path)):
        return torch.load(path_or_url, map_location=map_location, weights_only=False)

    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(str(path_or_url), map_location=map_location)

    from lightning_fabric.utilities.cloud_io import get_filesystem
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location, weights_only=False)

cloud_io._load = _patched_load

# =============================================================================
# Import thư viện và module nội bộ
# =============================================================================
# Nhóm import bên dưới phục vụ toàn bộ pipeline: xử lý audio, diarization,
# ASR, tách nguồn, gọi API phụ trợ và xuất kết quả.
import argparse
import json
import librosa
import numpy as np
import ast
import sys
import os
import shutil
import tqdm
import re
import warnings
import tempfile
from openai import OpenAI

import requests
from pydub import AudioSegment
import os
from tritony import InferenceClient
import numpy as np
import librosa
from pyannote.audio import Pipeline, Inference
import pandas as pd
#from prompt import DIAR_PROMPT, WEAK_DIAR_PROMPT, NEW_DIAR_PROMPT, SPK_SUMMERIZE_PROMPT, NEW_DIAR_PROMPT_with_spk_inform, DIAR_PROMPT_KO
from utils.tool import (
    export_to_mp3,
    export_to_mp3_new,
    load_cfg,
    get_audio_files,
    detect_gpu,
    check_env,
    calculate_audio_stats,
)
from utils.asr_quality import choose_asr_text, should_skip_sepreformer_pair
from utils.logger import Logger, time_logger
from models import separate_fast, dnsmos, whisper_asr, silero_vad, vietnamese_asr
import time
import datetime
from panns_inference import AudioTagging
import soundfile as sf

from nemo.collections.asr.models import SortformerEncLabelModel

import json
import re
import argparse
from g2pk import G2p
import collections
import difflib
from typing import List, Tuple, Dict
from itertools import zip_longest

warnings.filterwarnings("ignore")






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

    df = df.copy()
    df["start"] = (df["start"].astype(float) + pad_onset).clip(lower=0.0)
    df["end"] = df["end"].astype(float) + pad_offset
    if audio_duration is not None and audio_duration > 0:
        df["end"] = df["end"].clip(lower=0.0, upper=float(audio_duration))
    else:
        df["end"] = df["end"].clip(lower=0.0)
    df["end"] = df[["start", "end"]].max(axis=1)

    return df
# =============================================================================
# Hằng số dùng xuyên suốt pipeline
# =============================================================================
audio_count = 0

# Giới hạn mỗi chunk diarization để tránh model xử lý audio quá dài một lần.
# Ưu tiên cắt tại khoảng lặng do VAD tìm được để hạn chế cắt ngang câu nói.
MAX_DIA_CHUNK_DURATION = 3 * 60  # giây; giữ dưới ngưỡng dài để Sortformer ổn định hơn
MIN_SPLIT_SILENCE = 1  # giây im lặng tối thiểu để được chọn làm điểm cắt
SILERO_MIN_SILENCE_DURATION_MS = 100
MIN_EMBED_DURATION = 0.5  # giây; segment ngắn hơn mức này sẽ bỏ qua embedding
QWEN_3_OMNI_PORT = "11500"

class RoverEnsembler:
    """
    Bộ ensemble ROVER (Recognizer Output Voting Error Reduction).

    Ý tưởng:
    - Nhận nhiều transcript từ các model ASR khác nhau như Whisper,
      PhoWhisper và ChunkFormer.
    - Căn chỉnh token giữa các transcript bằng Confusion Network.
    - Bỏ bớt transcript quá lệch và vote từng vị trí token để tạo câu cuối.
    """

    @staticmethod
    def build_confusion_network(all_tokens: List[List[str]]) -> List[List[str]]:
        """
        Tạo Confusion Network từ nhiều chuỗi token.

        Confusion Network là danh sách các "ô" theo vị trí. Mỗi ô chứa những
        token ứng viên mà các model khác nhau dự đoán ở cùng một vị trí tương
        đối. Cấu trúc này giúp vote dù mỗi model có thể thiếu/thừa token.

        Tham số:
            all_tokens: Danh sách token của từng transcript [[tok1, tok2, ...], ...]

        Trả về:
            Danh sách token ứng viên theo từng vị trí.
        """
        if not all_tokens:
            return []

        if len(all_tokens) == 1:
            return [[tok] for tok in all_tokens[0]]

        # Chọn chuỗi dài nhất làm pivot để làm trục căn chỉnh.
        # Với ASR, transcript quá ngắn thường bị mất từ, nên pivot dài hơn là
        # lựa chọn thực dụng hơn.
        pivot_idx = max(range(len(all_tokens)), key=lambda i: len(all_tokens[i]))
        pivot = all_tokens[pivot_idx]

        # Khởi tạo mỗi vị trí bằng token của pivot.
        confusion_net = [[pivot[i]] for i in range(len(pivot))]

        # Căn chỉnh từng transcript còn lại với pivot rồi thêm token ứng viên vào
        # vị trí tương ứng trong confusion network.
        for idx, tokens in enumerate(all_tokens):
            if idx == pivot_idx:
                continue

            matcher = difflib.SequenceMatcher(None, pivot, tokens)

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    # Khớp hoàn toàn: thêm token vào đúng vị trí của pivot.
                    for i, j in zip(range(i1, i2), range(j1, j2)):
                        if i < len(confusion_net):
                            confusion_net[i].append(tokens[j])

                elif tag == 'replace':
                    # Khác nhau: phân bổ token ứng viên vào các vị trí pivot gần nhất.
                    pivot_len = i2 - i1
                    cand_len = j2 - j1

                    if pivot_len == cand_len:
                        # Hai phía dài bằng nhau nên map 1-1.
                        for i, j in zip(range(i1, i2), range(j1, j2)):
                            if i < len(confusion_net):
                                confusion_net[i].append(tokens[j])
                    elif pivot_len > cand_len:
                        # Pivot dài hơn: rải token ứng viên vào các vị trí pivot.
                        for i in range(i1, i2):
                            offset = (i - i1) * cand_len // pivot_len
                            if j1 + offset < j2 and i < len(confusion_net):
                                confusion_net[i].append(tokens[j1 + offset])
                    else:
                        # Chuỗi ứng viên dài hơn: gom nhiều token ứng viên vào vị trí pivot đầu.
                        if i1 < len(confusion_net):
                            for j in range(j1, j2):
                                confusion_net[i1].append(tokens[j])

                elif tag == 'delete':
                    # Chỉ pivot có token: đã nằm sẵn trong confusion_net nên không cần thêm.
                    pass

                elif tag == 'insert':
                    # Chỉ candidate có token: gắn vào vị trí pivot gần nhất.
                    insert_pos = min(i1, len(confusion_net) - 1) if confusion_net else 0
                    if insert_pos >= 0 and insert_pos < len(confusion_net):
                        for j in range(j1, j2):
                            confusion_net[insert_pos].append(tokens[j])

        return confusion_net

    @staticmethod
    def has_local_repetition(output: List[str], word: str, window: int = 3) -> bool:
        """
        Kiểm tra một từ có bị lặp quá gần trong đoạn output vừa tạo hay không.

        Hàm này giúp tránh lỗi ASR kiểu "hello hello hello" do model bị lặp token.

        Tham số:
            output: Các từ đã được chọn trước đó.
            word: Từ đang chuẩn bị thêm.
            window: Số token gần nhất cần kiểm tra.

        Trả về:
            True nếu phát hiện lặp cục bộ, ngược lại False.
        """
        if len(output) < 1:
            return False

        # Chỉ nhìn vào vài từ gần nhất để không phạt các từ lặp hợp lý ở xa nhau.
        recent = output[-window:] if len(output) >= window else output

        # Nếu cùng một từ đã xuất hiện từ 2 lần trở lên trong cửa sổ gần nhất,
        # coi đó là dấu hiệu lặp bất thường.
        return recent.count(word) >= 2

    @staticmethod
    def calculate_transcript_similarity(t1_tokens: List[str], t2_tokens: List[str]) -> float:
        """
        Tính độ tương đồng giữa hai transcript bằng Jaccard similarity.

        Tham số:
            t1_tokens: Token của transcript thứ nhất.
            t2_tokens: Token của transcript thứ hai.

        Trả về:
            Điểm tương đồng từ 0 đến 1.
        """
        if not t1_tokens or not t2_tokens:
            return 0.0

        set1 = set(t1_tokens)
        set2 = set(t2_tokens)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def align_and_vote(transcripts: List[str]) -> str:
        """
        Căn chỉnh nhiều transcript rồi vote để tạo transcript cuối.

        Quy trình:
        - Bỏ transcript rỗng.
        - Tokenize theo khoảng trắng.
        - Tính độ tương đồng để phát hiện transcript quá lệch.
        - Tạo Confusion Network.
        - Vote từng vị trí, đồng thời tránh lặp token cục bộ.

        Tham số:
            transcripts: Kết quả từ nhiều model ASR, ví dụ [whisper, phowhisper, chunkformer].

        Trả về:
            Transcript cuối sau ensemble.
        """
        if not transcripts:
            return ""

        # Loại bỏ transcript rỗng để không làm nhiễu bước vote.
        transcripts = [t.strip() for t in transcripts if t and t.strip()]
        if not transcripts:
            return ""

        if len(transcripts) == 1:
            return transcripts[0]

        # Tách token đơn giản theo khoảng trắng; đủ dùng cho vote ở mức từ.
        all_tokens = [t.split() for t in transcripts]

        # Tính độ giống nhau giữa từng transcript với phần còn lại để phát hiện
        # model nào đang trả kết quả quá lệch.
        similarities = []
        for i in range(len(all_tokens)):
            sim_scores = []
            for j in range(len(all_tokens)):
                if i != j:
                    sim = RoverEnsembler.calculate_transcript_similarity(all_tokens[i], all_tokens[j])
                    sim_scores.append(sim)
            avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
            similarities.append(avg_sim)

        # Bản chép lời có độ tương đồng trung bình thấp hơn ngưỡng sẽ bị giảm tin cậy.
        outlier_threshold = 0.3
        trusted_indices = [i for i, sim in enumerate(similarities) if sim >= outlier_threshold]

        # Nếu tất cả đều bị coi là outlier, giữ lại hai transcript "ít lệch nhất"
        # để pipeline vẫn có dữ liệu vote thay vì trả rỗng.
        if len(trusted_indices) == 0:
            trusted_indices = [i for i, _ in sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:2]]

        # Căn chỉnh các token ứng viên theo vị trí tương đối.
        confusion_net = RoverEnsembler.build_confusion_network(all_tokens)

        # Bỏ phiếu từng vị trí trong confusion network.
        final_output = []
        for pos_idx, candidates in enumerate(confusion_net):
            if not candidates:
                continue

            # Bỏ token rỗng trước khi đếm phiếu.
            valid_candidates = [c for c in candidates if c]
            if not valid_candidates:
                continue

            # Ưu tiên token từ các transcript đáng tin cậy.
            trusted_candidates = []
            for i, cand in enumerate(valid_candidates):
                # Ước lượng nguồn token theo thứ tự đã thêm vào confusion_net.
                # Pivot luôn được giữ vì là trục căn chỉnh chính.
                if i == 0 or (i - 1) in trusted_indices:
                    trusted_candidates.append(cand)

            # Nếu không còn token tin cậy, quay về toàn bộ ứng viên để tránh mất chữ.
            if not trusted_candidates:
                trusted_candidates = valid_candidates

            # Chọn từ có nhiều phiếu nhất.
            votes = collections.Counter(trusted_candidates)
            best_word, count = votes.most_common(1)[0]

            # Nếu từ thắng vote gây lặp cục bộ, thử lấy ứng viên đứng thứ hai.
            if RoverEnsembler.has_local_repetition(final_output, best_word):
                if len(votes) > 1:
                    best_word = votes.most_common(2)[1][0]
                else:
                    # Không có ứng viên thay thế nên bỏ qua token này.
                    continue

            # Nếu đạt đa số thì nhận kết quả vote; nếu không, ưu tiên token pivot.
            if count >= len(trusted_candidates) / 2:
                final_output.append(best_word)
            else:
                pivot_word = candidates[0] if candidates[0] else best_word
                # Pivot cũng phải qua kiểm tra lặp để tránh kéo lỗi vào output.
                if RoverEnsembler.has_local_repetition(final_output, pivot_word):
                    if pivot_word != best_word:
                        final_output.append(best_word)
                else:
                    final_output.append(pivot_word)

        return " ".join(final_output)


class RepetitionFilter:
    """
    Bộ lọc transcript chất lượng thấp dựa trên n-gram bị lặp.

    Tiêu chí đang dùng: nếu một cụm 15 từ xuất hiện quá 5 lần, transcript có khả
    năng là lỗi hallucination/lặp của ASR và nên bị loại.
    """

    def __init__(self, use_mock_tokenizer=True):
        self.use_mock_tokenizer = use_mock_tokenizer

    def tokenize(self, text: str) -> List[str]:
        """Tokenize đơn giản theo khoảng trắng; có thể thay bằng SentencePiece khi cần."""
        if self.use_mock_tokenizer:
            return text.split()
        else:
            # Vị trí dự phòng nếu sau này muốn dùng tokenizer thật như SentencePiece.
            pass

    def filter(self, text: str) -> bool:
        """
        Điều kiện lọc:
        1. Loại transcript rỗng.
        2. Loại transcript có 15-gram lặp quá nhiều lần.

        Trả về:
            True nếu giữ lại, False nếu loại bỏ.
        """
        # Bản chép lời rỗng không có giá trị huấn luyện/đánh giá.
        if not text or not text.strip():
            logger.debug(f"[RepetitionFilter] Empty text detected.")
            return False

        tokens = self.tokenize(text)

        # Kiểm tra lặp cụm dài 15 token.
        N = 15
        THRESHOLD = 5

        if len(tokens) < N:
            return True  # Bản chép lời ngắn hơn 15 token thì không đủ điều kiện kiểm tra.

        # Sinh toàn bộ 15-gram liên tiếp.
        ngrams = [tuple(tokens[i:i+N]) for i in range(len(tokens) - N + 1)]

        # Đếm tần suất từng cụm.
        counts = collections.Counter(ngrams)

        # Nếu một cụm xuất hiện quá ngưỡng, coi là lỗi lặp.
        for ngram, count in counts.items():
            if count > THRESHOLD:
                logger.debug(f"[RepetitionFilter] Repetition detected! Span '{' '.join(ngram[:3])}...' occurs {count} times.")
                return False

        return True

import subprocess
import tempfile
import hashlib
from pathlib import Path
import subprocess
import os

# =============================================================================
# Chuyển đổi OPUS/OGG sang WAV
# =============================================================================
# Một số thư viện downstream đọc OPUS/OGG không ổn định. Vì vậy pipeline đổi các
# file này sang WAV trước khi xử lý. Bản cached dùng cho tiền xử lý hàng loạt,
# bản temporary dùng khi xử lý một file lẻ.

def convert_opus_to_wav_cached(audio_path: str, target_sr: int, cache_dir: str, logger, ffmpeg_threads: int = 1):
    """
    Chuyển .opus/.ogg sang .wav và lưu vào cache_dir.

    Cách hoạt động:
    - Nếu file không phải OPUS/OGG thì trả về path gốc.
    - Tên file cache gồm stem đã sanitize + hash theo path, mtime, size để tránh
      đụng tên giữa các file khác nhau.
    - Nếu file WAV cache đã tồn tại thì dùng lại, không decode lại.
    """
    lower = audio_path.lower()
    if not (lower.endswith(".opus") or lower.endswith(".ogg")):
        return audio_path

    os.makedirs(cache_dir, exist_ok=True)

    p = Path(audio_path)
    
    # Băm theo path + thời gian sửa + dung lượng để tránh trùng cache khi hai
    # file có cùng tên nhưng nội dung khác nhau.
    key = f"{str(p.resolve())}|{p.stat().st_mtime}|{p.stat().st_size}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    
    # Chuẩn hoá tên file: thay mọi ký tự không an toàn bằng "_".
    # Ví dụ: "![CDATA[Title]]" -> "__CDATA_Title__".
    safe_stem = re.sub(r'[^\w\-\.]', '_', p.stem)
    
    # Tạo tên file cache an toàn cho filesystem.
    out_wav = os.path.join(cache_dir, f"{safe_stem}.{h}.wav")

    # Nếu cache đã có sẵn thì trả về ngay.
    if os.path.exists(out_wav):
        return out_wav

    cmd = [
        "ffmpeg", "-y",
        "-threads", str(int(ffmpeg_threads)),   
        "-i", audio_path,
        "-ac", "1",
        "-ar", str(int(target_sr)),
        "-sample_fmt", "s16",
        out_wav,
    ]

    # Ghi log rõ nguồn và đích để dễ trace lỗi decode.
    logger.info(f"[OPUS][CACHE] Converting...\n  Src: {audio_path}\n  Dst: {out_wav}")
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or (not os.path.exists(out_wav)):
        logger.error(f"[OPUS][CACHE] ffmpeg failed.\nSTDERR:\n{proc.stderr}")
        raise RuntimeError(f"ffmpeg opus->wav conversion failed for {audio_path}")

    return out_wav

def convert_opus_to_wav_if_needed(audio_path: str, target_sr: int, logger):
    """
    Nếu input là .opus/.ogg thì đổi sang WAV tạm để xử lý.

    Trả về:
        (processing_path, temp_dir_or_None)
        - processing_path: path WAV tạm hoặc path gốc nếu không cần đổi.
        - temp_dir_or_None: thư mục tạm cần xoá sau khi xử lý xong.
    """
    lower = audio_path.lower()
    if not (lower.endswith(".opus") or lower.endswith(".ogg")):
        return audio_path, None

    temp_dir = tempfile.mkdtemp(prefix="opus2wav_")
    out_wav = os.path.join(temp_dir, "converted.wav")

    # Ưu tiên ffmpeg vì decode OPUS/OGG ổn định hơn pydub trong nhiều môi trường.
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ac", "1",
        "-ar", str(int(target_sr)),
        "-sample_fmt", "s16",
        out_wav,
    ]

    try:
        logger.info(f"[OPUS] Converting to wav via ffmpeg: {audio_path} -> {out_wav}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or (not os.path.exists(out_wav)):
            logger.error(f"[OPUS] ffmpeg conversion failed.\nSTDERR:\n{proc.stderr}")
            raise RuntimeError("ffmpeg opus->wav conversion failed")
        return out_wav, temp_dir
    except Exception as e:
        # Dự phòng sang pydub nếu gọi ffmpeg trực tiếp lỗi. Lưu ý pydub thường
        # vẫn cần ffmpeg bên dưới.
        logger.warning(f"[OPUS] ffmpeg path failed, trying pydub fallback: {e}")
        try:
            seg = AudioSegment.from_file(audio_path)
            seg = seg.set_channels(1).set_frame_rate(int(target_sr)).set_sample_width(2)
            seg.export(out_wav, format="wav")
            if not os.path.exists(out_wav):
                raise RuntimeError("pydub export failed")
            return out_wav, temp_dir
        except Exception as e2:
            logger.error(f"[OPUS] pydub fallback also failed: {e2}")
            # Dọn thư mục tạm nếu cả ffmpeg và pydub đều lỗi.
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

@time_logger
def standardization(audio):
    """
    Chuẩn hoá audio đầu vào trước khi đưa vào diarization/ASR.

    Tham số:
        audio (str or AudioSegment): Path file audio hoặc AudioSegment đã load.

    Trả về:
        dict chứa waveform đã chuẩn hoá, tên audio, sample rate và AudioSegment:
              {
                  "waveform": np.ndarray float32 dạng mono,
                  "name": str,
                  "sample_rate": int,
                  "audio_segment": AudioSegment đã chuẩn hoá
              }

    Ngoại lệ:
        ValueError: Nếu audio không phải path hoặc AudioSegment.
    """
    global audio_count
    name = "audio"

    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")

    logger.debug("Entering the preprocessing of audio")

    # Quy đổi về chuẩn chung của pipeline: sample rate cấu hình, 16-bit, mono.
    audio = audio.set_frame_rate(cfg["entrypoint"]["SAMPLE_RATE"])
    audio = audio.set_sample_width(2)  # 16-bit PCM
    audio = audio.set_channels(1)  # mono

    logger.debug("Audio file converted to WAV format")

    # Tính gain cần áp dụng để kéo âm lượng về mức mục tiêu.
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    # Giới hạn gain trong [-3, 3] dB để tránh tăng/giảm âm lượng quá mạnh.
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)

    # Bảo đảm waveform là 1 chiều vì các bước sau kỳ vọng mono.
    if waveform.ndim > 1:
        logger.warning(f"Waveform has {waveform.ndim} dimensions with shape {waveform.shape}, converting to mono")
        waveform = waveform.flatten()

    max_amplitude = np.max(np.abs(waveform))
    if max_amplitude > 0:
        waveform /= max_amplitude  # Đưa biên độ về khoảng [-1, 1].
    else:
        logger.warning("Audio has zero amplitude, skipping normalization")

    logger.debug(f"waveform shape: {waveform.shape}")
    logger.debug("waveform in np ndarray, dtype=" + str(waveform.dtype))

    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"],
        "audio_segment": normalized_audio,
    }


# =============================================================================
# Phát hiện và xử lý nhạc nền
# =============================================================================
@time_logger
def detect_background_music(audio, panns_model, threshold=0.3):
    """
    Phát hiện nhạc nền trên toàn bộ audio bằng PANNs.

    Tham số:
        audio: dict có waveform và sample_rate.
        panns_model: model PANNs đã load.
        threshold: ngưỡng xác suất để coi là có nhạc nền.

    Trả về:
        (has_music, music_prob)
    """
    if panns_model is None:
        logger.warning("PANNs model is not loaded, skipping music detection")
        return False, 0.0

    logger.debug("Detecting background music using PANNs")

    # PANNs kỳ vọng audio 32 kHz nên cần resample trước khi inference.
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if sample_rate != 32000:
        waveform_32k = librosa.resample(waveform, orig_sr=sample_rate, target_sr=32000)
    else:
        waveform_32k = waveform

    # Chạy inference PANNs; model đã được load ở phần __main__.
    (clipwise_output, embedding) = panns_model.inference(waveform_32k[None, :])

    # Lấy danh sách label AudioSet tương ứng với output.
    labels = panns_model.labels

    # Tìm xác suất của label "Music".
    music_idx = labels.index('Music') if 'Music' in labels else None
    if music_idx is not None:
        music_prob = float(clipwise_output[0, music_idx])
        logger.info(f"Music probability: {music_prob:.3f}")
        has_music = music_prob > threshold
        return has_music, music_prob
    else:
        logger.warning("Music label not found in PANNs output")
        return False, 0.0


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


@time_logger
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


@time_logger
def speaker_diarization(audio):
    """
    Chạy speaker diarization bằng pyannote pipeline.

    Tham số:
        audio: dict chứa waveform và sample_rate.

    Trả về:
        DataFrame gồm segment, label, speaker, start, end.
    """
    logger.debug(f"Start speaker diarization")
    logger.debug(f"audio waveform shape: {audio['waveform'].shape}")

    waveform = torch.tensor(audio["waveform"]).to(device)
    waveform = torch.unsqueeze(waveform, 0)

    segments = dia_pipeline(
        {
            "waveform": waveform,
            "sample_rate": audio["sample_rate"],
            "channel": 0,
        },
        max_speakers=4
    )

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    logger.debug(f"diarize_df: {diarize_df}")

    return diarize_df


@time_logger
def cut_by_speaker_label(vad_list):
    """
    Gộp/cắt segment theo speaker label để tạo các đoạn ASR hợp lý.

    Luật chính:
    - Segment quá dài bị cắt nhỏ theo MAX_SEGMENT_LENGTH.
    - Segment ngắn cùng speaker và sát nhau có thể được gộp theo MERGE_GAP.
    - Segment quá ngắn sau khi xử lý sẽ bị loại.

    Tham số:
        vad_list: danh sách segment có start, end, speaker.

    Trả về:
        Danh sách segment đã gộp/cắt/lọc.
    """
    MERGE_GAP = args.merge_gap  # giây; nhỏ hơn ngưỡng này thì có thể gộp
    MIN_SEGMENT_LENGTH = 3  # giây; đoạn ngắn hơn sẽ bị loại
    MAX_SEGMENT_LENGTH = 30  # giây; đoạn dài hơn sẽ bị cắt nhỏ

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            logger.warning(
                f"cut_by_speaker_label > segment longer than 30s, force trimming to 30s smaller segments"
            )
            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                vad["end"] = current_start + MAX_SEGMENT_LENGTH  # cập nhật end của chunk hiện tại
                updated_list.append(vad)
                vad = vad.copy()
                current_start += MAX_SEGMENT_LENGTH
                vad["start"] = current_start  # cập nhật start của chunk tiếp theo
                vad["end"] = segment_end
            updated_list.append(vad)
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]  # gộp bằng cách kéo dài end time

    logger.debug(
        f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments"
    )

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]

    logger.debug(
        f"cut_by_speaker_label > removed: {len(updated_list) - len(filter_list)} segments by length"
    )

    return filter_list

@time_logger
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


@time_logger
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


@time_logger
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


@time_logger
def asr(vad_segments, audio):
    """
    Chạy ASR Whisper cho từng segment.

    Điểm cần chú ý:
    - Hàm xử lý từng segment độc lập để có thể ưu tiên `enhanced_audio` nếu
      segment đã qua SepReformer.
    - Whisper nhận audio đã cắt riêng, nên timestamp nội bộ bắt đầu từ 0.
      Sau khi transcribe, start/end phải cộng lại `start_time` của segment để
      trở về timeline tuyệt đối của file gốc.
    """
    if len(vad_segments) == 0:
        return []

    # Waveform đầy đủ dùng làm dự phòng khi segment không có enhanced_audio.
    full_waveform = audio["waveform"]
    global_sample_rate = audio["sample_rate"]

    final_results = []

    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    # Xử lý từng segment một để metadata/enhanced_audio không bị lẫn giữa các đoạn.
    batch_size = 1

    if multilingual_flag:
        # Nhánh multilingual hiện chưa được triển khai trong flow segment-by-segment này.
        # Nếu cần bật lại, phải đảm bảo vẫn chuyển timestamp relative -> absolute.
        pass
        return []

    logger.info(f"ASR Processing: {len(vad_segments)} segments (Iterative Mode)")

    for idx, segment in enumerate(vad_segments):
        start_time = segment["start"]
        end_time = segment["end"]
        speaker = segment.get("speaker", "Unknown")

        # Bước 1: chọn nguồn audio cho segment.
        # Nếu SepReformer đã tách overlap thì dùng enhanced_audio; nếu không thì
        # cắt trực tiếp từ waveform đầy đủ.
        segment_audio = None
        is_enhanced = False

        if "enhanced_audio" in segment:
            # Ưu tiên audio đã tách giọng nói chồng nhau.
            raw_audio = segment["enhanced_audio"]
            is_enhanced = True
        else:
            # Không có enhanced_audio thì cắt theo start/end gốc.
            start_frame = int(start_time * global_sample_rate)
            end_frame = int(end_time * global_sample_rate)
            raw_audio = full_waveform[start_frame:end_frame]
            is_enhanced = False

        # Whisper/faster-whisper kỳ vọng 16 kHz.
        if global_sample_rate != 16000:
            segment_audio_16k = librosa.resample(raw_audio, orig_sr=global_sample_rate, target_sr=16000)
        else:
            segment_audio_16k = raw_audio

        # Bỏ qua đoạn quá ngắn, tránh model/feature extractor lỗi hoặc trả rỗng.
        if len(segment_audio_16k) < 160: 
            continue

        # Bước 2: tạo dummy VAD cho audio đã cắt.
        # Vì input vào Whisper chỉ là segment riêng lẻ, timeline local luôn là
        # 0 -> duration_sec.
        duration_sec = len(segment_audio_16k) / 16000
        dummy_vad = [{"start": 0.0, "end": duration_sec}]

        try:
            # Pipeline hiện cố định tiếng Anh. Nếu cần detect theo segment, có thể
            # bật lại dòng detect_language bên dưới.
            # language, prob = asr_model.detect_language(segment_audio_16k)
            language = "vi"

            transcribe_result = asr_model.transcribe(
                segment_audio_16k,
                dummy_vad,
                batch_size=batch_size,
                language=language,
                print_progress=False,
            )
            
            # Bước 3: chuẩn hoá kết quả về timeline gốc và gắn metadata segment.
            if transcribe_result and "segments" in transcribe_result:
                for res_seg in transcribe_result["segments"]:
                    # Chỉ giữ transcript có text.
                    if res_seg["text"].strip():
                        # Chuyển timestamp local của segment về timestamp tuyệt đối.
                        res_seg["start"] += start_time
                        res_seg["end"] += start_time
                        
                        # Gắn lại speaker và các flag xử lý audio.
                        res_seg["speaker"] = speaker
                        res_seg["language"] = transcribe_result.get("language", language)
                        res_seg["sepreformer"] = segment.get("sepreformer", False)
                        res_seg["is_separated"] = is_enhanced
                        
                        if is_enhanced:
                            res_seg["enhanced_audio"] = raw_audio

                        # Word timestamp cũng cần cộng offset giống segment timestamp.
                        if "words" in res_seg:
                            for w in res_seg["words"]:
                                w["start"] += start_time
                                w["end"] += start_time

                        final_results.append(res_seg)

        except Exception as e:
            logger.error(f"ASR failed for segment {idx} ({start_time:.2f}-{end_time:.2f}): {e}")
            continue

    return final_results

import concurrent.futures

@time_logger
def asr_MoE(
    vad_segments,
    audio,
    segment_demucs_flags=None,
    enable_word_timestamps=False,
    device="cuda",
    asr_quality_guard=True,
    asr_micro_segment_seconds=0.5,
    asr_short_segment_seconds=1.0,
    asr_vi_agreement_threshold=0.75,
    asr_context_pad_before=0.0,
    asr_context_pad_after=0.0,
):
    """
    Chạy ASR MoE cho từng segment bằng ba model tiếng Việt: Whisper,
    PhoWhisper và ChunkFormer.

    Quy trình mỗi segment:
    1. Chọn audio tốt nhất (`enhanced_audio` nếu có, nếu không cắt từ waveform gốc).
    2. Resample về 16 kHz.
    3. Gửi song song vào Whisper, PhoWhisper và ChunkFormer.
    4. Dùng ROVER để vote transcript cuối.
    5. Chạy guard chống hallucination cho các segment quá ngắn hoặc Whisper outlier.
    6. Gắn metadata và timestamp tuyệt đối để xuất JSON/MP3.
    """
    if len(vad_segments) == 0:
        return [], 0.0, 0.0

    if segment_demucs_flags is None:
        segment_demucs_flags = [False] * len(vad_segments)

    # Waveform đầy đủ dùng làm dự phòng nếu segment chưa có enhanced_audio.
    full_waveform = audio["waveform"]
    global_sample_rate = audio["sample_rate"]

    final_results = []
    total_whisper_time = 0.0
    total_alignment_time = 0.0
    
    rover = RoverEnsembler()
    phowhisper_transcriber = globals().get("phowhisper_model", None)
    chunkformer_transcriber = globals().get("chunkformer_model", None)

    def slice_full_audio(start_sec, end_sec):
        start_frame = max(0, int(round(float(start_sec) * global_sample_rate)))
        end_frame = min(len(full_waveform), int(round(float(end_sec) * global_sample_rate)))
        if end_frame <= start_frame:
            return np.zeros(0, dtype=np.float32)
        return full_waveform[start_frame:end_frame]

    # Hàm phụ chạy Whisper cho một segment. Trả về text, language, word timestamps
    # nếu bật và thời gian xử lý để tính RT factor.
    def run_whisper_task(segment_audio_16k, dummy_vad):
        w_start = time.time()
        try:
            transcribe_result = asr_model.transcribe(
                segment_audio_16k, 
                dummy_vad, 
                batch_size=1, 
                print_progress=False
            )
            
            text_whisper = ""
            detected_language = "vi"
            words = []

            if transcribe_result and "segments" in transcribe_result and len(transcribe_result["segments"]) > 0:
                text_whisper = " ".join([s["text"] for s in transcribe_result["segments"]]).strip()
                detected_language = transcribe_result.get("language", "vi")
                if enable_word_timestamps:
                    for s in transcribe_result["segments"]:
                        if "words" in s: words.extend(s["words"])
            
            w_end = time.time()
            return {
                "text": text_whisper,
                "language": detected_language,
                "words": words,
                "time": w_end - w_start
            }
        except Exception as e:
            logger.error(f"Whisper failed: {e}")
            return {"text": "", "language": "vi", "words": [], "time": 0.0}

    def run_phowhisper_task(segment_audio_16k):
        try:
            return vietnamese_asr.transcribe(phowhisper_transcriber, segment_audio_16k)
        except Exception as e:
            logger.error(f"PhoWhisper failed: {e}")
            return ""

    def run_chunkformer_task(segment_audio_16k):
        try:
            return vietnamese_asr.transcribe(chunkformer_transcriber, segment_audio_16k)
        except Exception as e:
            logger.error(f"ChunkFormer failed: {e}")
            return ""
    # ThreadPoolExecutor cho phép 3 model chạy gần như đồng thời.
    # Dù Python có GIL, các tác vụ C++/CUDA thường release GIL nên vẫn có ích.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

        for idx, segment in enumerate(vad_segments):
            start_time = segment["start"]
            end_time = segment["end"]
            speaker = segment.get("speaker", "Unknown")
            
            # Chọn nguồn audio cho segment. ASR có thể nghe thêm context trước/sau
            # nhưng timestamp xuất ra vẫn giữ start/end gốc.
            is_enhanced = False
            segment_duration_sec = max(0.0, float(end_time) - float(start_time))
            pad_before = max(0.0, float(asr_context_pad_before))
            pad_after = max(0.0, float(asr_context_pad_after))

            if "enhanced_audio" in segment:
                # Segment đã được tách overlap: giữ core enhanced, chỉ lấy context
                # trước/sau từ full waveform để không mất chữ sát biên.
                prefix = slice_full_audio(start_time - pad_before, start_time) if pad_before else np.zeros(0, dtype=np.float32)
                suffix = slice_full_audio(end_time, end_time + pad_after) if pad_after else np.zeros(0, dtype=np.float32)
                enhanced_core = np.asarray(segment["enhanced_audio"], dtype=np.float32).reshape(-1)
                raw_audio = np.concatenate([prefix, enhanced_core, suffix]).astype(np.float32)
                is_enhanced = True
            else:
                raw_audio = slice_full_audio(start_time - pad_before, end_time + pad_after)
            
            # Cả ba model ASR ở đây dùng input 16 kHz.
            if global_sample_rate != 16000:
                segment_audio_16k = librosa.resample(raw_audio, orig_sr=global_sample_rate, target_sr=16000)
            else:
                segment_audio_16k = raw_audio

            # Bỏ qua đoạn quá ngắn để tránh lỗi inference.
            if len(segment_audio_16k) < 160: 
                continue

            # Dummy VAD cho Whisper vì audio đã là segment cắt sẵn.
            padded_duration_sec = len(segment_audio_16k) / 16000
            dummy_vad = [{"start": 0.0, "end": padded_duration_sec}]

            # Gửi cùng một segment tới ba model ASR.
            future_whisper = executor.submit(run_whisper_task, segment_audio_16k, dummy_vad)
            future_phowhisper = executor.submit(run_phowhisper_task, segment_audio_16k)
            future_chunkformer = executor.submit(run_chunkformer_task, segment_audio_16k)

            # Điểm chờ: đợi đủ ba model trả kết quả trước khi ensemble.
            whisper_res = future_whisper.result()
            text_phowhisper = future_phowhisper.result()
            text_chunkformer = future_chunkformer.result()

            # Tách kết quả Whisper để vừa lấy text vừa lấy word timestamp/thời gian.
            text_whisper = whisper_res["text"]
            detected_language = whisper_res["language"]
            words = whisper_res["words"]
            total_whisper_time += whisper_res["time"]

            # Bỏ phiếu transcript cuối từ ba nguồn ASR.
            text_ensemble = rover.align_and_vote([text_whisper, text_phowhisper, text_chunkformer])
            quality_decision = choose_asr_text(
                rover_text=text_ensemble,
                text_whisper=text_whisper,
                text_phowhisper=text_phowhisper,
                text_chunkformer=text_chunkformer,
                duration_sec=segment_duration_sec,
                enabled=asr_quality_guard,
                micro_segment_seconds=asr_micro_segment_seconds,
                short_segment_seconds=asr_short_segment_seconds,
                vi_agreement_threshold=asr_vi_agreement_threshold,
            )
            text_ensemble = quality_decision["text"]
            if quality_decision["actions"]:
                logger.info(
                    f"ASR quality guard segment {idx} ({start_time:.2f}-{end_time:.2f}s): "
                    f"{quality_decision['actions']} -> {quality_decision['source']}"
                )

            seg_result = {
                "start": start_time,
                "end": end_time,
                "text": text_ensemble,
                "text_whisper": text_whisper,
                "text_phowhisper": text_phowhisper,
                "text_chunkformer": text_chunkformer,
                "speaker": speaker,
                "language": detected_language,
                "demucs": segment_demucs_flags[idx] if idx < len(segment_demucs_flags) else False,
                "is_separated": is_enhanced, 
                "sepreformer": segment.get("sepreformer", False),
                "asr_quality_source": quality_decision["source"],
                "asr_quality_actions": quality_decision["actions"],
                "asr_context_pad_before": pad_before,
                "asr_context_pad_after": pad_after,
            }
            
            if is_enhanced:
                seg_result["enhanced_audio"] = raw_audio

            if enable_word_timestamps and words:
                # Word timestamp từ Whisper đang là timeline local của segment,
                # cần cộng start_time để trở về timeline file gốc.
                for w in words:
                    w["start"] += start_time
                    w["end"] += start_time
                seg_result["words"] = words

            final_results.append(seg_result)

    return final_results, total_whisper_time, total_alignment_time

def add_qwen3omni_caption(filtered_list, audio, save_path):
    """
    Gọi Qwen3-Omni API cho từng segment ASR để thêm audio caption.

    Tham số:
        filtered_list: danh sách segment sau ASR.
        audio: dict audio có waveform và sample_rate.
        save_path: thư mục dùng để ghi file WAV tạm.

    Trả về:
        (danh sách segment đã thêm qwen3omni_caption, thời gian xử lý giây)
    """
    import soundfile as sf

    logger.info(f"Adding Qwen3-Omni captions to {len(filtered_list)} segments...")
    caption_start_time = time.time()

    for idx, segment in enumerate(filtered_list):
        try:
            # Cắt audio của segment. Nếu segment có enhanced_audio thì dùng nó để
            # caption khớp với file MP3 sẽ xuất.
            if "enhanced_audio" in segment:
                segment_audio = segment["enhanced_audio"]
                sample_rate = audio["sample_rate"]
            else:
                start_time = segment["start"]
                end_time = segment["end"]
                sample_rate = audio["sample_rate"]
                start_frame = int(start_time * sample_rate)
                end_frame = int(end_time * sample_rate)
                segment_audio = audio["waveform"][start_frame:end_frame]

            # Ghi WAV tạm vì API nội bộ nhận audio_url dạng file://.
            temp_audio_path = os.path.join(save_path, f"temp_segment_{idx:05d}.wav")
            sf.write(temp_audio_path, segment_audio, sample_rate)

            # Gọi service Qwen3-Omni đang chạy local.
            url = f"http://localhost:{QWEN_3_OMNI_PORT}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}

            # Ở môi trường này dùng file://. Khi deploy qua network có thể cần URL
            # mà service Qwen3-Omni truy cập được hoặc base64 audio.
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {"url": f"file://{temp_audio_path}"}
                            }
                        ]
                    }
                ]
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                # Lấy nội dung caption từ response OpenAI-compatible.
                caption = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                segment["qwen3omni_caption"] = caption
                logger.debug(f"Segment {idx}: Successfully added Qwen3-Omni caption")
            else:
                logger.warning(f"Segment {idx}: API call failed with status {response.status_code}")
                segment["qwen3omni_caption"] = ""

            # Xoá WAV tạm ngay sau khi gọi API xong.
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        except Exception as e:
            logger.error(f"Segment {idx}: Error calling Qwen3-Omni API: {e}")
            segment["qwen3omni_caption"] = ""

    caption_end_time = time.time()
    caption_processing_time = caption_end_time - caption_start_time

    logger.info("Qwen3-Omni caption addition completed")
    return filtered_list, caption_processing_time


# =============================================================================
# Tiện ích hậu xử lý văn bản/LLM
# =============================================================================
# Các hàm dưới đây dùng cho phần gắn speaker tag, parse kết quả LLM và tính chi
# phí ước lượng khi có gọi model ngôn ngữ.
def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Tính chi phí ước lượng theo bảng giá hard-code cho một số model OpenAI."""
    pricing = {
        "gpt-4.1": {
            "input": 2.00 / 1_000_000,
            "cached_input": 0.50 / 1_000_000,
            "output": 8.00 / 1_000_000,
        },
        "gpt-4.1-mini": {
            "input": 0.40 / 1_000_000,
            "cached_input": 0.10 / 1_000_000,
            "output": 1.60 / 1_000_000,
        },
        "gpt-4.1-nano": {
            "input": 0.10 / 1_000_000,
            "cached_input": 0.025 / 1_000_000,
            "output": 0.40 / 1_000_000,
        },
        "openai-o3": {
            "input": 2.00 / 1_000_000,
            "cached_input": 0.50 / 1_000_000,
            "output": 8.00 / 1_000_000,
        },
        "openai-o4-mini": {
            "input": 1.10 / 1_000_000,
            "cached_input": 0.275 / 1_000_000,
            "output": 4.40 / 1_000_000,
        },
    }

    if model_name not in pricing:
        raise ValueError(f"Model '{model_name}' not found in pricing table.")

    rates = pricing[model_name]
    input_cost = input_tokens * rates["input"]
    output_cost = output_tokens * rates["output"]
    total_cost = input_cost + output_cost

    return total_cost
import json
from collections import defaultdict



def speaker_tagged_text(data):
    """
    Thêm tag speaker vào text và đánh lại số speaker theo thứ tự xuất hiện.

    Ví dụ:
    - Speaker gốc có thể là SPEAKER_02, SPEAKER_00, SPEAKER_01.
    - Output sẽ đổi thành [s0], [s1], [s2] theo thứ tự xuất hiện trong transcript.
    """
    # Bước 1: tạo tag ban đầu và ghi nhận thứ tự xuất hiện của speaker.
    initially_tagged_data = []
    unique_speakers_in_order = []
    seen_speakers = set()

    for item in data:
        # Chuyển SPEAKER_01 -> [s1].
        speaker_num = item['speaker'].replace('SPEAKER_', '')
        original_tag = f"[s{int(speaker_num)}]"

        # Ghi nhận speaker mới theo đúng thứ tự xuất hiện trong transcript.
        if original_tag not in seen_speakers:
            unique_speakers_in_order.append(original_tag)
            seen_speakers.add(original_tag)
        
        # Lưu tạm tag gốc để lát nữa map sang tag tuần tự mới.
        initially_tagged_data.append({
            'text': item['text'],
            'start': item['start'],
            'end': item['end'],
            'original_tag': original_tag
        })

    # Bước 2: tạo mapping từ tag gốc sang tag tuần tự mới.
    # Ví dụ: {'[s2]': '[s0]', '[s0]': '[s1]', '[s1]': '[s2]'}.
    speaker_map = {
        original_tag: f"[s{i}]" 
        for i, original_tag in enumerate(unique_speakers_in_order)
    }

    # Bước 3: áp dụng mapping để tạo output cuối.
    result = []
    for item in initially_tagged_data:
        original_tag = item['original_tag']
        new_tag = speaker_map[original_tag]  # lấy tag mới từ mapping
        
        final_item = {
            'text': f"{new_tag}{item['text']}",  # thêm tag speaker vào đầu text
            'start': item['start'],
            'end': item['end']
        }
        result.append(final_item)
        
    return result

import re
import json
import ast

def parse_speaker_summary(llm_output: str) -> list | None:
    """
    Trích và parse JSON array từ output LLM.

    Hàm chịu được output có code fence, tiền tố json hoặc whitespace thừa, miễn
    là bên trong có một mảng JSON dạng [...].
    """
    if not llm_output:
        return None

    try:
        # Tìm phần nằm giữa dấu [ và ] đầu/cuối, kể cả khi LLM bọc trong code block.
        match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            # Chuyển chuỗi JSON thành list/dict Python.
            return json.loads(json_str)
        else:
            print("Parsing Error: Could not find a valid JSON array format ([]).")
            return None
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"Unknown parsing error: {e}")
        return None

def process_llm_diarization_output(llm_output: str) -> list[dict]:

    # Bước 1: ưu tiên lấy nội dung trong code block ```json ... ```.
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_output)
    if not json_match:
        # Nếu không có code block, thử parse toàn bộ string.
        json_string = llm_output
    else:
        json_string = json_match.group(1)

    # Bước 2: parse JSON string thành object Python.
    try:
        llm_data = json.loads(json_string)
    except json.JSONDecodeError:
        # Dự phòng khi LLM trả format gần giống Python literal.
        try:
            # ast.literal_eval an toàn hơn eval vì chỉ parse literal.
            import ast
            llm_data = ast.literal_eval(json_string)
        except (ValueError, SyntaxError) as e:
            print(f"Error: Both JSON and Python literal parsing failed. {e!r}")
            return []


    return llm_data

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


def deduplicate_segments_by_index(segments: list[dict], logger=None) -> list[dict]:
    """
    Bảo đảm mỗi index segment chỉ xuất hiện một lần, giữ bản đầu tiên.
    """
    seen = set()
    deduped = []
    for seg in segments:
        idx = seg.get("index")
        if idx is None or idx not in seen:
            if idx is not None:
                seen.add(idx)
            deduped.append(seg)
        else:
            if logger:
                logger.warning(f"Duplicate segment index detected and skipped: {idx}")
    return deduped

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
        embeddings = []
        for _, row in rows.sort_values("start").iterrows():
            emb = _extract_speaker_embedding(
                audio_info, row["start"], row["end"], embedder=embedder
            )
            if emb is not None:
                embeddings.append(emb)
            if len(embeddings) >= 3:
                break
        if embeddings:
            centroids[speaker] = np.mean(embeddings, axis=0)
    return centroids


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

    return aligned_frames


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

def ko_transliterate_english(text: str) -> str:
    """
    Tìm đoạn tiếng Anh trong text và chuyển sang cách đọc tiếng Hàn bằng G2P.
    """
    def _repl(m: re.Match) -> str:
        segment = m.group(0)
        return G2P(segment)
    return ENG_PATTERN.sub(_repl, text)


def ko_process_json(input_list: str) -> None:
    """Hậu xử lý list segment: nếu text có chữ Latin thì chuyển sang phát âm Hàn."""
    for entry in input_list:
        text = entry.get("text", "")
        # Chỉ chuyển khi text có ký tự Latin.
        if re.search(r"[A-Za-z]", text):
            entry["text"] = ko_transliterate_english(text)

def export_segments_with_enhanced_audio(audio_info, segment_list, save_dir, audio_name):
    """
    Xuất từng segment thành MP3.

    Quy tắc chọn audio:
    - Nếu segment có `is_separated=True` và `enhanced_audio`, dùng audio đã tách
      bởi SepReformer.
    - Nếu không, cắt trực tiếp từ audio gốc/đã chuẩn hoá.
    """
    import os
    from pydub import AudioSegment as PydubAudioSegment
    
    # Thư mục chứa MP3 của từng segment.
    segments_dir = os.path.join(save_dir, audio_name)
    os.makedirs(segments_dir, exist_ok=True)
    
    # AudioSegment đầy đủ dùng để cắt các segment không có enhanced_audio.
    full_audio_segment = audio_info.get("audio_segment")
    sample_rate = audio_info["sample_rate"]
    
    if full_audio_segment is None:
        # Dự phòng: dựng AudioSegment từ waveform float32.
        waveform_int16 = (audio_info["waveform"] * 32767).astype(np.int16)
        full_audio_segment = PydubAudioSegment(
            waveform_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

    logger.info(f"Exporting {len(segment_list)} segments with enhanced audio check...")

    for i, seg in enumerate(segment_list):
        # Tên file gồm index và speaker để dễ tra lại JSON.
        idx_str = seg.get("index", f"{i:05d}")
        spk = seg.get("speaker", "Unknown")
        filename = f"{idx_str}_{spk}.mp3"
        file_path = os.path.join(segments_dir, filename)

        # Ưu tiên audio đã tách từ SepReformer nếu segment có.
        if seg.get("is_separated", False) and "enhanced_audio" in seg:
            # Chuyển numpy waveform -> Pydub AudioSegment.
            enhanced_waveform = seg["enhanced_audio"]

            # float32 [-1, 1] -> int16; clip để tránh overflow.
            enhanced_waveform = np.clip(enhanced_waveform, -1.0, 1.0)
            wav_int16 = (enhanced_waveform * 32767).astype(np.int16)

            target_segment = PydubAudioSegment(
                wav_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            # Có thể bật log này nếu cần xác nhận segment đã dùng output SepReformer.

        else:
            # Không có enhanced_audio thì cắt từ full_audio_segment.
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            target_segment = full_audio_segment[start_ms:end_ms]

        # Ghi file MP3 segment.
        target_segment.export(file_path, format="mp3")
        
def main_process(audio_path, save_path=None, audio_name=None,
                 do_vad = False,
                 LLM = "",
                 use_demucs = False,
                 use_sepreformer = False,
                 overlap_threshold = 1.0,
                 min_sepreformer_overlap = 1.0,
                 min_sepreformer_segment = 1.0,
                 sepreformer_separator = None,
                 embedding_model = None,
                 panns_model = None,
                 speaker_embedder = None,
                 speaker_link_threshold: float = 0.75):

    """
    Xử lý một file audio từ đầu đến cuối.

    Luồng chính:
    1. Decode OPUS/OGG nếu cần.
    2. Chuẩn hoá audio.
    3. Chia chunk và chạy Sortformer diarization.
    4. Nối speaker giữa các chunk bằng embedding.
    5. Cắt segment dài, xử lý nhạc nền bằng Demucs nếu bật.
    6. Tách đoạn nói chồng bằng SepReformer nếu bật.
    7. Chạy ASR thường hoặc ASR MoE.
    8. Tuỳ chọn caption Qwen3-Omni, Korean G2P.
    9. Xuất MP3 từng segment và JSON metadata.
    """
    
    proc_audio_path = audio_path
    opus_temp_dir = None
    try:
        # Bước 0a: đảm bảo input OPUS/OGG được đổi sang WAV tạm trước khi load.
        target_sr = int(cfg["entrypoint"]["SAMPLE_RATE"])
        proc_audio_path, opus_temp_dir = convert_opus_to_wav_if_needed(
            audio_path, target_sr=target_sr, logger=logger
        )

        if not proc_audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac", ".opus")):
            logger.warning(f"Unsupported file type: {proc_audio_path}")

        # Tạo tên audio và thư mục output. Tên thư mục encode các flag quan trọng
        # để dễ phân biệt kết quả giữa nhiều lần chạy.
        audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
        suffix = "dia3" if args.dia3 else "ori"
        save_path = save_path or os.path.join(
            os.path.dirname(audio_path), "_final", f"-sepreformer-{args.sepreformer}" +f"-demucs-{args.demucs}"  + f"-vad-{do_vad}"+ f"-diaModel-{suffix}"
            # Ghi rõ trạng thái bật/tắt initial prompt trong tên thư mục kết quả.
            + f"-initPrompt-{args.initprompt}"
            + f"-merge_gap-{args.merge_gap}" +f"-seg_th-{args.seg_th}"+ f"-cl_min-{args.min_cluster_size}" +f"-cl-th-{args.clust_th}"+ f"-LLM-{LLM}", audio_name
        )
        os.makedirs(save_path, exist_ok=True)
        logger.debug(
            f"Processing audio: {audio_name}, from {audio_path}, save to: {save_path}"
        )

        logger.info(
            "Step 0: Preprocess all audio files --> 24k sample rate + wave format + loudnorm + bit depth 16"
        )
        # Bước 0b: chuẩn hoá audio về format chung của pipeline.
        audio = standardization(audio_path)

        # Bước 1: chuẩn bị chunk cho diarization. Mỗi chunk có offset để quy đổi
        # timestamp local của chunk về timestamp toàn file.
        diar_chunks, temp_chunk_dir = prepare_diarization_chunks(audio_path, audio)

        # Thời lượng audio dùng để tính RT factor cho từng stage.
        audio_duration = len(audio["waveform"]) / audio["sample_rate"]
        logger.info(f"Total audio duration: {audio_duration:.2f} seconds")

        logger.info("Step 2: Speaker Diarization")
        dia_start = time.time()


        diarization_frames = []
        try:
            # Chạy Sortformer trên từng chunk. Output của chunk được cộng offset để
            # quay về timeline tuyệt đối của file gốc.
            for chunk in diar_chunks:
                predicted_segments, _ = diar_model.diarize(
                    audio=chunk["path"], batch_size=1, include_tensor_outputs=True
                )
                chunk_df = sortformer_dia(predicted_segments)
                if not chunk_df.empty:
                    chunk_df["start"] += chunk["offset"]
                    chunk_df["end"] += chunk["offset"]
                    chunk_df = _apply_sortformer_segment_padding_from_args(
                        chunk_df, args=args, logger=logger, audio_duration=audio_duration
                    )
                diarization_frames.append(chunk_df)
        finally:
            # Dọn file chunk tạm sau khi diarization hoàn tất.
            if temp_chunk_dir:
                shutil.rmtree(temp_chunk_dir, ignore_errors=True)

        if diarization_frames:
            # Sortformer chạy theo chunk nên speaker label chỉ có nghĩa cục bộ.
            # Bước này dùng embedding để nối label speaker xuyên chunk.
            diarization_frames = align_speakers_across_chunks(
                diarization_frames,
                audio_info=audio,
                embedder=speaker_embedder,
                similarity_threshold=speaker_link_threshold,
            )

        if diarization_frames:
            speakerdia = pd.concat(diarization_frames, ignore_index=True)
        else:
            # Trường hợp model không trả segment nào, vẫn tạo DataFrame đúng schema.
            speakerdia = pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])
        ori_list = df_to_list(speakerdia)
        dia_end = time.time()

        # RT factor = thời gian xử lý / thời lượng audio. <1 nghĩa là nhanh hơn realtime.
        vad_sortformer_processing_time = dia_end - dia_start
        vad_sortformer_rt = vad_sortformer_processing_time / audio_duration if audio_duration > 0 else 0
        logger.info(f"VAD + Sortformer - Processing time: {vad_sortformer_processing_time:.2f}s, RT factor: {vad_sortformer_rt:.4f}")

        # Chuẩn hoá danh sách segment trước ASR: cắt các đoạn quá dài để ASR ổn định.
        segment_list = ori_list
        segment_list = split_long_segments(segment_list)

        # Bước 3 chạy trước SepReformer: nếu audio có nhạc nền, làm sạch trước để
        # tách giọng overlap dễ hơn.
        logger.info("Step 3: Background Music Detection and Removal")
        # Padding giúp Demucs xử lý cả phần nhạc nền sát biên segment.
        audio, segment_demucs_flags = preprocess_segments_with_demucs(segment_list, audio, panns_model=panns_model, use_demucs=use_demucs, padding=0.5)

        # Bước 2.5: sau khi audio đã được làm sạch nhạc nền, tách vùng hai người
        # nói chồng nhau bằng SepReformer nếu flag được bật.
        logger.info("Step 2.5: Overlap Control with SepReformer")
        separation_time = 0.0
        if use_sepreformer and sepreformer_separator is not None and embedding_model is not None:
            separation_start = time.time()
            # Tại đây audio đã qua Demucs nếu --demucs bật.
            audio, segment_list = process_overlapping_segments_with_separation(
                segment_list,
                audio,
                overlap_threshold=overlap_threshold,
                separator=sepreformer_separator,
                embedding_model=embedding_model,
                min_sepreformer_overlap=min_sepreformer_overlap,
                min_sepreformer_segment=min_sepreformer_segment,
            )
            separation_end = time.time()
            separation_time = separation_end - separation_start

            # Tính tốc độ xử lý SepReformer so với realtime.
            separation_rt = separation_time / audio_duration if audio_duration > 0 else 0
            logger.info(f"SepReformer separation - Processing time: {separation_time:.2f}s, RT factor: {separation_rt:.4f}")
        else:
            logger.info("SepReformer overlap separation skipped (flag disabled)")
            
        logger.info("Step 4: ASR (Automatic Speech Recognition)")
        if args.ASRMoE:
            # ASR MoE: chạy Whisper + PhoWhisper + ChunkFormer rồi ROVER vote.
            asr_start = time.time()

            asr_result, whisper_time, alignment_time = asr_MoE(
                segment_list,
                audio,
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

            asr_end = time.time()

            dia_time = dia_end-dia_start
            asr_time = whisper_time
        else:
            # ASR thường: chỉ dùng Whisper.
            asr_start = time.time()

            asr_result = asr(segment_list, audio)

            asr_end = time.time()

            dia_time = dia_end-dia_start
            asr_time = asr_end-asr_start
            alignment_time = 0.0

        # Tính RT factor cho Whisper large-v3.
        whisper_processing_time = asr_time
        whisper_rt = whisper_processing_time / audio_duration if audio_duration > 0 else 0

        # Alignment time hiện chỉ có ý nghĩa khi word timestamp được bật.
        alignment_rt = alignment_time / audio_duration if audio_duration > 0 else 0

        if LLM == "case_0":
            # case_0: không hậu xử lý LLM, dùng trực tiếp kết quả ASR.
            print("LLM case_0")
            filtered_list = asr_result
            print(f"ASR result contains {len(filtered_list)} segments")




        # case_2: nhánh giữ chỗ cho LLM post-processing. Hiện tại các hàm LLM
        # đang tắt nên dùng trực tiếp ASR result.
        elif LLM == "case_2":
            print(f"asr_result len: {len(asr_result)}")
            print("Warning: llm_inference functions are commented out. Using ASR results directly.")
            filtered_list = asr_result
            

        else:
            raise ValueError("LLM variable must be one of case_0, case_1, case_2.")

        # Bước 4.5: thêm caption audio bằng Qwen3-Omni nếu bật.
        caption_time = 0.0
        if args.qwen3omni:
            logger.info("Step 4.5: Adding Qwen3-Omni captions")
            filtered_list, caption_time = add_qwen3omni_caption(filtered_list, audio, save_path)
        else:
            logger.info("Step 4.5: Qwen3-Omni caption generation skipped (flag disabled)")

        # RT factor cho bước caption.
        caption_rt = caption_time / audio_duration if audio_duration > 0 else 0

        # In thống kê thời gian để dễ benchmark từng stage.
        print(f"\n{'='*60}")
        print(f"Audio duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)")
        print(f"{'='*60}")
        print(f"VAD + Sortformer:")
        print(f"  - Processing time: {dia_time:.2f} seconds")
        print(f"  - RT factor: {vad_sortformer_rt:.4f}")
        print(f"{'='*60}")
        if use_sepreformer:
            print(f"SepReformer Overlap Separation:")
            print(f"  - Processing time: {separation_time:.2f} seconds")
            print(f"  - RT factor: {separation_rt:.4f}")
            print(f"{'='*60}")
        print(f"Whisper large v3:")
        print(f"  - Processing time: {asr_time:.2f} seconds")
        print(f"  - RT factor: {whisper_rt:.4f}")
        print(f"{'='*60}")
        if args.whisperx_word_timestamps:
            print(f"WhisperX Alignment:")
            print(f"  - Processing time: {alignment_time:.2f} seconds")
            print(f"  - RT factor: {alignment_rt:.4f}")
            print(f"{'='*60}")
        if args.qwen3omni:
            print(f"Qwen3-Omni Caption:")
            print(f"  - Processing time: {caption_time:.2f} seconds")
            print(f"  - RT factor: {caption_rt:.4f}")
            print(f"{'='*60}")
        print()

        logger.info("Step 5: Write result into MP3 and JSON file")
        print(f"Exporting {len(filtered_list)} segments to MP3 and JSON...")
        # Xuất MP3 từng segment. Nếu segment có enhanced_audio, hàm export sẽ dùng
        # audio đã xử lý thay vì cắt từ bản gốc.
        export_segments_with_enhanced_audio(audio, filtered_list, save_path, audio_name)

        # Hậu xử lý Korean G2P nếu bật.
        if args.korean:
            ko_process_json(filtered_list)

        # JSON không nên chứa numpy array audio lớn. Trước khi dump, xoá
        # enhanced_audio và chuyển numpy scalar sang kiểu Python native.
        cleaned_list = []
        for item in filtered_list:
            # Copy nông để không làm mất enhanced_audio trong filtered_list trả về.
            clean_item = item.copy()
            
            # Bỏ ma trận audio khỏi JSON output.
            if "enhanced_audio" in clean_item:
                del clean_item["enhanced_audio"]
            # json.dump không serialize numpy scalar trực tiếp được.
            for k, v in clean_item.items():
                if hasattr(v, 'item'):  # numpy scalar
                    clean_item[k] = v.item()
                    
            cleaned_list.append(clean_item)

        # Chuẩn bị JSON cuối gồm metadata benchmark và danh sách segment.
        output_data = {
            "metadata": {
                "audio_duration_seconds": audio_duration,
                "audio_duration_minutes": audio_duration / 60,
                "vad_sortformer": {
                    "processing_time_seconds": vad_sortformer_processing_time,
                    "rt_factor": vad_sortformer_rt
                },
                "whisper_large_v3": {
                    "processing_time_seconds": whisper_processing_time,
                    "rt_factor": whisper_rt
                },
                # Dùng cleaned_list vì đây là dữ liệu thật sự được ghi ra JSON.
                "total_segments": len(cleaned_list)
            },
            # Không ghi enhanced_audio vào JSON để tránh file cực lớn.
            "segments": cleaned_list  
        }

        # Gắn metadata cho các stage tuỳ chọn nếu được bật.
        if args.whisperx_word_timestamps:
            output_data["metadata"]["whisperx_alignment"] = {
                "processing_time_seconds": alignment_time,
                "rt_factor": alignment_rt,
                "enabled": True
            }

        if args.qwen3omni:
            output_data["metadata"]["qwen3omni_caption"] = {
                "processing_time_seconds": caption_time,
                "rt_factor": caption_rt,
                "enabled": True
            }

        if use_sepreformer:
            output_data["metadata"]["sepreformer_separation"] = {
                "processing_time_seconds": separation_time,
                "rt_factor": separation_rt,
                "overlap_threshold_seconds": overlap_threshold,
                "min_sepreformer_overlap_seconds": min_sepreformer_overlap,
                "min_sepreformer_segment_seconds": min_sepreformer_segment,
                "enabled": True
            }

        final_path = os.path.join(save_path, audio_name + ".json")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"All done, Saved to: {final_path}")
        print(f"Processing complete! Results saved to: {final_path}")
        print(f"Total segments processed: {len(filtered_list)}")
        return final_path, filtered_list
    finally:
        # Xoá thư mục WAV tạm tạo từ OPUS/OGG nếu có.
        if opus_temp_dir:
            shutil.rmtree(opus_temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # =============================================================================
    # Điểm vào CLI
    # =============================================================================
    # Phần này đọc tham số dòng lệnh, load toàn bộ model dùng chung, quét thư mục
    # input và gọi `main_process` cho từng file audio.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="",
        help="input folder path, this will override config if set",
    )
    parser.add_argument(
        "--config_path", type=str, default="config.json", help="config path"
    )
    
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    # Kích thước batch chỉ dùng cho các nhánh/model có hỗ trợ batch.

    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help="The compute type to use for the model",
    )
    parser.add_argument(
        "--whisper_arch",
        type=str,
        default="large-v3",
        help="The name of the Whisper model to load.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="The number of CPU threads to use per worker, e.g. will be multiplied by num workers.",
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
    parser.add_argument(
        "--exit_pipeline",
        type=bool,
        default=False,
        help="Exit pipeline when task done.",
    )
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turning on vad.",
    )
    parser.add_argument(
        "--dia3",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turning on diaization 3.0 model",
    )
    parser.add_argument(
        "--LLM",
        type=str,
        default="case_2",
        help="LLM diarization cases",
    )

    # Siêu tham số cho diarization/segment merging.
    parser.add_argument(
        "--seg_th",
        type=float,
        default=0.15,
        help="diarization model segmentation threshold",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=10,
        help="diarization model clustering min_cluster_size",
    )
    parser.add_argument(
        "--clust_th",
        type=float,
        default=0.5,
        help="diarization model clustering threshold",
    )

    parser.add_argument(
        "--merge_gap",
        type=float,
        default=2,
        help="merge gap in seconds, if smaller than this, merge",
    )
    parser.add_argument(
        "--speaker-link-threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for linking speakers across diarization chunks",
    )

    parser.add_argument(
        "--initprompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Turning on initial prompt on whisper model",
    )
    parser.add_argument(
        "--korean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="korean_g2p",
    )

    parser.add_argument(
        "--ASRMoE",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Vietnamese ASR ensemble with Whisper, PhoWhisper, and ChunkFormer.",
    )
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
        help="Seconds of audio context to prepend during ASR while keeping original timestamps.",
    )
    parser.add_argument(
        "--asr_context_pad_after",
        type=float,
        default=0.0,
        help="Seconds of audio context to append during ASR while keeping original timestamps.",
    )

    parser.add_argument(
        "--demucs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable background music detection and removal using PANNs and Demucs",
    )

    parser.add_argument(
        "--whisperx_word_timestamps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable WhisperX word-level timestamps with alignment",
    )

    parser.add_argument(
        "--qwen3omni",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen3-Omni audio captioning for each segment",
    )

    parser.add_argument(
        "--sepreformer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable SepReformer for overlapping speech separation",
    )

    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=1.0,
        help="Minimum overlap duration in seconds to trigger SepReformer separation",
    )
    parser.add_argument(
        "--min_sepreformer_overlap",
        type=float,
        default=1.0,
        help="Skip SepReformer when an overlap region is shorter than this many seconds.",
    )
    parser.add_argument(
        "--min_sepreformer_segment",
        type=float,
        default=1.0,
        help="Skip SepReformer when either overlapped segment is shorter than this many seconds.",
    )

    # Tuỳ chọn chỉnh biên segment Sortformer sau khi model trả output.
    parser.add_argument(
        "--sortformer-param",
        "--sortformerParam",
        "--sortformerParma",
        dest="sortformer_param",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Sortformer segment boundary adjustment (pad_offset/pad_onset applied after model output).",
    )
    parser.add_argument( "--sortformer-pad-offset", 
                        type=float, 
                        default=-0.24, 
                        help="Seconds to add to segment end time (negative pulls ends earlier). Used with --sortformer-param.", )
    
    parser.add_argument( "--sortformer-pad-onset", 
                        type=float, 
                        default=0.0, 
                        help="Seconds to add to segment start time (negative pulls starts earlier). Used with --sortformer-param.", )
    
    parser.add_argument(
    "--opus_decode_workers",
    type=int,
    default=8,
    help="Number of parallel workers for opus/ogg -> wav decoding pre-stage.",
    )
    parser.add_argument(
        "--ffmpeg_threads_per_decode",
        type=int,
        default=1,
        help="ffmpeg -threads per decode process (keep small when using many workers).",
    )

    args = parser.parse_args()

    # Đọc config và khởi tạo logger dùng chung toàn file.
    batch_size = args.batch_size
    cfg = load_cfg(args.config_path)

    logger = Logger.get_logger()

    if args.input_folder_path:
        # Tham số CLI có độ ưu tiên cao hơn config.json.
        logger.info(f"Using input folder path: {args.input_folder_path}")
        cfg["entrypoint"]["input_folder_path"] = args.input_folder_path

    logger.debug("Loading models...")

    # Chọn thiết bị chạy model. Nếu không có GPU, ép compute_type về int8 để
    # Whisper chạy được trên CPU.
    if detect_gpu():
        logger.info("Using GPU")
        device_name = "cuda"
        device = torch.device(device_name)
    else:
        logger.info("Using CPU")
        device_name = "cpu"
        device = torch.device(device_name)
        # WhisperX trên CPU kỳ vọng compute_type là int8.
        logger.info("Overriding the compute type to int8")
        args.compute_type = "int8"

    cuda_device_count = torch.cuda.device_count() if device_name == "cuda" else 0
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

        vi_asr_device = torch.device(f"cuda:{selected_vi_asr_device_index}")
        logger.info(
            "ASR device plan: Whisper on cuda:%s, Vietnamese ASRMoE models on %s, CUDA devices=%s",
            whisper_device_index,
            vi_asr_device,
            cuda_device_count,
        )

    check_env(logger)

    # -------------------------------------------------------------------------
    # Nạp model diarization
    # -------------------------------------------------------------------------
    # Pyannote cần Hugging Face token hợp lệ và quyền truy cập model.
    logger.debug(" * Loading Speaker Diarization Model")
    if not cfg["huggingface_token"].startswith("hf"):
        raise ValueError(
            "huggingface_token must start with 'hf', check the config file. "
            "You can get the token at https://huggingface.co/settings/tokens. "
            "Remeber grant access following https://github.com/pyannote/pyannote-audio?tab=readme-ov-file#tldr"
        )
    if args.dia3 == True:
        print("Using diarization-3.1 model")
        dia_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        #"pyannote/speaker-diarization",
        use_auth_token=cfg["huggingface_token"],

    )
        dia_pipeline.to(device)
        
    else:
        dia_pipeline = Pipeline.from_pretrained(
            #"pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization",
            use_auth_token=cfg["huggingface_token"]
        )
        dia_pipeline.to(device)

        # Siêu tham số cho pyannote diarization bản cũ.
        dia_pipeline.instantiate({
        "segmentation": {
            "min_duration_off": 0.0, 
            "threshold": args.seg_th
        },
        "clustering": {
            "method": "centroid",
            "min_cluster_size": args.min_cluster_size,
            "threshold": args.clust_th   
        }
    })
    # -------------------------------------------------------------------------
    # Nạp model ASR Whisper
    # -------------------------------------------------------------------------
    logger.debug(" * Loading ASR Model")

    if args.initprompt == True:
        # Initial prompt giúp Whisper nhận diện filler words tốt hơn. Prompt hiện
        # giữ nhiều ngôn ngữ để giảm khả năng model bỏ sót các từ đệm.
        asr_options_dict = {
            #"log_prob_threshold": -1.0,
            #"no_speech_threshold": 0.6,
            # 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음

            # Prompt gốc được giữ lại làm tham chiếu khi cần so sánh hành vi.
            # "initial_prompt": "ha. heh. Mm, hmm. Mm hm. uh. Uh huh. Mm huh. Uh. hum Uh. Ah. Uh hu. Like. you know. Yeah. I mean. right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. Hoo. Hu. Hoo, hoo. Heah. Ha. Yu. Nah. Uh-huh. No way. Uh-oh. Jeez. Whoa. Dang. Gosh. Duh. Whoops. Phew. Woo. Ugh. Er. Geez. Oh wow. Oh man. Uh yeah. Uh huh. For real?",

            #"initial_prompt": "ha. heh. Mm, hmm. uh. "

            # Ghi chú thực nghiệm về initial_prompt:
            # - Không có initial_prompt: ít bị gap ASR nhưng nhận diện filler word kém.
            # - Prompt gốc: ít gap nhưng chưa nhận diện đúng filler mong muốn.
            # - Prompt khác hoàn toàn, bỏ CJK: đôi khi sinh gap ASR.
            # - Chỉnh prompt gốc và giữ ít filler word hơn: cân bằng tốt hơn.


            "initial_prompt": "Um. Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. Mm. So. Oh. Hoo hoo.生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음.",

        }
        # Bật word timestamp nếu cần WhisperX alignment.
        if args.whisperx_word_timestamps:
            asr_options_dict["word_timestamps"] = True

        asr_model = whisper_asr.load_asr_model(
            args.whisper_arch,
            device_name,
            device_index=whisper_device_index,
            compute_type=args.compute_type,
            threads=args.threads,
            language="vi",

        # Các option mặc định khác nằm trong models/whisper_asr.py.

            asr_options=asr_options_dict,
        )
    else:
        asr_options_dict = {}
        # Nếu tắt initial prompt nhưng vẫn cần word timestamp, chỉ truyền option này.
        if args.whisperx_word_timestamps:
            asr_options_dict["word_timestamps"] = True

        asr_model = whisper_asr.load_asr_model(
            args.whisper_arch,
            device_name,
            device_index=whisper_device_index,
            compute_type=args.compute_type,
            threads=args.threads,

            language="vi",
            asr_options=asr_options_dict if asr_options_dict else None,

            )
    if args.ASRMoE:
        # ASR MoE tiếng Việt: Whisper + PhoWhisper + ChunkFormer.
        logger.debug(" * Loading Vietnamese ASRMoE Models")
        phowhisper_model = vietnamese_asr.load_phowhisper_model(device=vi_asr_device)
        chunkformer_model = vietnamese_asr.load_chunkformer_model(device=vi_asr_device)
        logger.debug(f" * Vietnamese ASRMoE models loaded on {vi_asr_device}")
        # Client OpenAI từng dùng cho nhánh LLM; hiện không khởi tạo ở đây.
    # client = OpenAI(api_key="YOUR_API_KEY")
    model_name = "gpt-4.1"

    # VAD dùng để tìm khoảng lặng khi chia chunk diarization.
    logger.debug(" * Loading VAD Model")
    vad = silero_vad.SileroVAD(device=device)
    
    # G2P dùng cho hậu xử lý tiếng Hàn nếu bật --korean.
    G2P = G2p()

    # Regex tìm cụm tiếng Anh để chuyển sang phát âm Hàn.
    ENG_PATTERN = re.compile(r"[A-Za-z][A-Za-z']*(?: [A-Za-z][A-Za-z']*)*")

    # Speaker embedder dùng để nối speaker label giữa các chunk diarization.
    speaker_embedder = None
    try:
        speaker_embedder = Inference(
            "pyannote/embedding",
            device=device,
            use_auth_token=cfg["huggingface_token"],
            window="whole",
        )
        logger.debug(" * Speaker embedding model loaded for cross-chunk linking")
    except Exception as e:
        logger.error(f" * Failed to load speaker embedding model: {e}")
        speaker_embedder = None

    # Sortformer là model diarization chính trong flow hiện tại.
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    diar_model.eval()

    # Embedding model phục vụ SepReformer: sau khi tách hai source, dùng embedding
    # để gán source về đúng speaker.
    embedding_model = None
    if args.sepreformer:
        logger.debug(" * Loading Pyannote Embedding Model")
        try:
            from pyannote.audio import Model as PyannoteModel
            embedding_model = PyannoteModel.from_pretrained("pyannote/embedding", use_auth_token=cfg["huggingface_token"])
            embedding_model = embedding_model.to(device)
            logger.debug(" * Pyannote Embedding Model loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load Pyannote Embedding Model: {e}")
            embedding_model = None

    # Nạp SepReformer chỉ khi cần xử lý đoạn nói chồng.
    sepreformer_separator = None
    if args.sepreformer:
        logger.debug(" * Loading SepReformer Separator Model")
        try:
            sepreformer_separator = SepReformerSeparator(
                sepreformer_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SepReformer"),
                device=device
            )
            logger.debug(" * SepReformer Separator loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load SepReformer Separator: {e}")
            sepreformer_separator = None

    # PANNs dùng để quyết định segment nào có nhạc nền và cần Demucs.
    panns_model = None
    if args.demucs:
        logger.debug(" * Loading PANNs Model for background music detection")
        try:
            panns_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "panns_data")
            os.makedirs(panns_data_dir, exist_ok=True)
            os.environ['PANNS_DATA'] = panns_data_dir
            checkpoint_path = os.path.join(panns_data_dir, 'Cnn14_mAP=0.431.pth')
            panns_model = AudioTagging(checkpoint_path=checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            logger.debug(" * PANNs Model loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load PANNs Model: {e}")
            panns_model = None

    logger.debug("All models loaded")

    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    logger.debug(f"supported languages multilingual {supported_languages}")
    logger.debug(f"using multilingual asr {multilingual_flag}")

    input_folder_path = cfg["entrypoint"]["input_folder_path"]

    if not os.path.exists(input_folder_path):
        raise FileNotFoundError(f"input_folder_path: {input_folder_path} not found")

    # Quét file audio trực tiếp trong input_folder_path, không recursive.
    audio_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.opus', '.ogg')
    audio_paths = []
    
    for file in os.listdir(input_folder_path):
        # Chỉ nhận extension audio được hỗ trợ.
        if not file.lower().endswith(audio_extensions):
            continue
            
        # Bỏ file tạm.
        if ".temp" in file:
            continue
            
        # Không cho chạy trực tiếp trên thư mục cache OPUS để tránh xử lý lại file
        # trung gian như input thật.
        if "_opus_cache" in input_folder_path:
            logger.warning(f"Skipping execution because input path looks like a cache dir: {input_folder_path}")
            sys.exit(0)

        audio_paths.append(os.path.join(input_folder_path, file))

    logger.debug(f"Scanning {len(audio_paths)} audio files in {input_folder_path} (non-recursive)")

    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Tiền decode OPUS/OGG hàng loạt để pipeline chính chỉ nhận WAV/format dễ đọc.
    target_sr = int(cfg["entrypoint"]["SAMPLE_RATE"])
    opus_cache_dir = os.path.join(input_folder_path, f"_opus_cache_wav_{target_sr}")

    # Lọc riêng OPUS/OGG để decode song song.
    to_decode = [p for p in audio_paths if p.lower().endswith((".opus", ".ogg"))]
    if to_decode:
        logger.info(f"[OPUS] Pre-decoding {len(to_decode)} files with {args.opus_decode_workers} workers...")
        decoded_map = {}

        def _job(p):
            return p, convert_opus_to_wav_cached(
                p,
                target_sr=target_sr,
                cache_dir=opus_cache_dir,
                logger=logger,
                ffmpeg_threads=args.ffmpeg_threads_per_decode,
            )

        with ThreadPoolExecutor(max_workers=int(args.opus_decode_workers)) as ex:
            futures = [ex.submit(_job, p) for p in to_decode]
            
            for fut in as_completed(futures):
                try:
                    src, dst = fut.result()
                    decoded_map[src] = dst
                except Exception as e:
                    # Một file decode lỗi không được làm dừng cả batch.
                    logger.error(f"❌ [OPUS] Failed to decode file, skipping: {e}")
                    # File lỗi không được thêm vào decoded_map nên sẽ bị loại ở bước dưới.

        # Thay OPUS/OGG đã decode thành path WAV cache. File OPUS/OGG decode lỗi
        # sẽ bị loại khỏi danh sách xử lý.
        new_audio_paths = []
        for p in audio_paths:
            if p in decoded_map:
                new_audio_paths.append(decoded_map[p])
            elif p.lower().endswith((".opus", ".ogg")):
                # Không có trong decoded_map nghĩa là decode lỗi.
                logger.warning(f"Skipping failed file: {p}")
            else:
                # File không cần decode giữ nguyên.
                new_audio_paths.append(p)
        
        audio_paths = new_audio_paths
        logger.info(f"[OPUS] Pre-decoding done. Valid files: {len(audio_paths)}")
    else:
        logger.info("[OPUS] No opus/ogg files found; skipping pre-decoding.")

    start_time = time.time()
    start_time = time.time()
    
    # Đếm số file thành công/thất bại để tổng kết cuối batch.
    success_count = 0
    fail_count = 0

    for path in audio_paths:
        try:
            # Bỏ file quá nhỏ vì thường là file hỏng hoặc chưa ghi xong.
            if os.path.getsize(path) < 1024:
                logger.warning(f"⚠️ [Skip] File too small ({os.path.getsize(path)} bytes): {path}")
                fail_count += 1
                continue

            # Xử lý từng file độc lập. Lỗi ở một file sẽ được catch bên dưới.
            main_process(path, do_vad=args.vad, LLM=args.LLM, use_demucs=args.demucs,
                         use_sepreformer=args.sepreformer, overlap_threshold=args.overlap_threshold,
                         min_sepreformer_overlap=args.min_sepreformer_overlap,
                         min_sepreformer_segment=args.min_sepreformer_segment,
                         sepreformer_separator=sepreformer_separator,
                         embedding_model=embedding_model, panns_model=panns_model,
                         speaker_embedder=speaker_embedder,
                         speaker_link_threshold=args.speaker_link_threshold)
            
            success_count += 1

        except Exception as e:
            # Không dừng cả batch khi một file lỗi; log rồi xử lý file tiếp theo.
            logger.error(f"❌ [Error] Failed to process file: {path}")
            logger.error(f"   Reason: {e}")
            fail_count += 1
            continue

    end_time = time.time()
    print(f"Directory processing finished.")
    print(f" - Success: {success_count}")
    print(f" - Failed: {fail_count}")
    print(f"Total time: {end_time - start_time:.2f}s")
