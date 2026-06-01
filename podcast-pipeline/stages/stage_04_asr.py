from __future__ import annotations
import argparse
import shutil
import time
import json
import collections
import difflib
from itertools import zip_longest
from typing import List, Tuple, Dict
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np
import librosa
import torch
import concurrent.futures
import warnings

import stage_common
from utils.tool import load_cfg
from utils.asr_quality import choose_asr_text
try:
    from utils.logger import Logger
except ImportError:
    import logging
    class Logger:
        @staticmethod
        def get_logger():
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger("stage_04")

from models import whisper_asr, vietnamese_asr

warnings.filterwarnings("ignore")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 04: run ASR.")
    parser.add_argument("--overlap_json", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--config_path", default="config.json")
    parser.add_argument("--whisper_arch", default="large-v3", help="Model arch for WhisperX")
    parser.add_argument("--device", default="")
    parser.add_argument("--asr-moe", action=argparse.BooleanOptionalAction, default=True, help="Use multiple ASR models (Whisper, PhoWhisper, Chunkformer) and ROVER voting")
    parser.add_argument("--word-timestamps", action=argparse.BooleanOptionalAction, default=False, help="Enable word-level timestamps (slower)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logger = Logger.get_logger()
    cfg = load_cfg(args.config_path)

    device_name = args.device
    if not device_name:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    logger.info(f"Stage 04 device: {device_name}")

    start_time = time.time()
    seg_data = stage_common.load_json(args.overlap_json)
    audio_path = Path(seg_data["audio_path"])
    out_dir = audio_path.parent
    if args.out:
        out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_json_path = out_dir / "transcript.json"
    segments = seg_data.get("segments", [])
    sample_rate = seg_data.get("sample_rate", int(cfg["entrypoint"]["SAMPLE_RATE"]))
    
    pipe_args = SimpleNamespace(
        asr_moe=args.asr_moe,
        whisper_arch=args.whisper_arch,
        word_timestamps=args.word_timestamps,
    )

    if args.asr_moe:
        asr_model = whisper_asr.load_asr_model(
            args.whisper_arch,
            device_name,
            compute_type=cfg["model"]["whisper"]["compute_type"]
        )
        phowhisper_model = vietnamese_asr.load_phowhisper_model(
            "vinai/PhoWhisper-large", device=device_name
        )
        chunkformer_model = vietnamese_asr.load_chunkformer_model(
            "trungdt21/chunkformer-v2", device=device_name
        )
        
        segments, stats = asr_MoE(
            segments,
            str(audio_path),
            asr_model,
            phowhisper_model,
            chunkformer_model,
            pipe_args,
            logger
        )
    else:
        asr_model = whisper_asr.load_asr_model(
            args.whisper_arch,
            device_name,
            compute_type=cfg["model"]["whisper"]["compute_type"]
        )
        segments, stats = asr(
            segments,
            str(audio_path),
            asr_model,
            pipe_args,
            logger
        )

    elapsed = time.time() - start_time
    logger.info(f"ASR finished in {elapsed:.2f}s.")

    out_data = {
        "audio_path": str(audio_path),
        "source_audio_path": seg_data.get("source_audio_path", ""),
        "audio_name": seg_data.get("audio_name", ""),
        "sample_rate": sample_rate,
        "audio_duration_seconds": seg_data.get("audio_duration_seconds", 0.0),
        "segments": segments,
        "segment_demucs_flags": seg_data.get("segment_demucs_flags", []),
        "metadata": {
            "stage": "asr",
            "asr_moe": args.asr_moe,
            "whisper_arch": args.whisper_arch,
            "processing_time_seconds": elapsed,
            **stats
        },
    }
    stage_common.dump_json(out_data, out_json_path)
    logger.info(f"Saved transcripts to {out_json_path}")

if __name__ == "__main__":
    main()
