from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any


BOILERPLATE_MARKERS = (
    "thank you",
    "thanks for watching",
    "subscribe",
    "la la school",
    "ghien mi go",
    "ghiền mì gõ",
    "ini mah",
)

VIETNAMESE_DIACRITICS = set(
    "àáạảãâầấậẩẫăằắặẳẵ"
    "èéẹẻẽêềếệểễ"
    "ìíịỉĩ"
    "òóọỏõôồốộổỗơờớợởỡ"
    "ùúụủũưừứựửữ"
    "ỳýỵỷỹđ"
)

LOW_INFORMATION_SHORT_TOKENS = {
    "a",
    "à",
    "ờ",
    "ừ",
    "ừm",
    "um",
    "uh",
    "và",
    "với",
    "thì",
}

KNOWN_CONFUSION_PAIRS = (
    ("xướng", "sướng"),
    ("thành ra", "thật ra"),
)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def normalize_text(value: Any, *, keep_accents: bool = True) -> str:
    text = str(value or "").lower().strip()
    if not keep_accents:
        text = strip_accents(text)
    text = re.sub(r"[^\w\sà-ỹđ]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_similarity(left: Any, right: Any, *, keep_accents: bool = True) -> float:
    left_norm = normalize_text(left, keep_accents=keep_accents)
    right_norm = normalize_text(right, keep_accents=keep_accents)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def contains_boilerplate(text: Any) -> bool:
    normalized = normalize_text(text, keep_accents=False)
    return any(marker in normalized for marker in BOILERPLATE_MARKERS)


def contains_foreign_script(text: Any) -> bool:
    value = str(text or "")
    return any(
        "\u0e00" <= char <= "\u0e7f" or "\u4e00" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff"
        for char in value
    )


def has_vietnamese_diacritic(text: Any) -> bool:
    return any(char.lower() in VIETNAMESE_DIACRITICS for char in str(text or ""))


def _clean_candidate(text: Any) -> str:
    return str(text or "").strip()


def _is_viable_candidate(text: str) -> bool:
    if not normalize_text(text):
        return False
    if contains_boilerplate(text) or contains_foreign_script(text):
        return False
    return True


def _is_low_information_short_text(text: str) -> bool:
    normalized = normalize_text(text)
    tokens = normalized.split()
    return len(tokens) <= 1 and normalized in LOW_INFORMATION_SHORT_TOKENS


def _token_count(text: str) -> int:
    return len(normalize_text(text).split())


def _confusion_consensus_candidate(rover_text: str, text_phowhisper: str, text_chunkformer: str) -> str:
    pho_norm = normalize_text(text_phowhisper)
    chunk_norm = normalize_text(text_chunkformer)
    rover_norm = normalize_text(rover_text)
    if not pho_norm or pho_norm != chunk_norm or pho_norm == rover_norm:
        return ""

    for left, right in KNOWN_CONFUSION_PAIRS:
        for source_phrase, target_phrase in ((left, right), (right, left)):
            if source_phrase not in rover_norm or target_phrase not in pho_norm:
                continue
            candidate = rover_norm.replace(source_phrase, target_phrase)
            if normalize_text(candidate) == pho_norm:
                return _clean_candidate(text_phowhisper)
    return ""


def _best_vi_candidate(
    text_phowhisper: str,
    text_chunkformer: str,
    *,
    duration_sec: float,
    micro_segment_seconds: float,
    vi_agreement_threshold: float,
) -> tuple[str, str, bool]:
    candidates = []
    for source, text in (("phowhisper", text_phowhisper), ("chunkformer", text_chunkformer)):
        cleaned = _clean_candidate(text)
        if _is_viable_candidate(cleaned):
            candidates.append((source, cleaned))

    if not candidates:
        return "", "", False

    if len(candidates) == 1:
        source, text = candidates[0]
        if duration_sec < micro_segment_seconds and _is_low_information_short_text(text):
            return "", "", False
        return text, source, False

    pho = _clean_candidate(text_phowhisper)
    chunk = _clean_candidate(text_chunkformer)
    agreement = text_similarity(pho, chunk) >= vi_agreement_threshold
    if normalize_text(pho) and normalize_text(chunk):
        pho_norm = normalize_text(pho)
        chunk_norm = normalize_text(chunk)
        if pho_norm == chunk_norm:
            agreement = True

    source, text = max(candidates, key=lambda item: (len(normalize_text(item[1]).split()), len(item[1])))
    return text, source if not agreement else "vi_consensus", agreement


def choose_asr_text(
    *,
    rover_text: str,
    text_whisper: str,
    text_phowhisper: str,
    text_chunkformer: str,
    duration_sec: float,
    enabled: bool = True,
    micro_segment_seconds: float = 0.5,
    short_segment_seconds: float = 1.0,
    vi_agreement_threshold: float = 0.75,
) -> dict[str, Any]:
    """
    Select the final ASR text after ROVER, guarding against short-segment hallucinations.
    """
    rover_text = _clean_candidate(rover_text)
    text_whisper = _clean_candidate(text_whisper)
    text_phowhisper = _clean_candidate(text_phowhisper)
    text_chunkformer = _clean_candidate(text_chunkformer)
    actions: list[str] = []

    if not enabled:
        return {"text": rover_text, "source": "rover", "actions": actions}

    vi_text, vi_source, vi_agrees = _best_vi_candidate(
        text_phowhisper,
        text_chunkformer,
        duration_sec=duration_sec,
        micro_segment_seconds=micro_segment_seconds,
        vi_agreement_threshold=vi_agreement_threshold,
    )
    rover_is_bad = contains_boilerplate(rover_text) or contains_foreign_script(rover_text)
    whisper_is_bad = contains_boilerplate(text_whisper) or contains_foreign_script(text_whisper)
    rover_follows_whisper = text_similarity(rover_text, text_whisper) >= 0.85
    micro_long_text = duration_sec < micro_segment_seconds and _token_count(rover_text) >= 5

    confusion_candidate = _confusion_consensus_candidate(rover_text, text_phowhisper, text_chunkformer)
    if confusion_candidate:
        return {
            "text": confusion_candidate,
            "source": "vi_consensus",
            "actions": ["replace_known_confusion_with_vi_consensus"],
        }

    if micro_long_text:
        if vi_text and vi_agrees:
            return {
                "text": vi_text,
                "source": vi_source,
                "actions": ["replace_micro_hallucination_with_vi_consensus"],
            }
        return {"text": "", "source": "quality_guard", "actions": ["drop_micro_hallucination"]}

    if duration_sec < micro_segment_seconds and rover_is_bad and not vi_agrees:
        return {"text": "", "source": "quality_guard", "actions": ["drop_micro_boilerplate"]}

    if vi_text and vi_agrees and rover_follows_whisper:
        rover_vi_sim = text_similarity(rover_text, vi_text)
        accentless_sim = text_similarity(rover_text, vi_text, keep_accents=False)
        accent_mismatch_on_short = (
            duration_sec < short_segment_seconds
            and accentless_sim >= 0.8
            and has_vietnamese_diacritic(vi_text)
            and not has_vietnamese_diacritic(rover_text)
        )
        if rover_vi_sim < 0.55 or accent_mismatch_on_short:
            return {
                "text": vi_text,
                "source": "vi_consensus",
                "actions": ["replace_whisper_outlier_with_vi_consensus"],
            }

    if vi_text and rover_follows_whisper and (rover_is_bad or whisper_is_bad):
        return {
            "text": vi_text,
            "source": vi_source,
            "actions": ["replace_bad_whisper_with_vi_candidate"],
        }

    return {"text": rover_text, "source": "rover", "actions": actions}
