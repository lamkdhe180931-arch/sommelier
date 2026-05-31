from __future__ import annotations

from typing import Mapping, Sequence


def _score(source_scores: Mapping[str, float], speaker: str) -> float:
    value = source_scores.get(speaker)
    if value is None:
        return -1.0
    return float(value)


def _best_speaker(source_scores: Mapping[str, float], speakers: Sequence[str]) -> str | None:
    available = [(speaker, _score(source_scores, speaker)) for speaker in speakers if speaker in source_scores]
    if not available:
        return None
    return max(available, key=lambda item: item[1])[0]


def choose_stem_assignment(
    seg1_speaker: str,
    seg2_speaker: str,
    source_scores: Sequence[Mapping[str, float]],
    *,
    min_confidence: float = 0.55,
    min_margin: float = 0.08,
) -> dict:
    """Choose whether source 0/1 map directly or swapped to two diarized speakers.

    `source_scores` must contain two dictionaries:
    - source_scores[0][speaker] = similarity between stem/source 0 and speaker reference
    - source_scores[1][speaker] = similarity between stem/source 1 and speaker reference

    The function intentionally rejects ambiguous assignments. A wrong speaker mapping
    is worse than leaving the overlap as review-only for full-duplex training data.
    """

    if len(source_scores) < 2:
        return {
            "accepted": False,
            "mapping": None,
            "assignment_method": "embedding_rejected",
            "reject_reasons": ["missing_source_scores"],
            "confidence": 0.0,
            "margin": 0.0,
            "direct_score": -1.0,
            "swapped_score": -1.0,
        }

    speakers = [str(seg1_speaker), str(seg2_speaker)]
    src0_scores = source_scores[0]
    src1_scores = source_scores[1]

    direct_components = [_score(src0_scores, speakers[0]), _score(src1_scores, speakers[1])]
    swapped_components = [_score(src0_scores, speakers[1]), _score(src1_scores, speakers[0])]
    direct_score = sum(direct_components)
    swapped_score = sum(swapped_components)

    if direct_score >= swapped_score:
        mapping = "direct"
        assignment_method = "embedding_direct"
        confidence = min(direct_components)
        margin = direct_score - swapped_score
    else:
        mapping = "swapped"
        assignment_method = "embedding_swapped"
        confidence = min(swapped_components)
        margin = swapped_score - direct_score

    src0_best = _best_speaker(src0_scores, speakers)
    src1_best = _best_speaker(src1_scores, speakers)
    reject_reasons = []
    if src0_best is None or src1_best is None:
        reject_reasons.append("missing_speaker_score")
    elif src0_best == src1_best:
        reject_reasons.append("same_best_speaker")
    if confidence < float(min_confidence):
        reject_reasons.append("confidence_lt_min")
    if margin < float(min_margin):
        reject_reasons.append("margin_lt_min")

    accepted = not reject_reasons
    return {
        "accepted": accepted,
        "mapping": mapping if accepted else None,
        "assignment_method": assignment_method if accepted else "embedding_rejected",
        "reject_reasons": reject_reasons,
        "confidence": round(float(max(confidence, 0.0)), 6),
        "margin": round(float(max(margin, 0.0)), 6),
        "direct_score": round(float(direct_score), 6),
        "swapped_score": round(float(swapped_score), 6),
        "source_best_speakers": [src0_best, src1_best],
    }
