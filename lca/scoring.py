"""Combine detections and entities into a Finding with a Cao/Trung bình/Thấp level."""
from __future__ import annotations

import math
import re
from collections import defaultdict

from lca.detectors.regex_rules import CONTEXT_SCORE, RegexDetector
from lca.ner import obligor
from lca.types import AnalysisContext, Detection, Entity, Finding, Level, RiskLabel, Segment

COMMERCIAL_PENALTY_CAP = 8.0  # Art. 301 Luật Thương mại 2005
# Similarity alone is the noisiest signal: neutral sentences ("Hai bên thống nhất ký kết...") reach
# 0.45-0.57 against the prototypes, while real paraphrased risks score 0.60 and up. So a sentence
# flagged only by STS needs STS_ONLY_MIN, and STS alone counts as penalty evidence for the
# amount rules (8% cap, "penalty with a %") only from STS_PENALTY_MIN.
STS_METHOD = "sts"
STS_ONLY_MIN = 0.60
STS_PENALTY_MIN = 0.70
# Uncapped or full liability: "toàn bộ thiệt hại", or repaying 100% (e.g. of training costs).
UNCAPPED_RE = re.compile(r"toàn bộ thiệt hại|không giới hạn|uncapped|unlimited|without limit|\b100\s?%",
                         re.I)


def score(segment: Segment, detections: list[Detection], entities: list[Entity],
          ctx: AnalysisContext) -> Finding | None:
    if not detections:
        return None
    if all(d.method == STS_METHOD for d in detections) and max(d.score for d in detections) < STS_ONLY_MIN:
        return None
    totals: dict[RiskLabel, float] = defaultdict(float)
    for d in detections:
        totals[d.label] += d.score
    label = max(totals, key=totals.get)
    methods = sorted({d.method for d in detections})
    level = Level(min(len(methods), 3))

    percents = [e.value for e in entities if e.kind == "PERCENT" and e.value is not None]
    durations = [e.value for e in entities if e.kind == "DURATION" and e.value is not None]
    text = segment.text
    # A penalty can sit inside a sentence whose main label is something else
    # ("bồi thường ... và chịu phạt 10%"). Only the regex context pattern counts there: it needs
    # a penalty word next to the amount, so "thanh toán 30% giá trị hợp đồng" stays out.
    penalty_backed = any(d.label is RiskLabel.PENALTY
                         and (d.method != STS_METHOD or d.score >= STS_PENALTY_MIN) for d in detections)
    has_penalty = (label is RiskLabel.PENALTY and penalty_backed) or any(
        d.label is RiskLabel.PENALTY and d.method == RegexDetector.name and d.score >= CONTEXT_SCORE
        for d in detections)
    over_cap = (has_penalty and ctx.contract_type == "commercial"
                and any(p > COMMERCIAL_PENALTY_CAP for p in percents))
    if label is RiskLabel.PENALTY and penalty_backed and percents:
        level = max(level, Level.MEDIUM)
    elif label is RiskLabel.INDEMNITY and UNCAPPED_RE.search(text):
        level = Level.HIGH
    elif label is RiskLabel.CONFIDENTIALITY and durations:
        level = Level.HIGH if any(math.isinf(d) for d in durations) else max(level, Level.MEDIUM)
    if over_cap:
        level = Level.HIGH

    bound = obligor(text, entities)
    if ctx.my_party and bound and bound.casefold() != ctx.my_party.casefold():
        level = Level(max(Level.LOW, level - 1))

    return Finding(segment, label, list(detections), list(entities), level,
                   _reason(label, entities, bound, methods, over_cap))


def _reason(label: RiskLabel, entities: list[Entity], bound: str | None, methods: list[str],
            over_cap: bool = False) -> str:
    parts = [label.vi]
    for kind, prefix in (("PERCENT", "mức"), ("MONEY", "số tiền"), ("DURATION", "thời hạn")):
        values = [e.text for e in entities if e.kind == kind]
        if values:
            parts.append(f"{prefix} {', '.join(values)}")
    if over_cap:
        parts.append(f"vượt trần phạt {COMMERCIAL_PENALTY_CAP:g}% (Điều 301 Luật Thương mại 2005)")
    if bound:
        parts.append(f"{bound} chịu nghĩa vụ")
    return "; ".join(parts) + f" (phát hiện bởi: {', '.join(methods)})."
