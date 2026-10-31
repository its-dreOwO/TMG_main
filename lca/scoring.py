"""Combine detections and entities into a Finding with a Cao/Trung bình/Thấp level."""
from __future__ import annotations

import math
import re
from collections import defaultdict

from lca.ner import obligor
from lca.types import AnalysisContext, Detection, Entity, Finding, Level, RiskLabel, Segment

COMMERCIAL_PENALTY_CAP = 8.0  # Art. 301 Luật Thương mại 2005
UNCAPPED_RE = re.compile(r"toàn bộ thiệt hại|không giới hạn|uncapped|unlimited|without limit", re.I)


def score(segment: Segment, detections: list[Detection], entities: list[Entity],
          ctx: AnalysisContext) -> Finding | None:
    if not detections:
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
    if label is RiskLabel.PENALTY:
        if ctx.contract_type == "commercial" and any(p > COMMERCIAL_PENALTY_CAP for p in percents):
            level = Level.HIGH
        elif percents:
            level = max(level, Level.MEDIUM)
    elif label is RiskLabel.INDEMNITY and UNCAPPED_RE.search(text):
        level = Level.HIGH
    elif label is RiskLabel.CONFIDENTIALITY and durations:
        level = Level.HIGH if any(math.isinf(d) for d in durations) else max(level, Level.MEDIUM)

    bound = obligor(text, entities)
    if ctx.my_party and bound and bound.casefold() != ctx.my_party.casefold():
        level = Level(max(Level.LOW, level - 1))

    return Finding(segment, label, list(detections), list(entities), level,
                   _reason(label, entities, bound, methods))


def _reason(label: RiskLabel, entities: list[Entity], bound: str | None, methods: list[str]) -> str:
    parts = [label.vi]
    for kind, prefix in (("PERCENT", "mức"), ("MONEY", "số tiền"), ("DURATION", "thời hạn")):
        values = [e.text for e in entities if e.kind == kind]
        if values:
            parts.append(f"{prefix} {', '.join(values)}")
    if bound:
        parts.append(f"{bound} chịu nghĩa vụ")
    return "; ".join(parts) + f" (phát hiện bởi: {', '.join(methods)})."
