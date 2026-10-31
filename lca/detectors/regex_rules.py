"""Baseline detector: contextual regex patterns plus a VI/EN keyword dictionary."""
from __future__ import annotations

import re
import unicodedata

from lca.types import Detection, RiskLabel

CONTEXT_SCORE = 1.0
KEYWORD_SCORE = 0.6
NEGATION_WINDOW = 15

_F = re.IGNORECASE | re.UNICODE

# Contextual patterns exist only for the three required groups (proposal §3.2).
CONTEXT_PATTERNS: dict[RiskLabel, list[str]] = {
    RiskLabel.PENALTY: [
        r"(phạt|tiền phạt).{0,40}(\d+\s?%|giá trị hợp đồng)",
        r"(penalt\w*|liquidated damages).{0,60}\d+(\.\d+)?\s?%",
    ],
    RiskLabel.INDEMNITY: [
        r"bồi thường.{0,60}(toàn bộ|mọi|không giới hạn)",
        r"indemnif\w*.{0,80}(any and all|all (losses|claims|damages)|without limit\w*|unlimited|uncapped)",
    ],
    RiskLabel.CONFIDENTIALITY: [
        r"(bảo mật|thông tin mật|không được tiết lộ).{0,80}(\d+\s?(năm|tháng)|vô thời hạn)",
        r"(confidential\w*|non-disclosure).{0,80}(\d+\s?(\(\d+\)\s?)?(years?|months?)|indefinitely|perpetu\w*)",
    ],
}

KEYWORDS: dict[RiskLabel, list[str]] = {
    RiskLabel.PENALTY: [
        "phạt vi phạm", "tiền phạt", "chịu phạt", "bị phạt", "mức phạt", r"% giá trị hợp đồng",
        r"penalt(y|ies)", "liquidated damages",
    ],
    RiskLabel.INDEMNITY: [
        "bồi thường", "toàn bộ thiệt hại",
        r"indemnif(y|ies|ication)", "indemnity", "hold harmless", "uncapped", "unlimited liability",
    ],
    RiskLabel.CONFIDENTIALITY: [
        "bảo mật", "thông tin mật", "không tiết lộ", "không được tiết lộ",
        r"confidential(ity)?", "non-disclosure",
    ],
    RiskLabel.TERMINATION: [
        "đơn phương chấm dứt", "không cần báo trước",
        r"terminat(e|ion) for convenience", "without cause", "without prior notice",
    ],
    RiskLabel.NON_COMPETE: [
        "không được làm việc cho", "cạnh tranh",
        r"non-?compet(e|ition)", r"non-?solicit(ation)?", "competing business",
    ],
}

NEGATIONS = ["không bị", "miễn", "không phải chịu", "not be subject to"]
DEFINITION_RE = re.compile(
    r"^\W*[\"“']?(\w+\s){0,5}\w+[\"”']?\s+(là|có nghĩa là|means|shall mean)\b", _F)


def _compile(patterns: dict[RiskLabel, list[str]]) -> dict[RiskLabel, list[re.Pattern]]:
    return {label: [re.compile(p, _F) for p in ps] for label, ps in patterns.items()}


class RegexDetector:
    name = "regex"

    def __init__(self):
        self._context = _compile(CONTEXT_PATTERNS)
        self._keywords = _compile(KEYWORDS)

    def predict(self, texts: list[str]) -> list[list[Detection]]:
        return [self._predict_one(unicodedata.normalize("NFC", t)) for t in texts]

    def _predict_one(self, text: str) -> list[Detection]:
        if DEFINITION_RE.match(text):
            return []
        out = []
        for label in KEYWORDS:
            score = self._best(text, self._context.get(label, []), CONTEXT_SCORE) \
                or self._best(text, self._keywords[label], KEYWORD_SCORE)
            if score:
                out.append(Detection(self.name, label, score))
        return out

    @staticmethod
    def _best(text: str, patterns: list[re.Pattern], score: float) -> float:
        for pat in patterns:
            for m in pat.finditer(text):
                window = text[max(0, m.start() - NEGATION_WINDOW): m.end()].lower()
                if not any(neg in window for neg in NEGATIONS):
                    return score
        return 0.0
