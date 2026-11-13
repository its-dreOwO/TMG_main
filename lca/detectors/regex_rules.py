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
        # Repaying training costs on leaving early (Điều 40, 62 Bộ luật Lao động 2019).
        r"(hoàn trả|bồi hoàn|bồi thường).{0,40}(chi phí|phí) đào tạo",
        r"(repay|reimburse)\w*.{0,40}training (costs?|expenses?|fees?)",
    ],
    RiskLabel.CONFIDENTIALITY: [
        r"(bảo mật|giữ bí mật|thông tin mật|không được tiết lộ).{0,80}(\d+\s?(năm|tháng)|vô thời hạn)",
        r"(confidential\w*|non-disclosure).{0,80}(\d+\s?(\(\d+\)\s?)?(years?|months?)|indefinitely|perpetu\w*)",
    ],
}

KEYWORDS: dict[RiskLabel, list[str]] = {
    RiskLabel.PENALTY: [
        "phạt vi phạm", "tiền phạt", "chịu phạt", "bị phạt", "mức phạt", r"% giá trị hợp đồng",
        r"penalt(y|ies)", "liquidated damages",
    ],
    RiskLabel.INDEMNITY: [
        "bồi thường", "toàn bộ thiệt hại", r"(chi phí|phí) đào tạo",
        r"indemnif(y|ies|ication)", "indemnity", "hold harmless", "uncapped", "unlimited liability",
    ],
    RiskLabel.CONFIDENTIALITY: [
        "bảo mật", "giữ bí mật", "thông tin mật", "không tiết lộ", "không được tiết lộ",
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
NEGATION_RE = re.compile("|".join(re.escape(n) for n in NEGATIONS), _F)
_MAX_NEGATION = max(len(n) for n in NEGATIONS)
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
                if not RegexDetector._negated(text, m):
                    return score
        return 0.0

    @staticmethod
    def _negated(text: str, m: re.Match) -> bool:
        """A negation that ends at most NEGATION_WINDOW characters before the match, or runs into it.

        Measured from the negation's end, so a phrase longer than the window ("not be subject to")
        still counts.
        """
        earliest_end = m.start() - NEGATION_WINDOW
        start = max(0, earliest_end - _MAX_NEGATION)
        return any(n.end() >= earliest_end for n in NEGATION_RE.finditer(text, start, m.end()))


# Definition sentences ("“Fees” has the meaning ...") name a term and bind nobody, so the pipeline
# drops them for every detector. Stricter than DEFINITION_RE above: the term must be quoted or
# Title Case, so an obligation such as "Mức phạt là 12% giá trị hợp đồng" is never mistaken for one.
_NUMBERING = r"^\W*(?:[A-Z]{3,}\s+){0,3}(?:\d+(?:\.\d+)*[.)]?\s+)?"
_MEANS = r"(?:means|shall\s+mean|(?:has|shall\s+have)\s+the\s+meaning)\b"
QUOTED_DEFINITION_RE = re.compile(
    _NUMBERING + r"(?:the\s+terms?\s+)?[“\"][^“”\"]{1,80}[”\"](?:\s+[\w-]+){0,6}?\s*[:,]?\s+"
    r"(?:" + _MEANS + r"|là\b|có\s+nghĩa\s+là\b|được\s+hiểu\s+là\b)", _F)
_TERM_WORD = r"[A-Z][\w’'&-]*"
TITLE_CASE_DEFINITION_RE = re.compile(  # case-sensitive: "Initial Term has the meaning ..."
    _NUMBERING + r"(?!(?:This|That|It|These|Those|Which|Such)\b)"
    + _TERM_WORD + r"(?:\s+(?:" + _TERM_WORD + r"|of|and|the|for)){0,6}\s*:?\s+" + _MEANS, re.UNICODE)


def is_definition(text: str) -> bool:
    text = unicodedata.normalize("NFC", text)
    return bool(QUOTED_DEFINITION_RE.match(text) or TITLE_CASE_DEFINITION_RE.match(text))
