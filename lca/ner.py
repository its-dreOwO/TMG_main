"""Entity extraction used to explain and grade risks: PARTY, MONEY, PERCENT, DURATION, DATE."""
from __future__ import annotations

import math
import re
from functools import lru_cache

from lca.types import Entity

_I = re.IGNORECASE
VI_NUM = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10}
EN_NUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())} | {"thirty": 30, "sixty": 60, "ninety": 90}
_EN_WORDS = "|".join(sorted(EN_NUM, key=len, reverse=True))
_VI_WORDS = "|".join(VI_NUM)
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

PARTY_RE = re.compile(
    r"\b(Bên [A-Z]\b|Bên (?:mua|bán|thuê|cho thuê|giao|nhận|vay|cho vay)|Party [A-Z]\b|"
    r"Người sử dụng lao động|Người lao động|"
    r"(?:Buyer|Seller|Vendor|Supplier|Client|Customer|Company|Employee|Employer|Contractor|Consultant|"
    r"Distributor|Licensee|Licensor|Recipient|Discloser|Disclosing Party|Receiving Party)\b)")
DATE_RES = [
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", _I),
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}\b"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+day\s+of\s+(?:{_MONTHS}),?\s+\d{{4}}\b", _I),
]
PERCENT_RE = re.compile(rf"(\d+(?:[.,]\d+)?|{_VI_WORDS}|{_EN_WORDS})\s?(%|phần trăm|percent|per cent)", _I)
MONEY_VI_RE = re.compile(r"(\d{1,3}(?:\.\d{3})+|\d+(?:,\d+)?)\s*(tỷ|triệu|nghìn|ngàn)?\s*(đồng|VNĐ|VND)\b", _I)
MONEY_EN_RE = re.compile(r"(?:US\$|\$|USD\s?)\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s?(million|billion))?", _I)
DURATION_VI_RE = re.compile(r"(\d+)\s*(năm|tháng|tuần|ngày)\b", _I)
DURATION_EN_RE = re.compile(rf"\b(\d+|{_EN_WORDS})\s*(?:\((\d+)\)\s*)?(years?|months?|weeks?|days?)\b", _I)
INFINITE_RE = re.compile(r"vô thời hạn|không (?:có )?thời hạn|indefinitely|in perpetuity|perpetual", _I)
OBLIGATION_RE = re.compile(r"\b(phải|chịu|có nghĩa vụ|shall|must|agrees? to)\b", _I)
_SCALE = {"tỷ": 1e9, "triệu": 1e6, "nghìn": 1e3, "ngàn": 1e3, "million": 1e6, "billion": 1e9}


def _number(token: str, lang: str) -> float:
    t = token.lower()
    if t in VI_NUM:
        return float(VI_NUM[t])
    if t in EN_NUM:
        return float(EN_NUM[t])
    return float(t.replace(",", ".") if lang == "vi" else t.replace(",", ""))


def _months(n: float, unit: str) -> float:
    u = unit.lower()
    if u.startswith(("năm", "year")):
        return n * 12
    if u.startswith(("tháng", "month")):
        return n
    days = n * 7 if u.startswith(("tuần", "week")) else n
    return max(1, round(days / 30))


@lru_cache(maxsize=1)
def _spacy_ner():
    import spacy
    return spacy.load("en_core_web_sm", exclude=["lemmatizer"])


def _orgs(text: str, lang: str) -> list[tuple[int, int, str]]:
    try:
        if lang == "en":
            return [(e.start_char, e.end_char, e.text) for e in _spacy_ner()(text).ents if e.label_ == "ORG"]
        from underthesea import ner
        out, cursor = [], 0
        words = ner(text)
        current: list[str] = []
        for w in words + [("", "", "", "O")]:
            tag = w[-1]
            if tag.endswith("ORG") and not (tag.startswith("B") and current):
                current.append(w[0])
                continue
            if current:
                name = " ".join(current)
                pos = text.find(name, cursor)
                if pos >= 0:
                    out.append((pos, pos + len(name), name))
                    cursor = pos + len(name)
            current = [w[0]] if tag.endswith("ORG") else []
        return out
    except Exception:  # NER is an enhancement; regex entities still stand
        return []


def _overlaps(s: int, e: int, spans: list[tuple[int, int]]) -> bool:
    return any(s < b and a < e for a, b in spans)


def extract_entities(text: str, lang: str) -> list[Entity]:
    found: list[tuple[int, int, Entity]] = []
    dates = [(m.start(), m.end()) for r in DATE_RES for m in r.finditer(text)]
    for s, e in dates:
        found.append((s, e, Entity("DATE", text[s:e])))
    for m in PARTY_RE.finditer(text):
        found.append((m.start(), m.end(), Entity("PARTY", m.group(0))))
    for s, e, name in _orgs(text, lang):
        found.append((s, e, Entity("PARTY", name)))
    for m in PERCENT_RE.finditer(text):
        found.append((m.start(), m.end(), Entity("PERCENT", m.group(0), _number(m.group(1), lang))))
    for m in MONEY_VI_RE.finditer(text):
        n = float(m.group(1).replace(".", "").replace(",", ".")) * _SCALE.get((m.group(2) or "").lower(), 1)
        found.append((m.start(), m.end(), Entity("MONEY", m.group(0), n)))
    for m in MONEY_EN_RE.finditer(text):
        n = float(m.group(1).replace(",", "") + (m.group(2) or "")) * _SCALE.get((m.group(3) or "").lower(), 1)
        found.append((m.start(), m.end(), Entity("MONEY", m.group(0), n)))
    for regex in (DURATION_VI_RE, DURATION_EN_RE):
        for m in regex.finditer(text):
            if _overlaps(m.start(), m.end(), dates):
                continue
            n = _number(m.group(2) if regex is DURATION_EN_RE and m.group(2) else m.group(1), lang)
            found.append((m.start(), m.end(), Entity("DURATION", m.group(0), _months(n, m.groups()[-1]))))
    for m in INFINITE_RE.finditer(text):
        found.append((m.start(), m.end(), Entity("DURATION", m.group(0), math.inf)))
    seen, out = set(), []
    for _, _, ent in sorted(found, key=lambda f: f[0]):
        key = (ent.kind, ent.text)
        if key not in seen:
            seen.add(key)
            out.append(ent)
    return out


def obligor(text: str, entities: list[Entity]) -> str | None:
    """The PARTY mentioned nearest before the first obligation word, if any."""
    m = OBLIGATION_RE.search(text)
    if not m:
        return None
    best, best_end = None, -1
    for ent in entities:
        if ent.kind != "PARTY":
            continue
        for occ in re.finditer(re.escape(ent.text), text[:m.start()]):
            if occ.end() > best_end:
                best, best_end = ent.text, occ.end()
    return best
