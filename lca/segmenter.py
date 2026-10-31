"""Split a Document into clause- or sentence-level Segments.

Offsets in Segment.spans always point into the *raw* Block.text, so writers can
highlight the original file; Segment.text is normalised (NFC, single spaces)
and is used for detection only.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Literal

from lca.types import Document, Segment

HEADING_RE = re.compile(r"^(Điều|Khoản|Article|Section|Clause)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)
SUBCLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s")
MARKER_RE = re.compile(r"(?:^|(?<=[\s:;]))(?:[a-zđ]|\d{1,2}|[ivx]{1,4})\)(?=\s)")
PREAMBLE = "Mở đầu"
MIN_ALPHA_FOR_LANG = 20
VI_RATIO = 0.03


# ---------------------------------------------------------------- normalising
def _normalize_with_map(raw: str) -> tuple[str, list[int], list[int]]:
    """NFC + whitespace collapse, with each output char mapped to a raw [start, end)."""
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    pending: tuple[int, int] | None = None
    i, n = 0, len(raw)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(raw[j]):
            j += 1
        chunk = raw[i:j]
        if chunk.isspace():
            if out and pending is None:
                pending = (i, j)
        else:
            if pending is not None:
                out.append(" ")
                starts.append(pending[0])
                ends.append(pending[1])
                pending = None
            for ch in unicodedata.normalize("NFC", chunk):
                out.append(ch)
                starts.append(i)
                ends.append(j)
        i = j
    return "".join(out), starts, ends


def normalize(text: str) -> str:
    """Same result as _normalize_with_map(text)[0], but fast enough for bulk data."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def detect_lang(text: str) -> Literal["vi", "en"]:
    alpha = [c for c in unicodedata.normalize("NFC", text) if c.isalpha()]
    if not alpha:
        return "en"
    non_ascii = sum(1 for c in alpha if ord(c) > 127)
    return "vi" if non_ascii / len(alpha) >= VI_RATIO else "en"


def _lang_or_default(text: str, default: str) -> str:
    if sum(c.isalpha() for c in text) < MIN_ALPHA_FOR_LANG:
        return default
    return detect_lang(text)


# ---------------------------------------------------------------- tokenisers
@lru_cache(maxsize=1)
def _spacy():
    import spacy
    return spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])


def _sentences(text: str, lang: str) -> list[str]:
    if lang == "vi":
        from underthesea import sent_tokenize
        return sent_tokenize(text)
    return [s.text for s in _spacy()(text).sents]


def _split_sentences(norm: str, start: int, end: int, lang: str) -> list[tuple[int, int]]:
    """Sentence ranges (in norm offsets) inside norm[start:end]."""
    piece = norm[start:end]
    stripped = piece.strip(" ;")
    if not stripped:
        return []
    out, cursor = [], start
    for sent in _sentences(stripped, lang):
        sent = sent.strip()
        if not sent:
            continue
        pos = norm.find(sent, cursor, end)
        if pos < 0:  # tokenizer altered the text; fall back to the whole piece
            s = start + (len(piece) - len(piece.lstrip(" ;")))
            return [(s, s + len(stripped))]
        out.append((pos, pos + len(sent)))
        cursor = pos + len(sent)
    return out


def _trim(norm: str, s: int, e: int) -> tuple[int, int]:
    while s < e and norm[s] in " ;":
        s += 1
    while e > s and norm[e - 1] in " ;":
        e -= 1
    return s, e


# ---------------------------------------------------------------- segmenting
def segment(doc: Document, mode: Literal["clause", "sentence"]) -> list[Segment]:
    doc_lang = detect_lang(" ".join(b.text for b in doc.blocks))
    if mode == "clause":
        return _clauses(doc, doc_lang)
    if mode == "sentence":
        return _sentence_segments(doc, doc_lang)
    raise ValueError(f"unknown mode {mode!r}")


def _clauses(doc: Document, doc_lang: str) -> list[Segment]:
    groups: list[tuple[str, list]] = []
    for block in doc.blocks:
        norm, starts, ends = _normalize_with_map(block.text)
        if not norm:
            continue
        span = (block.index, starts[0], ends[-1])
        m = HEADING_RE.match(norm)
        if m or not groups:
            loc = f"{m.group(1)} {m.group(2)}" if m else PREAMBLE
            groups.append((loc, []))
        groups[-1][1].append((norm, span))
    segs = []
    for loc, items in groups:
        text = " ".join(t for t, _ in items)
        segs.append(Segment(len(segs), text, loc, [s for _, s in items],
                            _lang_or_default(text, doc_lang)))
    return segs


def _sentence_segments(doc: Document, doc_lang: str) -> list[Segment]:
    segs: list[Segment] = []
    keyword = "Điều" if doc_lang == "vi" else "Section"
    location = PREAMBLE

    def add(text: str, span: tuple[int, int, int]):
        segs.append(Segment(len(segs), text, location, [span], _lang_or_default(text, doc_lang)))

    for block in doc.blocks:
        norm, starts, ends = _normalize_with_map(block.text)
        if not norm:
            continue
        head = HEADING_RE.match(norm)
        sub = SUBCLAUSE_RE.match(norm)
        if head:
            keyword = head.group(1)
            location = f"{head.group(1)} {head.group(2)}"
        elif sub:
            location = f"{keyword} {sub.group(1)}"
        lang = _lang_or_default(norm, doc_lang)

        def raw_span(s: int, e: int) -> tuple[int, int, int]:
            return (block.index, starts[s], ends[e - 1])

        markers = list(MARKER_RE.finditer(norm))
        if markers:
            lead = _split_sentences(norm, 0, markers[0].start(), lang)
            context = ""
            if lead:
                *full, last = lead
                for s, e in full:
                    add(norm[s:e], raw_span(s, e))
                context = norm[last[0]:last[1]]
            bounds = [m.start() for m in markers] + [len(norm)]
            for a, b in zip(bounds, bounds[1:]):
                for s, e in _split_sentences(norm, a, b, lang):
                    s, e = _trim(norm, s, e)
                    if s < e:
                        text = norm[s:e]
                        add(f"{context} {text}" if context else text, raw_span(s, e))
        else:
            cuts = [-1] + [i for i, c in enumerate(norm) if c == ";"] + [len(norm)]
            for a, b in zip(cuts, cuts[1:]):
                for s, e in _split_sentences(norm, a + 1, b, lang):
                    s, e = _trim(norm, s, e)
                    if s < e:
                        add(norm[s:e], raw_span(s, e))
    return segs
