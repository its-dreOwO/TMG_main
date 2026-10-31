"""Route (b): translate Vietnamese segments to English locally, then classify."""
from __future__ import annotations

from dataclasses import replace

from lca.segmenter import detect_lang
from lca.types import Detection

MT_MODEL = "Helsinki-NLP/opus-mt-vi-en"


class Translator:
    """Local MarianMT vi→en translation; text never leaves the machine."""

    def __init__(self, model: str = MT_MODEL, device: str | None = None, batch_size: int = 8,
                 max_length: int = 512):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = MarianTokenizer.from_pretrained(model)
        self.model = MarianMTModel.from_pretrained(model).to(self.device).eval()
        self.batch_size, self.max_length = batch_size, max_length

    def translate(self, texts: list[str]) -> list[str]:
        import torch

        out: list[str] = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(texts[i:i + self.batch_size], return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_length).to(self.device)
            with torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=self.max_length)
            out.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return out


class TranslatingDetector:
    def __init__(self, inner, translator, name: str):
        self.inner, self.translator, self.name = inner, translator, name

    def predict(self, texts: list[str]) -> list[list[Detection]]:
        vi_idx = [i for i, t in enumerate(texts) if detect_lang(t) == "vi"]
        routed = list(texts)
        if vi_idx:
            for i, en in zip(vi_idx, self.translator.translate([texts[i] for i in vi_idx])):
                routed[i] = en
        return [[replace(d, method=self.name) for d in ds] for ds in self.inner.predict(routed)]
