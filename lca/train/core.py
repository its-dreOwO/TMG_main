"""Training helpers shared by the Modal transformer job."""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def class_weights(labels: Sequence[str], classes: list[str]) -> list[float]:
    """Inverse-frequency weights in `classes` order, normalised to mean 1; absent classes get 0."""
    counts = Counter(labels)
    raw = [1.0 / counts[c] if counts[c] else 0.0 for c in classes]
    total = sum(raw)
    return [w * len(classes) / total for w in raw] if total else [0.0] * len(classes)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_weighted_trainer(weights: list[float]):
    """A transformers.Trainer subclass using class-weighted cross-entropy."""
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            w = torch.tensor(weights, dtype=outputs.logits.dtype, device=outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer
