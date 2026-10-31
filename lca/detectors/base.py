"""Common interface every risk detector implements."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from lca.types import Detection


@runtime_checkable
class Detector(Protocol):
    name: str

    def predict(self, texts: list[str]) -> list[list[Detection]]:
        """One inner list per input text; never contains RiskLabel.NONE."""
        ...
