"""Shared data types for the Legal Clause Analyzer pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Literal


class RiskLabel(str, Enum):
    PENALTY = "penalty"
    INDEMNITY = "indemnity"
    CONFIDENTIALITY = "confidentiality"
    TERMINATION = "termination"
    NON_COMPETE = "non_compete"
    NONE = "none"

    @property
    def vi(self) -> str:
        return _RISK_VI[self]


_RISK_VI = {
    RiskLabel.PENALTY: "Phạt vi phạm",
    RiskLabel.INDEMNITY: "Bồi thường thiệt hại",
    RiskLabel.CONFIDENTIALITY: "Bảo mật",
    RiskLabel.TERMINATION: "Chấm dứt hợp đồng",
    RiskLabel.NON_COMPETE: "Hạn chế cạnh tranh",
    RiskLabel.NONE: "Không rủi ro",
}

RISK_LABELS: tuple[RiskLabel, ...] = tuple(l for l in RiskLabel if l is not RiskLabel.NONE)


class Level(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @property
    def vi(self) -> str:
        return {Level.LOW: "Thấp", Level.MEDIUM: "Trung bình", Level.HIGH: "Cao"}[self]


@dataclass(frozen=True)
class Block:
    index: int
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Document:
    path: Path
    kind: Literal["docx", "pdf"]
    blocks: list[Block]


@dataclass(frozen=True)
class Segment:
    seg_id: int
    text: str
    location: str
    spans: list[tuple[int, int, int]]
    lang: Literal["vi", "en"]


@dataclass(frozen=True)
class Detection:
    method: str
    label: RiskLabel
    score: float


@dataclass(frozen=True)
class Entity:
    kind: Literal["PARTY", "MONEY", "PERCENT", "DURATION", "DATE"]
    text: str
    value: float | None = None


@dataclass(frozen=True)
class Finding:
    segment: Segment
    label: RiskLabel
    detections: list[Detection]
    entities: list[Entity]
    level: Level
    reason: str


@dataclass(frozen=True)
class AnalysisContext:
    contract_type: Literal["commercial", "labour", "other"] = "commercial"
    my_party: str | None = None
