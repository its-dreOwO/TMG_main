"""Map CUAD categories and LEDGAR provision labels onto the project's risk groups."""
from __future__ import annotations

import re

from lca.types import RiskLabel

CUAD_MAP: dict[str, RiskLabel] = {
    "Liquidated Damages": RiskLabel.PENALTY,
    "Uncapped Liability": RiskLabel.INDEMNITY,
    "Cap On Liability": RiskLabel.INDEMNITY,
    "Non-Compete": RiskLabel.NON_COMPETE,
    "No-Solicit Of Employees": RiskLabel.NON_COMPETE,
    "No-Solicit Of Customers": RiskLabel.NON_COMPETE,
    "Termination For Convenience": RiskLabel.TERMINATION,
}

_LEDGAR_RULES: list[tuple[RiskLabel, re.Pattern]] = [
    (RiskLabel.PENALTY, re.compile(r"liquidated damages|^penalt(y|ies)$")),
    (RiskLabel.INDEMNITY, re.compile(r"indemn")),
    (RiskLabel.CONFIDENTIALITY, re.compile(r"confiden|non-?disclosure")),
    (RiskLabel.TERMINATION, re.compile(r"terminat")),
    (RiskLabel.NON_COMPETE, re.compile(r"non-?compet|non-?solicit")),
]
# A label that mentions a risk topic but matches no precise rule is ambiguous.
_RISK_STEMS = re.compile(r"penalt|liquidat|indemn|confiden|disclos|terminat|compet|solicit")


def map_ledgar_label(label: str) -> RiskLabel | None:
    """RiskLabel for one LEDGAR label, or None when the mapping is uncertain."""
    text = label.lower().strip()
    hits = {risk for risk, pat in _LEDGAR_RULES if pat.search(text)}
    if len(hits) == 1:
        return hits.pop()
    if hits or _RISK_STEMS.search(text):
        return None
    return RiskLabel.NONE
