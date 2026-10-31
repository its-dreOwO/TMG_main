import pytest

from lca.data.mapping import CUAD_MAP, map_ledgar_label
from lca.types import RiskLabel


@pytest.mark.parametrize("raw,expected", [
    ("indemnifications", RiskLabel.INDEMNITY),
    ("indemnification by the borrowers", RiskLabel.INDEMNITY),
    ("confidential information", RiskLabel.CONFIDENTIALITY),
    ("confidentiality", RiskLabel.CONFIDENTIALITY),
    ("termination for cause", RiskLabel.TERMINATION),
    ("terminations", RiskLabel.TERMINATION),
    ("noncompetition", RiskLabel.NON_COMPETE),
    ("non-competes", RiskLabel.NON_COMPETE),
    ("non-solicitation of employees", RiskLabel.NON_COMPETE),
    ("liquidated damages", RiskLabel.PENALTY),
    ("penalties", RiskLabel.PENALTY),
    ("governing laws", RiskLabel.NONE),
    ("Governing Laws", RiskLabel.NONE),
    ("no premium or penalty", None),
    ("confidentiality and non-solicitation", None),
])
def test_ledgar_mapping(raw, expected):
    assert map_ledgar_label(raw) == expected


def test_cuad_map_values():
    assert CUAD_MAP["Liquidated Damages"] == RiskLabel.PENALTY
    assert CUAD_MAP["Uncapped Liability"] == CUAD_MAP["Cap On Liability"] == RiskLabel.INDEMNITY
    assert CUAD_MAP["No-Solicit Of Customers"] == RiskLabel.NON_COMPETE
    assert CUAD_MAP["Termination For Convenience"] == RiskLabel.TERMINATION
    assert len(CUAD_MAP) == 7
