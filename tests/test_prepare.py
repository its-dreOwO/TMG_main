import json

import pandas as pd
import pytest

from lca.data.prepare import prepare

Q = 'Highlight the parts (if any) of this contract related to "{}" that should be reviewed by a lawyer.'
PEN = "Supplier {} shall pay liquidated damages equal to two percent of the price for each week of delay."
IND_UNLABELLED = "The Distributor shall indemnify the Company against all third party claims."
NEG = "This Agreement shall be governed by the laws of the State of New York, without regard to conflicts."


def _cuad(n):
    data = []
    for i in range(n):
        pen = PEN.format(i)
        ctx = f"Contract number {i} is entered into by the parties on the effective date stated below. {pen} {IND_UNLABELLED} {NEG}"
        start = ctx.index(pen)
        data.append({"title": f"contract_{i}", "paragraphs": [{"context": ctx, "qas": [
            {"id": f"{i}_ld", "question": Q.format("Liquidated Damages"), "is_impossible": False,
             "answers": [{"text": pen, "answer_start": start}]},
            {"id": f"{i}_gl", "question": Q.format("Governing Law"), "is_impossible": False,
             "answers": [{"text": NEG, "answer_start": ctx.index(NEG)}]},
            {"id": f"{i}_nc", "question": Q.format("Non-Compete"), "is_impossible": True, "answers": []},
        ]}]})
    return {"version": "x", "data": data}


def _ledgar_rows(n):
    rows = []
    for i in range(n):
        src = f"2019/QTR1/{i:06d}/doc.htm"
        rows.append({"provision": f"Provision {i}: the Vendor shall keep all information secret and confidential at all times.",
                     "label": ["confidentiality"], "source": src})
        for j in range(5):
            rows.append({"provision": f"Provision {i}-{j}: this Agreement may be executed in counterparts, each an original.",
                         "label": ["counterparts"], "source": src})
    rows.append({"provision": "Provision X: both indemnity and confidentiality apply here.",
                 "label": ["indemnity", "confidentiality"], "source": "2019/QTR2/999999/doc.htm"})
    rows.append({"provision": "Provision Y: the loan may be prepaid with no premium or penalty.",
                 "label": ["no premium or penalty"], "source": "2019/QTR2/999998/doc.htm"})
    return rows


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    (raw / "cuad/CUAD_v1").mkdir(parents=True)
    (raw / "ledgar").mkdir(parents=True)
    (raw / "cuad/CUAD_v1/CUAD_v1.json").write_text(json.dumps(_cuad(30)))
    with open(raw / "ledgar/LEDGAR_2016-2019_clean.jsonl", "w") as f:
        for r in _ledgar_rows(60):
            f.write(json.dumps(r) + "\n")
    return raw


@pytest.fixture
def prepared(raw_dir, tmp_path):
    out = tmp_path / "processed"
    stats = prepare(raw_dir, out)
    frames = {s: pd.read_parquet(out / f"{s}.parquet") for s in ("train", "val", "test")}
    return stats, frames, out


def test_outputs_have_schema(prepared):
    _, frames, out = prepared
    assert list(frames["train"].columns) == ["text", "label", "source", "group"]
    assert (out / "stats.json").exists()


def test_cuad_positive_mapped(prepared):
    _, frames, _ = prepared
    allf = pd.concat(frames.values())
    assert ((allf.source == "cuad") & (allf.label == "penalty")).sum() == 30


def test_cuad_negative_excludes_unlabelled_risk_topics(prepared):
    _, frames, _ = prepared
    allf = pd.concat(frames.values())
    none_texts = allf.query("label=='none' and source=='cuad'").text
    assert len(none_texts) > 0
    assert not none_texts.str.contains("indemnif|confidential", case=False).any()
    assert not none_texts.str.contains("liquidated damages", case=False).any()


def test_no_group_in_two_splits(prepared):
    _, f, _ = prepared
    assert not set(f["train"].group) & set(f["test"].group)
    assert not set(f["train"].group) & set(f["val"].group)
    assert not set(f["val"].group) & set(f["test"].group)


def test_no_duplicate_text_across_splits(prepared):
    _, f, _ = prepared
    assert not set(f["train"].text) & set(f["test"].text)
    assert not set(f["train"].text) & set(f["val"].text)


def test_conflicting_risk_labels_dropped(prepared):
    stats, _, _ = prepared
    assert stats["dropped_multi_risk"] == 1


def test_uncertain_label_dropped(prepared):
    stats, f, _ = prepared
    assert stats["dropped_uncertain"] == 1
    assert not pd.concat(f.values()).text.str.contains("no premium").any()


def test_none_capped_at_three_times_positives(prepared):
    _, f, _ = prepared
    allf = pd.concat(f.values())
    assert (allf.label == "none").sum() <= 3 * (allf.label != "none").sum()


def test_stats_counts_match_files(prepared):
    stats, f, _ = prepared
    assert stats["splits"]["train"] == f["train"].label.value_counts().to_dict()
