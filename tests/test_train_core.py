import pytest

from lca.train.core import class_weights


def test_class_weights_boost_rare_class():
    w = class_weights(["a", "a", "a", "b"], ["a", "b"])
    assert w[1] > w[0] and pytest.approx(sum(w) / 2) == 1.0


def test_class_weights_handles_absent_class():
    w = class_weights(["a"], ["a", "b"])
    assert w[1] == 0.0 and w[0] == pytest.approx(2.0)


def test_class_weights_follow_class_order():
    assert class_weights(["b", "b", "a"], ["b", "a"])[0] < class_weights(["b", "b", "a"], ["b", "a"])[1]


def test_limit_keeps_label_column_and_rare_classes():
    import pandas as pd
    from modal_train import _limit
    df = pd.DataFrame({"text": [f"t{i}" for i in range(100)], "label": ["none"] * 97 + ["penalty"] * 3})
    out = _limit(df, 10, 2)
    assert list(out.columns) == ["text", "label"] and len(out) <= 10 and (out.label == "penalty").sum() == 3


def test_limit_zero_means_all_rows():
    import pandas as pd
    from modal_train import _limit
    df = pd.DataFrame({"text": ["a", "b"], "label": ["none", "penalty"]})
    assert len(_limit(df, 0, 2)) == 2
