import math

import numpy as np
import pandas as pd
import pytest

from alphaops.engine import operators as op


def test_ts_delta_and_delay(small_frame):
    pd.testing.assert_frame_equal(op.ts_delay(small_frame, 2), small_frame.shift(2))
    assert op.ts_delta(small_frame, 1)["A"].iloc[1:].eq(1.0).all()


def test_ts_mean_min_periods(small_frame):
    x = small_frame.copy()
    x.iloc[3, 0] = np.nan
    out = op.ts_mean(x, 4)  # needs ceil(0.75*4)=3 valid values
    assert math.isnan(out["A"].iloc[1])  # only 2 values available
    assert out["A"].iloc[4] == pytest.approx((2 + 3 + 5) / 3)  # window [2, 3, nan, 5]


def test_ts_rank_last_value_percentile(small_frame):
    out = op.ts_rank(small_frame, 5)
    assert out["A"].iloc[-1] == 1.0  # increasing series: today is the max
    assert out["B"].iloc[-1] == pytest.approx(0.2)  # decreasing: today is the min


def test_ts_decay_linear_weights():
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    out = op.ts_decay_linear(x, 3)
    assert out["A"].iloc[-1] == pytest.approx((3 * 3 + 2 * 2 + 1 * 1) / 6)


def test_ts_decay_linear_skips_nan():
    x = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
    assert op.ts_decay_linear(x, 3)["A"].iloc[-1] == pytest.approx((3 * 3 + 1 * 1) / 4)


def test_ts_argmax_argmin(small_frame):
    assert op.ts_argmax(small_frame, 4)["A"].iloc[-1] == 0
    assert op.ts_argmax(small_frame, 4)["B"].iloc[-1] == 3
    assert op.ts_argmin(small_frame, 4)["B"].iloc[-1] == 0
    assert math.isnan(op.ts_argmax(small_frame, 4)["A"].iloc[2])


def test_ts_corr_perfect(small_frame):
    out = op.ts_corr(small_frame[["A"]], small_frame[["A"]] * 2 + 1, 5)
    assert out["A"].iloc[-1] == pytest.approx(1.0)


def test_rank_cross_sectional(small_frame):
    r = op.rank(small_frame)
    assert list(r.iloc[0]) == pytest.approx([0.5, 1.0, 0.5])  # A and C tie at 1
    assert ((r > 0) & (r <= 1)).all().all()
    assert r.iloc[-1].tolist() == pytest.approx([5 / 6, 1 / 3, 5 / 6])  # tie at the top


def test_zscore_and_normalize(small_frame):
    z = op.zscore(small_frame)
    assert np.allclose(z.mean(axis=1), 0)
    assert np.allclose(op.normalize(small_frame).sum(axis=1), 0)


def test_scale_sums_to_target(small_frame):
    s = op.scale(op.normalize(small_frame), 2.0)
    assert np.allclose(s.abs().sum(axis=1), 2.0)


def test_group_neutralize_zero_per_group(small_frame):
    groups = pd.Series({"A": "g1", "B": "g1", "C": "g2"})
    out = op.group_neutralize(small_frame, groups)
    assert np.allclose(out[["A", "B"]].sum(axis=1), 0)
    assert np.allclose(out["C"], 0)  # single-member group


def test_group_rank_within_group(small_frame):
    groups = pd.Series({"A": "g1", "B": "g1", "C": "g2"})
    out = op.group_rank(small_frame, groups)
    assert out.iloc[0]["B"] == 1.0 and out.iloc[0]["A"] == 0.5
    assert out["C"].eq(1.0).all()


def test_elementwise_nan_domain():
    x = pd.DataFrame({"A": [-1.0, 0.0, 4.0]})
    assert op.log(x)["A"].isna().tolist() == [True, True, False]
    assert op.sqrt(x)["A"].tolist()[2] == 2.0
    assert op.signed_power(x, 0.5)["A"].tolist() == [-1.0, 0.0, 2.0]


def test_max_min_frame_and_scalar(small_frame):
    assert op.max_(small_frame, 5.0)["A"].min() == 5.0
    assert op.min_(5.0, small_frame)["A"].max() == 5.0
    both = op.max_(
        small_frame["A"].to_frame(), small_frame["B"].to_frame().rename(columns={"B": "A"})
    )
    assert both["A"].tolist() == [10, 9, 8, 7, 6, 6, 7, 8, 9, 10]


def test_if_else_scalar_branches(small_frame):
    out = op.if_else(small_frame > 5, 1.0, -1.0)
    assert set(np.unique(out.to_numpy())) == {-1.0, 1.0}


def test_trade_when_enter_hold_exit():
    idx = pd.RangeIndex(6)
    alpha = pd.DataFrame({"A": [1.0, 2, 3, 4, 5, 6]}, index=idx)
    trigger = pd.DataFrame({"A": [1.0, 0, 0, 1, 0, 0]}, index=idx)
    exit_ = pd.DataFrame({"A": [0.0, 0, 1, 0, 0, 0]}, index=idx)
    out = op.trade_when(trigger, alpha, exit_)["A"].tolist()
    assert out[:2] == [1.0, 1.0]  # enter at 1, hold
    assert math.isnan(out[2])  # exit
    assert out[3:] == [4.0, 4.0, 4.0]  # re-enter at 4, hold


def test_comparison_keeps_nan():
    a = pd.DataFrame({"A": [1.0, np.nan]})
    out = op.BINARY_OPS[">"](a, 0.0)
    assert out["A"].iloc[0] == 1.0 and math.isnan(out["A"].iloc[1])


def test_registry_signatures_are_unique_and_documented():
    assert len(op.OPERATORS) >= 35
    for spec in op.OPERATORS.values():
        assert spec.doc and spec.signature.startswith(spec.name + "(")
