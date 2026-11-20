import numpy as np
import pandas as pd
import pytest

from alphaops.engine import ExpressionError, evaluate
from alphaops.engine.operators import rank, ts_delta


def test_matches_direct_operator_calls(panel):
    out = evaluate("rank(-ts_delta(close, 5))", panel)
    expected = rank(-ts_delta(panel.fields["close"], 5))
    pd.testing.assert_frame_equal(out, expected)


def test_assignments_and_reuse(panel):
    a = evaluate("m = ts_mean(returns, 10); s = ts_std(returns, 10); m / s", panel)
    b = evaluate("ts_mean(returns, 10) / ts_std(returns, 10)", panel)
    pd.testing.assert_frame_equal(a, b)


def test_ternary_matches_if_else(panel):
    a = evaluate("volume > adv20 / close ? 1 : -1", panel)
    b = evaluate("if_else(volume > adv20 / close, 1, -1)", panel)
    pd.testing.assert_frame_equal(a, b)


def test_division_by_zero_becomes_nan(panel):
    out = evaluate("close / (close - close)", panel)
    assert not np.isinf(out.to_numpy()).any()


def test_group_argument(panel):
    out = evaluate("group_neutralize(returns, sector)", panel)
    sectors = panel.groups["sector"]
    sums = out.T.groupby(sectors).sum().T
    assert np.allclose(sums.dropna(), 0, atol=1e-12)


@pytest.mark.parametrize(
    "expr, fragment",
    [
        ("rank(closee)", "unknown field 'closee'"),
        ("foo(close)", "unknown operator 'foo'"),
        ("ts_mean(close)", "takes 2 argument"),
        ("ts_mean(close, 2.5)", "window must be an integer"),
        ("ts_mean(close, 0)", "window must be an integer"),
        ("ts_mean(close, volume)", "must be a constant"),
        ("ts_mean(5, 3)", "expected an expression over fields"),
        ("group_neutralize(close, colour)", "expected a group name"),
        ("sector + 1", "is a group"),
        ("1 + 2", "constant"),
        ("x = rank(close)", "last statement must be an expression"),
        ("close = 1; close", "reserved name"),
        ("scale(close, size=2)", "has no parameter 'size'"),
        ("rank", "is an operator"),
    ],
)
def test_semantic_errors(panel, expr, fragment):
    with pytest.raises(ExpressionError, match=fragment):
        evaluate(expr, panel)


def test_no_python_eval(panel):
    # Anything that isn't in the grammar is rejected by the tokenizer/parser.
    for evil in ["__import__('os')", "close.__class__", "[c for c in close]"]:
        with pytest.raises(ExpressionError):
            evaluate(evil, panel)
