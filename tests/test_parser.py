import pytest

from alphaops.engine.parser import (
    Assign,
    BinOp,
    Call,
    ExpressionError,
    Name,
    Number,
    Ternary,
    UnaryOp,
    parse,
    tokenize,
)


def only(src: str):
    prog = parse(src)
    assert len(prog.statements) == 1
    return prog.statements[0]


def test_precedence_mul_over_add():
    assert only("1 + 2 * 3") == BinOp("+", Number(1), BinOp("*", Number(2), Number(3)))


def test_left_associativity():
    assert only("8 - 3 - 2") == BinOp("-", BinOp("-", Number(8), Number(3)), Number(2))


def test_power_binds_tighter_than_unary_minus():
    assert only("-x ^ 2") == UnaryOp("-", BinOp("^", Name("x"), Number(2)))


def test_negative_literal_folded():
    assert only("-5") == Number(-5.0)


def test_call_with_args_and_kwargs():
    node = only("scale(rank(close), scale=2)")
    assert node == Call("scale", (Call("rank", (Name("close"),)),), (("scale", Number(2)),))


def test_ternary_and_comparison():
    node = only("volume > 1e6 ? close : 0")
    assert isinstance(node, Ternary)
    assert node.cond == BinOp(">", Name("volume"), Number(1e6))


def test_logical_precedence():
    node = only("a > 1 || b > 2 && c > 3")
    assert node.op == "||"
    assert node.right.op == "&&"


def test_multiple_statements_with_assignment():
    prog = parse("x = ts_mean(close, 5); y = x * 2; rank(y);")
    assert [type(s) for s in prog.statements] == [Assign, Assign, Call]
    assert prog.statements[0].name == "x"


@pytest.mark.parametrize(
    "src, fragment",
    [
        ("", "empty"),
        ("rank(close", r"expected ',' or '\)'"),
        ("1 +", "expected a number"),
        ("close $ 2", "unexpected character"),
        ("f(a=1, 2)", "positional argument after keyword"),
        ("a < b < c", "chained comparisons"),
        ("rank(close) close", "unexpected"),
    ],
)
def test_syntax_errors(src, fragment):
    with pytest.raises(ExpressionError, match=fragment):
        parse(src)


def test_error_points_at_position():
    with pytest.raises(ExpressionError) as exc:
        parse("rank(close $ 1)")
    assert exc.value.pos == 11
    assert "^" in str(exc.value)


def test_tokenizer_numbers():
    kinds = [(t.kind, t.text) for t in tokenize("1 2.5 .5 1e-3")][:-1]
    assert kinds == [("number", "1"), ("number", "2.5"), ("number", ".5"), ("number", "1e-3")]
