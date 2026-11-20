"""Evaluate a parsed alpha expression over a Panel."""

from __future__ import annotations

import math

import pandas as pd

from alphaops.engine.operators import BINARY_OPS, OPERATORS, OpSpec, clean, if_else, truthy
from alphaops.engine.panel import Panel
from alphaops.engine.parser import (
    Assign,
    BinOp,
    Call,
    ExpressionError,
    Name,
    Node,
    Number,
    Program,
    Ternary,
    UnaryOp,
    parse,
)

MAX_WINDOW = 1000
CONSTANTS = {"nan": math.nan, "true": 1.0, "false": 0.0}


class Evaluator:
    def __init__(self, panel: Panel) -> None:
        self.panel = panel
        self._source = ""

    def evaluate(self, expression: str | Program) -> pd.DataFrame:
        if isinstance(expression, str):
            self._source = expression
            program = parse(expression)
        else:
            program = expression
        env: dict[str, object] = {}
        result: object = None
        for stmt in program.statements:
            if isinstance(stmt, Assign):
                if stmt.name in self.panel.fields or stmt.name in OPERATORS:
                    raise self._err(f"cannot assign to reserved name {stmt.name!r}", stmt)
                env[stmt.name] = self._eval(stmt.value, env)
                result = None
            else:
                result = self._eval(stmt, env)
        if result is None:
            raise ExpressionError(
                "the last statement must be an expression, not an assignment", self._source
            )
        if not isinstance(result, pd.DataFrame):
            raise ExpressionError(
                "expression is a constant; an alpha must depend on data fields", self._source
            )
        return result

    # ------------------------------------------------------------------ helpers
    def _err(self, msg: str, node: Node) -> ExpressionError:
        return ExpressionError(msg, self._source, node.pos)

    def _eval(self, node: Node, env: dict[str, object]) -> object:
        if isinstance(node, Number):
            return node.value
        if isinstance(node, Name):
            return self._resolve(node, env)
        if isinstance(node, UnaryOp):
            v = self._eval(node.operand, env)
            if node.op == "-":
                return clean(-v)
            if node.op == "!":
                return 1.0 - truthy(v)
        if isinstance(node, BinOp):
            a, b = self._eval(node.left, env), self._eval(node.right, env)
            for v, side in ((a, node.left), (b, node.right)):
                if isinstance(v, pd.Series):
                    raise self._err("group names can only be used as group arguments", side)
            return clean(BINARY_OPS[node.op](a, b))
        if isinstance(node, Ternary):
            return clean(
                if_else(*(self._eval(n, env) for n in (node.cond, node.then, node.otherwise)))
            )
        if isinstance(node, Call):
            return self._call(node, env)
        raise self._err(f"cannot evaluate {type(node).__name__}", node)  # pragma: no cover

    def _resolve(self, node: Name, env: dict[str, object]) -> object:
        if node.id in env:
            return env[node.id]
        if node.id in self.panel.fields:
            return self.panel.fields[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        if node.id in self.panel.groups:
            raise self._err(f"{node.id!r} is a group; use it inside group_* operators", node)
        if node.id in OPERATORS:
            raise self._err(f"{node.id!r} is an operator; call it like {node.id}(...)", node)
        available = ", ".join(sorted(self.panel.fields))
        raise self._err(f"unknown field {node.id!r}. Available fields: {available}", node)

    def _call(self, node: Call, env: dict[str, object]) -> object:
        spec: OpSpec | None = OPERATORS.get(node.func)
        if spec is None:
            raise self._err(f"unknown operator {node.func!r} (see `alphaops ops`)", node)
        n_req = len(spec.args) - spec.optional
        if not (n_req <= len(node.args) <= len(spec.args)):
            raise self._err(
                f"{spec.signature} takes {n_req} argument(s), got {len(node.args)}", node
            )
        args = [
            self._arg(kind, arg, spec, env) for kind, arg in zip(spec.args, node.args, strict=False)
        ]
        kwargs = {}
        for key, val in node.kwargs:
            if key not in spec.kwargs:
                raise self._err(f"{spec.name} has no parameter {key!r}", val)
            kwargs[key] = self._arg(spec.kwargs[key], val, spec, env)
        try:
            return clean(spec.func(*args, **kwargs))
        except (TypeError, ValueError) as exc:
            raise self._err(f"{spec.name}: {exc}", node) from exc

    def _arg(self, kind: str, node: Node, spec: OpSpec, env: dict[str, object]) -> object:
        if kind == "group":
            if not isinstance(node, Name) or node.id not in self.panel.groups:
                groups = ", ".join(sorted(self.panel.groups))
                raise self._err(f"{spec.name}: expected a group name ({groups})", node)
            return self.panel.groups[node.id]
        value = self._eval(node, env)
        if kind in ("int", "num"):
            if isinstance(value, pd.DataFrame):
                raise self._err(f"{spec.name}: this argument must be a constant number", node)
            if kind == "int":
                if (
                    not math.isfinite(value)
                    or value != int(value)
                    or not (1 <= value <= MAX_WINDOW)
                ):
                    raise self._err(
                        f"{spec.name}: window must be an integer in [1, {MAX_WINDOW}], "
                        f"got {value:g}",
                        node,
                    )
                return int(value)
            return float(value)
        if kind == "frame" and not isinstance(value, pd.DataFrame):
            raise self._err(
                f"{spec.name}: expected an expression over fields, got a constant", node
            )
        return value


def evaluate(expression: str, panel: Panel) -> pd.DataFrame:
    return Evaluator(panel).evaluate(expression)
