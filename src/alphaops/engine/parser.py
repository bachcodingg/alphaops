"""Tokenizer and recursive-descent parser for a Fast-Expression-like alpha language.

Grammar (lowest to highest precedence)::

    program    := statement (';' statement)* ';'?
    statement  := NAME '=' expr | expr
    expr       := or_expr ('?' expr ':' expr)?
    or_expr    := and_expr ('||' and_expr)*
    and_expr   := comparison ('&&' comparison)*
    comparison := additive (('<'|'<='|'>'|'>='|'=='|'!=') additive)?
    additive   := term (('+'|'-') term)*
    term       := unary (('*'|'/') unary)*
    unary      := ('-'|'+'|'!') unary | power
    power      := atom ('^' unary)?
    atom       := NUMBER | NAME | NAME '(' [arg (',' arg)*] ')' | '(' expr ')'
    arg        := NAME '=' expr | expr

Expressions are parsed into an AST and evaluated by ``evaluator.py``; nothing is ever
passed to Python's ``eval``, so user input cannot execute code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class ExpressionError(ValueError):
    """Syntax or semantic error in an alpha expression, with a caret pointer."""

    def __init__(self, message: str, source: str = "", pos: int | None = None) -> None:
        self.message, self.source, self.pos = message, source, pos
        text = message
        if source and pos is not None:
            text += f"\n  {source}\n  {' ' * pos}^"
        super().__init__(text)


# ---------------------------------------------------------------------- AST nodes
@dataclass(frozen=True)
class Node:
    pos: int = field(default=0, compare=False, kw_only=True)


@dataclass(frozen=True)
class Number(Node):
    value: float


@dataclass(frozen=True)
class Name(Node):
    id: str


@dataclass(frozen=True)
class Call(Node):
    func: str
    args: tuple[Node, ...]
    kwargs: tuple[tuple[str, Node], ...] = ()


@dataclass(frozen=True)
class BinOp(Node):
    op: str
    left: Node
    right: Node


@dataclass(frozen=True)
class UnaryOp(Node):
    op: str
    operand: Node


@dataclass(frozen=True)
class Ternary(Node):
    cond: Node
    then: Node
    otherwise: Node


@dataclass(frozen=True)
class Assign(Node):
    name: str
    value: Node


@dataclass(frozen=True)
class Program(Node):
    statements: tuple[Node, ...]


# ---------------------------------------------------------------------- tokenizer
_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op><=|>=|==|!=|&&|\|\||[-+*/^(),<>?:;=!])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    kind: str  # number | name | op | eof
    text: str
    pos: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(source):
        m = _TOKEN_RE.match(source, pos)
        if not m:
            raise ExpressionError(f"unexpected character {source[pos]!r}", source, pos)
        kind = m.lastgroup
        assert kind is not None
        if kind != "ws":
            tokens.append(Token(kind, m.group(), pos))
        pos = m.end()
    tokens.append(Token("eof", "", len(source)))
    return tokens


# ------------------------------------------------------------------------- parser
class _Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.tokens = tokenize(source)
        self.i = 0

    @property
    def tok(self) -> Token:
        return self.tokens[self.i]

    def peek(self, k: int = 1) -> Token:
        return self.tokens[min(self.i + k, len(self.tokens) - 1)]

    def error(self, msg: str, tok: Token | None = None) -> ExpressionError:
        tok = tok or self.tok
        return ExpressionError(msg, self.source, tok.pos)

    def accept(self, *ops: str) -> Token | None:
        if self.tok.kind == "op" and self.tok.text in ops:
            t = self.tok
            self.i += 1
            return t
        return None

    def expect(self, op: str) -> Token:
        t = self.accept(op)
        if t is None:
            found = self.tok.text or "end of input"
            raise self.error(f"expected {op!r}, found {found!r}")
        return t

    # program / statements
    def program(self) -> Program:
        stmts: list[Node] = []
        while True:
            if self.tok.kind == "eof":
                break
            stmts.append(self.statement())
            if not self.accept(";"):
                break
        if self.tok.kind != "eof":
            raise self.error(f"unexpected {self.tok.text!r}")
        if not stmts:
            raise self.error("empty expression")
        return Program(tuple(stmts), pos=0)

    def statement(self) -> Node:
        if self.tok.kind == "name" and self.peek().kind == "op" and self.peek().text == "=":
            name_tok = self.tok
            self.i += 2
            return Assign(name_tok.text, self.expr(), pos=name_tok.pos)
        return self.expr()

    def expr(self) -> Node:
        node = self.or_expr()
        q = self.accept("?")
        if q:
            then = self.expr()
            self.expect(":")
            otherwise = self.expr()
            return Ternary(node, then, otherwise, pos=q.pos)
        return node

    def _left_assoc(self, sub, ops: tuple[str, ...]) -> Node:
        node = sub()
        while (t := self.accept(*ops)) is not None:
            node = BinOp(t.text, node, sub(), pos=t.pos)
        return node

    def or_expr(self) -> Node:
        return self._left_assoc(self.and_expr, ("||",))

    def and_expr(self) -> Node:
        return self._left_assoc(self.comparison, ("&&",))

    def comparison(self) -> Node:
        node = self.additive()
        t = self.accept("<", "<=", ">", ">=", "==", "!=")
        if t:
            node = BinOp(t.text, node, self.additive(), pos=t.pos)
            if self.tok.kind == "op" and self.tok.text in ("<", "<=", ">", ">=", "==", "!="):
                raise self.error("chained comparisons are not supported; use &&")
        return node

    def additive(self) -> Node:
        return self._left_assoc(self.term, ("+", "-"))

    def term(self) -> Node:
        return self._left_assoc(self.unary, ("*", "/"))

    def unary(self) -> Node:
        t = self.accept("-", "+", "!")
        if t:
            operand = self.unary()
            if t.text == "+":
                return operand
            if t.text == "-" and isinstance(operand, Number):
                return Number(-operand.value, pos=t.pos)
            return UnaryOp(t.text, operand, pos=t.pos)
        return self.power()

    def power(self) -> Node:
        node = self.atom()
        t = self.accept("^")
        if t:
            node = BinOp("^", node, self.unary(), pos=t.pos)
        return node

    def atom(self) -> Node:
        t = self.tok
        if t.kind == "number":
            self.i += 1
            return Number(float(t.text), pos=t.pos)
        if t.kind == "name":
            self.i += 1
            if self.accept("("):
                return self.call(t)
            return Name(t.text, pos=t.pos)
        if self.accept("("):
            node = self.expr()
            self.expect(")")
            return node
        found = t.text or "end of input"
        raise self.error(f"expected a number, field or '(', found {found!r}")

    def call(self, name_tok: Token) -> Call:
        args: list[Node] = []
        kwargs: list[tuple[str, Node]] = []
        if not self.accept(")"):
            while True:
                if self.tok.kind == "name" and self.peek().kind == "op" and self.peek().text == "=":
                    key = self.tok.text
                    self.i += 2
                    kwargs.append((key, self.expr()))
                else:
                    if kwargs:
                        raise self.error("positional argument after keyword argument")
                    args.append(self.expr())
                if self.accept(")"):
                    break
                if not self.accept(","):
                    found = self.tok.text or "end of input"
                    raise self.error(
                        f"expected ',' or ')' in call to {name_tok.text}(), found {found!r}"
                    )
        return Call(name_tok.text, tuple(args), tuple(kwargs), pos=name_tok.pos)


def parse(source: str) -> Program:
    if not source or not source.strip():
        raise ExpressionError("empty expression")
    return _Parser(source).program()
