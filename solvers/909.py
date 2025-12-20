#!/usr/bin/env python3
"""
Project Euler 909: L-Expressions I

We avoid brute-force rewriting for the target expression by using the algebraic
meaning of the combinators Z and S (Church numerals) and a small amount of
hand-derived simplification for the specific expression in the problem.

No external libraries are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union, Tuple


# ----------------------------
# Small parser + reducer (for the examples / asserts only)
# ----------------------------


@dataclass(frozen=True)
class Sym:
    name: str  # 'A', 'Z', 'S'


@dataclass(frozen=True)
class Num:
    value: int


@dataclass(frozen=True)
class App:
    f: "Term"
    x: "Term"


Term = Union[Sym, Num, App]


def _tokenize(s: str) -> List[Union[str, int]]:
    toks: List[Union[str, int]] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "()":
            toks.append(c)
            i += 1
            continue
        if c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            toks.append(int(s[i:j]))
            i = j
            continue
        if c.isalpha():
            # In this problem the only symbols are single letters: A, Z, S.
            toks.append(c)
            i += 1
            continue
        raise ValueError(f"Unexpected character: {c!r}")
    return toks


def parse_expr(s: str) -> Term:
    toks = _tokenize(s)
    pos = 0

    def atom() -> Term:
        nonlocal pos
        if pos >= len(toks):
            raise ValueError("Unexpected end of input")
        tok = toks[pos]
        if tok == "(":
            pos += 1
            t = expr()
            if pos >= len(toks) or toks[pos] != ")":
                raise ValueError("Missing ')'")
            pos += 1
            return t
        pos += 1
        if isinstance(tok, int):
            return Num(tok)
        if tok in ("A", "Z", "S"):
            return Sym(tok)
        raise ValueError(f"Unknown symbol: {tok!r}")

    def expr() -> Term:
        nonlocal pos
        t = atom()
        # Implicit application: u(v)(w) is represented as ((u v) w)
        while pos < len(toks) and toks[pos] != ")":
            t = App(t, atom())
        return t

    out = expr()
    if pos != len(toks):
        raise ValueError("Trailing tokens")
    return out


def _is_nat(t: Term) -> bool:
    return isinstance(t, Num) and t.value >= 0


def _spine(t: Term) -> Tuple[Term, List[Term]]:
    """Return head and argument list for left-associated application."""
    args: List[Term] = []
    while isinstance(t, App):
        args.append(t.x)
        t = t.f
    args.reverse()
    return t, args


def _rebuild(head: Term, args: List[Term]) -> Term:
    t = head
    for a in args:
        t = App(t, a)
    return t


def _step(t: Term) -> Tuple[Term, bool]:
    """One leftmost-outermost rewrite step."""
    head, args = _spine(t)

    if isinstance(head, Sym):
        if head.name == "A" and len(args) >= 1 and _is_nat(args[0]):
            # A(x) -> x+1
            new = Num(args[0].value + 1)
            return _rebuild(new, args[1:]), True

        if head.name == "Z" and len(args) >= 2:
            # Z(u)(v) -> v
            v = args[1]
            return _rebuild(v, args[2:]), True

        if head.name == "S" and len(args) >= 3:
            # S(u)(v)(w) -> v(u(v)(w))
            u, v, w = args[0], args[1], args[2]
            new = App(v, App(App(u, v), w))
            return _rebuild(new, args[3:]), True

    # normal-order: reduce function position first, then argument
    if isinstance(t, App):
        nf, changed = _step(t.f)
        if changed:
            return App(nf, t.x), True
        nx, changed = _step(t.x)
        if changed:
            return App(t.f, nx), True

    return t, False


def normalize(t: Term, step_limit: int = 200_000) -> Term:
    for _ in range(step_limit):
        t, changed = _step(t)
        if not changed:
            return t
    raise RuntimeError(
        "Normalization step limit reached (expected only for large expressions)."
    )


def eval_small(expr: str) -> int:
    """Evaluate a small L-expression expected to reduce to a natural number."""
    t = parse_expr(expr)
    t = normalize(t, step_limit=500_000)
    if not isinstance(t, Num):
        raise ValueError("Expression did not reduce to a natural number")
    return t.value


# ----------------------------
# Fast evaluation for the target expression
# ----------------------------

MOD = 1_000_000_000


def F_mod(n: int, mod: int = MOD) -> int:
    """
    Let g(x) = x(x+1).
    For a Church numeral n, the term D = S(S) transforms n into g(n).

    For the specific expression in problem 909 one can show it equals:
        F(F(1))
    where
        F(n) = g(g(n^2(n+1))).

    We compute F(n) modulo mod.
    """
    n %= mod
    # a = n^2 (n+1)
    a = (n * n) % mod
    a = (a * ((n + 1) % mod)) % mod
    # b = g(a) = a(a+1)
    b = (a * ((a + 1) % mod)) % mod
    # c = g(b) = b(b+1)
    c = (b * ((b + 1) % mod)) % mod
    return c


def main() -> None:
    # Examples stated in the problem statement:
    assert eval_small("S(Z)(A)(0)") == 1
    assert eval_small("S(S)(S(S))(S(Z))(A)(0)") == 6

    seed = F_mod(1)
    ans = F_mod(seed)

    # Print last 9 digits (zero-padded).
    print(f"{ans:09d}")


if __name__ == "__main__":
    main()
