from __future__ import annotations

NETWORK_NAME = "drosophila"
NODES = ['D',
 'Twi',
 'Sna',
 'Fog',
 'FogR',
 'Cta',
 'GEF',
 'Csk',
 'Src',
 'GAP',
 'Rho',
 'rock',
 'GDI',
 'Moe',
 'mlcp',
 'mlck',
 'mlc',
 'Actin']
RULES = {
    "D": "1",
    "Twi": "Twi | D",
    "Sna": "D",
    "Fog": "Twi | Sna",
    "FogR": "Fog",
    "Cta": "FogR",
    "GEF": "Cta",
    "Csk": "Cta",
    "Src": "!Csk",
    "GAP": "Src | GAP",
    "Rho": "!GDI & GEF & !GAP",
    "rock": "Rho",
    "GDI": "!Moe",
    "Moe": "rock & !mlcp & Moe",
    "mlcp": "!rock",
    "mlck": "1",
    "mlc": "!mlcp & rock & mlck",
    "Actin": "Moe & mlc",
}

# -*- coding: utf-8 -*-
"""
Boolean network source for Experiment 3.

Rule syntax:
    !X       negation
    X & Y    conjunction
    X | Y    disjunction
    0, 1     constants
"""

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Set, Tuple


class Expr:
    def eval(self, env: Dict[str, int]) -> int:
        raise NotImplementedError

    def variables(self) -> Set[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Const(Expr):
    value: int

    def eval(self, env: Dict[str, int]) -> int:
        return int(self.value)

    def variables(self) -> Set[str]:
        return set()


@dataclass(frozen=True)
class Var(Expr):
    name: str

    def eval(self, env: Dict[str, int]) -> int:
        return int(env[self.name])

    def variables(self) -> Set[str]:
        return {self.name}


@dataclass(frozen=True)
class Not(Expr):
    arg: Expr

    def eval(self, env: Dict[str, int]) -> int:
        return 1 - self.arg.eval(env)

    def variables(self) -> Set[str]:
        return self.arg.variables()


@dataclass(frozen=True)
class And(Expr):
    left: Expr
    right: Expr

    def eval(self, env: Dict[str, int]) -> int:
        return int(self.left.eval(env) and self.right.eval(env))

    def variables(self) -> Set[str]:
        return self.left.variables() | self.right.variables()


@dataclass(frozen=True)
class Or(Expr):
    left: Expr
    right: Expr

    def eval(self, env: Dict[str, int]) -> int:
        return int(self.left.eval(env) or self.right.eval(env))

    def variables(self) -> Set[str]:
        return self.left.variables() | self.right.variables()


class RuleParser:
    def __init__(self, text: str):
        self.text = (
            text.replace("¬", "!")
                .replace("~", "!")
                .replace("⋀", "&")
                .replace("∧", "&")
                .replace("⋁", "|")
                .replace("∨", "|")
                .replace(" ", "")
        )
        self.pos = 0

    def parse(self) -> Expr:
        expr = self._parse_or()
        if self.pos != len(self.text):
            raise ValueError(
                f"Unexpected character at position {self.pos} in rule {self.text!r}"
            )
        return expr

    def _parse_or(self) -> Expr:
        expr = self._parse_and()
        while self._peek("|"):
            self.pos += 1
            expr = Or(expr, self._parse_and())
        return expr

    def _parse_and(self) -> Expr:
        expr = self._parse_factor()
        while self._peek("&"):
            self.pos += 1
            expr = And(expr, self._parse_factor())
        return expr

    def _parse_factor(self) -> Expr:
        if self._peek("!"):
            self.pos += 1
            return Not(self._parse_factor())

        if self._peek("("):
            self.pos += 1
            expr = self._parse_or()
            if not self._peek(")"):
                raise ValueError(f"Missing closing parenthesis in rule {self.text!r}")
            self.pos += 1
            return expr

        if self._peek("0") or self._peek("1"):
            value = int(self.text[self.pos])
            self.pos += 1
            return Const(value)

        name = self._parse_name()
        if not name:
            raise ValueError(
                f"Expected variable, constant, or parenthesis at position "
                f"{self.pos} in rule {self.text!r}"
            )
        return Var(name)

    def _parse_name(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and (
            self.text[self.pos].isalnum() or self.text[self.pos] == "_"
        ):
            self.pos += 1
        return self.text[start:self.pos]

    def _peek(self, token: str) -> bool:
        return self.text.startswith(token, self.pos)


def parse_rule(rule: str) -> Expr:
    return RuleParser(rule).parse()


def compiled_network() -> Dict:
    """Return nodes, parsed rules, and node indices for this network."""
    missing = set(NODES) ^ set(RULES)
    if missing:
        raise ValueError(f"Nodes/rules mismatch: {sorted(missing)}")

    parsed = {node: parse_rule(rule) for node, rule in RULES.items()}

    all_vars = set()
    for expr in parsed.values():
        all_vars |= expr.variables()

    unknown = all_vars - set(NODES)
    if unknown:
        raise ValueError(f"Unknown variables in rules: {sorted(unknown)}")

    return {
        "name": NETWORK_NAME,
        "nodes": NODES,
        "index": {node: i for i, node in enumerate(NODES)},
        "rules": RULES,
        "parsed": parsed,
    }


def eval_node(network: Dict, node: str, x) -> int:
    """Evaluate one update rule on a global binary vector x."""
    env = {name: int(x[i]) for i, name in enumerate(network["nodes"])}
    return network["parsed"][node].eval(env)


def eval_network_state(network: Dict, x) -> List[int]:
    """Evaluate the complete Boolean network on one global binary state."""
    return [eval_node(network, node, x) for node in network["nodes"]]


def infer_signed_support(network: Dict, node: str) -> Tuple[List[int], List[int], List[int]]:
    """
    Infer signed support from the truth table of the node's local rule.
    """
    expr = network["parsed"][node]
    local_vars = sorted(expr.variables(), key=lambda name: network["index"][name])
    n = len(network["nodes"])

    signs_by_name = {}

    for var in local_vars:
        other_vars = [v for v in local_vars if v != var]
        has_pos = False
        has_neg = False

        for bits in product([0, 1], repeat=len(other_vars)):
            env0 = dict(zip(other_vars, bits))
            env0[var] = 0
            env1 = dict(env0)
            env1[var] = 1

            y0 = expr.eval(env0)
            y1 = expr.eval(env1)

            if y0 < y1:
                has_pos = True
            elif y0 > y1:
                has_neg = True

            if has_pos and has_neg:
                raise ValueError(
                    f"Node {node!r} is not unate in variable {var!r} "
                    f"under the provided Boolean rule."
                )

        if has_pos:
            signs_by_name[var] = 1
        elif has_neg:
            signs_by_name[var] = -1

    support = [network["index"][v] for v in signs_by_name]
    support_signs = [signs_by_name[v] for v in signs_by_name]

    sign_vector = [0] * n
    for idx, sign in zip(support, support_signs):
        sign_vector[idx] = sign

    return support, support_signs, sign_vector


def node_type(network: Dict, node: str, support: List[int]) -> str:
    """Classify a node as constant, identity, or regulated."""
    if len(support) == 0:
        return "constant"
    if len(support) == 1 and network["nodes"][support[0]] == node:
        return "identity"
    return "regulated"


def nonconstant_nodes(network: Dict) -> List[str]:
    """Return nodes whose update rules have nonempty signed support."""
    out = []
    for node in network["nodes"]:
        support, _, _ = infer_signed_support(network, node)
        if len(support) > 0:
            out.append(node)
    return out


def node_metadata(network: Dict, node: str) -> Dict:
    """Return support/sign metadata for one node."""
    support, signs, sign_vector = infer_signed_support(network, node)
    return {
        "node": node,
        "node_type": node_type(network, node, support),
        "support": support,
        "signs": signs,
        "sign_vector": sign_vector,
        "true_k": len(support),
    }
