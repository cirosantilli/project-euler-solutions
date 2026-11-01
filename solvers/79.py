#!/usr/bin/env python
"""
@formalsw: theorem solve_list_source_equivalence
  (heap : FormalSW.PyAst.Mutable.Heap) (address : Nat) (attempts : List String) :
  heap.Valid → heap[address]? = some (.list (attempts.map (fun s => .scalar (.string s)))) →
    solvers.«79».source_equivalence.Verified heap address
"""
from __future__ import annotations

import heapq

KEYLOG_PATH = "0079_keylog.txt"


def load_attempts() -> list[str]:
    with open(KEYLOG_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_graph(
    attempts: list[str],
) -> tuple[set[str], dict[str, set[str]], dict[str, int]]:
    nodes: set[str] = set()
    adj: dict[str, set[str]] = {}
    indeg: dict[str, int] = {}

    def add_node(v: str) -> None:
        if v not in nodes:
            nodes.add(v)
            adj[v] = set()
            indeg[v] = 0

    def add_edge(u: str, v: str) -> None:
        # Avoid double-counting indegree.
        if v not in adj[u]:
            adj[u].add(v)
            indeg[v] += 1

    for s in attempts:
        if len(s) != 3 or not s.isdigit():
            raise ValueError(f"Invalid attempt: {s!r}")
        a, b, c = s[0], s[1], s[2]
        add_node(a)
        add_node(b)
        add_node(c)
        add_edge(a, b)
        add_edge(b, c)

    return nodes, adj, indeg


def topo_sort(nodes: set[str], adj: dict[str, set[str]], indeg: dict[str, int]) -> str:
    # Use a heap for deterministic output when multiple choices exist.
    heap: list[str] = [v for v in nodes if indeg[v] == 0]
    heapq.heapify(heap)

    out: list[str] = []
    indeg2 = dict(indeg)

    while heap:
        u = heapq.heappop(heap)
        out.append(u)
        for v in adj[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                heapq.heappush(heap, v)

    if len(out) != len(nodes):
        raise ValueError("Constraints contain a cycle; no valid passcode exists.")

    return "".join(out)


def solve(attempts: list[str]) -> str:
    nodes, adj, indeg = build_graph(attempts)
    return topo_sort(nodes, adj, indeg)


if __name__ == "__main__":
    print(solve(load_attempts()))
