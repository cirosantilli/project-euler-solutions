#!/usr/bin/env python
"""
@formalsw: theorem generate_4digit_polygonals_source_result
  (s : Int) (heap : FormalSW.PyAst.Mutable.Heap)
  (globals : FormalSW.PyAst.Mutable.Globals) :
  3 ≤ s → FormalSW.PyAst.HeapCalls.ReturnsValue Solvers.P61Ast.module "generate_4digit_polygonals"
    [.scalar (.int s)] heap (.ref heap.length)
    (heap ++ [.list ((solvers.«61».generate_4digit_polygonals.answer s).map
      (fun value => .scalar (.int value)))]) globals

@formalsw: theorem build_prefix_map_source_result
  (values : List Int) (heap : FormalSW.PyAst.Mutable.Heap) (input : Nat)
  (globals : FormalSW.PyAst.Mutable.Globals) :
  heap[input]? = some (.list (values.map (fun value => .scalar (.int value)))) →
  ∃ finalHeap, Nonempty (solvers.«61».build_prefix_map.Store heap finalHeap values) ∧
    FormalSW.PyAst.HeapCalls.ReturnsValue Solvers.P61Ast.module "build_prefix_map"
      [.ref input] heap (.ref heap.length) finalHeap globals
"""
from __future__ import annotations


def polygonal(s: int, n: int) -> int:
    """
    @formalsw: Returns
    @formalsw: Py from project_euler_statements.p61 import polygonal as reference_polygonal
    @formalsw: Equiv(reference_polygonal)
    s-gonal number P_{s,n} = n((s-2)n - (s-4)) / 2, for s >= 3
    """
    return n * ((s - 2) * n - (s - 4)) // 2


def generate_4digit_polygonals(s: int) -> list[int]:
    nums: list[int] = []
    n = 1
    while True:
        val = polygonal(s, n)
        if val >= 10000:
            break
        if 1000 <= val <= 9999:
            # last two digits must be a valid next "prefix" (10..99) for a 4-digit number
            if val % 100 >= 10:
                nums.append(val)
        n += 1
    return nums


def build_prefix_map(nums: list[int]) -> dict[int, list[int]]:
    mp: dict[int, list[int]] = {}
    for x in nums:
        pref = x // 100
        mp.setdefault(pref, []).append(x)
    return mp


def solve() -> int:
    types = [3, 4, 5, 6, 7, 8]
    nums_by_type: dict[int, list[int]] = {
        t: generate_4digit_polygonals(t) for t in types
    }
    pref_map: dict[int, dict[int, list[int]]] = {
        t: build_prefix_map(nums_by_type[t]) for t in types
    }

    def dfs(
        path: list[tuple[int, int]], used_types: set[int], used_nums: set[int]
    ) -> list[tuple[int, int]] | None:
        if len(path) == 6:
            last_num = path[-1][0]
            first_num = path[0][0]
            if last_num % 100 == first_num // 100:
                return path
            return None

        suffix = path[-1][0] % 100
        for t in types:
            if t in used_types:
                continue
            for nxt in pref_map[t].get(suffix, []):
                if nxt in used_nums:
                    continue
                res = dfs(path + [(nxt, t)], used_types | {t}, used_nums | {nxt})
                if res is not None:
                    return res
        return None

    # Fix the start type to 8 (octagonal) to break rotational symmetry of the cycle.
    for start in nums_by_type[8]:
        res = dfs([(start, 8)], {8}, {start})
        if res is not None:
            return sum(x for x, _t in res)

    raise RuntimeError("No cycle found")


if __name__ == "__main__":
    # Problem statement examples
    assert polygonal(3, 127) == 8128
    assert polygonal(4, 91) == 8281
    assert polygonal(5, 44) == 2882

    print(solve())
