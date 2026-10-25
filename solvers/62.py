#!/usr/bin/env python
"""
@formalsw: theorem digit_signature_source_result
  (n : Nat) (heap : FormalSW.PyAst.Mutable.Heap) :
  FormalSW.PyAst.HeapCalls.ReturnsValue Solvers.P62Ast.module "digit_signature" [.scalar (.int n)] heap
    (.ref (heap.length + 2))
    (heap ++ [.list [.scalar (.int 0)],
      .list ((ProjectEulerSolutions.CubicPermutations.signature n).map (fun (value : Nat) => .scalar (.int value))),
      .tuple ((ProjectEulerSolutions.CubicPermutations.signature n).map (fun (value : Nat) => .scalar (.int value)))])

@formalsw: theorem candidates_source_result
  (k : Int) (heap : FormalSW.PyAst.Mutable.Heap) (env : FormalSW.PyAst.HeapCalls.Env)
  (dictionary : Nat) (rows : List solvers.«62».candidates.Row)
  (view : solvers.«62».candidates.View heap dictionary rows) :
  env "groups" = some (.ref dictionary) → env "k" = some (.scalar (.int k)) →
  ∃ extra, FormalSW.PyAst.HeapCalls.BlockContinues Solvers.P62Ast.module env
    (solvers.«62».groups.widthBody.take 1) heap
    (FormalSW.PyAst.HeapCalls.setLocal env "candidates" (.ref heap.length))
    (ProjectEulerSolutions.HeapCollection.working heap
      ((solvers.«62».candidates.selected k rows).map (fun (value : Int) => .scalar (.int value))) extra)

@formalsw: theorem width_source_result
  (k : Int) (heap : FormalSW.PyAst.Mutable.Heap) (env : FormalSW.PyAst.HeapCalls.Env)
  (dictionary : Nat) (rows : List solvers.«62».candidates.Row)
  (view : solvers.«62».candidates.View heap dictionary rows)
  (first : Int) (rest : List Int) :
  env "groups" = some (.ref dictionary) → env "k" = some (.scalar (.int k)) →
  solvers.«62».candidates.selected k rows = first :: rest →
  ∃ extra, FormalSW.PyAst.HeapCalls.BlockReturns Solvers.P62Ast.module env
    solvers.«62».groups.widthBody heap (.scalar (.int (rest.foldl min first)))
    (ProjectEulerSolutions.HeapCollection.working heap
      ((solvers.«62».candidates.selected k rows).map (fun (value : Int) => .scalar (.int value))) extra)

@formalsw: theorem grouping_store_source_result
  (cube : Nat) (heap : FormalSW.PyAst.Mutable.Heap) (env : FormalSW.PyAst.HeapCalls.Env)
  (dictionary : Nat) (cells : List solvers.«62».store.Cell)
  (view : solvers.«62».store.Store heap dictionary cells) (floor : Nat) :
  env "groups" = some (.ref dictionary) → env "cube" = some (.scalar (.int cube)) →
  floor ≤ dictionary → solvers.«62».store.Above floor cells →
  ∃ finalHeap finalCells, FormalSW.PyAst.HeapCalls.BlockContinues Solvers.P62Ast.module env
    solvers.«62».groups.groupingCode heap
    (FormalSW.PyAst.HeapCalls.setLocal env "sig" (.ref (heap.length + 2))) finalHeap ∧
    solvers.«62».store.Store finalHeap dictionary finalCells ∧
    solvers.«62».store.logical finalCells =
      ProjectEulerSolutions.CubicGroups.record cube (solvers.«62».store.logical cells) ∧
    solvers.«62».store.Above floor finalCells ∧ finalHeap.take floor = heap.take floor ∧
    heap.length ≤ finalHeap.length

@formalsw: theorem solve_source_result
  (target answer : Int) (caller : FormalSW.PyAst.Mutable.Heap) :
  ProjectEulerSolutions.CubicSearch.Returns target ProjectEulerSolutions.CubicSearch.start answer →
  ∃ finalHeap, FormalSW.PyAst.HeapCalls.ReturnsValue Solvers.P62Ast.module "solve"
    [.scalar (.int target)] caller (.scalar (.int answer)) finalHeap ∧
    finalHeap.take caller.length = caller

@formalsw: theorem grouped_reference_source_result
  (limit target : Int) (caller : FormalSW.PyAst.Mutable.Heap)
  (groups : List ProjectEulerSolutions.CubicGroups.Group)
  (summary : ProjectEulerSolutions.CubicGroupCounts.Summary
    (solvers.«62».reference_model.cubes limit) groups) :
  ∃ extra, FormalSW.PyAst.HeapCalls.ReturnsValue P62Ast.module "naive"
    [.scalar (.int limit), .scalar (.int target)] caller
    (.scalar (.int (ProjectEulerSolutions.CubicGroupCounts.choose target groups))) (caller ++ extra)

@formalsw: theorem closed_window_source_equivalence
  (limit : Nat) (target : Int) (caller : FormalSW.PyAst.Mutable.Heap)
  (closed : ProjectEulerSolutions.CubicSearch.width limit < ProjectEulerSolutions.CubicSearch.width (limit + 1))
  (success : data.«project-euler-statements».data.py.project_euler_statements.p62.naive.answer limit target ≠ 0) :
  ∃ finalHeap extra,
    FormalSW.PyAst.HeapCalls.ReturnsValue Solvers.P62Ast.module "solve" [.scalar (.int target)] caller
      (.scalar (.int (data.«project-euler-statements».data.py.project_euler_statements.p62.naive.answer limit target)))
      finalHeap ∧ finalHeap.take caller.length = caller ∧
    FormalSW.PyAst.HeapCalls.ReturnsValue P62Ast.module "naive" [.scalar (.int limit), .scalar (.int target)] caller
      (.scalar (.int (data.«project-euler-statements».data.py.project_euler_statements.p62.naive.answer limit target)))
      (caller ++ extra)
"""
from __future__ import annotations

Signature = tuple[int, ...]


def digit_signature(x: int) -> Signature:
    cnt = [0] * 10
    for ch in str(x):
        cnt[ord(ch) - 48] += 1
    return tuple(cnt)


def solve(k: int) -> int:
    """
    Returns the smallest cube for which exactly k permutations of its digits are cubes.
    """
    groups: dict[Signature, list[int]] = {}  # sig -> [count, min_cube]
    current_len = 1

    n = 1
    while True:
        cube = n * n * n
        length = len(str(cube))

        if length > current_len:
            candidates = [
                min_cube for (count, min_cube) in groups.values() if count == k
            ]
            if candidates:
                return min(candidates)
            groups.clear()
            current_len = length

        sig = digit_signature(cube)
        if sig in groups:
            groups[sig][0] += 1
            if cube < groups[sig][1]:
                groups[sig][1] = cube
        else:
            groups[sig] = [1, cube]

        n += 1


if __name__ == "__main__":
    # Given example from the statement (Project Euler 62 known fact)
    assert solve(3) == 41063625

    print(solve(5))
