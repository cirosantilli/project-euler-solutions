import solvers.«9.py.lean.d».Source
import lang.py.FormalSW.HeapRuntimeCostSorting

namespace solvers.«9».runtimeCost
open FormalSW.PyAst HeapCalls.Runtime solvers.«9».runtimeSource
private abbrev program := Solvers.P9Ast.module

/-- Every integer triple, including unsuccessful candidates, has a bounded
sorting and unpacking cost. Both allocated containers remain in the heap. -/
theorem sorted_any_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (a b c : Int)
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) :
    ∃ x y z, BlockContinuesIn .word program env [sortedStatement] heap 72
      (HeapCalls.assignNames env ["a", "b", "c"] (triple x y z))
      (heap ++ [.tuple (triple a b c), .list (triple x y z)]) := by
  have length : (ScalarSort.ints [a, b, c]).length = 3 := (ScalarSort.ints_perm [a, b, c]).length_eq
  obtain ⟨x, y, z, values⟩ : ∃ x y z, ScalarSort.ints [a, b, c] = [x, y, z] := by
    generalize ScalarSort.ints [a, b, c] = xs at length ⊢
    match xs with
    | [] | [_] | [_, _] => simp at length
    | x :: y :: z :: rest =>
        have empty : rest = [] := by simpa using length
        exact ⟨x, y, z, by rw [empty]⟩
  have tupleRun : EvaluatesIn .word program env (.tuple [.name "a", .name "b", .name "c"])
      heap 15 (.ref heap.length) (heap ++ [.tuple (triple a b c)]) :=
    EvaluatesIn.tuple (ArgumentsIn.cons (EvaluatesIn.lookup ha) (ArgumentsIn.cons (EvaluatesIn.lookup hb)
      (ArgumentsIn.cons (EvaluatesIn.lookup hc) .nil))) (by simp [triple])
  have tupleFound : (heap ++ [.tuple (triple a b c)])[heap.length]? = some (.tuple (triple a b c)) := by simp
  have sorted : EvaluatesIn .word program env sortedExpression heap 61 (.ref (heap.length + 1))
      (heap ++ [.tuple (triple a b c), .list (triple x y z)]) := by
    have returned := HeapCalls.primitive_sorted_tuple (heap ++ [.tuple (triple a b c)])
      heap.length [.int a, .int b, .int c] [.int x, .int y, .int z]
      (by simp [triple]) (by simpa [values] using ScalarSort.sort_ints [a, b, c])
    have priced : operation .word (heap ++ [.tuple (triple a b c)]) (.builtin .sorted) [.ref heap.length] = some 40 := by
      simp only [operation, HeapCalls.SortWork.operation, tupleFound]
      change (HeapCalls.SortWork.scalars (([a, b, c].map Constant.int).map Mutable.Value.scalar)).map _ = some 40
      rw [HeapCalls.SortWork.scalars_map, Option.map_some, HeapCalls.SortWork.sorting_ints]
      simp only [List.length_cons, List.length_nil]
      decide
    simpa [triple, List.append_assoc] using EvaluatesIn.sorted (work := 40)
      (output := triple x y z) tupleRun returned priced
  refine ⟨x, y, z, ?_⟩
  apply BlockContinuesIn.unpack_list (names := ["a", "b", "c"]) (items := triple x y z) rfl sorted
  · simp
  · rfl

end solvers.«9».runtimeCost
