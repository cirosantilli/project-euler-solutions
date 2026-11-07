import solvers.«14.py.lean.d».SearchRuntime
import lang.py.FormalSW.HeapRuntimeCostSorting
import lang.py.FormalSW.HeapRuntimeCostSetup
import Mathlib.Tactic.Linarith

/-! Complete source runtime with fixed arithmetic and size-sensitive dictionaries.
The bound depends on Collatz trajectory work under the existing Collatz assumption. -/
namespace solvers.«14».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.HeapCalls FormalSW.PyAst.HeapCalls.Runtime
open ProjectEulerStatements.P14 ProjectEulerStatements.Collatz

/-- Initialize both caches, visit the range, and allocate the returned pair. -/
theorem longest_returns (n : Nat) (positive : 0 < n) :
    ∃ heap, ReturnsIn .word program "longest_collatz_under" [.scalar (.int (n + 1))] []
      (10000 * (prefixWork n + 1)^2)
      (.ref heap.length) (heap ++ [.tuple [.scalar (.int (ProjectEulerStatements.P14.naive n)),
        .scalar (.int (collatzLen (ProjectEulerStatements.P14.naive n)))]]) := by
  have initialLocals : SearchLocals (searchEnv (n + 1)) (n + 1) 1 2 1 := by
    refine ⟨rfl, rfl, rfl, rfl, ?_⟩
    simp [searchEnv, setLocal, length_one]
  obtain ⟨heap, small, big, env, smallFound, bigFound, valid, locals, _, scanned⟩ :=
    search_loop n 1 (n + 1) 1 2 1 (by omega) (by omega) (by decide) (searchEnv (n + 1))
      (searchHeap (n + 1)) (Cache.initialSmall (n + 1)) [] (Cache.initial_valid _ (by omega)) rfl rfl initialLocals
  rw [initial_winner n positive] at locals
  have range : BlockContinuesIn .word program (searchEnv (n + 1))
      [.forIn (.name "n") (.call (.name "range") [.constant (.int 1), .name "limit"]) searchBody []]
      (searchHeap (n + 1)) (2000 * (rangeWork 1 n + 1) * (rangeWork 1 n + 1) + 9) env heap := by
    exact (BlockContinuesIn.for_range_pair EvaluatesIn.constant
      (EvaluatesIn.lookup initialLocals.bound) scanned).mono (by simp; omega)
  have returned : BlockReturnsIn .word program env [.return (.tuple [.name "best_n", .name "best_len"])]
      heap 11 (.ref heap.length)
      (heap ++ [.tuple [.scalar (.int (ProjectEulerStatements.P14.naive n)),
        .scalar (.int (collatzLen (ProjectEulerStatements.P14.naive n)))]]) := by
    apply BlockReturnsIn.return_value
    exact EvaluatesIn.tuple (.cons (EvaluatesIn.lookup locals.best)
      (.cons (EvaluatesIn.lookup locals.length) .nil)) (by simp)
  have body := (search_initialize (n + 1) (by omega)).returns (range.returns returned)
  have called := BlockReturnsIn.function (name := "longest_collatz_under")
    (args := searchDefinition.1) (body := searchDefinition.2) (values := [.scalar (.int (n + 1))]) rfl rfl body
  refine ⟨heap, called.mono ?_⟩
  change 3 + (100 * (n + 1 + 1) + (2000 * (rangeWork 1 n + 1) * (rangeWork 1 n + 1) + 9 + 11)) ≤ _
  rw [rangeWork_prefix]
  have bound := input_le_prefixWork n
  nlinarith [Nat.zero_le (prefixWork n)]

/-- The public scalar wrapper adds one, calls the search, and reads its pair. -/
theorem solve_positive (n : Nat) (positive : 0 < n) :
    ∃ heap, ReturnsIn .word program "solve" [.scalar (.int n)] []
      (20000 * (prefixWork n + 1)^2) (.scalar (.int (ProjectEulerStatements.P14.naive n))) heap := by
  obtain ⟨heap, called⟩ := longest_returns n positive
  let finalHeap := heap ++ [.tuple [.scalar (.int (ProjectEulerStatements.P14.naive n)),
      .scalar (.int (collatzLen (ProjectEulerStatements.P14.naive n)))]]
  let definition := (findFunctionIn program.body "solve").getD ([], [])
  refine ⟨finalHeap, ?_⟩
  apply ReturnsIn.mono (small := 3 + (10000 * (prefixWork n + 1)^2 + 27))
  · apply BlockReturnsIn.function (args := definition.1) (body := definition.2)
      (budget := 10000 * (prefixWork n + 1)^2 + 27) rfl rfl
    apply BlockReturnsIn.mono (small := 7 + (10000 * (prefixWork n + 1)^2 + 18) + 2) _ (by omega)
    apply BlockReturnsIn.branch (cc := 7) (cb := 10000 * (prefixWork n + 1)^2 + 18)
      (value := .bool false) (flag := false) (middle := [])
    · apply EvaluatesIn.compare (ca := 1) (cb := 1) (work := 2)
        (EvaluatesIn.lookup rfl) EvaluatesIn.constant
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega
      · rfl
    · rfl
    · apply BlockReturnsIn.return_value (cost := 10000 * (prefixWork n + 1)^2 + 17)
      apply EvaluatesIn.mono (small := (6 + 10000 * (prefixWork n + 1)^2 + 2) + 1 + 1 + 1 + 6) _ (by omega)
      apply EvaluatesIn.tuple_subscript (address := heap.length) (middle := finalHeap) (afterIndex := finalHeap) (ca := 6 + 10000 * (prefixWork n + 1)^2 + 2) (cb := 1)
        (items := [.scalar (.int (ProjectEulerStatements.P14.naive n)),
          .scalar (.int (collatzLen (ProjectEulerStatements.P14.naive n)))])
        (by intro a b c; simp)
      · apply EvaluatesIn.call_single (ca := 6) (cc := 10000 * (prefixWork n + 1)^2)
          (value := .scalar (.int (n + 1))) (middle := [])
        · apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
            (EvaluatesIn.lookup rfl) EvaluatesIn.constant
          · simp [Mutable.binary, scalarBinary]
          · rfl
        · exact called
      · exact EvaluatesIn.constant
      · simp [finalHeap]
      · simp [finalHeap, Mutable.getItem, Mutable.index, Except.map]
  · nlinarith [Nat.zero_le (prefixWork n)]

end solvers.«14».runtimeCost

namespace solvers.«14»
open FormalSW.PyAst

theorem solve.«BigO((ProjectEulerStatements.Collatz.prefixWork n + 1)^2, n = limit)» :
    HeapCalls.Runtime.IsBigO .word program "solve"
      (fun n : Nat => [.scalar (.int n)]) (fun _ => []) (fun _ => True) id
      (fun n => (ProjectEulerStatements.Collatz.prefixWork n + 1)^2) := by
  refine ⟨20001, 1, fun n _ large => ?_⟩
  obtain ⟨heap, run⟩ := runtimeCost.solve_positive n (by simpa using large)
  have bound := HeapCalls.Runtime.ReturnsIn.timeBound_append (extra := heap) run
  apply RuntimeCost.TimeBound.mono bound
  simp only [id_eq]
  nlinarith [Nat.zero_le (ProjectEulerStatements.Collatz.prefixWork n)]

end solvers.«14»
