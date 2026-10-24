import solvers.«70.scan.loop.source»
import ProjectEulerSolutions.P70

/-! Initialized source proofs for the optimized totient-permutation search. -/
namespace solvers.«70»
open FormalSW.PyAst HeapCalls ProjectEulerStatements TotientPermutation
set_option Elab.async false

namespace scan
abbrev solverArgs : List Arg := ((findFunctionIn program.body "solve").getD ([], [])).1
abbrev guardCondition : Expr := .compare (.name "limit") [(.lt, .constant (.int 2))]

theorem solver_function_source : findFunctionIn program.body "solve" =
    some (solverArgs, solverBody) := rfl

theorem solver_code_source : executableBody solverBody =
    [.ifThen guardCondition [.return (.constant (.int 0))] [],
      .assign [.name "phi"] (.call (.name "compute_phi_linear") [.name "limit"]),
      .assign [.name "best_n"] (.constant (.int 0)),
      .assign [.name "best_phi"] (.constant (.int 1)),
      .assign [.name "low"] (.constant (.int 1)),
      .assign [.name "high"] (.constant (.int 10)),
      .forIn (.name "n") (.call (.name "range")
        [.constant (.int 2), .binOp (.name "limit") .add (.constant (.int 1))]) scanBody [],
      .return (.name "best_n")] := rfl

theorem guard_result (limit : Int) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit)) :
    Evaluates program env guardCondition heap (flag (decide (limit < 2))) heap :=
  Evaluates.compare (middle := heap) (afterRight := heap)
    (Evaluates.lookup getLimit) Evaluates.constant
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

/-- The initialized table and scan execute to the same signed mathematical minimum. -/
theorem positive_body_result (limit : Nat) (large : 2 ≤ limit) (base : Mutable.Heap)
    (env : HeapCalls.Env) (getLimit : env "limit" = some (number limit)) :
    ∃ finalHeap, BlockReturns program env (executableBody solverBody) base
      (number (TotientPermutation.result (limit : Int))) finalHeap := by
  obtain ⟨tableHeap, sieved, tableFound⟩ := sieve.result (limit : Int) base
  have size : ((limit : Int) + 1).toNat = limit + 1 := by omega
  have found : PhiTable limit (base.length + 1) tableHeap := by
    simpa only [PhiTable, size] using tableFound
  let phiEnv := setLocal env "phi" (.ref (base.length + 1))
  let bestEnv := setLocal phiEnv "best_n" (number 0)
  let bestPhiEnv := setLocal bestEnv "best_phi" (number 1)
  let lowEnv := setLocal bestPhiEnv "low" (number 1)
  let readyEnv := setLocal lowEnv "high" (number 10)
  have stable : Stable limit 0 1 0 (base.length + 1) readyEnv := by
    constructor
    · simpa [readyEnv, lowEnv, bestPhiEnv, bestEnv, phiEnv, setLocal] using getLimit
    · rfl
    · rfl
    · rfl
    · rfl
    · rfl
  have matched : PhiMatches 0 1 := fun nonzero => False.elim (nonzero rfl)
  obtain ⟨finalHeap, scanned⟩ := loop_result limit 2 (limit - 1) 0 1 0 (base.length + 1)
    (by omega) (by decide) (by constructor <;> decide) matched tableHeap readyEnv stable found
  rw [fold_from_two limit (by omega)] at scanned
  refine ⟨finalHeap, ?_⟩
  rw [solver_code_source]
  apply BlockReturns.branch (flag := false)
    (by simpa only [show ¬ (limit : Int) < 2 by omega, decide_false] using
      guard_result limit base env getLimit) rfl
  apply BlockReturns.assign (value := .ref (base.length + 1))
    (Evaluates.call (.cons (Evaluates.lookup getLimit) .nil) sieved)
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_range (values := [number 2, number (limit + 1 : Nat)])
    (start := 2) (stop := (limit + 1 : Nat)) (stride := 1) (middle := tableHeap)
    (.cons Evaluates.constant (.cons
      (Evaluates.binary (Evaluates.lookup stable.getLimit) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary])) .nil)) rfl
  simpa only [Nat.cast_add, Nat.cast_one] using scanned

/-- Small signed bounds return zero; all other bounds use the universal initialized scan. -/
theorem body_result (limit : Int) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit)) :
    ∃ finalHeap, BlockReturns program env (executableBody solverBody) base
      (number (TotientPermutation.result limit)) finalHeap := by
  by_cases small : limit < 2
  · refine ⟨base, ?_⟩
    rw [solver_code_source, result_small limit small]
    apply BlockReturns.branch (flag := true)
      (by simpa only [small, decide_true] using guard_result limit base env getLimit) rfl
    exact BlockReturns.return_value Evaluates.constant
  · have cast : (limit.toNat : Int) = limit := Int.toNat_of_nonneg (by omega)
    obtain ⟨finalHeap, returned⟩ := positive_body_result limit.toNat (by omega) base env
      (by simpa only [cast] using getLimit)
    exact ⟨finalHeap, by simpa only [cast] using returned⟩
end scan

/-- Supplied and default arguments share the same universal body proof. -/
theorem solve.result (limit : Int) (base : Mutable.Heap) (values : List Mutable.Value)
    (arguments : completeArguments scan.solverArgs values = some [number limit]) :
    ∃ finalHeap, ReturnsValue program "solve" values base
      (number (TotientPermutation.result limit)) finalHeap := by
  obtain ⟨finalHeap, returned⟩ := scan.body_result limit base
    (HeapCalls.argumentEnv scan.solverArgs [number limit]) rfl
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function scan.solver_function_source arguments
  exact returned

theorem solve.full_result (limit : Int) (base : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [number limit] base
      (number (TotientPermutation.result limit)) finalHeap := solve.result limit base _ rfl

theorem solve.default_result (base : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [] base
      (number (TotientPermutation.result 9_999_999)) finalHeap :=
  solve.result 9_999_999 base _ rfl

/-- The actual Python source returns the same value as the standalone array solver. -/
theorem solve.model_result (limit : Int) (base : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [number limit] base
      (number (ProjectEulerSolutions.P70.solve limit)) finalHeap := by
  rw [ProjectEulerSolutions.P70.solve_eq_reference]
  exact solve.full_result limit base

end solvers.«70»
