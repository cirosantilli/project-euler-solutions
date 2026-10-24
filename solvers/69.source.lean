import solvers.«69.loop.source»
import ProjectEulerSolutions.P69

namespace solvers.«69»
open FormalSW.PyAst HeapCalls ProjectEulerStatements TotientMaximum
set_option Elab.async false

abbrev integer (n : Int) : Mutable.Value := .scalar (.int n)
abbrev guardExpression : Expr := .compare (.name "limit") [(.lt, .constant (.int 2))]

theorem solver_function_source : findFunctionIn program.body "solve" =
    some (solverArgs, solverBody) := rfl

theorem solver_code_source : executableBody solverBody =
    [.ifThen guardExpression [.return (.constant (.int 1))] [],
     .assign [.name "product"] (.constant (.int 2)),
     .annAssign (.name "primes") (.subscript (.name "list") (.name "int"))
       (some (.list [.constant (.int 2)])),
     .assign [.name "candidate"] (.constant (.int 3)),
     .whileDo condition scanBody [], .return (.name "product")] := rfl

theorem guard_result (limit : Int) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (integer limit)) :
    Evaluates program env guardExpression heap (.scalar (.bool (decide (limit < 2)))) heap :=
  Evaluates.compare (middle := heap) (afterRight := heap)
    (Evaluates.lookup getLimit) Evaluates.constant
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

theorem positive_result (limit : Nat) (large : 2 ≤ limit) (base : Mutable.Heap) :
    ∃ finalCache, ReturnsValue program "solve" [number limit] base
      (number (scan limit 3 ⟨2, by decide⟩)) (working base finalCache) := by
  let env := HeapCalls.argumentEnv solverArgs [number limit]
  let productEnv := setLocal env "product" (number 2)
  let cacheEnv := setLocal productEnv "primes" (.ref base.length)
  let readyEnv := setLocal cacheEnv "candidate" (number 3)
  have ready : Ready limit 3 2 base.length readyEnv := by constructor <;> rfl
  obtain ⟨finalCache, scanned⟩ := loop_result limit 3 ⟨2, by decide⟩ [2]
    (by decide) (by decide) (by simp) PrimeTrialDivision.initial base readyEnv ready
  have tested := guard_result limit base env rfl
  have guarded : ¬ (limit : Int) < 2 := by omega
  refine ⟨finalCache, ?_⟩
  apply BlockReturns.function solver_function_source rfl
  rw [solver_code_source]
  apply BlockReturns.branch (flag := false)
    (by simpa only [guarded, decide_false] using tested) rfl
  apply BlockReturns.assign Evaluates.constant
  apply (BlockContinues.annAssign (Evaluates.list (.cons Evaluates.constant .nil))).returns
  apply BlockReturns.assign Evaluates.constant
  exact scanned

theorem solve.result (limit : Int) (base : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [integer limit] base
      (number (optimizedResult limit)) finalHeap := by
  by_cases small : limit < 2
  · refine ⟨base, ?_⟩
    apply BlockReturns.function solver_function_source rfl
    rw [solver_code_source]
    apply BlockReturns.branch (flag := true)
      (by simpa only [small, decide_true] using
        guard_result limit base (HeapCalls.argumentEnv solverArgs [integer limit]) rfl) rfl
    simpa only [optimizedResult, if_pos small] using
      (BlockReturns.return_value (module := program) (value := number 1) Evaluates.constant)
  · have cast : (limit.toNat : Int) = limit := Int.toNat_of_nonneg (by omega)
    obtain ⟨finalCache, returned⟩ := positive_result limit.toNat (by omega) base
    exact ⟨working base finalCache, by simpa only [optimizedResult, if_neg small, number,
      integer, cast] using returned⟩

theorem solve.model_result (limit : Int) (base : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [integer limit] base
      (number (ProjectEulerSolutions.P69.solve limit)) finalHeap := solve.result limit base

end solvers.«69»
