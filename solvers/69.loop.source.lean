import solvers.«69.step.source»

namespace solvers.«69»
open FormalSW.PyAst HeapCalls ProjectEulerStatements TotientMaximum
set_option Elab.async false

abbrev condition : Expr := .compare (.binOp (.name "product") .mult (.name "candidate"))
  [(.ltE, .name "limit")]

theorem loop_continuation_source (next : Stmt) : Calls.loopContinuation scanBody next = next := rfl

theorem condition_result (limit cursor product : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (address : Nat) (ready : Ready limit cursor product address env) :
    Evaluates program env condition heap (.scalar (.bool (decide (product * cursor ≤ limit))))
      heap := by
  apply Evaluates.compare (av := number (product * cursor)) (bv := number limit)
    (middle := heap) (afterRight := heap)
  · exact Evaluates.binary (middle := heap) (afterRight := heap)
      (Evaluates.lookup ready.getProduct) (Evaluates.lookup ready.getCandidate)
      (by simp [Mutable.binary, scalarBinary])
  · exact Evaluates.lookup ready.getLimit
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    norm_cast

theorem loop_result (limit cursor : Nat) (product : Positive) (cache : List Nat)
    (large : 3 ≤ cursor) (odd : cursor % 2 = 1) (ordered : cache.Pairwise (· < ·))
    (complete : PrimeTrialDivision.CompleteBelow cache cursor)
    (base : Mutable.Heap) (env : HeapCalls.Env)
    (ready : Ready limit cursor product.val base.length env) :
    ∃ finalCache, BlockReturns program env
      [.whileDo condition scanBody [], .return (.name "product")] (working base cache)
      (number (scan limit cursor product)) (working base finalCache) := by
  have tested :=
    condition_result limit cursor product.val (working base cache) env base.length ready
  by_cases active : product.val * cursor ≤ limit
  · obtain ⟨nextEnv, advanced, nextReady⟩ := iteration_result limit cursor product cache
      large ordered complete base env ready
    have nextOrdered : (nextCache cache cursor).Pairwise (· < ·) :=
      PrimeTrialDivision.advance_ordered cursor cache ordered complete
    have nextComplete : PrimeTrialDivision.CompleteBelow (nextCache cache cursor) (cursor + 2) :=
      PrimeTrialDivision.advance_complete cursor cache large odd complete
    obtain ⟨finalCache, continued⟩ := loop_result limit (cursor + 2) (nextProduct cursor product)
      (nextCache cache cursor) (by omega) (by omega) nextOrdered nextComplete base nextEnv nextReady
    refine ⟨finalCache, ?_⟩
    rw [scan, dif_pos active]
    apply BlockReturns.while_loop (flag := true)
      (by simpa only [active, decide_true] using tested) rfl
    rw [loop_continuation_source]
    exact advanced.returns continued
  · refine ⟨cache, ?_⟩
    rw [scan, dif_neg active]
    apply BlockReturns.while_loop (flag := false)
      (by simpa only [active, decide_false] using tested) rfl
    exact BlockReturns.return_value (Evaluates.lookup ready.getProduct)
termination_by limit + 2 - cursor
decreasing_by
  have positive := product.property
  have lower : cursor ≤ product.val * cursor := by nlinarith
  omega

end solvers.«69»
