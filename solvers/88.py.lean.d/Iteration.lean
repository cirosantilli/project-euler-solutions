import solvers.«88.py.lean.d».State

namespace solvers.«88»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

/-- Each accepted factor updates the shared table before recursively visiting its descendants. -/
theorem iteration (maxSize limit : Nat) (before : List Nat) (valid : Admissible before)
    (factor : Nat) (lower : 2 ≤ factor) (inside : before.prod * factor ≤ limit)
    (context : Context) (cells : Cells context maxSize limit)
    (descendants : Descendants maxSize limit before context)
    (best : List Nat) (length : best.length = maxSize + 1)
    (env : HeapCalls.Env) (ready : Locals env context before)
    (getFactor : env "f" = some (number factor)) :
    BlockContinues program env dfsBody (working context best) (measuredEnv env before factor)
      (working context (accumulate (branch maxSize limit before.prod before.sum before.length factor) best)) := by
  let measured := measuredEnv env before factor
  let size := effectiveSize (before ++ [factor])
  have sizeFormula : size = before.length + 1 + before.prod * factor - (before.sum + factor) := by
    simp [size, effectiveSize]
  have measuredReady := ready.measured factor
  have getSize : measured "k" = some (number size) := measured_lookup env before factor
  have getValue : measured "new_prod" = some (number (before.prod * factor)) := by
    simp [measured, measuredEnv, setLocal]
  have arithmetic := measure_factor before factor valid lower env (working context best)
    ready.product ready.summ ready.terms getFactor
  apply arithmetic.append
  change BlockContinues program measured [dfsBody[4]'(by decide)] (working context best) measured _
  have tested := size_evaluate size maxSize limit context cells best measured getSize measuredReady.maximum
  by_cases accepted : size ≤ maxSize
  · let updated := relax best (size, before.prod * factor)
    have improved : BlockContinues program measured (acceptedBody.take 1) (working context best)
        measured (working context updated) := by
      simpa only [working_store] using improve best size (before.prod * factor) (by omega)
        (working context best) context.bestCell context.base.length measured measuredReady.best getSize getValue
        (cells.best best) (working_found context best)
    have returned := descendants factor lower inside updated (by simpa [updated, relax_length] using length)
    have called : Evaluates program measured
        (.call (.name "__formalsw_local_solve_dfs") [.name "f", .name "new_prod", .name "new_sum", .name "new_terms",
          .name "__formalsw_cell_solve_best", .name "__formalsw_cell_solve_k_max", .name "__formalsw_cell_solve_limit"])
        (working context updated) (.scalar .none)
        (working context (accumulate
          (search maxSize limit factor (before.prod * factor) (before.sum + factor) (before.length + 1)) updated)) := by
      apply Evaluates.call (values := [number factor, number (before.prod * factor), number (before.sum + factor),
        number (before.length + 1), .ref context.bestCell, .ref context.maxCell, .ref context.limitCell])
      · exact .cons (Evaluates.lookup (by simpa [measured, measuredEnv, setLocal] using getFactor))
          (.cons (Evaluates.lookup getValue)
          (.cons (Evaluates.lookup (by simp [measured, measuredEnv, setLocal]))
          (.cons (Evaluates.lookup (by simp [measured, measuredEnv, setLocal]))
          (.cons (Evaluates.lookup measuredReady.best)
          (.cons (Evaluates.lookup measuredReady.maximum)
          (.cons (Evaluates.lookup measuredReady.bound) .nil))))))
      · simpa [SearchReturns] using returned
    rw [branch, ← sizeFormula, if_pos accepted]
    apply BlockContinues.branch (flag := true) tested (by simp [Mutable.truth, scalarTruth, accepted])
    exact improved.append (BlockContinues.discard called)
  · rw [branch, ← sizeFormula, if_neg accepted]
    exact BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, accepted]) .nil

end solvers.«88»
