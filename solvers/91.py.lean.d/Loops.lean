import solvers.«91.py.lean.d».Arithmetic

namespace solvers.«91»
open FormalSW.PyAst Calls ProjectEulerSolutions ProjectEulerSolutions.LatticeRightTriangles
set_option Elab.async false

/-- The source's row traversal adds every partner count, including its early continue at the origin. -/
theorem row_result (n x y total : Nat) (env : ScalarEnv) (ready : Ready env n x total)
    (xBound : x ≤ n) (following : List Stmt) (answer : Constant)
    (rest : ∀ after, Ready after n x (total + rowSum n x y) → BlockReturns program after following answer) :
    BlockReturns program env (.rangeStepLoop "y" y (n + 1) 1 innerBody [] :: following) answer := by
  by_cases inside : y ≤ n
  · let entered := env.set "y" (number y)
    have enteredReady : Ready entered n x total := by
      rcases ready with ⟨hn, hx, ht⟩
      constructor <;> simpa [entered, ScalarEnv.set] using ‹_›
    have getY : entered "y" = some (number y) := by simp [entered, ScalarEnv.set]
    have step := NaturalIntervalSum.step (pointContribution n x) y n inside
    change rowSum n x y = pointContribution n x y + rowSum n x (y + 1) at step
    apply BlockReturns.range_step (by simp [rangeContinues]; omega)
    by_cases origin : x = 0 ∧ y = 0
    · apply BlockReturns.branch (value := .bool true)
        (by simpa only [entered, origin, true_and, decide_true] using origin_test x y entered enteredReady.first getY)
      apply BlockReturns.continue_loop rfl
      have continued := row_result n x (y + 1) total entered enteredReady xBound following answer
        (fun after afterReady => rest after (by
          simpa only [step, pointContribution, if_pos origin, Nat.zero_add] using afterReady))
      simpa only [Int.natCast_add, Int.natCast_one] using continued
    · apply BlockReturns.branch (value := .bool false)
        (by simpa only [origin, decide_false] using origin_test x y entered enteredReady.first getY)
      by_cases axis : x = 0 ∨ y = 0
      · apply BlockReturns.branch (value := .bool true)
          (by simpa only [axis, decide_true] using axis_test x y entered enteredReady.first getY)
        apply BlockReturns.augAssign (value := number (total + n))
          (Evaluates.binary (Evaluates.lookup enteredReady.count) (Evaluates.lookup enteredReady.bound)
            (by simp [scalarBinary, number]))
        apply BlockReturns.loop_resume
        let updated := entered.set "total" (number (total + n))
        have updatedReady : Ready updated n x (total + n) := by
          constructor
          · simpa [updated, ScalarEnv.set] using enteredReady.bound
          · simpa [updated, ScalarEnv.set] using enteredReady.first
          · rfl
        have continued := row_result n x (y + 1) (total + n) updated updatedReady xBound following answer
          (fun after afterReady => rest after (by
            simpa only [step, pointContribution, if_neg origin, if_pos axis, Nat.add_assoc] using afterReady))
        simpa only [Int.natCast_add, Int.natCast_one] using continued
      · apply BlockReturns.branch (value := .bool false)
          (by simpa only [axis, decide_false] using axis_test x y entered enteredReady.first getY)
        apply interior n x y total entered enteredReady getY (by omega) (by omega) xBound inside
        intro after afterReady
        apply BlockReturns.loop_resume
        have continued := row_result n x (y + 1) (total + upperSteps n x y + lowerSteps n x y)
          after afterReady xBound following answer (fun final finalReady => rest final (by
            simpa only [step, pointContribution, if_neg origin, if_neg axis, Nat.add_assoc] using finalReady))
        simpa only [Int.natCast_add, Int.natCast_one] using continued
  · apply BlockReturns.range_done (by simp [rangeContinues]; omega)
    apply rest env
    have empty := NaturalIntervalSum.empty (pointContribution n x) y n (by omega)
    simpa only [rowSum, empty, Nat.add_zero] using ready
termination_by n + 1 - y
decreasing_by all_goals omega

/-- Completing one row retains the shared bound and passes its exact count to the next row. -/
theorem rows_result (n x total : Nat) (env : ScalarEnv)
    (getN : env "n" = some (number n)) (getTotal : env "total" = some (number total))
    (following : List Stmt) (answer : Constant)
    (rest : ∀ after, after "n" = some (number n) → after "total" = some (number (total + rowsSum n x)) →
      BlockReturns program after following answer) :
    BlockReturns program env (.rangeStepLoop "x" x (n + 1) 1 outerBody [] :: following) answer := by
  by_cases inside : x ≤ n
  · let entered := env.set "x" (number x)
    have ready : Ready entered n x total := by
      constructor
      · simpa [entered, ScalarEnv.set] using getN
      · rfl
      · simpa [entered, ScalarEnv.set] using getTotal
    have step := NaturalIntervalSum.step (fun x => rowSum n x 0) x n inside
    change rowsSum n x = rowSum n x 0 + rowsSum n (x + 1) at step
    apply BlockReturns.range_step (by simp [rangeContinues]; omega)
    apply BlockReturns.for_range (values := [number (n + 1)])
    · exact .cons (Evaluates.binary (Evaluates.lookup ready.bound) Evaluates.constant
        (by simp [scalarBinary, number])) .nil
    · rfl
    apply row_result n x 0 total entered ready inside
    intro after afterReady
    have continued := rows_result n (x + 1) (total + rowSum n x 0) after afterReady.bound afterReady.count
      following answer (fun final getFinalN getFinalTotal => rest final getFinalN (by
        simpa only [step, Nat.add_assoc] using getFinalTotal))
    simpa only [Int.natCast_add, Int.natCast_one] using continued
  · apply BlockReturns.range_done (by simp [rangeContinues]; omega)
    apply rest env getN
    have empty := NaturalIntervalSum.empty (fun x => rowSum n x 0) x n (by omega)
    simpa only [rowsSum, empty, Nat.add_zero] using getTotal
termination_by n + 1 - x
decreasing_by omega

end solvers.«91»
