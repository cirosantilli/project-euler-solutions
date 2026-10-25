import solvers.«63.powers.source»

set_option Elab.async false

namespace solvers.«63».solve
open FormalSW.PyAst Calls ProjectEulerSolutions.PowerfulDigits
private abbrev program := Solvers.P63Ast.module
private abbrev number (n : Nat) : Constant := .int n

def seedBody : List Stmt :=
  [.assign [.name "power"] (.name "a"),
   .assign [.name "lower"] (.constant (.int 1)), innerLoop]

/-- The seed traversal adds the complete count for each seed below the numeral base. -/
theorem seed_loop (base first count total : Nat) (valid : 2 ≤ base) (positive : 1 ≤ first)
    (endpoint : first + count = base) (env : ScalarEnv)
    (getBase : env "base" = some (number base)) (getTotal : env "total" = some (number total)) :
    BlockReturns program env
      [.rangeStepLoop "a" first base 1 seedBody [], .return (.name "total")]
      (number (total + sumWindow (seedCount base) first count)) := by
  induction count generalizing first total env with
  | zero =>
      have done : rangeContinues (first : Int) base 1 = false := by
        simp [rangeContinues, show first = base by omega]
      apply BlockReturns.range_done done
      exact BlockReturns.return_value (by simpa using Evaluates.lookup getTotal)
  | succ count ih =>
      have inside : first < base := by omega
      have step : rangeContinues (first : Int) base 1 = true := by
        simp [rangeContinues, show (first : Int) < base by exact_mod_cast inside]
      let entered := env.set "a" (number first)
      let powered := entered.set "power" (number first)
      let ready := powered.set "lower" (number 1)
      apply BlockReturns.range_step step
      apply BlockReturns.assign (value := number first) (Evaluates.lookup (by simp [ScalarEnv.set]))
      apply BlockReturns.assign (value := number 1) Evaluates.constant
      apply inner_loop base first 1 total valid inside (by omega) ready
        (by simpa [ready, powered, entered, ScalarEnv.set] using getBase)
        (by simp [ready, powered, entered, ScalarEnv.set])
        (by simp [ready, powered, ScalarEnv.set])
        (by simp [ready, ScalarEnv.set])
        (by simpa [ready, powered, entered, ScalarEnv.set] using getTotal)
      intro after keptBase _ keptTotal
      have rest := ih (first + 1) (total + seedCount base first) (by omega) (by omega)
        after keptBase (by simpa only [seed_window] using keptTotal)
      have tally : total + sumWindow (seedCount base) first (count + 1) =
          total + seedCount base first + sumWindow (seedCount base) (first + 1) count := by
        rw [sumWindow_succ, Nat.add_assoc]
      simpa only [loopContinuation, show loopControl seedBody = false from rfl,
        Bool.false_eq_true, if_false, Nat.cast_add, Nat.cast_one, tally] using rest

theorem body_result (base : Nat) (valid : 2 ≤ base) (env : ScalarEnv)
    (getBase : env "base" = some (number base)) :
    BlockReturns program env
      (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)
      (number (answer base)) := by
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_range
    (.cons Evaluates.constant (.cons (Evaluates.lookup (by simpa [ScalarEnv.set] using getBase)) .nil)) rfl
  have executed := seed_loop base 1 (base - 1) 0 valid (by omega) (by omega)
    (env.set "total" (number 0))
    (by simpa [ScalarEnv.set] using getBase) (by simp [ScalarEnv.set])
  simpa only [seed_sum, Nat.zero_add] using executed

theorem result (base : Nat) (valid : 2 ≤ base) :
    ReturnsValue program "solve" [number base] (number (answer base)) := by
  apply BlockReturns.function rfl rfl
  exact body_result base valid _ rfl

theorem default_result : ReturnsValue program "solve" [] (number (answer 10)) := by
  apply BlockReturns.function_defaults rfl (by decide) rfl
  exact body_result 10 (by omega) _ rfl

end solvers.«63».solve
