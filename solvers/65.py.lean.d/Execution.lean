import «.formalsw».solvers.«65.py.ast»
import ProjectEulerSolutions.Lib.EConvergentCalls
import ProjectEulerSolutions.Lib.DecimalDigitSumCalls

namespace solvers.«65»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.EConvergents ProjectEulerSolutions.EConvergentCalls
set_option Elab.async false
private abbrev program := Solvers.P65Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
variable {globals : Mutable.Globals}

theorem digit_sum.result (n : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "digit_sum" [number n] heap (number (Nat.digits 10 n).sum) heap globals := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  exact ProjectEulerSolutions.DecimalDigitSumCalls.string_sum program _ "x" n heap globals
    (by cases globals <;> rfl)

namespace solve
def loopBody : List Stmt := coefficientStatements ++
  [.assign [.name "previous"] (.name "numerator"),
   .assign [.name "numerator"]
     (.binOp (.binOp (.name "coefficient") .mult (.name "numerator")) .add (.name "older")),
   .assign [.name "older"] (.name "previous")]
private abbrev finish : Stmt := .return (.call (.name "digit_sum") [.name "numerator"])

/-- Each source iteration consumes one coefficient and retains just two consecutive numerators. -/
theorem loop_result (n first count older current : Nat) (endpoint : first + count = n)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getOlder : env "older" = some (number older)) (getCurrent : env "numerator" = some (number current)) :
    BlockReturns program env [.rangeStepLoop "k" first n 1 loopBody [], finish]
      heap (number (Nat.digits 10 (run ((List.range' first count).map coefficient) older current)).sum) heap globals := by
  induction count generalizing first older current env with
  | zero =>
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value
        (Evaluates.call (.cons (Evaluates.lookup getCurrent) .nil) (digit_sum.result current heap globals))
  | succ count ih =>
      let entered := setLocal env "k" (number first)
      let ready := setLocal entered "coefficient" (number (coefficient first))
      let saved := setLocal ready "previous" (number current)
      let next := setLocal saved "numerator" (number (coefficient first * current + older))
      let after := setLocal next "older" (number current)
      have rest := ih (first + 1) current (coefficient first * current + older) (by omega) after
        (by simp [after, setLocal]) (by simp [after, next, setLocal])
      have updated : BlockReturns program ready
          [.assign [.name "previous"] (.name "numerator"),
           .assign [.name "numerator"]
             (.binOp (.binOp (.name "coefficient") .mult (.name "numerator")) .add (.name "older")),
           .assign [.name "older"] (.name "previous"),
           .rangeStepLoop "k" ((first : Int) + 1) n 1 loopBody [], finish]
          heap (number (Nat.digits 10 (run ((List.range' first (count + 1)).map coefficient) older current)).sum) heap globals := by
        apply BlockReturns.assign (value := number current)
          (Evaluates.lookup (by simpa [ready, entered, setLocal] using getCurrent))
        apply BlockReturns.assign (value := number (coefficient first * current + older)) (middle := heap)
        · apply Evaluates.binary (av := number (coefficient first * current)) (bv := number older)
            (middle := heap) (afterRight := heap)
          · exact Evaluates.binary (av := number (coefficient first)) (bv := number current)
              (middle := heap) (afterRight := heap)
              (Evaluates.lookup (by simp [ready, setLocal]))
              (Evaluates.lookup (by simpa [saved, ready, entered, setLocal] using getCurrent))
              (by simp [Mutable.binary, scalarBinary, number])
          · exact Evaluates.lookup (by simpa [saved, ready, entered, setLocal] using getOlder)
          · simp [Mutable.binary, scalarBinary, number]
        apply BlockReturns.assign (value := number current) (middle := heap)
          (Evaluates.lookup (by simp [setLocal]))
        simpa only [List.range'_succ, List.map_cons, run_cons, Int.natCast_add, Int.natCast_one] using rest
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [loopBody, Calls.loopContinuation, show Calls.loopControl loopBody = false from rfl,
        Bool.false_eq_true, if_false, List.append_assoc] using
          (coefficient_assignment program first heap entered (by simp [entered, setLocal])).returns updated

theorem body_result (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) :
    BlockReturns program env (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)
      heap (number (answer n)) heap globals := by
  apply BlockReturns.assign (value := number 0) Evaluates.constant
  apply BlockReturns.assign (value := number 1) Evaluates.constant
  apply BlockReturns.for_range (.cons (Evaluates.lookup (by simpa [setLocal] using getN)) .nil) rfl
  simpa only [answer, numerator, coefficients, ProjectEulerStatements.P65.coeffs,
    List.range_eq_range', Nat.zero_add] using loop_result (globals := globals) n 0 n 0 1 (by omega) heap
      (setLocal (setLocal env "older" (number 0)) "numerator" (number 1))
      (by simp [setLocal]) (by simp [setLocal])

theorem result (n : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "solve" [number n] heap (number (answer n)) heap globals := by
  apply BlockReturns.function rfl rfl
  exact body_result n heap _ (by cases globals <;> rfl)

theorem default_result (heap : Mutable.Heap) :
    ReturnsValue program "solve" [] heap (number (answer 100)) heap :=
  ReturnsValue.with_defaults rfl rfl rfl (result 100 heap)

end solve
end solvers.«65»
