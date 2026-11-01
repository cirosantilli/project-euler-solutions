import solvers.«64.py.lean.d».Period
import ProjectEulerSolutions.Lib.SqrtPeriodCounts

namespace solvers.«64».solve
open FormalSW.PyAst Calls ProjectEulerStatements.SqrtReducedStates ProjectEulerSolutions.SqrtPeriodCounts
set_option Elab.async false
private abbrev program := Solvers.P64Ast.module
private abbrev number (value : Int) : Constant := .int value

def test : Expr := .compare
  (.binOp (.call (.name "period_length_sqrt") [.name "n"]) .mod (.constant (.int 2)))
  [(.eq, .constant (.int 1))]
def loopBody : List Stmt := [.ifThen test [.augAssign (.name "count") .add (.constant (.int 1))] []]

theorem test_evaluates (n : Nat) (env : ScalarEnv) (getN : env "n" = some (number n)) :
    Evaluates program env test (.bool (decide (length n % 2 = 1))) := by
  apply Evaluates.compare (av := number ((length n % 2 : Nat) : Int)) (bv := number 1)
  · exact Evaluates.binary (av := number (length n)) (bv := number 2)
      (Evaluates.call (.cons (Evaluates.lookup getN) .nil) (period_length_sqrt.result n)) Evaluates.constant
      (by simp [scalarBinary, number])
  · exact Evaluates.constant
  · simp [scalarCompare, evalCmpOp]
    norm_cast

theorem loop_result (stop : Int) (first size total : Nat) (window : (stop - first).toNat = size)
    (env : ScalarEnv) (getCount : env "count" = some (number total)) :
    BlockReturns program env [.rangeStepLoop "n" first stop 1 loopBody [], .return (.name "count")]
      (number ((total + count first size : Nat) : Int)) := by
  induction size generalizing first total env with
  | zero =>
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (by simpa using Evaluates.lookup getCount)
  | succ size ih =>
      let entered := env.set "n" (number first)
      have tested := test_evaluates first entered (by simp [entered, ScalarEnv.set])
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      by_cases odd : length first % 2 = 1
      · apply BlockReturns.branch (value := .bool true) (by simpa only [odd, decide_true] using tested)
        apply BlockReturns.augAssign (value := number ((total + 1 : Nat) : Int))
          (Evaluates.binary (Evaluates.lookup (by simpa [entered, ScalarEnv.set] using getCount))
            Evaluates.constant (by simp [scalarBinary, number]))
        have rest := ih (first + 1) (total + 1) (by omega)
          (entered.set "count" (number ((total + 1 : Nat) : Int))) (by simp [ScalarEnv.set])
        have tally : total + count first (size + 1) = total + 1 + count (first + 1) size := by
          rw [count_succ]
          simp only [weight, if_pos odd, Nat.add_assoc]
        simpa only [Calls.loopContinuation, show Calls.loopControl loopBody = false from rfl,
          Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one, tally] using rest
      · apply BlockReturns.branch (value := .bool false) (by simpa only [odd, decide_false] using tested)
        have rest := ih (first + 1) total (by omega) entered (by simpa [entered, ScalarEnv.set] using getCount)
        have tally : total + count first (size + 1) = total + count (first + 1) size := by
          rw [count_succ]
          simp only [weight, if_neg odd, Nat.zero_add]
        simpa only [Calls.loopContinuation, show Calls.loopControl loopBody = false from rfl,
          scalarTruth, Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one, tally] using rest

theorem result (limit : Int) : ReturnsValue program "solve" [number limit] (number (answer limit)) := by
  let env := argumentEnv ((findFunctionIn program.body "solve").getD ([], [])).1 [number limit]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := number 0) Evaluates.constant
  apply BlockReturns.for_range (values := [number 2, number (limit + 1)])
    (.cons Evaluates.constant (.cons (Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant
      (by simp [scalarBinary, number])) .nil)) rfl
  simpa only [answer, Nat.zero_add] using
    loop_result (limit + 1) 2 (limit - 1).toNat 0 (by omega) (env.set "count" (number 0))
      (by simp [ScalarEnv.set])

end solvers.«64».solve
