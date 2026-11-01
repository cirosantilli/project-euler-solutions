import «.formalsw».solvers.«63.py.ast»
import ProjectEulerSolutions.Lib.PowerfulDigitCounts
import lang.py.FormalSW.Calls

set_option Elab.async false

namespace solvers.«63».solve
open FormalSW.PyAst Calls ProjectEulerSolutions.PowerfulDigits
private abbrev program := Solvers.P63Ast.module
private abbrev number (n : Nat) : Constant := .int n
private abbrev flag (b : Bool) : Constant := .bool b

def condition : Expr := .compare (.name "power") [(.gtE, .name "lower")]
def innerBody : List Stmt :=
  [.augAssign (.name "total") .add (.constant (.int 1)),
   .augAssign (.name "power") .mult (.name "a"),
   .augAssign (.name "lower") .mult (.name "base")]
def innerLoop : Stmt := .whileDo condition innerBody []

/-- The real multiplication loop consumes exactly the successful exponents before its first failure. -/
theorem inner_loop (base seed exponent total : Nat) (valid : 2 ≤ base) (inside : seed < base)
    (positive : 1 ≤ exponent) (env : ScalarEnv)
    (getBase : env "base" = some (number base)) (getSeed : env "a" = some (number seed))
    (getPower : env "power" = some (number (seed ^ exponent)))
    (getLower : env "lower" = some (number (base ^ (exponent - 1))))
    (getTotal : env "total" = some (number total)) (following : List Stmt) (result : Constant)
    (continued : ∀ after : ScalarEnv, after "base" = some (number base) → after "a" = some (number seed) →
      after "total" = some (number (total + sumWindow (hit base seed) exponent (bound base - exponent))) →
      BlockReturns program after following result) :
    BlockReturns program env (innerLoop :: following) result := by
  have tested : Evaluates program env condition
      (flag (decide (base ^ (exponent - 1) ≤ seed ^ exponent))) := by
    apply Evaluates.compare (Evaluates.lookup getPower) (Evaluates.lookup getLower)
    simp [scalarCompare, evalCmpOp]
    norm_cast
  by_cases yes : base ^ (exponent - 1) ≤ seed ^ exponent
  · have before := successful_before_bound base seed exponent valid inside yes
    have tally : total + sumWindow (hit base seed) exponent (bound base - exponent) =
        total + 1 + sumWindow (hit base seed) (exponent + 1) (bound base - (exponent + 1)) := by
      rw [window_tail_step _ base exponent before]
      simp only [hit, if_pos yes]
      omega
    let e1 := env.set "total" (number (total + 1))
    let e2 := e1.set "power" (number (seed ^ (exponent + 1)))
    let e3 := e2.set "lower" (number (base ^ exponent))
    have rest := inner_loop base seed (exponent + 1) (total + 1) valid inside (by omega) e3
      (by simpa [e3, e2, e1, ScalarEnv.set] using getBase)
      (by simpa [e3, e2, e1, ScalarEnv.set] using getSeed)
      (by simp [e3, e2, ScalarEnv.set]) (by simp [e3, ScalarEnv.set])
      (by simp [e3, e2, e1, ScalarEnv.set]) following result
      (fun after keptBase keptSeed keptTotal => continued after keptBase keptSeed (by simpa only [tally] using keptTotal))
    apply BlockReturns.while_loop tested
    simp only [yes, decide_true, scalarTruth, if_true]
    apply BlockReturns.augAssign (value := number (total + 1))
      (Evaluates.binary (Evaluates.lookup getTotal) Evaluates.constant (by simp [scalarBinary, number]))
    apply BlockReturns.augAssign (value := number (seed ^ (exponent + 1)))
      (Evaluates.binary
        (Evaluates.lookup (by simpa [e1, ScalarEnv.set] using getPower))
        (Evaluates.lookup (by simpa [e1, ScalarEnv.set] using getSeed))
        (by simp [scalarBinary, number, Nat.pow_succ]))
    have lowerStep : base ^ (exponent - 1) * base = base ^ exponent := by
      rw [← Nat.pow_succ]
      congr 1
      omega
    apply BlockReturns.augAssign (value := number (base ^ exponent))
      (Evaluates.binary
        (Evaluates.lookup (by simpa [e2, e1, ScalarEnv.set] using getLower))
        (Evaluates.lookup (by simpa [e2, e1, ScalarEnv.set] using getBase))
        (by simp only [scalarBinary, scalarNumber, number, ← Int.natCast_mul, lowerStep]))
    simpa only [loopContinuation, show loopControl innerBody = false from rfl, Bool.false_eq_true, if_false] using rest
  · have failed : seed ^ exponent < base ^ (exponent - 1) := by omega
    have zero := seed_window_zero base seed exponent (bound base - exponent) valid inside positive failed
    apply BlockReturns.while_loop tested
    simp only [yes, decide_false, scalarTruth, Bool.false_eq_true, if_false, List.nil_append]
    apply continued env getBase getSeed
    simpa only [zero, Nat.add_zero] using getTotal
termination_by bound base - exponent
decreasing_by omega

end solvers.«63».solve
