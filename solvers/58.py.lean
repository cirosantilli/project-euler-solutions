import «.formalsw».solvers.«58.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.MillerRabin

namespace solvers.«58».miller_check
open FormalSW.PyAst
open ProjectEulerSolutions.MillerRabin

private abbrev program := Solvers.P58Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private abbrev lastExpression : Expr := .binOp (.name "n") .sub (.constant (.int 1))
private abbrev lastTest : Expr := .compare (.name "x") [(.eq, lastExpression)]
private abbrev initialTest : Expr := .boolOp .or
  [.compare (.name "x") [(.eq, .constant (.int 1))], lastTest]
private abbrev squareExpression : Expr :=
  .binOp (.binOp (.name "x") .mult (.name "x")) .mod (.name "n")
private abbrev squareBody : List Stmt :=
  [.assign [.name "x"] squareExpression,
   .ifThen lastTest [.return (.constant (.bool true))] []]

private theorem last_test (n : Nat) (x : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) (getX : env "x" = some (number x))
    (globals : Mutable.Globals) :
    HeapCalls.Evaluates program env lastTest heap (flag (decide (x = (n : Int) - 1))) heap globals := by
  apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup getX)
    (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getN) HeapCalls.Evaluates.constant rfl)
  simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]

private theorem initial_test (n : Nat) (x : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) (getX : env "x" = some (number x))
    (globals : Mutable.Globals) :
    HeapCalls.Evaluates program env initialTest heap
      (flag (decide (x = 1 ∨ x = (n : Int) - 1))) heap globals := by
  have first : HeapCalls.Evaluates program env
      (.compare (.name "x") [(.eq, .constant (.int 1))]) heap
      (flag (decide (x = 1))) heap globals := by
    apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup getX) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]
  by_cases one : x = 1
  · simp only [one, true_or, decide_true]
    apply HeapCalls.Evaluates.boolean_stop (flag := true)
      (by simpa [one] using first) rfl rfl
  · apply HeapCalls.Evaluates.boolean_next (flag := false)
      (by simpa [one] using first) rfl rfl
    apply HeapCalls.Evaluates.boolean_last
    · simpa [one] using last_test n x env heap getN getX globals
    · cases decide (x = (n : Int) - 1) <;> rfl

/-- The loop proof includes Python's clipping of a nonpositive range stop to zero iterations. -/
private theorem square_loop (remaining : Nat) (n : Nat) (positive : 2 ≤ n)
    (next stop : Int) (equation : (stop - next).toNat = remaining)
    (x : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) (getX : env "x" = some (number x))
    (globals : Mutable.Globals) :
    HeapCalls.BlockReturns program env
      [.rangeStepLoop "_" next stop 1 squareBody [], .return (.constant (.bool false))]
      heap (flag (squareSearch n remaining x)) heap globals := by
  induction remaining generalizing next x env with
  | zero =>
      apply HeapCalls.BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
  | succ remaining ih =>
      let bound := HeapCalls.setLocal env "_" (number next)
      let squared := x * x % (n : Int)
      let updated := HeapCalls.setLocal bound "x" (number squared)
      have boundN : bound "n" = some (number n) := by simpa [bound, HeapCalls.setLocal] using getN
      have boundX : bound "x" = some (number x) := by simpa [bound, HeapCalls.setLocal] using getX
      have updatedN : updated "n" = some (number n) := by simpa [updated, HeapCalls.setLocal] using boundN
      have updatedX : updated "x" = some (number squared) := by simp [updated, HeapCalls.setLocal]
      have evaluated : HeapCalls.Evaluates program bound squareExpression heap (number squared) heap globals := by
        apply HeapCalls.Evaluates.binary
          (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup boundX) (HeapCalls.Evaluates.lookup boundX) rfl)
          (HeapCalls.Evaluates.lookup boundN)
        simp [squared, Mutable.binary, number, scalarBinary,
          show n ≠ 0 by omega, show 0 < n by omega]
      apply HeapCalls.BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply HeapCalls.BlockReturns.assign evaluated
      by_cases hit : squared = (n : Int) - 1
      · have score : squareSearch n (remaining + 1) x = true := by
          change (if squared = (n : Int) - 1 then true else squareSearch n remaining squared) = true
          rw [if_pos hit]
        rw [score]
        apply HeapCalls.BlockReturns.branch (flag := true)
          (by simpa only [show decide (squared = (n : Int) - 1) = true by simp only [hit, decide_true]]
            using last_test n squared updated heap updatedN updatedX globals) rfl
        exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
      · have score : squareSearch n (remaining + 1) x = squareSearch n remaining squared := by
          change (if squared = (n : Int) - 1 then true else squareSearch n remaining squared) = _
          rw [if_neg hit]
        rw [score]
        apply HeapCalls.BlockReturns.branch (flag := false)
          (by simpa only [show decide (squared = (n : Int) - 1) = false by simp only [hit, decide_false]]
            using last_test n squared updated heap updatedN updatedX globals) rfl
        have rest := ih (next + 1) (by omega) squared updated updatedN updatedX
        simpa [squareBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest

private abbrev params := ((findFunctionIn program.body "__formalsw_local_is_prime_check").getD ([], [])).1
private abbrev sourceBody := ((findFunctionIn program.body "__formalsw_local_is_prime_check").getD ([], [])).2

/-- The actual lifted local helper follows all its early returns and preserves every caller heap object. -/
theorem result (a : Int) (n d s : Nat) (positive : 2 ≤ n)
    (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    HeapCalls.ReturnsValue program "__formalsw_local_is_prime_check"
      [number a, number n, number d, number s] heap (flag (millerCheck a n d s)) heap globals := by
  let env := HeapCalls.functionEnv params sourceBody [number a, number n, number d, number s] globals
  have getA : env "a" = some (number a) := by cases globals <;> rfl
  have getN : env "n" = some (number n) := by cases globals <;> rfl
  have getD : env "d" = some (number d) := by cases globals <;> rfl
  have getS : env "s" = some (number s) := by cases globals <;> rfl
  have remainder : HeapCalls.Evaluates program env
      (.compare (.binOp (.name "a") .mod (.name "n")) [(.eq, .constant (.int 0))])
      heap (flag (decide (a % (n : Int) = 0))) heap globals := by
    have reduced : HeapCalls.Evaluates program env
        (.binOp (.name "a") .mod (.name "n")) heap (number (a % (n : Int))) heap globals := by
      apply HeapCalls.Evaluates.binary (middle := heap) (finalHeap := heap)
        (HeapCalls.Evaluates.lookup getA) (HeapCalls.Evaluates.lookup getN)
      simp [Mutable.binary, number, scalarBinary, show n ≠ 0 by omega, show 0 < n by omega]
    apply HeapCalls.Evaluates.compare (middle := heap) (afterRight := heap)
      (av := number (a % (n : Int))) (bv := number 0) reduced HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]
  apply HeapCalls.BlockReturns.function rfl rfl
  by_cases divisible : a % (n : Int) = 0
  · rw [millerCheck, if_pos divisible]
    apply HeapCalls.BlockReturns.branch (flag := true) (by simpa [divisible] using remainder) rfl
    exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
  · rw [millerCheck, if_neg divisible]
    apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [divisible] using remainder) rfl
    let x := FormalSW.Py.Builtin.Int.powModBinary a d n
    have power : HeapCalls.Evaluates program env
        (.builtinCall .pow [.name "a", .name "d", .name "n"]) heap (number x) heap globals := by
      apply HeapCalls.Evaluates.builtin
        (.cons (HeapCalls.Evaluates.lookup getA)
          (.cons (HeapCalls.Evaluates.lookup getD) (.cons (HeapCalls.Evaluates.lookup getN) .nil)))
      simp [HeapCalls.primitive, HeapCalls.scalarResult, Calls.modularPower, scalarNumber,
        FormalSW.Py.Builtin.Int.powMod_nonnegative a d n (by omega),
        ← FormalSW.Py.Builtin.Int.powModBinary_correct a d n, number, x]
    apply HeapCalls.BlockReturns.assign power
    let updated := HeapCalls.setLocal env "x" (number x)
    have updatedN : updated "n" = some (number n) := by simpa [updated, HeapCalls.setLocal] using getN
    have updatedX : updated "x" = some (number x) := by simp [updated, HeapCalls.setLocal]
    have updatedS : updated "s" = some (number s) := by simpa [updated, HeapCalls.setLocal] using getS
    change HeapCalls.BlockReturns program updated
      [.ifThen initialTest [.return (.constant (.bool true))] [],
       .forIn (.name "_") (.call (.name "range") [.binOp (.name "s") .sub (.constant (.int 1))]) squareBody [],
       .return (.constant (.bool false))] heap
      (flag (if x = 1 ∨ x = (n : Int) - 1 then true else squareSearch n (s - 1) x)) heap globals
    have tested := initial_test n x updated heap updatedN updatedX globals
    by_cases accepted : x = 1 ∨ x = (n : Int) - 1
    · rw [if_pos accepted]
      apply HeapCalls.BlockReturns.branch (flag := true) (by simpa [accepted] using tested) rfl
      exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
    · rw [if_neg accepted]
      apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [accepted] using tested) rfl
      apply HeapCalls.BlockReturns.for_range
        (.cons (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup updatedS) HeapCalls.Evaluates.constant rfl) .nil) rfl
      apply square_loop (s - 1) n positive 0 ((s : Int) - 1) (by omega) x updated heap updatedN updatedX globals

end solvers.«58».miller_check
