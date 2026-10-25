import ProjectEulerSolutions.Lib.EConvergents
import lang.py.FormalSW.HeapCalls

namespace ProjectEulerSolutions.EConvergentCalls
open FormalSW.PyAst HeapCalls EConvergents
set_option Elab.async false
variable {globals : Mutable.Globals}
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)

def zeroTest : Expr := .compare (.name "k") [(.eq, .constant (.int 0))]
def periodicTest : Expr := .compare (.binOp (.name "k") .mod (.constant (.int 3))) [(.eq, .constant (.int 2))]
def periodicValue : Expr := .binOp (.constant (.int 2)) .mult
  (.binOp (.binOp (.name "k") .floorDiv (.constant (.int 3))) .add (.constant (.int 1)))
def coefficientExpr : Expr := .ifExp zeroTest (.constant (.int 2))
  (.ifExp periodicTest periodicValue (.constant (.int 1)))
def coefficientStatements : List Stmt :=
  [.ifThen zeroTest [.assign [.name "coefficient"] (.constant (.int 2))]
    [.ifThen periodicTest [.assign [.name "coefficient"] periodicValue]
      [.assign [.name "coefficient"] (.constant (.int 1))]]]

theorem zero_test (program : Module) (k : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (get : env "k" = some (number k)) :
    Evaluates program env zeroTest heap (flag (decide (k = 0))) heap globals := by
  exact Evaluates.compare (Evaluates.lookup get) Evaluates.constant
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag])

theorem periodic_test (program : Module) (k : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (get : env "k" = some (number k)) :
    Evaluates program env periodicTest heap (flag (decide (k % 3 = 2))) heap globals := by
  apply Evaluates.compare (av := number (k % 3)) (bv := number 2) (middle := heap) (afterRight := heap)
  · exact Evaluates.binary (Evaluates.lookup get) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number])
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]
    norm_cast

theorem periodic_value (program : Module) (k : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (get : env "k" = some (number k)) :
    Evaluates program env periodicValue heap (number (2 * (k / 3 + 1))) heap globals := by
  apply Evaluates.binary (av := number 2) (bv := number (k / 3 + 1))
    (middle := heap) (afterRight := heap) Evaluates.constant
  · apply Evaluates.binary (av := number (k / 3)) (bv := number 1)
      (middle := heap) (afterRight := heap)
    · exact Evaluates.binary (Evaluates.lookup get) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number])
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary, number]
  · simp [Mutable.binary, scalarBinary, number]

theorem coefficient_expression (program : Module) (k : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (get : env "k" = some (number k)) :
    Evaluates program env coefficientExpr heap (number (coefficient k)) heap globals := by
  by_cases zero : k = 0
  · simp only [coefficient, ProjectEulerStatements.P65.coeffE, if_pos zero]
    apply Evaluates.conditional (flag := true) (by simpa only [zero, decide_true] using zero_test program k heap env get) rfl
    exact Evaluates.constant
  · simp only [coefficient, ProjectEulerStatements.P65.coeffE, if_neg zero]
    apply Evaluates.conditional (flag := false) (by simpa only [zero, decide_false] using zero_test program k heap env get) rfl
    by_cases periodic : k % 3 = 2
    · simp only [if_pos periodic]
      apply Evaluates.conditional (flag := true)
        (by simpa only [periodic, decide_true] using periodic_test program k heap env get) rfl
      exact periodic_value program k heap env get
    · simp only [if_neg periodic]
      apply Evaluates.conditional (flag := false)
        (by simpa only [periodic, decide_false] using periodic_test program k heap env get) rfl
      exact Evaluates.constant

theorem coefficient_assignment (program : Module) (k : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (get : env "k" = some (number k)) :
    BlockContinues program env coefficientStatements heap
      (setLocal env "coefficient" (number (coefficient k))) heap globals := by
  by_cases zero : k = 0
  · simp only [coefficient, ProjectEulerStatements.P65.coeffE, if_pos zero]
    apply BlockContinues.branch (flag := true) (by simpa only [zero, decide_true] using zero_test program k heap env get) rfl
    exact BlockContinues.assign Evaluates.constant
  · simp only [coefficient, ProjectEulerStatements.P65.coeffE, if_neg zero]
    apply BlockContinues.branch (flag := false) (by simpa only [zero, decide_false] using zero_test program k heap env get) rfl
    by_cases periodic : k % 3 = 2
    · simp only [if_pos periodic]
      apply BlockContinues.branch (flag := true)
        (by simpa only [periodic, decide_true] using periodic_test program k heap env get) rfl
      exact BlockContinues.assign (periodic_value program k heap env get)
    · simp only [if_neg periodic]
      apply BlockContinues.branch (flag := false)
        (by simpa only [periodic, decide_false] using periodic_test program k heap env get) rfl
      exact BlockContinues.assign Evaluates.constant

end ProjectEulerSolutions.EConvergentCalls
