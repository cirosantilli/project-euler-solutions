import lang.py.FormalSW.HeapCalls
import ProjectEulerStatements.Lib.SquareRootDigits

namespace ProjectEulerSolutions.SquareRootDigitCalls
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements
set_option Elab.async false
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
variable {program : Module} {env : HeapCalls.Env} {heap : Mutable.Heap} {globals : Mutable.Globals}

theorem isqrt_call (n : Nat) (expression : Expr)
    (evaluated : Evaluates program env expression heap (number n) heap globals)
    (absent : findFunctionIn program.body "isqrt" = none)
    (imported : libraryFunction program "isqrt" = some Math.isqrtCall) :
    Evaluates program env (.call (.name "isqrt") [expression]) heap (number (Nat.sqrt n)) heap globals := by
  apply Evaluates.library_call (.cons evaluated .nil) absent
  have root : Math.isqrt n = Nat.sqrt n := Nat.eq_sqrt.mpr (Math.isqrt_spec n)
  simp [library, imported, scalarResult, Math.isqrtCall, root, show ¬(n : Int) < 0 by omega]

theorem width_evaluate (n : Nat) (getRoot : env "root" = some (number (Nat.sqrt n))) :
    Evaluates program env (.builtinCall .len [.builtinCall .str [.name "root"]])
      heap (number (P80.integerWidth n)) heap globals := by
  apply Evaluates.builtin (values := [.scalar (.string (toString (Nat.sqrt n : Int)))])
    (.cons (Evaluates.builtin (.cons (Evaluates.lookup getRoot) .nil) rfl) .nil)
  change (some (.ok (number (toString (Nat.sqrt n : Int)).length), heap) : Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
  simp [P80.integerWidth, toString, Int.repr, Nat.repr]

def prefixCode : List Stmt :=
  [.ifThen (.compare (.name "digits") [(.lt, .name "width")])
    [.assign [.name "prefix"] (.binOp (.name "root") .floorDiv
      (.binOp (.constant (.int 10)) .pow (.binOp (.name "width") .sub (.name "digits"))))]
    [.assign [.name "prefix"] (.call (.name "isqrt")
      [.binOp (.name "n") .mult (.binOp (.constant (.int 10)) .pow
        (.binOp (.constant (.int 2)) .mult (.binOp (.name "digits") .sub (.name "width"))))])]]

theorem prefix_result (n digits : Nat)
    (getN : env "n" = some (number n))
    (getDigits : env "digits" = some (number digits))
    (getRoot : env "root" = some (number (Nat.sqrt n)))
    (getWidth : env "width" = some (number (P80.integerWidth n)))
    (absent : findFunctionIn program.body "isqrt" = none)
    (imported : libraryFunction program "isqrt" = some Math.isqrtCall) :
    BlockContinues program env prefixCode heap
      (setLocal env "prefix" (number (P80.sqrtPrefix n digits))) heap globals := by
  have tested : Evaluates program env (.compare (.name "digits") [(.lt, .name "width")])
      heap (.scalar (.bool (decide (digits < P80.integerWidth n)))) heap globals := by
    apply Evaluates.compare (Evaluates.lookup getDigits) (Evaluates.lookup getWidth)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  unfold prefixCode
  by_cases short : digits < P80.integerWidth n
  · simp only [short, decide_true] at tested
    apply BlockContinues.branch tested rfl
    apply BlockContinues.assign
    rw [P80.sqrtPrefix, if_pos short]
    apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (av := number (Nat.sqrt n))
      (bv := number (10 ^ (P80.integerWidth n - digits))) (Evaluates.lookup getRoot)
    · apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (av := number 10) (bv := number (P80.integerWidth n - digits)) Evaluates.constant
      · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (Evaluates.lookup getWidth) (Evaluates.lookup getDigits)
          (by simp [Mutable.binary, scalarBinary, number, Int.ofNat_sub (Nat.le_of_lt short)])
      · simp [Mutable.binary, scalarBinary, number, show ¬((P80.integerWidth n - digits : Nat) : Int) < 0 by omega]
    · simp [Mutable.binary, scalarBinary, number, Int.natCast_ediv]
  · simp only [short, decide_false] at tested
    apply BlockContinues.branch tested rfl
    apply BlockContinues.assign
    rw [P80.sqrtPrefix, if_neg short]
    apply isqrt_call _ _ _ absent imported
    apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (av := number n) (bv := number (10 ^ (2 * (digits - P80.integerWidth n))))
      (Evaluates.lookup getN)
    · apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (av := number 10) (bv := number (2 * (digits - P80.integerWidth n))) Evaluates.constant
      · apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (av := number 2) (bv := number (digits - P80.integerWidth n)) Evaluates.constant
        · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (Evaluates.lookup getDigits) (Evaluates.lookup getWidth)
            (by simp [Mutable.binary, scalarBinary, number, Int.ofNat_sub (by omega : P80.integerWidth n ≤ digits)])
        · simp [Mutable.binary, scalarBinary, number]
      · have nonnegative : ¬(2 * ((digits - P80.integerWidth n : Nat) : Int) < 0) := by omega
        simp [Mutable.binary, scalarBinary, number, nonnegative, Int.toNat_mul]
    · simp [Mutable.binary, scalarBinary, number]

end ProjectEulerSolutions.SquareRootDigitCalls
