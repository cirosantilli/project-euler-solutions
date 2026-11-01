import «.formalsw».solvers.«80.py.ast»
import ProjectEulerSolutions.Lib.DecimalDigitSumCalls
import ProjectEulerSolutions.Lib.SquareRootDigitCalls

namespace solvers.«80»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.SquareRootDigitCalls
open ProjectEulerStatements
set_option Elab.async false
private abbrev program := Solvers.P80Ast.module
variable {globals : Mutable.Globals}

theorem digit_sum.result (n : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "digit_sum" [number n] heap (number (Nat.digits 10 n).sum) heap globals := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  exact ProjectEulerSolutions.DecimalDigitSumCalls.string_sum program _ "n" n heap globals
    (by cases globals <;> rfl)

theorem sqrt_digits_sum.result (n digits : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "sqrt_digits_sum" [number n, number digits] heap
      (number (P80.sqrtDigitsSum n digits)) heap globals := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := number (Nat.sqrt n))
    (isqrt_call n (.name "n") (Evaluates.lookup (by cases globals <;> rfl)) rfl rfl)
  have tested : ∀ env : HeapCalls.Env, env "root" = some (number (Nat.sqrt n)) →
      env "n" = some (number n) →
      Evaluates program env (.compare (.binOp (.name "root") .mult (.name "root")) [(.eq, .name "n")])
        heap (.scalar (.bool (decide (Nat.sqrt n * Nat.sqrt n = n)))) heap globals := by
    intro env getRoot getN
    apply Evaluates.compare (av := number (Nat.sqrt n * Nat.sqrt n)) (bv := number n) (middle := heap) (afterRight := heap)
    · exact Evaluates.binary (av := number (Nat.sqrt n)) (bv := number (Nat.sqrt n)) (middle := heap) (afterRight := heap) (finalHeap := heap) (Evaluates.lookup getRoot) (Evaluates.lookup getRoot)
        (by simp [Mutable.binary, scalarBinary, number])
    · exact Evaluates.lookup getN
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      norm_cast
  by_cases square : Nat.sqrt n * Nat.sqrt n = n
  · apply BlockReturns.branch (flag := true)
      (tested _ (by simp [setLocal]) (by cases globals <;> rfl))
      (by simp [Mutable.truth, scalarTruth, square])
    simpa [P80.sqrtDigitsSum, P80.isSquare, square] using
      (BlockReturns.return_value (module := program) (env := _) (rest := _) (globals := globals)
        (heap := heap) (Evaluates.constant (value := .int 0)))
  · apply BlockReturns.branch (flag := false)
      (tested _ (by simp [setLocal]) (by cases globals <;> rfl))
      (by simp [Mutable.truth, scalarTruth, square])
    apply BlockReturns.assign (value := number (P80.integerWidth n)) (width_evaluate n (by simp [setLocal]))
    apply (prefix_result n digits (by cases globals <;> rfl) (by cases globals <;> rfl)
      (by simp [setLocal]) (by simp [setLocal]) (by rfl) (by rfl)).returns
    apply BlockReturns.return_value
    simpa [P80.sqrtDigitsSum, P80.isSquare, square, P80.digitSum] using
      (Evaluates.call (.cons (Evaluates.lookup (by simp [setLocal])) .nil)
        (digit_sum.result (P80.sqrtPrefix n digits) heap globals))

end solvers.«80»
