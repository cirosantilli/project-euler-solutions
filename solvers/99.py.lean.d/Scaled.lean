import solvers.«99.py.lean.d».Power

namespace solvers.«99»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PowerIntervals
set_option Elab.async false

def intComparison (left right : Int) : Int :=
  (if right < left then 1 else 0) - (if left < right then 1 else 0)

theorem comparison_expression (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (leftName rightName : String) (left right : Int)
    (getLeft : env leftName = some (number left)) (getRight : env rightName = some (number right)) :
    Evaluates program env
      (.binOp (.compare (.name leftName) [(.gt, .name rightName)]) .sub
        (.compare (.name leftName) [(.lt, .name rightName)])) heap
      (number (intComparison left right)) heap globals := by
  apply Evaluates.binary (middle := heap) (afterRight := heap)
    (av := .scalar (.bool (decide (right < left)))) (bv := .scalar (.bool (decide (left < right))))
  · apply Evaluates.compare (Evaluates.lookup getLeft) (Evaluates.lookup getRight)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  · apply Evaluates.compare (Evaluates.lookup getLeft) (Evaluates.lookup getRight)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  · by_cases less : left < right <;> by_cases greater : right < left <;>
      simp [Mutable.binary, scalarBinary, intComparison, number, less, greater]

theorem comparison_expression_nat (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (leftName rightName : String) (left right : Nat)
    (getLeft : env leftName = some (number left)) (getRight : env rightName = some (number right)) :
    Evaluates program env
      (.binOp (.compare (.name leftName) [(.gt, .name rightName)]) .sub
        (.compare (.name leftName) [(.lt, .name rightName)])) heap
      (number (comparison left right)) heap globals := by
  simpa [intComparison, ProjectEulerSolutions.PowerIntervals.comparison] using
    comparison_expression env heap globals leftName rightName left right getLeft getRight

namespace compare_scaled

abbrev args := ((findFunctionIn program.body "compare_scaled").getD ([], [])).1
abbrev body := ((findFunctionIn program.body "compare_scaled").getD ([], [])).2

theorem zero_test (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (left right : Nat) (getLeft : env "left" = some (number left)) (getRight : env "right" = some (number right)) :
    Evaluates program env
      (.boolOp .or [.compare (.name "left") [(.eq, .constant (.int 0))],
        .compare (.name "right") [(.eq, .constant (.int 0))]]) heap
      (.scalar (.bool (decide (left = 0 ∨ right = 0)))) heap globals := by
  by_cases zero : left = 0
  · simp only [zero, true_or, decide_true]
    apply Evaluates.boolean_stop (flag := true) _ rfl rfl
    apply Evaluates.compare (Evaluates.lookup getLeft) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero]
  · have first : Evaluates program env (.compare (.name "left") [(.eq, .constant (.int 0))])
        heap (.scalar (.bool false)) heap globals := by
      apply Evaluates.compare (Evaluates.lookup getLeft) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero]
    apply Evaluates.boolean_next (flag := false) first rfl rfl
    apply Evaluates.boolean_last (flag := decide (left = 0 ∨ right = 0))
    · apply Evaluates.compare (Evaluates.lookup getRight) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero]
    · rfl

theorem result (left leftShift right rightShift : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "compare_scaled" [number left, number leftShift, number right, number rightShift]
      heap (number (scaledComparison left leftShift right rightShift)) heap globals := by
  let entry := functionEnv args body [number left, number leftShift, number right, number rightShift] globals
  have getLeft : entry "left" = some (number left) := by cases globals <;> rfl
  have getRight : entry "right" = some (number right) := by cases globals <;> rfl
  apply BlockReturns.function (args := args) (body := body) rfl rfl
  by_cases zero : left = 0 ∨ right = 0
  · apply BlockReturns.branch (flag := true)
      (by simpa [zero] using zero_test entry heap globals left right getLeft getRight) rfl
    apply BlockReturns.return_value
    simpa only [scaledComparison, if_pos zero] using
      comparison_expression_nat entry heap globals "left" "right" left right getLeft getRight
  · apply BlockReturns.branch (flag := false)
      (by simpa [zero] using zero_test entry heap globals left right getLeft getRight) rfl
    apply BlockReturns.assign (value := number (left.size + leftShift : Nat)) (middle := heap)
    · apply Evaluates.binary (middle := heap) (afterRight := heap)
        (av := number left.size) (bv := number leftShift)
      · rw [← bitLength_eq_size left]
        exact Evaluates.builtin (.cons (Evaluates.lookup getLeft) .nil) rfl
      · exact Evaluates.lookup (by cases globals <;> rfl)
      · simp [Mutable.binary, scalarBinary, number]
    apply BlockReturns.assign (value := number (right.size + rightShift : Nat)) (middle := heap)
    · apply Evaluates.binary (middle := heap) (afterRight := heap)
        (av := number right.size) (bv := number rightShift)
      · rw [← bitLength_eq_size right]
        exact Evaluates.builtin (.cons (Evaluates.lookup (by cases globals <;> rfl)) .nil) rfl
      · exact Evaluates.lookup (by cases globals <;> rfl)
      · simp [Mutable.binary, scalarBinary, number]
    by_cases different : left.size + leftShift ≠ right.size + rightShift
    · apply BlockReturns.branch (flag := true) (value := .scalar (.bool true)) (middle := heap)
      · apply Evaluates.compare (Evaluates.lookup rfl) (Evaluates.lookup rfl)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        omega
      · rfl
      · apply BlockReturns.return_value
        simpa only [scaledComparison, if_neg zero, if_pos different] using
          comparison_expression_nat _ heap globals "left_size" "right_size" (left.size + leftShift)
            (right.size + rightShift) rfl rfl
    · apply BlockReturns.branch (flag := false) (value := .scalar (.bool false)) (middle := heap)
      · apply Evaluates.compare (Evaluates.lookup rfl) (Evaluates.lookup rfl)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        omega
      · rfl
      · by_cases direction : rightShift ≤ leftShift
        · apply BlockReturns.branch (flag := true) (value := .scalar (.bool true)) (middle := heap)
          · apply Evaluates.compare (Evaluates.lookup (by cases globals <;> rfl))
              (Evaluates.lookup (by cases globals <;> rfl))
            simp [HeapCalls.compare, scalarCompare, evalCmpOp, direction]
          · rfl
          · apply BlockReturns.augAssign (left := number left) (right := number (leftShift - rightShift : Nat))
              (value := number (left * 2 ^ (leftShift - rightShift) : Nat)) (middle := heap) (afterOperation := heap)
              (by cases globals <;> rfl)
            · apply Evaluates.binary (middle := heap) (afterRight := heap)
                (av := number leftShift) (bv := number rightShift)
                (Evaluates.lookup (by cases globals <;> rfl)) (Evaluates.lookup (by cases globals <;> rfl))
              simp [Mutable.binary, scalarBinary, number]
              omega
            · have nonnegative : ¬((leftShift - rightShift : Nat) : Int) < 0 := by omega
              simp [inplace, Mutable.binary, scalarShiftLeft, integerShiftLeft, number, nonnegative]
            · apply BlockReturns.return_value
              simpa only [scaledComparison, if_neg zero, if_neg different, if_pos direction] using
                comparison_expression_nat _ heap globals "left" "right"
                  (left * 2 ^ (leftShift - rightShift)) right rfl (by cases globals <;> rfl)
        · apply BlockReturns.branch (flag := false) (value := .scalar (.bool false)) (middle := heap)
          · apply Evaluates.compare (Evaluates.lookup (by cases globals <;> rfl))
              (Evaluates.lookup (by cases globals <;> rfl))
            simp [HeapCalls.compare, scalarCompare, evalCmpOp, direction]
          · rfl
          · apply BlockReturns.augAssign (left := number right) (right := number (rightShift - leftShift : Nat))
              (value := number (right * 2 ^ (rightShift - leftShift) : Nat)) (middle := heap) (afterOperation := heap)
              (by cases globals <;> rfl)
            · apply Evaluates.binary (middle := heap) (afterRight := heap)
                (av := number rightShift) (bv := number leftShift)
                (Evaluates.lookup (by cases globals <;> rfl)) (Evaluates.lookup (by cases globals <;> rfl))
              simp [Mutable.binary, scalarBinary, number]
              omega
            · have nonnegative : ¬((rightShift - leftShift : Nat) : Int) < 0 := by omega
              simp [inplace, Mutable.binary, scalarShiftLeft, integerShiftLeft, number, nonnegative]
            · apply BlockReturns.return_value
              simpa only [scaledComparison, if_neg zero, if_neg different, if_neg direction] using
                comparison_expression_nat _ heap globals "left" "right" left
                  (right * 2 ^ (rightShift - leftShift)) (by cases globals <;> rfl) rfl

theorem correct (left leftShift right rightShift : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "compare_scaled" [number left, number leftShift, number right, number rightShift]
      heap (number (comparison (left * 2 ^ leftShift) (right * 2 ^ rightShift))) heap globals := by
  simpa only [scaledComparison_eq] using result left leftShift right rightShift heap globals

end compare_scaled
end solvers.«99»
