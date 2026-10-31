import «.formalsw».solvers.«90.py.ast»
import ProjectEulerSolutions.Lib.DigitDice
import lang.py.FormalSW.HeapCalls

/-! Source execution of the optimized die-label rotation helper. -/
namespace solvers.«90»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.DigitDice
set_option Elab.async false

abbrev program := Solvers.P90Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

private theorem bit_evaluates (mask index : Nat) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (found : env "mask" = some (number mask)) :
    Evaluates program env (.binOp (.binOp (.name "mask") .rightShift
      (.constant (.int index))) .bitAnd (.constant (.int 1))) heap
      (number (mask.testBit index).toNat) heap globals := by
  rw [Nat.toNat_testBit]
  apply Evaluates.binary (av := number (mask / 2 ^ index)) (bv := number 1)
  · apply Evaluates.binary (Evaluates.lookup found) Evaluates.constant
    simp only [Mutable.binary, scalarShiftRight, scalarNumber,
      show ¬ (index : Int) < 0 by omega, if_false, Int.toNat_natCast, integerShiftRight_nat]
    rfl
  · exact Evaluates.constant
  · change (some (Except.ok (Mutable.Value.scalar (.int (integerBitAnd (mask / 2 ^ index : Nat) 1))), heap) :
      Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
    rw [integerBitAnd_nat_one]

private theorem bits_or (a b : Bool) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals)
    (first : env "has6" = some (number a.toNat))
    (second : env "has9" = some (number b.toNat)) :
    Evaluates program env (.boolOp .or [.name "has6", .name "has9"])
      heap (number (a || b).toNat) heap globals := by
  cases a with
  | false =>
      apply Evaluates.boolean_next (Evaluates.lookup first) (flag := false)
      · rfl
      · rfl
      · exact Evaluates.boolean_last (flag := b) (Evaluates.lookup second)
          (by cases b <;> rfl)
  | true =>
      exact Evaluates.boolean_stop (Evaluates.lookup first) rfl rfl

/-- The helper covers every nonnegative mask and preserves the heap and globals. -/
theorem expand_6_9.result (mask : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "expand_6_9" [number mask] heap
      (number (rotateMask mask)) heap globals := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (bit_evaluates mask 6 _ heap globals (by cases globals <;> rfl))
  apply BlockReturns.assign (bit_evaluates mask 9 _ heap globals (by cases globals <;> rfl))
  apply BlockReturns.branch (bits_or (mask.testBit 6) (mask.testBit 9) _ heap globals rfl rfl)
    (flag := mask.testBit 6 || mask.testBit 9)
  · cases mask.testBit 6 <;> cases mask.testBit 9 <;> rfl
  · cases active : (mask.testBit 6 || mask.testBit 9) with
    | false =>
        simp only [Bool.false_eq_true, ↓reduceIte, List.nil_append, rotateMask, active]
        exact BlockReturns.return_value (Evaluates.lookup (by cases globals <;> rfl))
    | true =>
        simp only [↓reduceIte, List.cons_append, List.nil_append, rotateMask, active]
        apply BlockReturns.augAssign (left := number mask) (right := number 576)
          (value := number (mask ||| 576)) (afterOperation := heap)
        · cases globals <;> rfl
        · apply Evaluates.binary (av := number 64) (bv := number 512)
          · exact Evaluates.binary Evaluates.constant Evaluates.constant rfl
          · exact Evaluates.binary Evaluates.constant Evaluates.constant rfl
          · rfl
        · simp [inplace, Mutable.binary, scalarBitOr, integerBitOr, number]
        · exact BlockReturns.return_value (Evaluates.lookup rfl)

end solvers.«90»
