import «.formalsw».solvers.«99.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.ScaledPowerComparison

namespace solvers.«99»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PowerIntervals
set_option Elab.async false

abbrev program := Solvers.P99Ast.module
abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
def boundsObject (bounds : Bounds) : Mutable.Object :=
  .tuple [number bounds.lower, number bounds.upper, number bounds.shift]

namespace round_power_bounds

theorem result (bounds : Bounds) (precision : Nat) (heap : Mutable.Heap)
    (globals : Mutable.Globals := []) :
    ReturnsValue program "round_power_bounds"
      [number bounds.lower, number bounds.upper, number bounds.shift, number precision] heap
      (.ref heap.length) (heap ++ [boundsObject (trim bounds precision)]) globals := by
  let drop := bounds.upper.size - precision
  have nonnegative : ¬(drop : Int) < 0 := by omega
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := number drop) (middle := heap)
  · apply Evaluates.builtin (middle := heap)
    · apply Evaluates.Arguments.cons (middle := heap) Evaluates.constant
      apply Evaluates.Arguments.cons (middle := heap) (value := number ((bounds.upper.size : Int) - precision)) _ .nil
      apply Evaluates.binary (middle := heap) (afterRight := heap)
        (av := number bounds.upper.size) (bv := number precision)
      · have length : FormalSW.Py.Builtin.Int.bitLength bounds.upper = bounds.upper.size := bitLength_eq_size _
        rw [← length]
        exact Evaluates.builtin (.cons (Evaluates.lookup (by cases globals <;> rfl)) .nil) rfl
      · exact Evaluates.lookup (by cases globals <;> rfl)
      · simp [Mutable.binary, scalarBinary, number]
    · change (scalarResult (Calls.positionalMaximum [.int 0, .int ((bounds.upper.size : Int) - precision)])) heap = _
      simp only [Calls.positionalMaximum, Calls.maximumFrom, scalarCompare, evalCmpOp]
      split <;> simp_all [scalarResult, number, drop]
      omega
  apply BlockReturns.return_value
  change Evaluates program _
    (.tuple [.binOp (.name "lower") .rightShift (.name "drop"),
      .binOp (.binOp (.binOp (.name "upper") .add (.binOp (.constant (.int 1)) .leftShift (.name "drop")))
        .sub (.constant (.int 1))) .rightShift (.name "drop"),
      .binOp (.name "shift") .add (.name "drop")]) heap
    (.ref heap.length) (heap ++ [boundsObject (trim bounds precision)]) globals
  apply Evaluates.tuple
    (.cons (middle := heap) ?_ (.cons (middle := heap) ?_ (.cons (middle := heap) ?_ .nil))) (by simp)
  · apply Evaluates.binary (middle := heap) (afterRight := heap)
      (av := number bounds.lower) (bv := number drop)
      (Evaluates.lookup (by cases globals <;> rfl)) (Evaluates.lookup rfl)
    simp [Mutable.binary, scalarShiftRight, integerShiftRight, number, trim, rounded, nonnegative]
    rfl
  · apply Evaluates.binary (middle := heap) (afterRight := heap)
      (av := number (bounds.upper + 2 ^ drop - 1)) (bv := number drop)
    · apply Evaluates.binary (middle := heap) (afterRight := heap)
        (av := number (bounds.upper + 2 ^ drop)) (bv := number 1)
      · apply Evaluates.binary (middle := heap) (afterRight := heap)
          (av := number bounds.upper) (bv := number (2 ^ drop))
        · exact Evaluates.lookup (by cases globals <;> rfl)
        · apply Evaluates.binary (middle := heap) (afterRight := heap)
            (av := number 1) (bv := number drop) Evaluates.constant (Evaluates.lookup rfl)
          simp [Mutable.binary, scalarShiftLeft, integerShiftLeft, number, nonnegative]
        · simp [Mutable.binary, scalarBinary, number]
      · exact Evaluates.constant
      · simp [Mutable.binary, scalarBinary, number]
    · exact Evaluates.lookup rfl
    · simp [Mutable.binary, scalarShiftRight, integerShiftRight, number, trim, rounded, nonnegative]
      rfl
  · apply Evaluates.binary (middle := heap) (afterRight := heap)
      (av := number bounds.shift) (bv := number drop)
      (Evaluates.lookup (by cases globals <;> rfl)) (Evaluates.lookup rfl)
    simp [Mutable.binary, scalarBinary, number, trim, rounded, drop]

end round_power_bounds
end solvers.«99»
