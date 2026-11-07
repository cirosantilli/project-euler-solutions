import «.formalsw».solvers.«96.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.SudokuCandidates

namespace solvers.«96»
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev program := Solvers.P96Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)

/-- The source helper returns the bit length for every integer, preserving caller memory. -/
theorem bit_to_digit.result (bit : Int) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "bit_to_digit" [number bit] heap
      (number (FormalSW.Py.Builtin.Int.bitLength bit)) heap globals := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply Evaluates.binary (middle := heap) (afterRight := heap)
    (av := number ((FormalSW.Py.Builtin.Int.bitLength bit : Int) - 1))
    (bv := number 1)
  · apply Evaluates.binary (middle := heap) (afterRight := heap)
      (av := number (FormalSW.Py.Builtin.Int.bitLength bit)) (bv := number 1)
    · apply Evaluates.builtin (.cons (Evaluates.lookup (by cases globals <;> rfl)) .nil)
      rfl
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary, number]
  · exact Evaluates.constant
  · simp [Mutable.binary, scalarBinary, number]

theorem bit_to_digit.single (index : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "bit_to_digit" [number (2 ^ index)] heap
      (number (index + 1)) heap globals := by
  have length : FormalSW.Py.Builtin.Int.bitLength (2 ^ index) = index + 1 := by
    simp [FormalSW.Py.Builtin.Int.bitLength, Int.natAbs_pow, Nat.log2_two_pow]
  simpa [length] using result (2 ^ index) heap globals

theorem bit_to_digit.«Returns» (bit : Int) :
    (HeapCalls.call program "bit_to_digit" [number bit] []).Returns :=
  (result bit []).returns

theorem bit_to_digit.«Supported» (bit : Int) :
    (HeapCalls.call program "bit_to_digit" [number bit] []).Supported :=
  («Returns» bit).supported

theorem top_left_number.result (a b c : Nat) (suffix : List Mutable.Value)
    (heap : Mutable.Heap) (address : Nat) (globals : Mutable.Globals := [])
    (found : heap[address]? = some (.list
      (number (2 ^ a) :: number (2 ^ b) :: number (2 ^ c) :: suffix))) :
    ReturnsValue program "top_left_number" [.ref address] heap
      (number (100 * (a + 1) + 10 * (b + 1) + (c + 1))) heap globals := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := number (a + 1)) (middle := heap)
  · apply Evaluates.call _ (bit_to_digit.single a heap globals)
    apply Evaluates.Arguments.cons (middle := heap) _ .nil
    apply Evaluates.subscript (by intros; simp)
      (Evaluates.lookup (by cases globals <;> rfl)) Evaluates.constant
    simp [Mutable.getItem, Mutable.index, found, Except.map]
  apply BlockReturns.assign (value := number (b + 1)) (middle := heap)
  · apply Evaluates.call _ (bit_to_digit.single b heap globals)
    apply Evaluates.Arguments.cons (middle := heap) _ .nil
    apply Evaluates.subscript (by intros; simp)
      (Evaluates.lookup (by cases globals <;> rfl)) Evaluates.constant
    simp [Mutable.getItem, Mutable.index, found, Except.map]
    split
    · omega
    · rfl
  apply BlockReturns.assign (value := number (c + 1)) (middle := heap)
  · apply Evaluates.call _ (bit_to_digit.single c heap globals)
    apply Evaluates.Arguments.cons (middle := heap) _ .nil
    apply Evaluates.subscript (by intros; simp)
      (Evaluates.lookup (by cases globals <;> rfl)) Evaluates.constant
    simp [Mutable.getItem, Mutable.index, found, Except.map]
    split
    · omega
    · rfl
  apply BlockReturns.return_value
  apply Evaluates.binary (middle := heap) (afterRight := heap)
    (av := number (100 * (a + 1) + 10 * (b + 1))) (bv := number (c + 1))
  · apply Evaluates.binary (middle := heap) (afterRight := heap)
      (av := number (100 * (a + 1))) (bv := number (10 * (b + 1)))
    · apply Evaluates.binary (middle := heap) (afterRight := heap)
        Evaluates.constant (Evaluates.lookup (by cases globals <;> rfl))
      simp [Mutable.binary, scalarBinary, number]
    · apply Evaluates.binary (middle := heap) (afterRight := heap)
        Evaluates.constant (Evaluates.lookup (by cases globals <;> rfl))
      simp [Mutable.binary, scalarBinary, number]
    · simp [Mutable.binary, scalarBinary, number]
  · exact Evaluates.lookup (by cases globals <;> rfl)
  · simp [Mutable.binary, scalarBinary, number]

end solvers.«96»
