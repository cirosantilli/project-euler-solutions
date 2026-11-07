import ProjectEulerSolutions.Lib.SudokuMasks
import ProjectEulerSolutions.Lib.LowBit

/-! The recursive search visits candidate digits by extracting and clearing the least bit. -/
namespace ProjectEulerSolutions.Sudoku.Masks
set_option Elab.async false
noncomputable section

theorem least_bit (digits : Finset Digit) (nonempty : digits ≠ ∅) :
    ∃ digit ∈ digits, FormalSW.Py.Builtin.Int.bitAnd (encode digits) (-(encode digits : Int)) = (2 : Int) ^ digit.val ∧
      ∀ other ∈ digits, digit ≤ other := by
  have nonzero : encode digits ≠ 0 := fun zero => nonempty ((Masks.zero_iff digits).mp zero)
  obtain ⟨index, extracted, present, least⟩ := LowBit.power (encode digits) nonzero
  have inside : index < 9 := by
    by_contra outside
    have zero := Nat.testBit_lt_two_pow ((bounded digits).trans_le
      (show 2 ^ 9 ≤ 2 ^ index from Nat.pow_le_pow_right (by decide) (by omega)))
    simp [present] at zero
  let digit : Digit := ⟨index, inside⟩
  have member : digit ∈ digits := by
    change (encode digits).testBit digit.val = true at present
    simpa only [Masks.testBit, decide_eq_true_eq] using present
  refine ⟨digit, member, ?_, ?_⟩
  · rw [LowBit.integer, extracted]
    simp [digit]
  · intro other selected
    by_contra below
    have falseBit := least other.val (by change ¬ index ≤ other.val at below; omega)
    have absent : other ∉ digits := by simpa only [Masks.testBit, decide_eq_false_iff_not] using falseBit
    exact absent selected

theorem subtract_bit (digits : Finset Digit) (digit : Digit) (member : digit ∈ digits) :
    encode digits - 2 ^ digit.val = encode (digits.erase digit) := by
  rw [LowBit.subtract (encode digits) digit.val (by simp [testBit, member])]
  have cleared := erase digits digit
  have cast : (2 : Int) ^ digit.val = ((2 ^ digit.val : Nat) : Int) := by simp
  rw [cast] at cleared
  change ((encode digits ^^^ (encode digits &&& 2 ^ digit.val) : Nat) : Int) = (encode (digits.erase digit) : Int) at cleared
  rw [LowBit.xor_and] at cleared
  exact_mod_cast cleared

theorem subtract_bit_int (digits : Finset Digit) (digit : Digit) (member : digit ∈ digits) :
    (encode digits : Int) - (2 : Int) ^ digit.val = encode (digits.erase digit) := by
  have bound : 2 ^ digit.val ≤ encode digits := by
    by_contra outside
    have zero := Nat.testBit_lt_two_pow (by omega : encode digits < 2 ^ digit.val)
    simp [testBit, member] at zero
  have cast : (2 : Int) ^ digit.val = ((2 ^ digit.val : Nat) : Int) := by simp
  rw [cast, ← Int.natCast_sub bound, subtract_bit digits digit member]

end
end ProjectEulerSolutions.Sudoku.Masks
