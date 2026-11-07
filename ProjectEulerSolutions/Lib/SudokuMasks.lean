import ProjectEulerSolutions.Lib.SudokuCandidates
import ProjectEulerSolutions.Lib.BitSubsets

/-! The nine-bit Python representation of Sudoku candidates. -/
namespace ProjectEulerSolutions.Sudoku.Masks

set_option Elab.async false
noncomputable section

def indices (digits : Finset Digit) : List Nat := digits.toList.map Fin.val

def encode (digits : Finset Digit) : Nat := DigitMask.mask (indices digits)

def decode (bits : Nat) : Finset Digit := Finset.univ.filter fun digit => bits.testBit digit.val

theorem mem_indices (digits : Finset Digit) (index : Nat) :
    index ∈ indices digits ↔ ∃ digit ∈ digits, digit.val = index := by
  simp [indices]

theorem indices_bound (digits : Finset Digit) (index : Nat) (member : index ∈ indices digits) :
    index < 9 := by
  obtain ⟨digit, _, rfl⟩ := (mem_indices digits index).mp member
  exact digit.isLt

theorem indices_nodup (digits : Finset Digit) : (indices digits).Nodup :=
  digits.nodup_toList.map Fin.val_injective

theorem bounded (digits : Finset Digit) : encode digits < 512 :=
  BitSubsets.mask_bound _ 9 (indices_bound digits)

theorem testBit (digits : Finset Digit) (digit : Digit) :
    (encode digits).testBit digit.val = decide (digit ∈ digits) := by
  simp [encode, DigitMask.testBit_mask, indices, Fin.val_inj]

theorem decode_encode (digits : Finset Digit) : decode (encode digits) = digits := by
  ext digit
  simp [decode, testBit]

theorem encode_decode (bits : Nat) (bound : bits < 512) : encode (decode bits) = bits := by
  apply Nat.eq_of_testBit_eq
  intro index
  by_cases inside : index < 9
  · simpa [decode] using testBit (decode bits) ⟨index, inside⟩
  · have zero : bits.testBit index = false := Nat.testBit_lt_two_pow
      (bound.trans_le (show 2 ^ 9 ≤ 2 ^ index from Nat.pow_le_pow_right (by decide) (by omega)))
    rw [encode, DigitMask.testBit_mask, zero]
    simp only [decide_eq_false_iff_not]
    intro member
    exact inside (indices_bound _ _ member)

theorem injective : Function.Injective encode := by
  intro left right equal
  simpa only [decode_encode] using congrArg decode equal

@[simp] theorem empty : encode ∅ = 0 := by simp [encode, indices, DigitMask.mask]

@[simp] theorem singleton (digit : Digit) : encode {digit} = 2 ^ digit.val := by
  simp [encode, indices, DigitMask.mask]

theorem all : encode Finset.univ = 511 := by
  apply Nat.eq_of_testBit_eq
  intro index
  change (encode Finset.univ).testBit index = (2 ^ 9 - 1).testBit index
  rw [encode, DigitMask.testBit_mask, Nat.testBit_two_pow_sub_one]
  congr 1
  apply propext
  simp only [mem_indices, Finset.mem_univ, true_and]
  exact ⟨fun ⟨digit, same⟩ => same ▸ digit.isLt, fun bound => ⟨⟨index, bound⟩, rfl⟩⟩

theorem zero_iff (digits : Finset Digit) : encode digits = 0 ↔ digits = ∅ := by
  rw [← empty]
  exact injective.eq_iff

theorem bitCount (digits : Finset Digit) :
    FormalSW.Py.Builtin.Int.bitCount (encode digits) = digits.card := by
  simpa only [indices, List.length_map, Finset.length_toList] using
    BitSubsets.bitCount_mask (indices digits) 9 (indices_nodup digits) (indices_bound digits)

theorem single_iff (digits : Finset Digit) :
    FormalSW.Py.Builtin.Int.bitCount (encode digits) = 1 ↔ ∃ digit, digits = {digit} := by
  rw [bitCount, Finset.card_eq_one]

theorem and_bit (digits : Finset Digit) (digit : Digit) :
    FormalSW.Py.Builtin.Int.bitAnd (encode digits) (2 ^ digit.val) =
      if digit ∈ digits then (2 ^ digit.val : Int) else 0 := by
  have cast : (2 : Int) ^ digit.val = ((2 ^ digit.val : Nat) : Int) := by simp
  rw [cast]
  change ((DigitMask.mask (indices digits) &&& 2 ^ digit.val : Nat) : Int) = _
  rw [DigitMask.and_bit]
  have membership : digit.val ∈ indices digits ↔ digit ∈ digits := by simp [indices, Fin.val_inj]
  simp only [membership]
  split <;> rfl

theorem and_bit_nonzero (digits : Finset Digit) (digit : Digit) :
    FormalSW.Py.Builtin.Int.bitAnd (encode digits) (2 ^ digit.val) ≠ 0 ↔ digit ∈ digits := by
  rw [and_bit]
  by_cases member : digit ∈ digits
  · simp only [member, if_true, iff_true]
    exact Int.ne_of_gt (Int.pow_pos (by decide))
  · simp [member]

/-- Python's complement is unbounded and signed; intersecting it still removes exactly one bit. -/
theorem erase (digits : Finset Digit) (digit : Digit) :
    FormalSW.Py.Builtin.Int.bitAnd (encode digits)
      (FormalSW.Py.Builtin.Int.bitNot (2 ^ digit.val)) = encode (digits.erase digit) := by
  have cast : (2 : Int) ^ digit.val = ((2 ^ digit.val : Nat) : Int) := by simp
  rw [cast]
  change ((encode digits ^^^ (encode digits &&& 2 ^ digit.val) : Nat) : Int) = _
  congr 1
  apply Nat.eq_of_testBit_eq
  intro index
  by_cases inside : index < 9
  · have asDigit := testBit digits ⟨index, inside⟩
    have erased := testBit (digits.erase digit) ⟨index, inside⟩
    simp only [Nat.testBit_xor, Nat.testBit_and, Nat.testBit_two_pow, asDigit, erased]
    by_cases same : digit.val = index
    · have equal : (⟨index, inside⟩ : Digit) = digit := Fin.ext same.symm
      simp [same, equal]
    · have distinct : (⟨index, inside⟩ : Digit) ≠ digit := fun equal => same (congrArg Fin.val equal).symm
      simp [same, distinct]
  · have zero (items : Finset Digit) : (encode items).testBit index = false :=
      Nat.testBit_lt_two_pow ((bounded items).trans_le
        (show 2 ^ 9 ≤ 2 ^ index from Nat.pow_le_pow_right (by decide) (by omega)))
    simp [zero]

end
end ProjectEulerSolutions.Sudoku.Masks
