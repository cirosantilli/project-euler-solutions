import ProjectEulerSolutions.P13
import ProjectEulerStatements.Lib.DecimalDigits

/-! Leading decimal digits agree with positional notation. -/
namespace ProjectEulerSolutions.P13

export ProjectEulerStatements.DecimalDigits
  (digitChar_val foldl_digitChars_aux foldl_digitChars parseNat_digitChars toDigits_eq_digitChars
    firstDigits_nat foldl_add_eq_sum_aux foldl_add_eq_sum)

theorem equiv (nums : List Nat) (k : Nat) :
    ProjectEulerStatements.P13.naive nums k = solve nums k := by
  unfold ProjectEulerStatements.P13.naive solve firstDigitsOfSum
  rw [foldl_add_eq_sum nums]
  exact (firstDigits_nat nums.sum k).symm
end ProjectEulerSolutions.P13
