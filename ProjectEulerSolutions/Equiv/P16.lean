import ProjectEulerSolutions.P16
import ProjectEulerStatements.Lib.DecimalDigitSum

/-! Reuse the decimal-digit sum theorem for powers of two. -/
namespace ProjectEulerSolutions.P16

export ProjectEulerStatements.DecimalDigits
  (digitChar_val toDigits_eq_digitChars foldl_digitChars_sum_aux foldl_digitChars_sum)

lemma digitSum_eq_digits_sum (n : Nat) : digitSum n = (Nat.digits 10 n).sum :=
  ProjectEulerStatements.DecimalDigits.sum_string_digits n

theorem equiv (n : Nat) : ProjectEulerStatements.P16.naive n = solve n := by
  unfold ProjectEulerStatements.P16.naive ProjectEulerStatements.P16.digitSum solve
  exact (digitSum_eq_digits_sum (2 ^ n)).symm
end ProjectEulerSolutions.P16
