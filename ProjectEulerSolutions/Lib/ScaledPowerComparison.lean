import ProjectEulerSolutions.Lib.PowerIntervals

namespace ProjectEulerSolutions.PowerIntervals

set_option Elab.async false

def comparison (left right : Nat) : Int :=
  (if right < left then 1 else 0) - (if left < right then 1 else 0)

theorem comparison_mul_right (left right factor : Nat) (positive : 0 < factor) :
    comparison (left * factor) (right * factor) = comparison left right := by
  simp only [comparison, mul_lt_mul_iff_left₀ positive]

theorem comparison_pos (left right : Nat) : 0 < comparison left right ↔ right < left := by
  unfold comparison
  split <;> split <;> omega

theorem comparison_neg (left right : Nat) : comparison left right < 0 ↔ left < right := by
  unfold comparison
  split <;> split <;> omega

theorem comparison_zero (left right : Nat) : comparison left right = 0 ↔ left = right := by
  unfold comparison
  split <;> split <;> omega

theorem comparison_swap (left right : Nat) : comparison right left = -comparison left right := by
  unfold comparison
  omega

theorem comparison_pow (left right exponent : Nat) (positive : 0 < exponent) :
    comparison (left ^ exponent) (right ^ exponent) = comparison left right := by
  simp only [comparison, Nat.pow_lt_pow_iff_left (by omega : exponent ≠ 0)]

theorem comparison_gcd (a m b n : Nat) (positive : 0 < m) :
    comparison (a ^ (m / m.gcd n)) (b ^ (n / m.gcd n)) = comparison (a ^ m) (b ^ n) := by
  have left : (a ^ (m / m.gcd n)) ^ m.gcd n = a ^ m := by
    rw [← pow_mul, Nat.div_mul_cancel (Nat.gcd_dvd_left m n)]
  have right : (b ^ (n / m.gcd n)) ^ m.gcd n = b ^ n := by
    rw [← pow_mul, Nat.div_mul_cancel (Nat.gcd_dvd_right m n)]
  rw [← left, ← right, comparison_pow _ _ _ (Nat.gcd_pos_of_pos_left n positive)]

def scaledComparison (left leftShift right rightShift : Nat) : Int :=
  if left = 0 ∨ right = 0 then comparison left right
  else if left.size + leftShift ≠ right.size + rightShift then
    comparison (left.size + leftShift) (right.size + rightShift)
  else if rightShift ≤ leftShift then comparison (left * 2 ^ (leftShift - rightShift)) right
  else comparison left (right * 2 ^ (rightShift - leftShift))

theorem scaledComparison_eq (left leftShift right rightShift : Nat) :
    scaledComparison left leftShift right rightShift =
      comparison (left * 2 ^ leftShift) (right * 2 ^ rightShift) := by
  by_cases leftZero : left = 0
  · simp [scaledComparison, leftZero, comparison]
  by_cases rightZero : right = 0
  · simp [scaledComparison, rightZero, comparison]
  have leftSize : (left * 2 ^ leftShift).size = left.size + leftShift := by
    simpa [Nat.shiftLeft_eq_mul_pow] using Nat.size_shiftLeft leftZero leftShift
  have rightSize : (right * 2 ^ rightShift).size = right.size + rightShift := by
    simpa [Nat.shiftLeft_eq_mul_pow] using Nat.size_shiftLeft rightZero rightShift
  have size_lt {a b : Nat} (less : a.size < b.size) : a < b := by
    by_contra notLess
    have := Nat.size_le_size (Nat.le_of_not_gt notLess)
    omega
  simp only [scaledComparison, leftZero, rightZero, false_or, if_false]
  by_cases different : left.size + leftShift ≠ right.size + rightShift
  · rw [if_pos different]
    rcases lt_or_gt_of_ne different with less | greater
    · have values : left * 2 ^ leftShift < right * 2 ^ rightShift :=
        size_lt (by simpa only [leftSize, rightSize] using less)
      simp [comparison, less, Nat.not_lt_of_ge less.le, values, Nat.not_lt_of_ge values.le]
    · have values : right * 2 ^ rightShift < left * 2 ^ leftShift :=
        size_lt (by simpa only [leftSize, rightSize] using greater)
      simp [comparison, greater, Nat.not_lt_of_ge greater.le, values, Nat.not_lt_of_ge values.le]
  · rw [if_neg different]
    by_cases direction : rightShift ≤ leftShift
    · rw [if_pos direction]
      have aligned : left * 2 ^ leftShift = (left * 2 ^ (leftShift - rightShift)) * 2 ^ rightShift := by
        rw [Nat.mul_assoc, ← pow_add, Nat.sub_add_cancel direction]
      rw [aligned, comparison_mul_right _ _ _ (by positivity)]
    · rw [if_neg direction]
      have aligned : right * 2 ^ rightShift = (right * 2 ^ (rightShift - leftShift)) * 2 ^ leftShift := by
        rw [Nat.mul_assoc, ← pow_add, Nat.sub_add_cancel (by omega)]
      rw [aligned, comparison_mul_right _ _ _ (by positivity)]

theorem inexact_precision_lt (a m b n precision : Nat) (positiveA : 0 < a) (positiveB : 0 < b)
    (inexact : ¬ ((power a m precision).lower = (power a m precision).upper ∧
      (power b n precision).lower = (power b n precision).upper)) :
    precision < max (a ^ m).size (b ^ n).size := by
  by_contra notLess
  have leftSmall : a ^ m < 2 ^ precision := Nat.size_le.mp (by omega)
  have rightSmall : b ^ n < 2 ^ precision := Nat.size_le.mp (by omega)
  have leftExact := power_exact a m precision positiveA leftSmall
  have rightExact := power_exact b n precision positiveB rightSmall
  exact inexact (by simp [leftExact, rightExact, exact])

/-- Refine until bounds separate or are exact; the precision has no external cutoff. -/
def refine (a m b n : Nat) (positiveA : 0 < a) (positiveB : 0 < b)
    (precision : Nat) (positivePrecision : 0 < precision) : Int :=
  let left := power a m precision
  let right := power b n precision
  if 0 < scaledComparison left.lower left.shift right.upper right.shift then 1
  else if scaledComparison left.upper left.shift right.lower right.shift < 0 then -1
  else if exactBounds : left.lower = left.upper ∧ right.lower = right.upper then
    scaledComparison left.lower left.shift right.lower right.shift
  else refine a m b n positiveA positiveB (2 * precision) (by omega)
termination_by max (a ^ m).size (b ^ n).size - precision
decreasing_by
  have below := inexact_precision_lt a m b n precision positiveA positiveB exactBounds
  omega

theorem refine_eq (a m b n : Nat) (positiveA : 0 < a) (positiveB : 0 < b)
    (precision : Nat) (positivePrecision : 0 < precision) :
    refine a m b n positiveA positiveB precision positivePrecision = comparison (a ^ m) (b ^ n) := by
  induction bound : max (a ^ m).size (b ^ n).size - precision using Nat.strong_induction_on
      generalizing precision with
  | h bound ih =>
      rw [refine]
      by_cases greater : 0 < scaledComparison (power a m precision).lower (power a m precision).shift
          (power b n precision).upper (power b n precision).shift
      · rw [if_pos greater]
        rw [scaledComparison_eq, comparison_pos] at greater
        have values : b ^ n < a ^ m := disjoint_lt (power_encloses b n precision)
          (power_encloses a m precision) greater
        simp [comparison, values, Nat.not_lt_of_ge values.le]
      · rw [if_neg greater]
        by_cases less : scaledComparison (power a m precision).upper (power a m precision).shift
            (power b n precision).lower (power b n precision).shift < 0
        · rw [if_pos less]
          rw [scaledComparison_eq, comparison_neg] at less
          have values : a ^ m < b ^ n := disjoint_lt (power_encloses a m precision)
            (power_encloses b n precision) less
          simp [comparison, values, Nat.not_lt_of_ge values.le]
        · rw [if_neg less]
          by_cases exactBounds : (power a m precision).lower = (power a m precision).upper ∧
              (power b n precision).lower = (power b n precision).upper
          · rw [dif_pos exactBounds, scaledComparison_eq]
            rw [← exact_value (power_encloses a m precision) exactBounds.1,
              ← exact_value (power_encloses b n precision) exactBounds.2]
          · rw [dif_neg exactBounds]
            have below := inexact_precision_lt a m b n precision positiveA positiveB exactBounds
            exact ih (max (a ^ m).size (b ^ n).size - 2 * precision) (by omega)
              (2 * precision) (by omega) rfl

end ProjectEulerSolutions.PowerIntervals
