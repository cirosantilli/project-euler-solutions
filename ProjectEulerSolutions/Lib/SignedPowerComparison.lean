import ProjectEulerSolutions.Lib.ScaledPowerComparison

namespace ProjectEulerSolutions.PowerIntervals

set_option Elab.async false

def integerComparison (left right : Int) : Int :=
  (if right < left then 1 else 0) - (if left < right then 1 else 0)

/-- Compare rational powers by cross multiplying their positive denominators. -/
def signedComparison (a : Nat) (m : Int) (b : Nat) (n : Int) : Int :=
  comparison (a ^ m.toNat * b ^ (-n).toNat) (b ^ n.toNat * a ^ (-m).toNat)

theorem signedComparison_normalize (a : Nat) (m : Int) (b : Nat) (n : Int) :
    signedComparison a (if a = 1 then 0 else m) b (if b = 1 then 0 else n) =
      signedComparison a m b n := by
  by_cases left : a = 1 <;> by_cases right : b = 1 <;>
    simp [signedComparison, left, right]

theorem signedComparison_nonnegative (a : Nat) (m : Int) (b : Nat) (n : Int)
    (left : 0 ≤ m) (right : 0 ≤ n) :
    signedComparison a m b n = comparison (a ^ m.toNat) (b ^ n.toNat) := by
  simp [signedComparison, Int.toNat_eq_zero.mpr (by omega : -m ≤ 0),
    Int.toNat_eq_zero.mpr (by omega : -n ≤ 0)]

theorem signedComparison_negate (a : Nat) (m : Int) (b : Nat) (n : Int) :
    signedComparison a (-m) b (-n) = -signedComparison a m b n := by
  simpa [signedComparison, mul_comm] using
    comparison_swap (a ^ m.toNat * b ^ (-n).toNat) (b ^ n.toNat * a ^ (-m).toNat)

theorem signedComparison_same_base (a : Nat) (large : 1 < a) (m n : Int) :
    signedComparison a m a n = integerComparison m n := by
  have left : n.toNat + (-m).toNat < m.toNat + (-n).toNat ↔ n < m := by omega
  have right : m.toNat + (-n).toNat < n.toNat + (-m).toNat ↔ m < n := by omega
  simp only [signedComparison, ← pow_add, comparison, pow_lt_pow_iff_right₀ large,
    left, right, integerComparison]

theorem signedComparison_opposite (a : Nat) (m : Int) (b : Nat) (n : Int)
    (positiveA : 0 < a) (positiveB : 0 < b)
    (normalizedA : a = 1 → m = 0) (normalizedB : b = 1 → n = 0)
    (left : m ≤ 0) (right : 0 ≤ n) (different : ¬(m = 0 ∧ n = 0)) :
    signedComparison a m b n = -1 := by
  have lowerA : 1 ≤ a ^ (-m).toNat := Nat.one_le_pow _ _ positiveA
  have lowerB : 1 ≤ b ^ n.toNat := Nat.one_le_pow _ _ positiveB
  have greater : 1 < b ^ n.toNat * a ^ (-m).toNat := by
    by_cases zero : m = 0
    · have strict : 1 < b ^ n.toNat := one_lt_pow₀ (by omega : 1 < b) (by omega)
      nlinarith
    · have strict : 1 < a ^ (-m).toNat := one_lt_pow₀ (by omega : 1 < a) (by omega)
      nlinarith
  simp [signedComparison, Int.toNat_eq_zero.mpr left,
    Int.toNat_eq_zero.mpr (by omega : -n ≤ 0), comparison, greater, Nat.not_lt_of_ge greater.le]

/-- The scalar cases and common-exponent reduction used before interval refinement. -/
def normalizedComparison (a : Nat) (m : Int) (b : Nat) (n : Int) : Int :=
  if m = 0 ∧ n = 0 then 0
  else if m ≤ 0 ∧ 0 ≤ n then -1
  else if 0 ≤ m ∧ n ≤ 0 then 1
  else if a = b then integerComparison m n
  else
    let left := (if m < 0 then -m else m).toNat
    let right := (if m < 0 then -n else n).toNat
    (if m < 0 then -1 else 1) *
      comparison (a ^ (left / left.gcd right)) (b ^ (right / left.gcd right))

def comparisonAlgorithm (a : Nat) (m : Int) (b : Nat) (n : Int) : Int :=
  normalizedComparison a (if a = 1 then 0 else m) b (if b = 1 then 0 else n)

theorem comparisonAlgorithm_eq (a : Nat) (m : Int) (b : Nat) (n : Int)
    (positiveA : 0 < a) (positiveB : 0 < b) :
    comparisonAlgorithm a m b n = signedComparison a m b n := by
  rw [← signedComparison_normalize a m b n]
  let left : Int := if a = 1 then 0 else m
  let right : Int := if b = 1 then 0 else n
  have normalizedA : a = 1 → left = 0 := by intro one; simp [left, one]
  have normalizedB : b = 1 → right = 0 := by intro one; simp [right, one]
  change (if left = 0 ∧ right = 0 then 0
    else if left ≤ 0 ∧ 0 ≤ right then -1
    else if 0 ≤ left ∧ right ≤ 0 then 1
    else if a = b then integerComparison left right
    else (if left < 0 then -1 else 1) * comparison
      (a ^ ((if left < 0 then -left else left).toNat /
        (if left < 0 then -left else left).toNat.gcd (if left < 0 then -right else right).toNat))
      (b ^ ((if left < 0 then -right else right).toNat /
        (if left < 0 then -left else left).toNat.gcd (if left < 0 then -right else right).toNat))) =
      signedComparison a left b right
  by_cases zero : left = 0 ∧ right = 0
  · simp [zero.1, zero.2, signedComparison, comparison]
  rw [if_neg zero]
  by_cases opposite : left ≤ 0 ∧ 0 ≤ right
  · rw [if_pos opposite, signedComparison_opposite a left b right positiveA positiveB
      normalizedA normalizedB opposite.1 opposite.2 zero]
  rw [if_neg opposite]
  by_cases opposite' : 0 ≤ left ∧ right ≤ 0
  · rw [if_pos opposite']
    have reverse := signedComparison_opposite a (-left) b (-right) positiveA positiveB
      (by simp_all) (by simp_all) (by omega) (by omega) (by omega)
    rw [signedComparison_negate] at reverse
    omega
  rw [if_neg opposite']
  by_cases same : a = b
  · rw [if_pos same, ← same, signedComparison_same_base a (by omega)]
  rw [if_neg same]
  by_cases negative : left < 0
  · simp only [if_pos negative, neg_one_mul]
    rw [comparison_gcd _ _ _ _ (by omega),
      ← signedComparison_nonnegative a (-left) b (-right) (by omega) (by omega),
      signedComparison_negate, neg_neg]
  · simp only [if_neg negative, one_mul]
    rw [comparison_gcd _ _ _ _ (by omega),
      signedComparison_nonnegative a left b right (by omega) (by omega)]

end ProjectEulerSolutions.PowerIntervals
