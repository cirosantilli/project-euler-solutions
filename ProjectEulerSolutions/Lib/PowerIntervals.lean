import Mathlib.Data.Nat.Size
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import lang.py.builtin.type.int

/-! Integer power intervals with outward rounding and no floating-point assumptions. -/
namespace ProjectEulerSolutions.PowerIntervals

set_option Elab.async false

structure Bounds where
  lower : Nat
  upper : Nat
  shift : Nat
  deriving DecidableEq

def Encloses (bounds : Bounds) (value : Nat) : Prop :=
  bounds.lower * 2 ^ bounds.shift ≤ value ∧ value ≤ bounds.upper * 2 ^ bounds.shift

def exact (value : Nat) : Bounds := ⟨value, value, 0⟩

theorem exact_encloses (value : Nat) : Encloses (exact value) value := by
  simp [Encloses, exact]

def rounded (bounds : Bounds) (drop : Nat) : Bounds :=
  ⟨bounds.lower / 2 ^ drop, (bounds.upper + 2 ^ drop - 1) / 2 ^ drop, bounds.shift + drop⟩

theorem rounded_encloses (bounds : Bounds) (drop value : Nat) (inside : Encloses bounds value) :
    Encloses (rounded bounds drop) value := by
  have down := Nat.div_mul_le_self bounds.lower (2 ^ drop)
  have positive : 0 < 2 ^ drop := by positivity
  have up : bounds.upper ≤ (bounds.upper + 2 ^ drop - 1) / 2 ^ drop * 2 ^ drop := by
    have remainder := Nat.mod_lt (bounds.upper + 2 ^ drop - 1) positive
    have division := Nat.mod_add_div (bounds.upper + 2 ^ drop - 1) (2 ^ drop)
    rw [Nat.mul_comm (2 ^ drop)] at division
    omega
  constructor
  · calc
      (rounded bounds drop).lower * 2 ^ (rounded bounds drop).shift =
          (bounds.lower / 2 ^ drop * 2 ^ drop) * 2 ^ bounds.shift := by
            simp only [rounded, pow_add]
            ring
      _ ≤ bounds.lower * 2 ^ bounds.shift := Nat.mul_le_mul_right _ down
      _ ≤ value := inside.1
  · calc
      value ≤ bounds.upper * 2 ^ bounds.shift := inside.2
      _ ≤ ((bounds.upper + 2 ^ drop - 1) / 2 ^ drop * 2 ^ drop) * 2 ^ bounds.shift :=
        Nat.mul_le_mul_right _ up
      _ = (rounded bounds drop).upper * 2 ^ (rounded bounds drop).shift := by
        simp only [rounded, pow_add]
        ring

theorem rounded_zero (bounds : Bounds) : rounded bounds 0 = bounds := by
  cases bounds
  simp [rounded]

def trim (bounds : Bounds) (precision : Nat) : Bounds :=
  rounded bounds (bounds.upper.size - precision)

theorem trim_encloses (bounds : Bounds) (precision value : Nat) (inside : Encloses bounds value) :
    Encloses (trim bounds precision) value := rounded_encloses bounds _ value inside

theorem trim_eq (bounds : Bounds) (precision : Nat) (small : bounds.upper < 2 ^ precision) :
    trim bounds precision = bounds := by
  simp only [trim, Nat.sub_eq_zero_of_le (Nat.size_le.mpr small), rounded_zero]

def multiply (left right : Bounds) : Bounds :=
  ⟨left.lower * right.lower, left.upper * right.upper, left.shift + right.shift⟩

theorem multiply_encloses (left right : Bounds) (a b : Nat)
    (leftInside : Encloses left a) (rightInside : Encloses right b) :
    Encloses (multiply left right) (a * b) := by
  constructor
  · calc
      (multiply left right).lower * 2 ^ (multiply left right).shift =
          (left.lower * 2 ^ left.shift) * (right.lower * 2 ^ right.shift) := by
            simp only [multiply, pow_add]
            ring
      _ ≤ a * b := Nat.mul_le_mul leftInside.1 rightInside.1
  · calc
      a * b ≤ (left.upper * 2 ^ left.shift) * (right.upper * 2 ^ right.shift) :=
        Nat.mul_le_mul leftInside.2 rightInside.2
      _ = (multiply left right).upper * 2 ^ (multiply left right).shift := by
        simp only [multiply, pow_add]
        ring

theorem multiply_exact (a b : Nat) : multiply (exact a) (exact b) = exact (a * b) := rfl

theorem disjoint_lt {left right : Bounds} {a b : Nat}
    (leftInside : Encloses left a) (rightInside : Encloses right b)
    (separated : left.upper * 2 ^ left.shift < right.lower * 2 ^ right.shift) : a < b :=
  lt_of_le_of_lt leftInside.2 (lt_of_lt_of_le separated rightInside.1)

theorem exact_value {bounds : Bounds} {value : Nat} (inside : Encloses bounds value)
    (equal : bounds.lower = bounds.upper) : value = bounds.lower * 2 ^ bounds.shift := by
  have := inside.2
  rw [← equal] at this
  exact Nat.le_antisymm this inside.1

theorem bitLength_eq_size (value : Nat) : FormalSW.Py.Builtin.Int.bitLength value = value.size := by
  by_cases zero : value = 0
  · simp [zero, FormalSW.Py.Builtin.Int.bitLength]
  · simp only [FormalSW.Py.Builtin.Int.bitLength, Int.natAbs_natCast, if_neg zero]
    have limits := (Nat.log2_eq_iff zero).mp rfl
    have upper := Nat.size_le.mpr limits.2
    have lower := Nat.lt_size.mpr limits.1
    omega

theorem power_residual (a b remaining : Nat) : a * b ^ remaining =
    (if remaining % 2 = 1 then a * b else a) * (b * b) ^ (remaining / 2) := by
  have decompose : b ^ remaining = b ^ (remaining % 2) * (b * b) ^ (remaining / 2) := by
    calc
      b ^ remaining = b ^ (remaining % 2 + 2 * (remaining / 2)) :=
        congrArg (b ^ ·) (Nat.mod_add_div remaining 2).symm
      _ = _ := by rw [pow_add, pow_mul, pow_two]
  rw [decompose]
  by_cases odd : remaining % 2 = 1
  · simp [odd, Nat.mul_assoc]
  · have even : remaining % 2 = 0 := by omega
    simp [even]

def powerLoop (precision : Nat) (accumulator factor : Bounds) (remaining : Nat) : Bounds :=
  if remaining = 0 then accumulator
  else
    let next := if remaining % 2 = 1 then trim (multiply accumulator factor) precision else accumulator
    if remaining / 2 = 0 then next
    else powerLoop precision next (trim (multiply factor factor) precision) (remaining / 2)
termination_by remaining
decreasing_by exact Nat.div_lt_self (by omega) (by decide)

theorem powerLoop_encloses (precision remaining : Nat) (accumulator factor : Bounds) (a b : Nat)
    (accInside : Encloses accumulator a) (factorInside : Encloses factor b) :
    Encloses (powerLoop precision accumulator factor remaining) (a * b ^ remaining) := by
  induction remaining using Nat.strong_induction_on generalizing accumulator factor a b with
  | h remaining ih =>
      by_cases zero : remaining = 0
      · simpa [powerLoop, zero] using accInside
      · have nextInside : Encloses
            (if remaining % 2 = 1 then trim (multiply accumulator factor) precision else accumulator)
            (if remaining % 2 = 1 then a * b else a) := by
          split
          · exact trim_encloses _ _ _ (multiply_encloses _ _ _ _ accInside factorInside)
          · exact accInside
        rw [powerLoop, if_neg zero, power_residual]
        dsimp only
        by_cases finished : remaining / 2 = 0
        · simpa only [finished, if_pos, pow_zero, Nat.mul_one] using nextInside
        · simp only [if_neg finished]
          exact ih (remaining / 2) (Nat.div_lt_self (by omega) (by decide)) _ _ _ _ nextInside
            (trim_encloses _ precision _ (multiply_encloses _ _ _ _ factorInside factorInside))

def power (base exponent precision : Nat) : Bounds :=
  powerLoop precision (exact 1) (trim (exact base) precision) exponent

theorem power_encloses (base exponent precision : Nat) :
    Encloses (power base exponent precision) (base ^ exponent) := by
  simpa only [power, Nat.one_mul] using powerLoop_encloses precision exponent (exact 1)
    (trim (exact base) precision) 1 base (exact_encloses 1)
    (trim_encloses _ _ _ (exact_encloses base))

/-- Sufficient precision makes every intermediate operation exact, including equality cases. -/
theorem powerLoop_exact (precision remaining a b : Nat) (positiveA : 0 < a) (positiveB : 0 < b)
    (small : a * b ^ remaining < 2 ^ precision) :
    powerLoop precision (exact a) (exact b) remaining = exact (a * b ^ remaining) := by
  induction remaining using Nat.strong_induction_on generalizing a b with
  | h remaining ih =>
      by_cases zero : remaining = 0
      · simp [powerLoop, zero]
      · let next := if remaining % 2 = 1 then a * b else a
        have positiveNext : 0 < next := by
          dsimp only [next]
          split
          · exact Nat.mul_pos positiveA positiveB
          · exact positiveA
        have residual : a * b ^ remaining = next * (b * b) ^ (remaining / 2) := power_residual a b remaining
        have nextSmall : next < 2 ^ precision :=
          (Nat.le_mul_of_pos_right next (pow_pos (Nat.mul_pos positiveB positiveB) _)).trans_lt
            (residual ▸ small)
        have nextExact :
            (if remaining % 2 = 1 then trim (multiply (exact a) (exact b)) precision else exact a) = exact next := by
          dsimp only [next]
          split
          next odd =>
            rw [multiply_exact]
            apply trim_eq
            simpa only [next, if_pos odd, exact] using nextSmall
          next => rfl
        rw [powerLoop, if_neg zero]
        dsimp only
        rw [nextExact]
        by_cases finished : remaining / 2 = 0
        · simp only [if_pos finished]
          rw [residual, finished, pow_zero, Nat.mul_one]
        · rw [if_neg finished, multiply_exact]
          have factorSmall : b * b < 2 ^ precision := by
            have powerLe : b * b ≤ b ^ remaining := by
              simpa only [pow_two] using Nat.pow_le_pow_right positiveB (by omega : 2 ≤ remaining)
            exact (powerLe.trans (Nat.le_mul_of_pos_left _ positiveA)).trans_lt small
          rw [trim_eq _ precision factorSmall]
          rw [ih (remaining / 2) (Nat.div_lt_self (by omega) (by decide)) next (b * b)
            positiveNext (Nat.mul_pos positiveB positiveB) (residual ▸ small), ← residual]

theorem power_exact (base exponent precision : Nat) (positive : 0 < base)
    (small : base ^ exponent < 2 ^ precision) :
    power base exponent precision = exact (base ^ exponent) := by
  by_cases zero : exponent = 0
  · simp [power, powerLoop, zero]
  · have baseSmall : base < 2 ^ precision := by
      have powerLe : base ≤ base ^ exponent := by
        simpa only [pow_one] using Nat.pow_le_pow_right positive (by omega : 1 ≤ exponent)
      exact powerLe.trans_lt small
    rw [power, trim_eq _ precision baseSmall]
    simpa only [Nat.one_mul] using powerLoop_exact precision exponent 1 base (by decide) positive
      (by simpa only [Nat.one_mul] using small)

end ProjectEulerSolutions.PowerIntervals
