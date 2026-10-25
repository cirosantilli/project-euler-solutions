import Mathlib.Data.Nat.Digits.Lemmas
import Mathlib.Algebra.Order.Ring.Pow
import Mathlib.Tactic.Linarith

/-! Complete power-digit counting in any numeral base greater than one. -/
namespace ProjectEulerSolutions.PowerfulDigits

set_option Elab.async false

def bound (base : Nat) : Nat := (base - 1) * base

theorem bound_positive (base : Nat) (valid : 2 ≤ base) : 1 ≤ bound base := by
  change 1 * 1 ≤ (base - 1) * base
  exact Nat.mul_le_mul (by omega) (by omega)

/-- The first two binomial terms already double the predecessor's power. -/
theorem doubling (base : Nat) (valid : 2 ≤ base) :
    2 * (base - 1) ^ (base - 1) ≤ base ^ (base - 1) := by
  have expanded := pow_add_mul_le_add_pow (a := base - 1) (b := 1)
    (Nat.zero_le _) (Nat.zero_le _) (base - 1)
  simp only [Nat.cast_id] at expanded
  have scale : (base - 1) * (base - 1) ^ (base - 1 - 1) = (base - 1) ^ (base - 1) := by
    rw [← Nat.pow_succ']
    congr 1
    omega
  rw [show base - 1 + 1 = base by omega, Nat.mul_one, scale] at expanded
  omega

/-- A base-dependent bound follows from power growth; the source search needs no exponent cap. -/
theorem maximum_fails (base : Nat) (valid : 2 ≤ base) :
    (base - 1) ^ bound base < base ^ (bound base - 1) := by
  have raised := Nat.pow_le_pow_left (doubling base valid) base
  rw [Nat.mul_pow, ← Nat.pow_mul, ← Nat.pow_mul] at raised
  change 2 ^ base * (base - 1) ^ bound base ≤ base ^ bound base at raised
  have positive : 0 < (base - 1) ^ bound base := Nat.pow_pos (by omega)
  have strict := Nat.mul_lt_mul_of_pos_right (Nat.lt_two_pow_self (n := base)) positive
  have ratio : base * (base - 1) ^ bound base < base ^ bound base := strict.trans_le raised
  have split : base ^ bound base = base * base ^ (bound base - 1) := by
    rw [← Nat.pow_succ']
    congr 1
    have := bound_positive base valid
    omega
  rw [split] at ratio
  exact (Nat.mul_lt_mul_left (by omega : 0 < base)).mp ratio

/-- Once a power falls below its required place value, it cannot recover at a later exponent. -/
theorem failure_succ (base seed exponent : Nat) (valid : 2 ≤ base) (inside : seed ≤ base)
    (positive : 1 ≤ exponent) (failed : seed ^ exponent < base ^ (exponent - 1)) :
    seed ^ (exponent + 1) < base ^ exponent := by
  have scaled := Nat.mul_lt_mul_of_pos_right failed (by omega : 0 < base)
  have next := Nat.mul_le_mul_left (seed ^ exponent) inside
  have split : base ^ (exponent - 1) * base = base ^ exponent := by
    rw [← Nat.pow_succ]
    congr 1
    omega
  rw [split] at scaled
  rw [Nat.pow_succ]
  exact next.trans_lt scaled

theorem failure_later (base seed exponent extra : Nat) (valid : 2 ≤ base) (inside : seed ≤ base)
    (positive : 1 ≤ exponent) (failed : seed ^ exponent < base ^ (exponent - 1)) :
    seed ^ (exponent + extra) < base ^ (exponent + extra - 1) := by
  induction extra with
  | zero => simpa using failed
  | succ extra ih =>
      simpa only [Nat.add_assoc, Nat.add_sub_cancel] using
        failure_succ base seed (exponent + extra) valid inside (by omega) ih

theorem seed_fails (base seed : Nat) (valid : 2 ≤ base) (inside : seed < base) :
    seed ^ bound base < base ^ (bound base - 1) :=
  (Nat.pow_le_pow_left (by omega : seed ≤ base - 1) (bound base)).trans_lt (maximum_fails base valid)

theorem after_bound (base seed exponent : Nat) (valid : 2 ≤ base) (inside : seed < base)
    (late : bound base ≤ exponent) : seed ^ exponent < base ^ (exponent - 1) := by
  have failed := failure_later base seed (bound base) (exponent - bound base) valid (by omega)
    (bound_positive base valid) (seed_fails base seed valid inside)
  simpa only [Nat.add_sub_cancel' late] using failed

def width (base value : Nat) : Nat := (Nat.digits base value).length

theorem power_width_le (base seed exponent : Nat) (valid : 2 ≤ base) (inside : seed < base)
    (positive : 1 ≤ exponent) : width base (seed ^ exponent) ≤ exponent := by
  apply (Nat.digits_length_le_iff (by omega : 1 < base) (seed ^ exponent)).mpr
  exact Nat.pow_lt_pow_left inside (by omega)

theorem width_at_least (base value exponent : Nat) (valid : 2 ≤ base) (positive : 1 ≤ exponent) :
    exponent ≤ width base value ↔ base ^ (exponent - 1) ≤ value := by
  have shifted : exponent ≤ width base value ↔ exponent - 1 < width base value := by omega
  exact shifted.trans (Nat.lt_digits_length_iff (by omega : 1 < base) value)

theorem power_width_eq (base seed exponent : Nat) (valid : 2 ≤ base) (inside : seed < base)
    (positive : 1 ≤ exponent) :
    width base (seed ^ exponent) = exponent ↔ base ^ (exponent - 1) ≤ seed ^ exponent := by
  have upper := power_width_le base seed exponent valid inside positive
  rw [← width_at_least base (seed ^ exponent) exponent valid positive]
  omega

/-- Bases at least the numeral base cannot have the requested number of digits. -/
theorem seed_below_base (base seed exponent : Nat) (valid : 2 ≤ base)
    (qualifies : width base (seed ^ exponent) = exponent) : seed < base := by
  have upper : seed ^ exponent < base ^ exponent :=
    (Nat.digits_length_le_iff (by omega : 1 < base) _).mp (by simpa only [width] using qualifies.le)
  exact Nat.lt_of_not_ge (fun outside => Nat.not_le_of_lt upper (Nat.pow_le_pow_left outside exponent))

def hit (base seed exponent : Nat) : Nat := if base ^ (exponent - 1) ≤ seed ^ exponent then 1 else 0

theorem hit_zero (base seed exponent : Nat) (failed : seed ^ exponent < base ^ (exponent - 1)) :
    hit base seed exponent = 0 := by simp [hit, Nat.not_le_of_lt failed]

def seeds (base : Nat) : List Nat := List.range' 1 (base - 1)
def exponents (base : Nat) : List Nat := List.range' 1 (bound base - 1)
def seedCount (base seed : Nat) : Nat := ((exponents base).map (hit base seed)).sum
def exponentCount (base exponent : Nat) : Nat := ((seeds base).map (fun seed => hit base seed exponent)).sum
def answer (base : Nat) : Nat := ((seeds base).map (seedCount base)).sum

theorem eligible_bounded (base seed exponent : Nat) (valid : 2 ≤ base) (seedPositive : 0 < seed)
    (positive : 1 ≤ exponent) (qualifies : width base (seed ^ exponent) = exponent) :
    seed ∈ seeds base ∧ exponent ∈ exponents base := by
  have inside := seed_below_base base seed exponent valid qualifies
  have lower := (power_width_eq base seed exponent valid inside positive).mp qualifies
  have before : exponent < bound base := by
    by_contra late
    have failed := after_bound base seed exponent valid inside (by omega)
    omega
  simp only [seeds, exponents, List.mem_range'_1]
  have := bound_positive base valid
  omega

private theorem sum_swap (left right : List Nat) (weight : Nat → Nat → Nat) :
    (left.map (fun a => (right.map (weight a)).sum)).sum =
      (right.map (fun b => (left.map (fun a => weight a b)).sum)).sum := by
  induction left with
  | nil => simp
  | cons head rest ih =>
      simp only [List.map_cons, List.sum_cons, ih]
      rw [← List.sum_map_add]

/-- Counting by exponent and counting by seed enumerate the same finite pairs. -/
theorem transpose (base : Nat) : ((exponents base).map (exponentCount base)).sum = answer base := by
  exact sum_swap (exponents base) (seeds base) (fun exponent seed => hit base seed exponent)

end ProjectEulerSolutions.PowerfulDigits
