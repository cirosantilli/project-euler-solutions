import Mathlib.Data.Nat.Digits.Lemmas
import Lean.Elab.Tactic.Omega

/-! Exact decimal shifting for P60's original multiplication loop.
Independent of all old P60 native/termination/equivalence files. -/
namespace ProjectEulerSolutions.DecimalShift
set_option Elab.async false

/-- Every active source iteration strictly reduces the remaining natural gap. -/
theorem measure_step (n : Int) (p : Nat) (positive : 0 < p) (active : (p : Int) ≤ n) :
    n.toNat + 1 - p * 10 < n.toNat + 1 - p := by
  have nonnegative : 0 ≤ n := le_trans (Int.natCast_nonneg p) active
  have cast : (n.toNat : Int) = n := Int.toNat_of_nonneg nonnegative
  have inside : p ≤ n.toNat := by omega
  have growth : p < p * 10 := by omega
  omega

/-- This is the actual while-loop transition, generalized to any positive start.
The positivity proof excludes the source's unreachable divergent start p=0. -/
def decimalShiftFrom (n : Int) (p : Nat) (positive : 0 < p) : Nat :=
  if active : (p : Int) ≤ n then decimalShiftFrom n (p * 10) (by omega) else p
termination_by n.toNat + 1 - p
decreasing_by exact measure_step n p positive active

def decimalShift (n : Int) : Nat := decimalShiftFrom n 1 (by decide)

theorem from_step (n : Int) (p : Nat) (positive : 0 < p) (active : (p : Int) ≤ n) :
    decimalShiftFrom n p positive = decimalShiftFrom n (p * 10) (by omega) := by
  rw [decimalShiftFrom, dif_pos active]

theorem from_done (n : Int) (p : Nat) (positive : 0 < p) (finished : n < (p : Int)) :
    decimalShiftFrom n p positive = p := by
  rw [decimalShiftFrom, dif_neg (show ¬(p : Int) ≤ n by omega)]

/-- The output is an attained power multiple, exceeds n, and every earlier
multiplication state was at most n. These are exact least-hit facts. -/
theorem decimalShiftFrom_spec (n : Int) (p : Nat) (positive : 0 < p) :
    ∃ steps : Nat,
      decimalShiftFrom n p positive = p * 10 ^ steps ∧
      n < (decimalShiftFrom n p positive : Int) ∧
      ∀ earlier < steps, ((p * 10 ^ earlier : Nat) : Int) ≤ n := by
  by_cases active : (p : Int) ≤ n
  · have nextPositive : 0 < p * 10 := by omega
    obtain ⟨steps, attained, exceeds, previous⟩ := decimalShiftFrom_spec n (p * 10) nextPositive
    have transition := from_step n p positive active
    refine ⟨steps + 1, ?_, ?_, ?_⟩
    · rw [transition, attained, Nat.pow_succ]
      ac_rfl
    · simpa only [transition] using exceeds
    · intro earlier before
      cases earlier with
      | zero => simpa only [Nat.pow_zero, Nat.mul_one] using active
      | succ earlier =>
          have earlierBound : earlier < steps := by omega
          have earlierActive := previous earlier earlierBound
          have same : p * 10 ^ (earlier + 1) = (p * 10) * 10 ^ earlier := by
            rw [Nat.pow_succ]
            ac_rfl
          simpa only [same] using earlierActive
  · have stopped := from_done n p positive (by omega)
    exact ⟨0, by simpa only [Nat.pow_zero, Nat.mul_one] using stopped,
      by rw [stopped]; omega, fun earlier before => by omega⟩
termination_by n.toNat + 1 - p
decreasing_by exact measure_step n p positive active

/-- For signed inputs, zero and negative values correctly choose 10^0=1. -/
theorem decimalShift_spec (n : Int) :
    ∃ width : Nat, decimalShift n = 10 ^ width ∧ n < (decimalShift n : Int) ∧
      ∀ earlier < width, ((10 ^ earlier : Nat) : Int) ≤ n := by
  simpa only [decimalShift, Nat.one_mul] using decimalShiftFrom_spec n 1 (by decide)

theorem decimalShift_positive (n : Int) : 0 < decimalShift n := by
  obtain ⟨width, attained, _, _⟩ := decimalShift_spec n
  rw [attained]
  exact Nat.pow_pos (by decide)

theorem decimalShift_exceeds (n : Int) : n < (decimalShift n : Int) :=
  (decimalShift_spec n).choose_spec.2.1

/-- Least power of ten strictly greater than the original signed input. -/
theorem decimalShift_le_power (n : Int) (width : Nat) (exceeds : n < ((10 ^ width : Nat) : Int)) :
    decimalShift n ≤ 10 ^ width := by
  obtain ⟨chosen, attained, _, previous⟩ := decimalShift_spec n
  have enough : chosen ≤ width := by
    by_cases before : width < chosen
    · have active := previous width before
      omega
    · omega
  rw [attained]
  exact Nat.pow_le_pow_right (by decide : 0 < 10) enough

@[simp] theorem decimalShift_small (n : Int) (small : n < 1) : decimalShift n = 1 :=
  from_done n 1 (by decide) small

/-- Compatibility with ordinary decimal concatenation, including Nat zero's
empty positional digit list. No source string conversion is introduced. -/
theorem decimalShift_nat_eq_digits (n : Nat) :
    decimalShift n = 10 ^ (Nat.digits 10 n).length := by
  obtain ⟨width, attained, exceeds, previous⟩ := decimalShift_spec n
  have fits : n < 10 ^ (Nat.digits 10 n).length :=
    (Nat.digits_length_le_iff (b := 10) (k := (Nat.digits 10 n).length) (by decide) n).mp (Nat.le_refl _)
  have widthLower : (Nat.digits 10 n).length ≤ width := by
    apply (Nat.digits_length_le_iff (b := 10) (k := width) (by decide) n).mpr
    rw [attained] at exceeds
    omega
  have widthUpper : width ≤ (Nat.digits 10 n).length := by
    by_cases before : (Nat.digits 10 n).length < width
    · have active := previous (Nat.digits 10 n).length before
      omega
    · omega
  rw [attained, Nat.le_antisymm widthUpper widthLower]

end ProjectEulerSolutions.DecimalShift
