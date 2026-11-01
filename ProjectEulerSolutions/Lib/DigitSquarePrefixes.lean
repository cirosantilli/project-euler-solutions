import ProjectEulerSolutions.Lib.DigitSquareCounts
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-! Exact digit-prefix transitions for arbitrary decimal bounds, including leading zeroes. -/
namespace ProjectEulerSolutions.DigitSquareChains.Prefix
open scoped BigOperators
set_option Elab.async false

abbrev State := Nat × Bool

def state (bound value : Nat) : State := (nextTerm value, decide (value = bound))
def digits (ceiling : Nat) (tight : Bool) : Nat := (if tight then ceiling else 9) + 1
def advance (ceiling : Nat) (current : State) (digit : Nat) : State :=
  (current.1 + digit * digit, current.2 && decide (digit = ceiling))

theorem nextTerm_append (value digit : Nat) (small : digit < 10) :
    nextTerm (10 * value + digit) = nextTerm value + digit * digit := by
  rw [nextTerm_step]
  have quotient : (10 * value + digit) / 10 = value := by omega
  have remainder : (10 * value + digit) % 10 = digit := by omega
  rw [quotient, remainder, Nat.add_comm]

theorem advance_state (bound value ceiling digit : Nat) (ceilingBound : ceiling < 10) (digitBound : digit < 10) :
    advance ceiling (state bound value) digit = state (10 * bound + ceiling) (10 * value + digit) := by
  apply Prod.ext
  · exact (nextTerm_append value digit digitBound).symm
  · change (decide (value = bound) && decide (digit = ceiling)) =
      decide (10 * value + digit = 10 * bound + ceiling)
    rw [← Bool.decide_and]
    apply Bool.eq_iff_iff.mpr
    simp only [decide_eq_true_eq]
    omega

/-- Every integer below the extended prefix has one quotient/remainder pair in the allowed branches. -/
theorem extend_sum (bound ceiling : Nat) (small : ceiling < 10) (score : Nat → Nat) :
    (∑ value ∈ Finset.range (bound + 1), ∑ digit ∈ Finset.range (if value = bound then ceiling + 1 else 10),
      score (10 * value + digit)) = ∑ value ∈ Finset.range (10 * bound + ceiling + 1), score value := by
  let pairs := (Finset.range (bound + 1)).sigma
    (fun value => Finset.range (if value = bound then ceiling + 1 else 10))
  have pairBounds : ∀ pair ∈ pairs, pair.1 ≤ bound ∧ pair.2 < 10 ∧
      (pair.1 = bound → pair.2 ≤ ceiling) := by
    intro pair member
    obtain ⟨first, second⟩ := Finset.mem_sigma.mp member
    simp only [Finset.mem_range] at first second
    by_cases tight : pair.1 = bound
    · simp only [if_pos tight] at second
      exact ⟨by omega, by omega, by omega⟩
    · simp only [if_neg tight] at second
      exact ⟨by omega, second, by intro impossible; contradiction⟩
  have summed : (∑ pair ∈ pairs, score (10 * pair.1 + pair.2)) =
      ∑ value ∈ Finset.range (10 * bound + ceiling + 1), score value := by
    apply Finset.sum_bij (fun pair _ => 10 * pair.1 + pair.2)
    · intro pair member
      have constraints := pairBounds pair member
      rw [Finset.mem_range]
      by_cases tight : pair.1 = bound
      · have last := constraints.2.2 tight
        omega
      · omega
    · intro left hl right hr equal
      have leftBound := pairBounds left hl
      have rightBound := pairBounds right hr
      have first : left.1 = right.1 := by omega
      apply Sigma.ext first
      exact heq_of_eq (show left.2 = right.2 by omega)
    · intro value member
      have within : value ≤ 10 * bound + ceiling := by simpa only [Finset.mem_range, Nat.lt_succ_iff] using member
      refine ⟨⟨value / 10, value % 10⟩, ?_, ?_⟩
      · apply Finset.mem_sigma.mpr
        constructor
        · rw [Finset.mem_range]
          change value / 10 < bound + 1
          omega
        · rw [Finset.mem_range]
          change value % 10 < (if value / 10 = bound then ceiling + 1 else 10)
          by_cases tight : value / 10 = bound
          · rw [if_pos tight]; omega
          · rw [if_neg tight]; omega
      · change 10 * (value / 10) + value % 10 = value
        omega
    · intros; rfl
  simpa only [pairs, Finset.sum_sigma] using summed

end ProjectEulerSolutions.DigitSquareChains.Prefix
