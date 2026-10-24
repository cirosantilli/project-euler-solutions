import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Sqrt
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Lean.Elab.Tactic.Omega

/-! Correctness of trial division through the residue classes 5 and 1 modulo 6. -/
namespace ProjectEulerSolutions.SixStepPrimality

set_option Elab.async false

/-- The guard totalizes states outside the source loop's invariant. -/
def trial (n first : Nat) : Bool :=
  if active : 5 ≤ first ∧ first * first ≤ n then
    if n % first = 0 ∨ n % (first + 2) = 0 then false else trial n (first + 6)
  else true
termination_by n + 1 - first
decreasing_by
  have : first ≤ n := by nlinarith [active.1, active.2]
  omega

def Checked (n before : Nat) : Prop :=
  ∀ p, Nat.Prime p → p < before → ¬ p ∣ n

private theorem prime_residue (p : Nat) (prime : Nat.Prime p) (large : 5 ≤ p) :
    p % 6 = 1 ∨ p % 6 = 5 := by
  have odd : p % 2 = 1 := prime.mod_two_eq_one_iff_ne_two.mpr (by omega)
  have third : p % 3 ≠ 0 := by
    intro zero
    rcases (Nat.dvd_prime prime).mp (Nat.dvd_of_mod_eq_zero zero) with one | same <;> omega
  omega

private theorem reject_prime (n first : Nat) (large : 5 ≤ first) (active : first * first ≤ n)
    (divides : n % first = 0 ∨ n % (first + 2) = 0) : ¬ Nat.Prime n := by
  intro prime
  rcases divides with divides | divides
  · have proper : first < n := by nlinarith
    rcases (Nat.dvd_prime prime).mp (Nat.dvd_of_mod_eq_zero divides) with one | same <;> omega
  · have proper : first + 2 < n := by nlinarith
    rcases (Nat.dvd_prime prime).mp (Nat.dvd_of_mod_eq_zero divides) with one | same <;> omega

private theorem advance (n first : Nat) (large : 5 ≤ first) (residue : first % 6 = 5)
    (checked : Checked n first) (left : n % first ≠ 0) (right : n % (first + 2) ≠ 0) :
    Checked n (first + 6) := by
  intro p prime below
  by_cases earlier : p < first
  · exact checked p prime earlier
  · have positions := prime_residue p prime (by omega)
    have same : p = first ∨ p = first + 2 := by omega
    rcases same with rfl | rfl
    · exact fun divides => left (Nat.mod_eq_zero_of_dvd divides)
    · exact fun divides => right (Nat.mod_eq_zero_of_dvd divides)

/-- All earlier prime factors have already been excluded when the source tests first. -/
theorem trial_correct (n first : Nat) (lower : 2 ≤ n) (large : 5 ≤ first)
    (residue : first % 6 = 5) (checked : Checked n first) :
    trial n first = decide (Nat.Prime n) := by
  induction measure : n + 1 - first using Nat.strong_induction_on generalizing first with
  | h measure ih =>
      rw [trial]
      by_cases active : first * first ≤ n
      · rw [dif_pos ⟨large, active⟩]
        by_cases divides : n % first = 0 ∨ n % (first + 2) = 0
        · simp only [if_pos divides, reject_prime n first large active divides, decide_false]
        · rw [if_neg divides]
          apply ih (n + 1 - (first + 6))
          · have : first ≤ n := by nlinarith
            omega
          · omega
          · omega
          · exact advance n first large residue checked (fun zero => divides (Or.inl zero))
              (fun zero => divides (Or.inr zero))
          · rfl
      · rw [dif_neg (by omega : ¬ (5 ≤ first ∧ first * first ≤ n))]
        have prime : Nat.Prime n := by
          by_contra composite
          have factor := Nat.minFac_prime (by omega : n ≠ 1)
          have square := Nat.minFac_sq_le_self (by omega : 0 < n) composite
          have earlier : n.minFac < first := by nlinarith
          exact checked n.minFac factor earlier (Nat.minFac_dvd n)
        simp only [prime, decide_true]

/-- The initial tests for 2 and 3 establish the first trial-division invariant. -/
theorem initial (n : Nat) (even : n % 2 ≠ 0) (third : n % 3 ≠ 0) : Checked n 5 := by
  intro p prime below
  have lower := prime.two_le
  have possibilities : p = 2 ∨ p = 3 := by
    have : p ≠ 4 := by
      intro same
      subst p
      exact (by decide : ¬ Nat.Prime 4) prime
    omega
  rcases possibilities with rfl | rfl
  · exact fun divides => even (Nat.mod_eq_zero_of_dvd divides)
  · exact fun divides => third (Nat.mod_eq_zero_of_dvd divides)

end ProjectEulerSolutions.SixStepPrimality
