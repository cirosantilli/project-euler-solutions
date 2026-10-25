import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Sqrt
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.IntervalCases
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

/-- A complete primality test whose small trial loop also reduces efficiently in the kernel. -/
def isPrime (n : Nat) : Bool :=
  if n ≤ 3 then decide (2 ≤ n)
  else if n % 2 = 0 ∨ n % 3 = 0 then false
  else trial n 5

theorem isPrime_correct (n : Nat) : isPrime n = decide (Nat.Prime n) := by
  unfold isPrime
  by_cases small : n ≤ 3
  · rw [if_pos small]
    interval_cases n <;> decide
  · rw [if_neg small]
    by_cases divides : n % 2 = 0 ∨ n % 3 = 0
    · rw [if_pos divides]
      have composite : ¬ Nat.Prime n := by
        intro prime
        rcases divides with even | third
        · rcases (Nat.dvd_prime prime).mp (Nat.dvd_of_mod_eq_zero even) with one | same <;> omega
        · rcases (Nat.dvd_prime prime).mp (Nat.dvd_of_mod_eq_zero third) with one | same <;> omega
      simp only [composite, decide_false]
    · rw [if_neg divides]
      exact trial_correct n 5 (by omega) (by decide) (by decide)
        (initial n (fun h => divides (Or.inl h)) (fun h => divides (Or.inr h)))

/-- A square-root-bounded structural loop avoids reducing large well-founded measures in certificates. -/
def boundedTrial (n first : Nat) : Nat → Bool
  | 0 => true
  | fuel+1 =>
      if 5≤first ∧ first*first≤n then
        if n%first=0 ∨ n%(first+2)=0 then false else boundedTrial n (first+6) fuel
      else true

theorem boundedTrial_correct (n first fuel : Nat) (large : 5≤first)
    (enough : Nat.sqrt n<first+6*fuel) : boundedTrial n first fuel=trial n first := by
  induction fuel generalizing first with
  | zero =>
      have inactive : ¬ (5≤first ∧ first*first≤n) := by
        rintro ⟨_, square⟩
        have bound : first≤Nat.sqrt n := Nat.le_sqrt'.mpr (by simpa only [Nat.pow_two] using square)
        omega
      rw [boundedTrial, trial, dif_neg inactive]
  | succ fuel ih =>
      rw [boundedTrial, trial]
      by_cases active : 5≤first ∧ first*first≤n
      · rw [if_pos active, dif_pos active]
        by_cases divides : n%first=0 ∨ n%(first+2)=0
        · simp only [if_pos divides]
        · simp only [if_neg divides]
          exact ih (first+6) (by omega) (by omega)
      · rw [if_neg active, dif_neg active]

/-- A complete primality predicate with a small, structural kernel reduction bound. -/
def certifiedPrime (n : Nat) : Bool :=
  if n≤3 then decide (2≤n)
  else if n%2=0 ∨ n%3=0 then false
  else boundedTrial n 5 (n/6+1)

theorem certifiedPrime_correct (n : Nat) : certifiedPrime n=decide (Nat.Prime n) := by
  rw [← isPrime_correct]
  unfold certifiedPrime isPrime
  split
  · rfl
  · split
    · rfl
    · exact boundedTrial_correct n 5 (n/6+1) (by decide) (by have := Nat.sqrt_le_self n; omega)

end ProjectEulerSolutions.SixStepPrimality
