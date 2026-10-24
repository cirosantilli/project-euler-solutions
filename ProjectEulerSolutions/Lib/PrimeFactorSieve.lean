import ProjectEulerStatements.P47
import ProjectEulerStatements.Lib.DivisorSieve
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.List.Range
import Mathlib.Tactic.Linarith

/-! Exact counts of distinct prime divisors and the incremental sieve invariant. -/
namespace ProjectEulerSolutions.PrimeFactorSieve

set_option Elab.async false

/-- Only positive multiples are marked; the zero cell remains zero. -/
def countsBefore (next n : Nat) : Nat :=
  ((List.range next).filter (fun p => decide (Nat.Prime p ∧ p≤n ∧ p∣n))).length

@[simp] theorem countsBefore_two (n : Nat) : countsBefore 2 n=0 := by
  simp [countsBefore, show List.range 2=[0,1] from rfl, Nat.not_prime_zero, Nat.not_prime_one]

theorem countsBefore_step (p n : Nat) :
    countsBefore (p+1) n=countsBefore p n+if Nat.Prime p ∧ p≤n ∧ p∣n then 1 else 0 := by
  by_cases marked : Nat.Prime p ∧ p≤n ∧ p∣n <;>
    simp [countsBefore, List.range_succ, List.filter_append, marked]

/-- A candidate has no marked prime divisors exactly when it is prime. -/
theorem current_zero (p : Nat) (lower : 2≤p) : countsBefore p p=0 ↔ Nat.Prime p := by
  have empty : countsBefore p p=0 ↔ ∀ q<p, ¬ (Nat.Prime q ∧ q≤p ∧ q∣p) := by
    simp [countsBefore, List.filter_eq_nil_iff]
  rw [empty]
  constructor
  · intro checked
    by_contra composite
    have minPrime := Nat.minFac_prime (by omega : p≠1)
    have minBefore : p.minFac<p := (Nat.not_prime_iff_minFac_lt lower).mp composite
    exact checked p.minFac minBefore ⟨minPrime, by omega, Nat.minFac_dvd p⟩
  · intro prime q before ⟨qPrime, _, divides⟩
    rcases (Nat.dvd_prime prime).mp divides with one | same
    · have := qPrime.two_le
      omega
    · omega

/-- Once the processed prefix passes n, no additional prime can divide its positive value. -/
theorem countsBefore_stable (next n : Nat) (enough : n<next) :
    countsBefore next n=countsBefore (n+1) n := by
  induction next with
  | zero => omega
  | succ next ih =>
      by_cases same : next=n
      · subst next
        rfl
      · rw [countsBefore_step]
        have beyond : ¬next≤n := by omega
        simp only [beyond, false_and, and_false, if_false, Nat.add_zero]
        exact ih (by omega)

/-- The sieve's completed cells equal the statement's exhaustive distinct-factor count. -/
theorem countsBefore_complete (next n : Nat) (enough : n<next) :
    countsBefore next n=ProjectEulerStatements.P47.numDistinctPrimeFactors n := by
  rw [countsBefore_stable next n enough]
  unfold countsBefore ProjectEulerStatements.P47.numDistinctPrimeFactors
  congr 1
  apply List.filter_congr
  intro p member
  have bound : p≤n := by have := List.mem_range.mp member; omega
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq]
  constructor
  · rintro ⟨prime, _, divides⟩
    exact ⟨prime.two_le, Nat.mod_eq_zero_of_dvd divides, prime⟩
  · rintro ⟨_, divides, prime⟩
    exact ⟨prime, bound, Nat.dvd_of_mod_eq_zero divides⟩

/-- A partially completed inner pass increments precisely the earlier multiples. -/
def row (base : Nat → Nat) (p next n : Nat) : Nat :=
  base n+if p≤n ∧ n<next ∧ p∣n then 1 else 0

@[simp] theorem row_start (base : Nat → Nat) (p n : Nat) : row base p p n=base n := by
  simp [row, show ¬ (p≤n ∧ n<p ∧ p∣n) by omega]

theorem row_step (base : Nat → Nat) (p m n : Nat) (positive : 0<p)
    (lower : p≤m) (multiple : p∣m) :
    row base p (m+p) n=if n=m then row base p m n+1 else row base p m n := by
  by_cases same : n=m
  · subst n
    simp [row, lower, multiple, positive]
  · by_cases before : n<m
    · have earlier : n<m+p := by omega
      simp [row, same, before, earlier]
    · have after : ¬ (p≤n ∧ n<m+p ∧ p∣n) := by
        rintro ⟨_, upper, divides⟩
        exact same (ProjectEulerStatements.DivisorSieve.consecutive p m n multiple divides (by omega) upper)
      simp [row, same, before, after]

theorem row_complete (p next n : Nat) (prime : Nat.Prime p) (finished : n<next) :
    row (countsBefore p) p next n=countsBefore (p+1) n := by
  simp [row, finished, countsBefore_step, prime]

theorem skip_composite (p n : Nat) (composite : ¬ Nat.Prime p) :
    countsBefore p n=countsBefore (p+1) n := by
  simp [countsBefore_step, composite]

@[simp] theorem count_zero : ProjectEulerStatements.P47.numDistinctPrimeFactors 0=0 := by decide
@[simp] theorem count_one : ProjectEulerStatements.P47.numDistinctPrimeFactors 1=0 := by decide

end ProjectEulerSolutions.PrimeFactorSieve
