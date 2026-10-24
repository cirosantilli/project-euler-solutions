import Mathlib.Data.Nat.Prime.Defs
import Mathlib.Data.Nat.Sqrt
import Lean.Elab.Tactic.Omega

namespace ProjectEulerSolutions.TrialPrimeCertificate

/-- A balanced divisor block keeps the certificate's recursion depth
logarithmic. Checking the Boolean equality uses ordinary kernel reduction. -/
def block (n first : Nat) : Nat → Bool
  | 0 => decide (n % first ≠ 0)
  | depth + 1 => block n first depth && block n (first + 2 ^ depth) depth

theorem block_sound (n first depth : Nat) (checked : block n first depth = true) :
    ∀ divisor, first ≤ divisor → divisor < first + 2 ^ depth → n % divisor ≠ 0 := by
  induction depth generalizing first with
  | zero =>
    intro divisor lower upper
    have same : divisor = first := by simp only [Nat.pow_zero] at upper; omega
    simpa only [same, block, decide_eq_true_eq] using checked
  | succ depth ih =>
    have parts : block n first depth = true ∧ block n (first + 2 ^ depth) depth = true := by
      simpa only [block, Bool.and_eq_true] using checked
    intro divisor lower upper
    have power : 2 ^ (depth + 1) = 2 ^ depth + 2 ^ depth := by rw [Nat.pow_succ]; omega
    rw [power] at upper
    by_cases left : divisor < first + 2 ^ depth
    · exact ih first parts.1 divisor lower left
    · exact ih (first + 2 ^ depth) parts.2 divisor (by omega) (by omega)

theorem prime (n depth : Nat) (large : 2 ≤ n) (covers : Nat.sqrt n < 2 + 2 ^ depth)
    (checked : block n 2 depth = true) : Nat.Prime n := by
  apply Nat.prime_def_le_sqrt.mpr
  refine ⟨large, ?_⟩
  intro divisor lower upper divides
  have nonzero := block_sound n 2 depth checked divisor lower (by omega)
  exact nonzero (Nat.mod_eq_zero_of_dvd divides)


theorem block_complete (n first depth : Nat)
    (nonzero : ∀ divisor, first ≤ divisor → divisor < first + 2 ^ depth → n % divisor ≠ 0) :
    block n first depth = true := by
  induction depth generalizing first with
  | zero =>
    simp only [block, decide_eq_true_eq]
    exact nonzero first (by omega) (by simp)
  | succ depth ih =>
    simp only [block, Bool.and_eq_true]
    have power : 2 ^ (depth + 1) = 2 ^ depth + 2 ^ depth := by rw [Nat.pow_succ]; omega
    rw [power] at nonzero
    constructor
    · apply ih first
      intro divisor lower upper
      exact nonzero divisor lower (by omega)
    · apply ih (first + 2 ^ depth)
      intro divisor lower upper
      exact nonzero divisor (by omega) (by omega)

/-- A bounded vertex classifier, using balanced divisor checks above the
small direct-decision range. Its correctness is proved for every n<258². -/
def smallPrime (n : Nat) : Bool :=
  if n ≤ 17 then decide (Nat.Prime n) else if n ≤ 257 then block n 2 4 else block n 2 8

theorem smallPrime_iff (n : Nat) (bounded : n < 258 * 258) : smallPrime n = true ↔ Nat.Prime n := by
  unfold smallPrime
  by_cases tiny : n ≤ 17
  · simp only [if_pos tiny, decide_eq_true_eq]
  · rw [if_neg tiny]
    by_cases small : n ≤ 257
    · rw [if_pos small]
      constructor
      · intro checked
        exact prime n 4 (by omega) (Nat.sqrt_lt.mpr (by omega)) checked
      · intro isPrime
        apply block_complete
        intro divisor lower upper zero
        rcases (Nat.dvd_prime isPrime).mp (Nat.dvd_of_mod_eq_zero zero) with one | same
        · omega
        · change divisor < 18 at upper
          omega
    · rw [if_neg small]
      constructor
      · intro checked
        exact prime n 8 (by omega) (Nat.sqrt_lt.mpr bounded) checked
      · intro isPrime
        apply block_complete
        intro divisor lower upper zero
        rcases (Nat.dvd_prime isPrime).mp (Nat.dvd_of_mod_eq_zero zero) with one | same
        · omega
        · change divisor < 258 at upper
          omega

end ProjectEulerSolutions.TrialPrimeCertificate
