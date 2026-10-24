import Mathlib.Data.Nat.Prime.Defs
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

end ProjectEulerSolutions.TrialPrimeCertificate
