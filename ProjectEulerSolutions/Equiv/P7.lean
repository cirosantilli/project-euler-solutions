module

public import ProjectEulerSolutions.Lib.PrimeSieve
public import ProjectEulerStatements.Lib.NthPrimeBounds

public section
namespace ProjectEulerSolutions.P7

lemma nth_prime_le_two_pow_succ (n : Nat) : Nat.nth Nat.Prime n ≤ 2 ^ (n + 1) :=
  ProjectEulerStatements.NthPrimeBounds.nth_prime_le_two_pow_succ n

theorem equiv (n : Nat) : ProjectEulerStatements.P7.naive n = solve n := by
  cases n with
  | zero => simp [ProjectEulerStatements.P7.naive, solve]
  | succ k =>
      unfold ProjectEulerStatements.P7.naive solve
      have hbound : Nat.nth Nat.Prime k ≤ initialLimit (k + 1) * 2 ^ (k + 1) := by
        calc
          Nat.nth Nat.Prime k ≤ 2 ^ (k + 1) := nth_prime_le_two_pow_succ k
          _ = 1 * 2 ^ (k + 1) := by simp
          _ ≤ initialLimit (k + 1) * 2 ^ (k + 1) :=
            Nat.mul_le_mul_right _ (one_le_initialLimit (k + 1))
      exact (nthPrimeWithRetry_eq_nth k (initialLimit (k + 1)) (k + 1)
        hbound).symm

end ProjectEulerSolutions.P7
