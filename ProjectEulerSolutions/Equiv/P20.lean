import ProjectEulerSolutions.P20
import ProjectEulerStatements.Lib.DecimalDigitSum

namespace ProjectEulerSolutions.P20

theorem digitSum_eq_digits_sum (n : Nat) : digitSum n = (Nat.digits 10 n).sum :=
  ProjectEulerStatements.DecimalDigits.sum_string_digits n

theorem loop_factorial (n k : Nat) (positive : 0 < k) (bound : k ≤ n + 1) :
    solve.loop n k (Nat.factorial (k - 1)) = digitSum (Nat.factorial n) := by
  induction measure : n + 1 - k using Nat.strong_induction_on generalizing k with
  | h size ih =>
      rw [solve.loop.eq_1]
      by_cases finished : k > n
      · rw [if_pos finished]
        have last : k - 1 = n := by omega
        rw [last]
      · rw [if_neg finished]
        have next : Nat.factorial (k - 1) * k = Nat.factorial k := by
          obtain ⟨j, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : k ≠ 0)
          simp [Nat.factorial_succ, Nat.mul_comm]
        rw [next]
        exact ih (n + 1 - (k + 1)) (by omega) (k + 1) (by omega) (by omega) rfl

theorem equiv (n : Nat) : ProjectEulerStatements.P20.naive n = solve n := by
  unfold solve
  by_cases zero : n = 0
  · subst n
    rw [solve.loop.eq_1]
    simp [ProjectEulerStatements.P20.naive, ProjectEulerStatements.P20.digitSum,
      digitSum_eq_digits_sum]
  · have result := loop_factorial n 2 (by decide) (by omega)
    simpa [ProjectEulerStatements.P20.naive, ProjectEulerStatements.P20.digitSum,
      digitSum_eq_digits_sum] using result.symm

end ProjectEulerSolutions.P20
