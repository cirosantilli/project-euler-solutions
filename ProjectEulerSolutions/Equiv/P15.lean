import ProjectEulerSolutions.P15
import ProjectEulerStatements.Lib.Binomial

/-! The multiplicative loop computes the binomial coefficient exactly. -/
namespace ProjectEulerSolutions.P15

private theorem loop_correct (n count i : Nat) (bound : count ≤ n)
    (positive : 1 ≤ i) (before : i ≤ count + 1) :
    binom.loop n count i (Nat.choose (n - count + i - 1) (i - 1)) = Nat.choose n count := by
  induction measure : count + 1 - i using Nat.strong_induction_on generalizing i with
  | h remaining ih =>
      rw [binom.loop.eq_1]
      by_cases finished : i > count
      · rw [if_pos finished]
        have lower : i - 1 = count := by omega
        have upper : n - count + i - 1 = n := by omega
        rw [lower, upper]
      · rw [if_neg finished]
        have next : Nat.choose (n - count + i - 1) (i - 1) * (n - count + i) / i =
            Nat.choose (n - count + i) i := by
          have step := ProjectEulerStatements.Binomial.multiplicative_step (n - count) (i - 1)
          have succ : i - 1 + 1 = i := by omega
          have index : n - count + (i - 1) = n - count + i - 1 := by omega
          have upper : n - count + i - 1 + 1 = n - count + i := by omega
          simpa only [index, Nat.add_assoc, succ, upper] using step
        rw [next]
        have continuation := ih (count + 1 - (i + 1)) (by omega) (i + 1) (by omega) (by omega) rfl
        simpa only [Nat.add_sub_cancel] using continuation

theorem binom_eq_choose (n k : Nat) : binom n k = Nat.choose n k := by
  unfold binom
  by_cases large : k > n
  · rw [if_pos large, Nat.choose_eq_zero_of_lt large]
  · rw [if_neg large]
    have count : (if k < n - k then k else n - k) = min k (n - k) := by
      split_ifs <;> omega
    rw [count]
    have result := loop_correct n (min k (n - k)) 1 (by omega) (by omega) (by omega)
    simpa only [Nat.add_sub_cancel, Nat.sub_self, Nat.choose_zero_right,
      ProjectEulerStatements.Binomial.choose_min n k (by omega)] using result

theorem equiv (n : Nat) : ProjectEulerStatements.P15.naive n = solve n := by
  simp only [ProjectEulerStatements.P15.naive, solve, latticePaths, binom_eq_choose, two_mul]

end ProjectEulerSolutions.P15
