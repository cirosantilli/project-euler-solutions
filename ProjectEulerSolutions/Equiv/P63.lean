import ProjectEulerSolutions.P63

namespace ProjectEulerSolutions.P63

private theorem count_eq (base seed exponent remaining : Nat)
    (valid : 2 ≤ base) (inside : seed < base) (positive : 1 ≤ exponent) :
    countPowers base seed exponent remaining =
      PowerfulDigits.sumWindow (PowerfulDigits.hit base seed) exponent remaining := by
  induction remaining generalizing exponent with
  | zero => rfl
  | succ remaining ih =>
      rw [countPowers]
      split
      · rename_i yes
        rw [PowerfulDigits.sumWindow_succ, ih _ (by omega)]
        simp [PowerfulDigits.hit, yes]
      · rename_i no
        symm
        exact PowerfulDigits.seed_window_zero base seed exponent (remaining + 1)
          valid inside positive (Nat.lt_of_not_ge no)

theorem equiv (base : Nat) (valid : 2 ≤ base) :
    ProjectEulerStatements.P63.naive base = solve base := by
  rw [ProjectEulerStatements.P63.naive_eq_answer base valid]
  unfold solve PowerfulDigits.answer
  apply congrArg List.sum
  apply List.map_congr_left
  intro seed member
  have inside : seed < base := by
    have := List.mem_range'_1.mp member
    omega
  exact (count_eq base seed 1 _ valid inside (by omega)).symm

end ProjectEulerSolutions.P63
