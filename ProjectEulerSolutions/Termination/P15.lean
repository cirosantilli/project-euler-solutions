namespace ProjectEulerSolutions.P15.Termination

theorem sub_lt_succ_sub_of_not_gt {bound idx : Nat} (h : ¬ idx > bound) :
    bound - idx < bound + 1 - idx := by
  exact Nat.sub_lt_sub_right (Nat.le_of_not_gt h) (Nat.lt_succ_self bound)

end ProjectEulerSolutions.P15.Termination
