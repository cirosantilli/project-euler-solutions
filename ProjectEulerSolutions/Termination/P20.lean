namespace ProjectEulerSolutions.P20.Termination

theorem loop_decreases {n k : Nat} (h : ¬ k > n) :
    n - k < n + 1 - k := by
  exact Nat.sub_lt_sub_right (Nat.le_of_not_gt h) (Nat.lt_succ_self n)

end ProjectEulerSolutions.P20.Termination
