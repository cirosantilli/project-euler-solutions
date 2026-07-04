namespace ProjectEulerSolutions.P14.Termination

theorem decreases {a b : Nat} : a < b := by
  sorry

theorem sub_succ_lt_sub {bound idx : Nat} (h : ¬ idx ≥ bound) :
    bound - (idx + 1) < bound - idx := by
  exact Nat.sub_lt_sub_left (Nat.lt_of_not_ge h) (Nat.lt_succ_self idx)

end ProjectEulerSolutions.P14.Termination
