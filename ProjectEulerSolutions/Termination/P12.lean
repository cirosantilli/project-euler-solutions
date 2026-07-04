namespace ProjectEulerSolutions.P12.Termination

theorem div_lt_self {x p : Nat} (hx : x ≠ 0) (hp : ¬ p ≤ 1) : x / p < x :=
  Nat.div_lt_self (Nat.pos_of_ne_zero hx) (Nat.lt_of_not_ge hp)

theorem sub_lt_succ_sub_of_not_gt {bound idx : Nat} (h : ¬ idx > bound) :
    bound - idx < bound + 1 - idx := by
  exact Nat.sub_lt_sub_right (Nat.le_of_not_gt h) (Nat.lt_succ_self bound)

end ProjectEulerSolutions.P12.Termination
