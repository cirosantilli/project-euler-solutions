namespace ProjectEulerSolutions.P7.Termination

def goMeasure (bound candidate : Nat) : Nat :=
  bound + 1 - candidate

theorem go_decreases (bound candidate : Nat) (h : candidate ≤ bound) :
    goMeasure bound (candidate + 1) < goMeasure bound candidate := by
  unfold goMeasure
  omega

end ProjectEulerSolutions.P7.Termination
