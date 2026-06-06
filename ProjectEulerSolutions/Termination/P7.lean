namespace ProjectEulerSolutions.P7.Termination

def goMeasure (n candidate count : Nat) : Nat :=
  (n + 1 - count) * (2 * n * n + 11) + ((2 * n * n + 10) - candidate)

theorem go_decreases (n candidate count nextCount : Nat) :
    goMeasure n (candidate + 2) nextCount < goMeasure n candidate count := by
  sorry

end ProjectEulerSolutions.P7.Termination
