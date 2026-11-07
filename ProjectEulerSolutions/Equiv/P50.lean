import ProjectEulerSolutions.P50

namespace ProjectEulerSolutions.P50
open ConsecutivePrimeSums

theorem equiv (limit : Nat) : ProjectEulerStatements.P50.naive limit = solve limit := by
  unfold ProjectEulerStatements.P50.naive solve
  dsimp only
  have ordered : (ProjectEulerStatements.P50.primesBelow limit).Pairwise (· ≤ ·) :=
    ((List.pairwise_lt_range (n := limit)).imp (fun h => Nat.le_of_lt h)).filter _
  rw [scanLengths_eq_reference _ ordered _ _ (Nat.le_refl _) (fun candidate valid => by
    have := valid.2.1
    omega)]

end ProjectEulerSolutions.P50
