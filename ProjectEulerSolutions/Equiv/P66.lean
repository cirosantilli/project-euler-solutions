import ProjectEulerSolutions.P66

namespace ProjectEulerSolutions.P66
open ProjectEulerStatements

theorem minimalPellX_eq (d : Nat) : minimalPellX d = PellMinimum.minimum d := by
  unfold minimalPellX
  split
  · rename_i square
    exact (PellMinimum.minimum_of_square d square).symm
  · rename_i nonsquare
    rw [PellConvergents.firstHit_minimum d nonsquare, Int.toNat_natCast]

theorem equiv (limit : Nat) : ProjectEulerStatements.P66.naive limit = solve limit := by
  simp only [ProjectEulerStatements.P66.naive, ProjectEulerStatements.P66.bestD,
    PellSearch.answer, PellSearch.scan, solve, minimalPellX_eq]
  congr 3
  omega

end ProjectEulerSolutions.P66
