import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch205
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch206

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan90 : scan edge (search edge 4) 5 [] 0 tail90 best = best := by
  have cons : tail90 = 487 :: tail91 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates90_checked]
  change scan edge (search edge 4) 5 [] 0 tail91
    (if 4 ≤ rootCandidates90.length then search edge 4 [487] rootCandidates90 487 best else best) = best
  rw [if_pos (by decide), rootBranch90]
  exact outerScan91

set_option maxRecDepth 4096 in
theorem outerScan89 : scan edge (search edge 4) 5 [] 0 tail89 best = best := by
  have cons : tail89 = 479 :: tail90 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates89_checked]
  change scan edge (search edge 4) 5 [] 0 tail90
    (if 4 ≤ rootCandidates89.length then search edge 4 [479] rootCandidates89 479 best else best) = best
  rw [if_pos (by decide), rootBranch89]
  exact outerScan90

end ProjectEulerSolutions.PrimePairUpper.Search
