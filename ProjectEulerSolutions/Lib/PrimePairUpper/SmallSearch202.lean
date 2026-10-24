import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch201
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch202

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan98 : scan edge (search edge 4) 5 [] 0 tail98 best = best := by
  have cons : tail98 = 547 :: tail99 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates98_checked]
  change scan edge (search edge 4) 5 [] 0 tail99
    (if 4 ≤ rootCandidates98.length then search edge 4 [547] rootCandidates98 547 best else best) = best
  rw [if_pos (by decide), rootBranch98]
  exact outerScan99

set_option maxRecDepth 4096 in
theorem outerScan97 : scan edge (search edge 4) 5 [] 0 tail97 best = best := by
  have cons : tail97 = 541 :: tail98 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates97_checked]
  change scan edge (search edge 4) 5 [] 0 tail98
    (if 4 ≤ rootCandidates97.length then search edge 4 [541] rootCandidates97 541 best else best) = best
  rw [if_pos (by decide), rootBranch97]
  exact outerScan98

end ProjectEulerSolutions.PrimePairUpper.Search
