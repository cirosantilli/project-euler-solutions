import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch208
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch209

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan84 : scan edge (search edge 4) 5 [] 0 tail84 best = best := by
  have cons : tail84 = 449 :: tail85 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates84_checked]
  change scan edge (search edge 4) 5 [] 0 tail85
    (if 4 ≤ rootCandidates84.length then search edge 4 [449] rootCandidates84 449 best else best) = best
  rw [if_pos (by decide), rootBranch84]
  exact outerScan85

set_option maxRecDepth 4096 in
theorem outerScan83 : scan edge (search edge 4) 5 [] 0 tail83 best = best := by
  have cons : tail83 = 443 :: tail84 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates83_checked]
  change scan edge (search edge 4) 5 [] 0 tail84
    (if 4 ≤ rootCandidates83.length then search edge 4 [443] rootCandidates83 443 best else best) = best
  rw [if_pos (by decide), rootBranch83]
  exact outerScan84

end ProjectEulerSolutions.PrimePairUpper.Search
