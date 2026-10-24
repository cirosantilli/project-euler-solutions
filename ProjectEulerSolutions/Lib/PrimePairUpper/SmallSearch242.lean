import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch241
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch242

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan27 : scan edge (search edge 4) 5 [] 0 tail27 best = best := by
  have cons : tail27 = 113 :: tail28 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates27_checked]
  change scan edge (search edge 4) 5 [] 0 tail28
    (if 4 ≤ rootCandidates27.length then search edge 4 [113] rootCandidates27 113 best else best) = best
  rw [if_pos (by decide), rootBranch27]
  exact outerScan28

set_option maxRecDepth 4096 in
theorem outerScan26 : scan edge (search edge 4) 5 [] 0 tail26 best = best := by
  have cons : tail26 = 109 :: tail27 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates26_checked]
  change scan edge (search edge 4) 5 [] 0 tail27
    (if 4 ≤ rootCandidates26.length then search edge 4 [109] rootCandidates26 109 best else best) = best
  rw [if_pos (by decide), rootBranch26]
  exact outerScan27

end ProjectEulerSolutions.PrimePairUpper.Search
