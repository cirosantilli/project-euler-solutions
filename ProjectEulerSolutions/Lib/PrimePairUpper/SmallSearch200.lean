import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch199
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch200

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan102 : scan edge (search edge 4) 5 [] 0 tail102 best = best := by
  have cons : tail102 = 571 :: tail103 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates102_checked]
  change scan edge (search edge 4) 5 [] 0 tail103
    (if 4 ≤ rootCandidates102.length then search edge 4 [571] rootCandidates102 571 best else best) = best
  rw [if_pos (by decide), rootBranch102]
  exact outerScan103

set_option maxRecDepth 4096 in
theorem outerScan101 : scan edge (search edge 4) 5 [] 0 tail101 best = best := by
  have cons : tail101 = 569 :: tail102 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates101_checked]
  change scan edge (search edge 4) 5 [] 0 tail102
    (if 4 ≤ rootCandidates101.length then search edge 4 [569] rootCandidates101 569 best else best) = best
  rw [if_pos (by decide), rootBranch101]
  exact outerScan102

end ProjectEulerSolutions.PrimePairUpper.Search
