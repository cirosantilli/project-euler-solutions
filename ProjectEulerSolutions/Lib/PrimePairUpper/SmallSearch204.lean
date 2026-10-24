import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch203
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch204

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan94 : scan edge (search edge 4) 5 [] 0 tail94 best = best := by
  have cons : tail94 = 509 :: tail95 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates94_checked]
  change scan edge (search edge 4) 5 [] 0 tail95
    (if 4 ≤ rootCandidates94.length then search edge 4 [509] rootCandidates94 509 best else best) = best
  rw [if_pos (by decide), rootBranch94]
  exact outerScan95

set_option maxRecDepth 4096 in
theorem outerScan93 : scan edge (search edge 4) 5 [] 0 tail93 best = best := by
  have cons : tail93 = 503 :: tail94 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates93_checked]
  change scan edge (search edge 4) 5 [] 0 tail94
    (if 4 ≤ rootCandidates93.length then search edge 4 [503] rootCandidates93 503 best else best) = best
  rw [if_pos (by decide), rootBranch93]
  exact outerScan94

end ProjectEulerSolutions.PrimePairUpper.Search
