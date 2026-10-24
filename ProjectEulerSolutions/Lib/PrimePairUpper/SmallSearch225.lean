import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch224
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch225

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan52 : scan edge (search edge 4) 5 [] 0 tail52 best = best := by
  have cons : tail52 = 257 :: tail53 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates52_checked]
  change scan edge (search edge 4) 5 [] 0 tail53
    (if 4 ≤ rootCandidates52.length then search edge 4 [257] rootCandidates52 257 best else best) = best
  rw [if_pos (by decide), rootBranch52]
  exact outerScan53

end ProjectEulerSolutions.PrimePairUpper.Search
