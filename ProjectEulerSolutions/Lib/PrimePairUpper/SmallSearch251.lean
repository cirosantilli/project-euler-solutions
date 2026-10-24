import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch250
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch251

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan16 : scan edge (search edge 4) 5 [] 0 tail16 best = best := by
  have cons : tail16 = 67 :: tail17 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates16_checked]
  change scan edge (search edge 4) 5 [] 0 tail17
    (if 4 ≤ rootCandidates16.length then search edge 4 [67] rootCandidates16 67 best else best) = best
  rw [if_pos (by decide), rootBranch16]
  exact outerScan17

end ProjectEulerSolutions.PrimePairUpper.Search
