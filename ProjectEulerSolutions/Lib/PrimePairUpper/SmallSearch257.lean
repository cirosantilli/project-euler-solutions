import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch256
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch257

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan10 : scan edge (search edge 4) 5 [] 0 tail10 best = best := by
  have cons : tail10 = 41 :: tail11 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates10_checked]
  change scan edge (search edge 4) 5 [] 0 tail11
    (if 4 ≤ rootCandidates10.length then search edge 4 [41] rootCandidates10 41 best else best) = best
  rw [if_pos (by decide), rootBranch10]
  exact outerScan11

end ProjectEulerSolutions.PrimePairUpper.Search
