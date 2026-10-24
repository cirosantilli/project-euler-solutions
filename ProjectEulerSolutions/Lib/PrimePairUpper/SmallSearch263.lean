import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch262
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch263

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan4 : scan edge (search edge 4) 5 [] 0 tail4 best = best := by
  have cons : tail4 = 17 :: tail5 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates4_checked]
  change scan edge (search edge 4) 5 [] 0 tail5
    (if 4 ≤ rootCandidates4.length then search edge 4 [17] rootCandidates4 17 best else best) = best
  rw [if_pos (by decide), rootBranch4]
  exact outerScan5

end ProjectEulerSolutions.PrimePairUpper.Search
