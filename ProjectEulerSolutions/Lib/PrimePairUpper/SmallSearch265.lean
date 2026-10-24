import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch264
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch265

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan2 : scan edge (search edge 4) 5 [] 0 tail2 best = best := by
  have cons : tail2 = 11 :: tail3 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates2_checked]
  change scan edge (search edge 4) 5 [] 0 tail3
    (if 4 ≤ rootCandidates2.length then search edge 4 [11] rootCandidates2 11 best else best) = best
  rw [if_pos (by decide), rootBranch2]
  exact outerScan3

end ProjectEulerSolutions.PrimePairUpper.Search
