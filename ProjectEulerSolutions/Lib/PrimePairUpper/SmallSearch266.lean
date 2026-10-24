import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch265
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch266

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan1 : scan edge (search edge 4) 5 [] 0 tail1 best = best := by
  have cons : tail1 = 7 :: tail2 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1_checked]
  change scan edge (search edge 4) 5 [] 0 tail2
    (if 4 ≤ rootCandidates1.length then search edge 4 [7] rootCandidates1 7 best else best) = best
  rw [if_pos (by decide), rootBranch1]
  exact outerScan2

end ProjectEulerSolutions.PrimePairUpper.Search
