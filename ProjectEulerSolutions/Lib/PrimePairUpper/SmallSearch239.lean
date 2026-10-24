import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch238
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch239

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan30 : scan edge (search edge 4) 5 [] 0 tail30 best = best := by
  have cons : tail30 = 137 :: tail31 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates30_checked]
  change scan edge (search edge 4) 5 [] 0 tail31
    (if 4 ≤ rootCandidates30.length then search edge 4 [137] rootCandidates30 137 best else best) = best
  rw [if_pos (by decide), rootBranch30]
  exact outerScan31

end ProjectEulerSolutions.PrimePairUpper.Search
