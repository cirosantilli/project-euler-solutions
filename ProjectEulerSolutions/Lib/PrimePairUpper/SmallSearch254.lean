import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch253
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch254

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan13 : scan edge (search edge 4) 5 [] 0 tail13 best = best := by
  have cons : tail13 = 53 :: tail14 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates13_checked]
  change scan edge (search edge 4) 5 [] 0 tail14
    (if 4 ≤ rootCandidates13.length then search edge 4 [53] rootCandidates13 53 best else best) = best
  rw [if_pos (by decide), rootBranch13]
  exact outerScan14

end ProjectEulerSolutions.PrimePairUpper.Search
