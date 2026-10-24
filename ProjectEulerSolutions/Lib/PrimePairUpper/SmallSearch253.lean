import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch252
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch253

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan14 : scan edge (search edge 4) 5 [] 0 tail14 best = best := by
  have cons : tail14 = 59 :: tail15 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates14_checked]
  change scan edge (search edge 4) 5 [] 0 tail15
    (if 4 ≤ rootCandidates14.length then search edge 4 [59] rootCandidates14 59 best else best) = best
  rw [if_pos (by decide), rootBranch14]
  exact outerScan15

end ProjectEulerSolutions.PrimePairUpper.Search
