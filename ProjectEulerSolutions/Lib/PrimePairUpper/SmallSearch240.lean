import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch239
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch240

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan29 : scan edge (search edge 4) 5 [] 0 tail29 best = best := by
  have cons : tail29 = 131 :: tail30 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates29_checked]
  change scan edge (search edge 4) 5 [] 0 tail30
    (if 4 ≤ rootCandidates29.length then search edge 4 [131] rootCandidates29 131 best else best) = best
  rw [if_pos (by decide), rootBranch29]
  exact outerScan30

end ProjectEulerSolutions.PrimePairUpper.Search
