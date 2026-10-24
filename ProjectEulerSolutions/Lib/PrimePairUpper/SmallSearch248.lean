import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch247
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch248

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan19 : scan edge (search edge 4) 5 [] 0 tail19 best = best := by
  have cons : tail19 = 79 :: tail20 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates19_checked]
  change scan edge (search edge 4) 5 [] 0 tail20
    (if 4 ≤ rootCandidates19.length then search edge 4 [79] rootCandidates19 79 best else best) = best
  rw [if_pos (by decide), rootBranch19]
  exact outerScan20

end ProjectEulerSolutions.PrimePairUpper.Search
