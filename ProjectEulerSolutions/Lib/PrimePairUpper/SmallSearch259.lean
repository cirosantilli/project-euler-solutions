import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch258
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch259

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan8 : scan edge (search edge 4) 5 [] 0 tail8 best = best := by
  have cons : tail8 = 31 :: tail9 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates8_checked]
  change scan edge (search edge 4) 5 [] 0 tail9
    (if 4 ≤ rootCandidates8.length then search edge 4 [31] rootCandidates8 31 best else best) = best
  rw [if_pos (by decide), rootBranch8]
  exact outerScan9

end ProjectEulerSolutions.PrimePairUpper.Search
