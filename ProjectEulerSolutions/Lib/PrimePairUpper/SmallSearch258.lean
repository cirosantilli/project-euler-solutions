import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch257
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch258

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan9 : scan edge (search edge 4) 5 [] 0 tail9 best = best := by
  have cons : tail9 = 37 :: tail10 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates9_checked]
  change scan edge (search edge 4) 5 [] 0 tail10
    (if 4 ≤ rootCandidates9.length then search edge 4 [37] rootCandidates9 37 best else best) = best
  rw [if_pos (by decide), rootBranch9]
  exact outerScan10

end ProjectEulerSolutions.PrimePairUpper.Search
