import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch225
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch226

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan51 : scan edge (search edge 4) 5 [] 0 tail51 best = best := by
  have cons : tail51 = 251 :: tail52 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates51_checked]
  change scan edge (search edge 4) 5 [] 0 tail52
    (if 4 ≤ rootCandidates51.length then search edge 4 [251] rootCandidates51 251 best else best) = best
  rw [if_pos (by decide), rootBranch51]
  exact outerScan52

end ProjectEulerSolutions.PrimePairUpper.Search
