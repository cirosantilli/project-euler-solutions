import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch237
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch238

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan31 : scan edge (search edge 4) 5 [] 0 tail31 best = best := by
  have cons : tail31 = 139 :: tail32 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates31_checked]
  change scan edge (search edge 4) 5 [] 0 tail32
    (if 4 ≤ rootCandidates31.length then search edge 4 [139] rootCandidates31 139 best else best) = best
  rw [if_pos (by decide), rootBranch31]
  exact outerScan32

end ProjectEulerSolutions.PrimePairUpper.Search
