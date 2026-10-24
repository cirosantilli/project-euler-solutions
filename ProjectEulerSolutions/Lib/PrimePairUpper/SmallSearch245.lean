import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch244
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch245

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan22 : scan edge (search edge 4) 5 [] 0 tail22 best = best := by
  have cons : tail22 = 97 :: tail23 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates22_checked]
  change scan edge (search edge 4) 5 [] 0 tail23
    (if 4 ≤ rootCandidates22.length then search edge 4 [97] rootCandidates22 97 best else best) = best
  rw [if_pos (by decide), rootBranch22]
  exact outerScan23

end ProjectEulerSolutions.PrimePairUpper.Search
