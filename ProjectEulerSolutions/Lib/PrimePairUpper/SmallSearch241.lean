import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch240
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch241

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan28 : scan edge (search edge 4) 5 [] 0 tail28 best = best := by
  have cons : tail28 = 127 :: tail29 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates28_checked]
  change scan edge (search edge 4) 5 [] 0 tail29
    (if 4 ≤ rootCandidates28.length then search edge 4 [127] rootCandidates28 127 best else best) = best
  rw [if_pos (by decide), rootBranch28]
  exact outerScan29

end ProjectEulerSolutions.PrimePairUpper.Search
