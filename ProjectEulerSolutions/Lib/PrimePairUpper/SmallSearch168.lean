import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch167
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch168

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan176 : scan edge (search edge 4) 5 [] 0 tail176 best = best := by
  have cons : tail176 = 1063 :: tail177 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates176_checked]
  change scan edge (search edge 4) 5 [] 0 tail177
    (if 4 ≤ rootCandidates176.length then search edge 4 [1063] rootCandidates176 1063 best else best) = best
  rw [if_pos (by decide), rootBranch176]
  exact outerScan177

set_option maxRecDepth 4096 in
theorem outerScan175 : scan edge (search edge 4) 5 [] 0 tail175 best = best := by
  have cons : tail175 = 1061 :: tail176 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates175_checked]
  change scan edge (search edge 4) 5 [] 0 tail176
    (if 4 ≤ rootCandidates175.length then search edge 4 [1061] rootCandidates175 1061 best else best) = best
  rw [if_pos (by decide), rootBranch175]
  exact outerScan176

end ProjectEulerSolutions.PrimePairUpper.Search
