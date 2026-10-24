import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch158
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch159

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan198 : scan edge (search edge 4) 5 [] 0 tail198 best = best := by
  have cons : tail198 = 1229 :: tail199 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates198_checked]
  change scan edge (search edge 4) 5 [] 0 tail199
    (if 4 ≤ rootCandidates198.length then search edge 4 [1229] rootCandidates198 1229 best else best) = best
  rw [if_pos (by decide), rootBranch198]
  exact outerScan199

set_option maxRecDepth 4096 in
theorem outerScan197 : scan edge (search edge 4) 5 [] 0 tail197 best = best := by
  have cons : tail197 = 1223 :: tail198 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates197_checked]
  change scan edge (search edge 4) 5 [] 0 tail198
    (if 4 ≤ rootCandidates197.length then search edge 4 [1223] rootCandidates197 1223 best else best) = best
  rw [if_pos (by decide), rootBranch197]
  exact outerScan198

end ProjectEulerSolutions.PrimePairUpper.Search
