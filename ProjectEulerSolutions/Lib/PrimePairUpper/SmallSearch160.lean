import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch159
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch160

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan196 : scan edge (search edge 4) 5 [] 0 tail196 best = best := by
  have cons : tail196 = 1217 :: tail197 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates196_checked]
  change scan edge (search edge 4) 5 [] 0 tail197
    (if 4 ≤ rootCandidates196.length then search edge 4 [1217] rootCandidates196 1217 best else best) = best
  rw [if_pos (by decide), rootBranch196]
  exact outerScan197

set_option maxRecDepth 4096 in
theorem outerScan195 : scan edge (search edge 4) 5 [] 0 tail195 best = best := by
  have cons : tail195 = 1213 :: tail196 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates195_checked]
  change scan edge (search edge 4) 5 [] 0 tail196
    (if 4 ≤ rootCandidates195.length then search edge 4 [1213] rootCandidates195 1213 best else best) = best
  rw [if_pos (by decide), rootBranch195]
  exact outerScan196

end ProjectEulerSolutions.PrimePairUpper.Search
