import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch217
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch218

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan66 : scan edge (search edge 4) 5 [] 0 tail66 best = best := by
  have cons : tail66 = 347 :: tail67 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates66_checked]
  change scan edge (search edge 4) 5 [] 0 tail67
    (if 4 ≤ rootCandidates66.length then search edge 4 [347] rootCandidates66 347 best else best) = best
  rw [if_pos (by decide), rootBranch66]
  exact outerScan67

set_option maxRecDepth 4096 in
theorem outerScan65 : scan edge (search edge 4) 5 [] 0 tail65 best = best := by
  have cons : tail65 = 337 :: tail66 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates65_checked]
  change scan edge (search edge 4) 5 [] 0 tail66
    (if 4 ≤ rootCandidates65.length then search edge 4 [337] rootCandidates65 337 best else best) = best
  rw [if_pos (by decide), rootBranch65]
  exact outerScan66

end ProjectEulerSolutions.PrimePairUpper.Search
