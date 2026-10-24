import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch221
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch222

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan58 : scan edge (search edge 4) 5 [] 0 tail58 best = best := by
  have cons : tail58 = 283 :: tail59 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates58_checked]
  change scan edge (search edge 4) 5 [] 0 tail59
    (if 4 ≤ rootCandidates58.length then search edge 4 [283] rootCandidates58 283 best else best) = best
  rw [if_pos (by decide), rootBranch58]
  exact outerScan59

set_option maxRecDepth 4096 in
theorem outerScan57 : scan edge (search edge 4) 5 [] 0 tail57 best = best := by
  have cons : tail57 = 281 :: tail58 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates57_checked]
  change scan edge (search edge 4) 5 [] 0 tail58
    (if 4 ≤ rootCandidates57.length then search edge 4 [281] rootCandidates57 281 best else best) = best
  rw [if_pos (by decide), rootBranch57]
  exact outerScan58

end ProjectEulerSolutions.PrimePairUpper.Search
