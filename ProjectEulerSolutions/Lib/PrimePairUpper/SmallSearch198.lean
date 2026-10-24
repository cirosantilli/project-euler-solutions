import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch197
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch198

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan106 : scan edge (search edge 4) 5 [] 0 tail106 best = best := by
  have cons : tail106 = 599 :: tail107 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates106_checked]
  change scan edge (search edge 4) 5 [] 0 tail107
    (if 4 ≤ rootCandidates106.length then search edge 4 [599] rootCandidates106 599 best else best) = best
  rw [if_pos (by decide), rootBranch106]
  exact outerScan107

set_option maxRecDepth 4096 in
theorem outerScan105 : scan edge (search edge 4) 5 [] 0 tail105 best = best := by
  have cons : tail105 = 593 :: tail106 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates105_checked]
  change scan edge (search edge 4) 5 [] 0 tail106
    (if 4 ≤ rootCandidates105.length then search edge 4 [593] rootCandidates105 593 best else best) = best
  rw [if_pos (by decide), rootBranch105]
  exact outerScan106

end ProjectEulerSolutions.PrimePairUpper.Search
