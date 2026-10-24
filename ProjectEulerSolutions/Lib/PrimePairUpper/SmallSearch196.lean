import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch195
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch196

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan110 : scan edge (search edge 4) 5 [] 0 tail110 best = best := by
  have cons : tail110 = 617 :: tail111 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates110_checked]
  change scan edge (search edge 4) 5 [] 0 tail111
    (if 4 ≤ rootCandidates110.length then search edge 4 [617] rootCandidates110 617 best else best) = best
  rw [if_pos (by decide), rootBranch110]
  exact outerScan111

set_option maxRecDepth 4096 in
theorem outerScan109 : scan edge (search edge 4) 5 [] 0 tail109 best = best := by
  have cons : tail109 = 613 :: tail110 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates109_checked]
  change scan edge (search edge 4) 5 [] 0 tail110
    (if 4 ≤ rootCandidates109.length then search edge 4 [613] rootCandidates109 613 best else best) = best
  rw [if_pos (by decide), rootBranch109]
  exact outerScan110

end ProjectEulerSolutions.PrimePairUpper.Search
