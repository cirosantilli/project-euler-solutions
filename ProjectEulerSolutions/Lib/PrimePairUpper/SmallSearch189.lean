import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch188
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch189

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan126 : scan edge (search edge 4) 5 [] 0 tail126 best = best := by
  have cons : tail126 = 727 :: tail127 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates126_checked]
  change scan edge (search edge 4) 5 [] 0 tail127
    (if 4 ≤ rootCandidates126.length then search edge 4 [727] rootCandidates126 727 best else best) = best
  rw [if_pos (by decide), rootBranch126]
  exact outerScan127

set_option maxRecDepth 4096 in
theorem outerScan125 : scan edge (search edge 4) 5 [] 0 tail125 best = best := by
  have cons : tail125 = 719 :: tail126 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates125_checked]
  change scan edge (search edge 4) 5 [] 0 tail126
    (if 4 ≤ rootCandidates125.length then search edge 4 [719] rootCandidates125 719 best else best) = best
  rw [if_pos (by decide), rootBranch125]
  exact outerScan126

end ProjectEulerSolutions.PrimePairUpper.Search
