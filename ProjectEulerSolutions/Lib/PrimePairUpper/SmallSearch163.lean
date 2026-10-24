import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch162
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch163

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan188 : scan edge (search edge 4) 5 [] 0 tail188 best = best := by
  have cons : tail188 = 1153 :: tail189 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates188_checked]
  change scan edge (search edge 4) 5 [] 0 tail189
    (if 4 ≤ rootCandidates188.length then search edge 4 [1153] rootCandidates188 1153 best else best) = best
  rw [if_pos (by decide), rootBranch188]
  exact outerScan189

set_option maxRecDepth 4096 in
theorem outerScan187 : scan edge (search edge 4) 5 [] 0 tail187 best = best := by
  have cons : tail187 = 1151 :: tail188 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates187_checked]
  change scan edge (search edge 4) 5 [] 0 tail188
    (if 4 ≤ rootCandidates187.length then search edge 4 [1151] rootCandidates187 1151 best else best) = best
  rw [if_pos (by decide), rootBranch187]
  exact outerScan188

end ProjectEulerSolutions.PrimePairUpper.Search
