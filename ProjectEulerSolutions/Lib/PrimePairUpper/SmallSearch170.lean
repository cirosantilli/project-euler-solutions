import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch169
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch170

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan172 : scan edge (search edge 4) 5 [] 0 tail172 best = best := by
  have cons : tail172 = 1039 :: tail173 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates172_checked]
  change scan edge (search edge 4) 5 [] 0 tail173
    (if 4 ≤ rootCandidates172.length then search edge 4 [1039] rootCandidates172 1039 best else best) = best
  rw [if_pos (by decide), rootBranch172]
  exact outerScan173

set_option maxRecDepth 4096 in
theorem outerScan171 : scan edge (search edge 4) 5 [] 0 tail171 best = best := by
  have cons : tail171 = 1033 :: tail172 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates171_checked]
  change scan edge (search edge 4) 5 [] 0 tail172
    (if 4 ≤ rootCandidates171.length then search edge 4 [1033] rootCandidates171 1033 best else best) = best
  rw [if_pos (by decide), rootBranch171]
  exact outerScan172

set_option maxRecDepth 4096 in
theorem outerScan170 : scan edge (search edge 4) 5 [] 0 tail170 best = best := by
  have cons : tail170 = 1031 :: tail171 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates170_checked]
  change scan edge (search edge 4) 5 [] 0 tail171
    (if 4 ≤ rootCandidates170.length then search edge 4 [1031] rootCandidates170 1031 best else best) = best
  rw [if_pos (by decide), rootBranch170]
  exact outerScan171

end ProjectEulerSolutions.PrimePairUpper.Search
