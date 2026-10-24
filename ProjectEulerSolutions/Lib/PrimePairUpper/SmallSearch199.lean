import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch198
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch199

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan104 : scan edge (search edge 4) 5 [] 0 tail104 best = best := by
  have cons : tail104 = 587 :: tail105 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates104_checked]
  change scan edge (search edge 4) 5 [] 0 tail105
    (if 4 ≤ rootCandidates104.length then search edge 4 [587] rootCandidates104 587 best else best) = best
  rw [if_pos (by decide), rootBranch104]
  exact outerScan105

set_option maxRecDepth 4096 in
theorem outerScan103 : scan edge (search edge 4) 5 [] 0 tail103 best = best := by
  have cons : tail103 = 577 :: tail104 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates103_checked]
  change scan edge (search edge 4) 5 [] 0 tail104
    (if 4 ≤ rootCandidates103.length then search edge 4 [577] rootCandidates103 577 best else best) = best
  rw [if_pos (by decide), rootBranch103]
  exact outerScan104

end ProjectEulerSolutions.PrimePairUpper.Search
