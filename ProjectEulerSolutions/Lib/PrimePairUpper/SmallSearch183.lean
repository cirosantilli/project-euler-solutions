import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch182
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch183

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan141 : scan edge (search edge 4) 5 [] 0 tail141 best = best := by
  have cons : tail141 = 827 :: tail142 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates141_checked]
  change scan edge (search edge 4) 5 [] 0 tail142
    (if 4 ≤ rootCandidates141.length then search edge 4 [827] rootCandidates141 827 best else best) = best
  rw [if_pos (by decide), rootBranch141]
  exact outerScan142

set_option maxRecDepth 4096 in
theorem outerScan140 : scan edge (search edge 4) 5 [] 0 tail140 best = best := by
  have cons : tail140 = 823 :: tail141 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates140_checked]
  change scan edge (search edge 4) 5 [] 0 tail141
    (if 4 ≤ rootCandidates140.length then search edge 4 [823] rootCandidates140 823 best else best) = best
  rw [if_pos (by decide), rootBranch140]
  exact outerScan141

end ProjectEulerSolutions.PrimePairUpper.Search
