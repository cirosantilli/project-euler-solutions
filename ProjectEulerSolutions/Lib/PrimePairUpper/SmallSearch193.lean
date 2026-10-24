import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch192
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch193

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan118 : scan edge (search edge 4) 5 [] 0 tail118 best = best := by
  have cons : tail118 = 661 :: tail119 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates118_checked]
  change scan edge (search edge 4) 5 [] 0 tail119
    (if 4 ≤ rootCandidates118.length then search edge 4 [661] rootCandidates118 661 best else best) = best
  rw [if_pos (by decide), rootBranch118]
  exact outerScan119

set_option maxRecDepth 4096 in
theorem outerScan117 : scan edge (search edge 4) 5 [] 0 tail117 best = best := by
  have cons : tail117 = 659 :: tail118 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates117_checked]
  change scan edge (search edge 4) 5 [] 0 tail118
    (if 4 ≤ rootCandidates117.length then search edge 4 [659] rootCandidates117 659 best else best) = best
  rw [if_pos (by decide), rootBranch117]
  exact outerScan118

end ProjectEulerSolutions.PrimePairUpper.Search
