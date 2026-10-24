import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch210
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch211

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan80 : scan edge (search edge 4) 5 [] 0 tail80 best = best := by
  have cons : tail80 = 431 :: tail81 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates80_checked]
  change scan edge (search edge 4) 5 [] 0 tail81
    (if 4 ≤ rootCandidates80.length then search edge 4 [431] rootCandidates80 431 best else best) = best
  rw [if_pos (by decide), rootBranch80]
  exact outerScan81

set_option maxRecDepth 4096 in
theorem outerScan79 : scan edge (search edge 4) 5 [] 0 tail79 best = best := by
  have cons : tail79 = 421 :: tail80 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates79_checked]
  change scan edge (search edge 4) 5 [] 0 tail80
    (if 4 ≤ rootCandidates79.length then search edge 4 [421] rootCandidates79 421 best else best) = best
  rw [if_pos (by decide), rootBranch79]
  exact outerScan80

end ProjectEulerSolutions.PrimePairUpper.Search
