import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch184
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch185

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan137 : scan edge (search edge 4) 5 [] 0 tail137 best = best := by
  have cons : tail137 = 809 :: tail138 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates137_checked]
  change scan edge (search edge 4) 5 [] 0 tail138
    (if 4 ≤ rootCandidates137.length then search edge 4 [809] rootCandidates137 809 best else best) = best
  rw [if_pos (by decide), rootBranch137]
  exact outerScan138

set_option maxRecDepth 4096 in
theorem outerScan136 : scan edge (search edge 4) 5 [] 0 tail136 best = best := by
  have cons : tail136 = 797 :: tail137 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates136_checked]
  change scan edge (search edge 4) 5 [] 0 tail137
    (if 4 ≤ rootCandidates136.length then search edge 4 [797] rootCandidates136 797 best else best) = best
  rw [if_pos (by decide), rootBranch136]
  exact outerScan137

end ProjectEulerSolutions.PrimePairUpper.Search
