import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch242
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch243

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan25 : scan edge (search edge 4) 5 [] 0 tail25 best = best := by
  have cons : tail25 = 107 :: tail26 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates25_checked]
  change scan edge (search edge 4) 5 [] 0 tail26
    (if 4 ≤ rootCandidates25.length then search edge 4 [107] rootCandidates25 107 best else best) = best
  rw [if_pos (by decide), rootBranch25]
  exact outerScan26

set_option maxRecDepth 4096 in
theorem outerScan24 : scan edge (search edge 4) 5 [] 0 tail24 best = best := by
  have cons : tail24 = 103 :: tail25 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates24_checked]
  change scan edge (search edge 4) 5 [] 0 tail25
    (if 4 ≤ rootCandidates24.length then search edge 4 [103] rootCandidates24 103 best else best) = best
  rw [if_pos (by decide), rootBranch24]
  exact outerScan25

end ProjectEulerSolutions.PrimePairUpper.Search
