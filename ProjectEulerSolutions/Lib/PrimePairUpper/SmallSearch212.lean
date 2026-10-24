import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch211
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch212

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan78 : scan edge (search edge 4) 5 [] 0 tail78 best = best := by
  have cons : tail78 = 419 :: tail79 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates78_checked]
  change scan edge (search edge 4) 5 [] 0 tail79
    (if 4 ≤ rootCandidates78.length then search edge 4 [419] rootCandidates78 419 best else best) = best
  rw [if_pos (by decide), rootBranch78]
  exact outerScan79

set_option maxRecDepth 4096 in
theorem outerScan77 : scan edge (search edge 4) 5 [] 0 tail77 best = best := by
  have cons : tail77 = 409 :: tail78 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates77_checked]
  change scan edge (search edge 4) 5 [] 0 tail78
    (if 4 ≤ rootCandidates77.length then search edge 4 [409] rootCandidates77 409 best else best) = best
  rw [if_pos (by decide), rootBranch77]
  exact outerScan78

end ProjectEulerSolutions.PrimePairUpper.Search
