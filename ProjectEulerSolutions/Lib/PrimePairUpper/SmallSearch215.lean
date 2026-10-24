import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch214
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch215

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan72 : scan edge (search edge 4) 5 [] 0 tail72 best = best := by
  have cons : tail72 = 379 :: tail73 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates72_checked]
  change scan edge (search edge 4) 5 [] 0 tail73
    (if 4 ≤ rootCandidates72.length then search edge 4 [379] rootCandidates72 379 best else best) = best
  rw [if_pos (by decide), rootBranch72]
  exact outerScan73

set_option maxRecDepth 4096 in
theorem outerScan71 : scan edge (search edge 4) 5 [] 0 tail71 best = best := by
  have cons : tail71 = 373 :: tail72 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates71_checked]
  change scan edge (search edge 4) 5 [] 0 tail72
    (if 4 ≤ rootCandidates71.length then search edge 4 [373] rootCandidates71 373 best else best) = best
  rw [if_pos (by decide), rootBranch71]
  exact outerScan72

end ProjectEulerSolutions.PrimePairUpper.Search
