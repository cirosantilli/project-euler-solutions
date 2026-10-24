import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch097
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch098

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan402 : scan edge (search edge 4) 5 [] 0 tail402 best = best := by
  have cons : tail402 = 2789 :: tail403 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates402_checked]
  change scan edge (search edge 4) 5 [] 0 tail403
    (if 4 ≤ rootCandidates402.length then search edge 4 [2789] rootCandidates402 2789 best else best) = best
  rw [if_pos (by decide), rootBranch402]
  exact outerScan403

set_option maxRecDepth 4096 in
theorem outerScan401 : scan edge (search edge 4) 5 [] 0 tail401 best = best := by
  have cons : tail401 = 2777 :: tail402 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates401_checked]
  change scan edge (search edge 4) 5 [] 0 tail402
    (if 4 ≤ rootCandidates401.length then search edge 4 [2777] rootCandidates401 2777 best else best) = best
  rw [if_pos (by decide), rootBranch401]
  exact outerScan402

set_option maxRecDepth 4096 in
theorem outerScan400 : scan edge (search edge 4) 5 [] 0 tail400 best = best := by
  have cons : tail400 = 2767 :: tail401 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates400_checked]
  change scan edge (search edge 4) 5 [] 0 tail401
    (if 4 ≤ rootCandidates400.length then search edge 4 [2767] rootCandidates400 2767 best else best) = best
  rw [if_pos (by decide), rootBranch400]
  exact outerScan401

end ProjectEulerSolutions.PrimePairUpper.Search
