import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch215
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch216

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan70 : scan edge (search edge 4) 5 [] 0 tail70 best = best := by
  have cons : tail70 = 367 :: tail71 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates70_checked]
  change scan edge (search edge 4) 5 [] 0 tail71
    (if 4 ≤ rootCandidates70.length then search edge 4 [367] rootCandidates70 367 best else best) = best
  rw [if_pos (by decide), rootBranch70]
  exact outerScan71

set_option maxRecDepth 4096 in
theorem outerScan69 : scan edge (search edge 4) 5 [] 0 tail69 best = best := by
  have cons : tail69 = 359 :: tail70 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates69_checked]
  change scan edge (search edge 4) 5 [] 0 tail70
    (if 4 ≤ rootCandidates69.length then search edge 4 [359] rootCandidates69 359 best else best) = best
  rw [if_pos (by decide), rootBranch69]
  exact outerScan70

end ProjectEulerSolutions.PrimePairUpper.Search
