import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch186
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch187

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan132 : scan edge (search edge 4) 5 [] 0 tail132 best = best := by
  have cons : tail132 = 761 :: tail133 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates132_checked]
  change scan edge (search edge 4) 5 [] 0 tail133
    (if 4 ≤ rootCandidates132.length then search edge 4 [761] rootCandidates132 761 best else best) = best
  rw [if_pos (by decide), rootBranch132]
  exact outerScan133

set_option maxRecDepth 4096 in
theorem outerScan131 : scan edge (search edge 4) 5 [] 0 tail131 best = best := by
  have cons : tail131 = 757 :: tail132 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates131_checked]
  change scan edge (search edge 4) 5 [] 0 tail132
    (if 4 ≤ rootCandidates131.length then search edge 4 [757] rootCandidates131 757 best else best) = best
  rw [if_pos (by decide), rootBranch131]
  exact outerScan132

set_option maxRecDepth 4096 in
theorem outerScan130 : scan edge (search edge 4) 5 [] 0 tail130 best = best := by
  have cons : tail130 = 751 :: tail131 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates130_checked]
  change scan edge (search edge 4) 5 [] 0 tail131
    (if 4 ≤ rootCandidates130.length then search edge 4 [751] rootCandidates130 751 best else best) = best
  rw [if_pos (by decide), rootBranch130]
  exact outerScan131

end ProjectEulerSolutions.PrimePairUpper.Search
