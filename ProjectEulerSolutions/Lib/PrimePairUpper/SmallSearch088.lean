import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch087
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch088

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan442 : scan edge (search edge 4) 5 [] 0 tail442 best = best := by
  have cons : tail442 = 3121 :: tail443 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates442_checked]
  change scan edge (search edge 4) 5 [] 0 tail443
    (if 4 ≤ rootCandidates442.length then search edge 4 [3121] rootCandidates442 3121 best else best) = best
  rw [if_pos (by decide), rootBranch442]
  exact outerScan443

set_option maxRecDepth 4096 in
theorem outerScan441 : scan edge (search edge 4) 5 [] 0 tail441 best = best := by
  have cons : tail441 = 3119 :: tail442 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates441_checked]
  change scan edge (search edge 4) 5 [] 0 tail442
    (if 4 ≤ rootCandidates441.length then search edge 4 [3119] rootCandidates441 3119 best else best) = best
  rw [if_pos (by decide), rootBranch441]
  exact outerScan442

set_option maxRecDepth 4096 in
theorem outerScan440 : scan edge (search edge 4) 5 [] 0 tail440 best = best := by
  have cons : tail440 = 3109 :: tail441 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates440_checked]
  change scan edge (search edge 4) 5 [] 0 tail441
    (if 4 ≤ rootCandidates440.length then search edge 4 [3109] rootCandidates440 3109 best else best) = best
  rw [if_pos (by decide), rootBranch440]
  exact outerScan441

set_option maxRecDepth 4096 in
theorem outerScan439 : scan edge (search edge 4) 5 [] 0 tail439 best = best := by
  have cons : tail439 = 3089 :: tail440 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates439_checked]
  change scan edge (search edge 4) 5 [] 0 tail440
    (if 4 ≤ rootCandidates439.length then search edge 4 [3089] rootCandidates439 3089 best else best) = best
  rw [if_pos (by decide), rootBranch439]
  exact outerScan440

end ProjectEulerSolutions.PrimePairUpper.Search
