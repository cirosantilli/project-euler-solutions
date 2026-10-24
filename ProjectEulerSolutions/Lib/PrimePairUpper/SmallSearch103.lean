import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch102
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch103

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan382 : scan edge (search edge 4) 5 [] 0 tail382 best = best := by
  have cons : tail382 = 2659 :: tail383 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates382_checked]
  change scan edge (search edge 4) 5 [] 0 tail383
    (if 4 ≤ rootCandidates382.length then search edge 4 [2659] rootCandidates382 2659 best else best) = best
  rw [if_pos (by decide), rootBranch382]
  exact outerScan383

set_option maxRecDepth 4096 in
theorem outerScan381 : scan edge (search edge 4) 5 [] 0 tail381 best = best := by
  have cons : tail381 = 2657 :: tail382 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates381_checked]
  change scan edge (search edge 4) 5 [] 0 tail382
    (if 4 ≤ rootCandidates381.length then search edge 4 [2657] rootCandidates381 2657 best else best) = best
  rw [if_pos (by decide), rootBranch381]
  exact outerScan382

set_option maxRecDepth 4096 in
theorem outerScan380 : scan edge (search edge 4) 5 [] 0 tail380 best = best := by
  have cons : tail380 = 2647 :: tail381 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates380_checked]
  change scan edge (search edge 4) 5 [] 0 tail381
    (if 4 ≤ rootCandidates380.length then search edge 4 [2647] rootCandidates380 2647 best else best) = best
  rw [if_pos (by decide), rootBranch380]
  exact outerScan381

set_option maxRecDepth 4096 in
theorem outerScan379 : scan edge (search edge 4) 5 [] 0 tail379 best = best := by
  have cons : tail379 = 2633 :: tail380 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates379_checked]
  change scan edge (search edge 4) 5 [] 0 tail380
    (if 4 ≤ rootCandidates379.length then search edge 4 [2633] rootCandidates379 2633 best else best) = best
  rw [if_pos (by decide), rootBranch379]
  exact outerScan380

end ProjectEulerSolutions.PrimePairUpper.Search
