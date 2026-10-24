import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch078
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch079

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan482 : scan edge (search edge 4) 5 [] 0 tail482 best = best := by
  have cons : tail482 = 3463 :: tail483 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates482_checked]
  change scan edge (search edge 4) 5 [] 0 tail483
    (if 4 ≤ rootCandidates482.length then search edge 4 [3463] rootCandidates482 3463 best else best) = best
  rw [if_pos (by decide), rootBranch482]
  exact outerScan483

set_option maxRecDepth 4096 in
theorem outerScan481 : scan edge (search edge 4) 5 [] 0 tail481 best = best := by
  have cons : tail481 = 3461 :: tail482 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates481_checked]
  change scan edge (search edge 4) 5 [] 0 tail482
    (if 4 ≤ rootCandidates481.length then search edge 4 [3461] rootCandidates481 3461 best else best) = best
  rw [if_pos (by decide), rootBranch481]
  exact outerScan482

set_option maxRecDepth 4096 in
theorem outerScan480 : scan edge (search edge 4) 5 [] 0 tail480 best = best := by
  have cons : tail480 = 3457 :: tail481 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates480_checked]
  change scan edge (search edge 4) 5 [] 0 tail481
    (if 4 ≤ rootCandidates480.length then search edge 4 [3457] rootCandidates480 3457 best else best) = best
  rw [if_pos (by decide), rootBranch480]
  exact outerScan481

set_option maxRecDepth 4096 in
theorem outerScan479 : scan edge (search edge 4) 5 [] 0 tail479 best = best := by
  have cons : tail479 = 3449 :: tail480 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates479_checked]
  change scan edge (search edge 4) 5 [] 0 tail480
    (if 4 ≤ rootCandidates479.length then search edge 4 [3449] rootCandidates479 3449 best else best) = best
  rw [if_pos (by decide), rootBranch479]
  exact outerScan480

end ProjectEulerSolutions.PrimePairUpper.Search
