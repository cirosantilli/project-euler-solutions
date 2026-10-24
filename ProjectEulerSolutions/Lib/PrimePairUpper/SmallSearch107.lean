import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch106
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch107

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan368 : scan edge (search edge 4) 5 [] 0 tail368 best = best := by
  have cons : tail368 = 2539 :: tail369 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates368_checked]
  change scan edge (search edge 4) 5 [] 0 tail369
    (if 4 ≤ rootCandidates368.length then search edge 4 [2539] rootCandidates368 2539 best else best) = best
  rw [if_pos (by decide), rootBranch368]
  exact outerScan369

set_option maxRecDepth 4096 in
theorem outerScan367 : scan edge (search edge 4) 5 [] 0 tail367 best = best := by
  have cons : tail367 = 2531 :: tail368 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates367_checked]
  change scan edge (search edge 4) 5 [] 0 tail368
    (if 4 ≤ rootCandidates367.length then search edge 4 [2531] rootCandidates367 2531 best else best) = best
  rw [if_pos (by decide), rootBranch367]
  exact outerScan368

set_option maxRecDepth 4096 in
theorem outerScan366 : scan edge (search edge 4) 5 [] 0 tail366 best = best := by
  have cons : tail366 = 2521 :: tail367 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates366_checked]
  change scan edge (search edge 4) 5 [] 0 tail367
    (if 4 ≤ rootCandidates366.length then search edge 4 [2521] rootCandidates366 2521 best else best) = best
  rw [if_pos (by decide), rootBranch366]
  exact outerScan367

set_option maxRecDepth 4096 in
theorem outerScan365 : scan edge (search edge 4) 5 [] 0 tail365 best = best := by
  have cons : tail365 = 2503 :: tail366 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates365_checked]
  change scan edge (search edge 4) 5 [] 0 tail366
    (if 4 ≤ rootCandidates365.length then search edge 4 [2503] rootCandidates365 2503 best else best) = best
  rw [if_pos (by decide), rootBranch365]
  exact outerScan366

end ProjectEulerSolutions.PrimePairUpper.Search
