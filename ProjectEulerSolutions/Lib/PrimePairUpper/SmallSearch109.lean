import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch108
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch109

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan360 : scan edge (search edge 4) 5 [] 0 tail360 best = best := by
  have cons : tail360 = 2447 :: tail361 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates360_checked]
  change scan edge (search edge 4) 5 [] 0 tail361
    (if 4 ≤ rootCandidates360.length then search edge 4 [2447] rootCandidates360 2447 best else best) = best
  rw [if_pos (by decide), rootBranch360]
  exact outerScan361

set_option maxRecDepth 4096 in
theorem outerScan359 : scan edge (search edge 4) 5 [] 0 tail359 best = best := by
  have cons : tail359 = 2441 :: tail360 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates359_checked]
  change scan edge (search edge 4) 5 [] 0 tail360
    (if 4 ≤ rootCandidates359.length then search edge 4 [2441] rootCandidates359 2441 best else best) = best
  rw [if_pos (by decide), rootBranch359]
  exact outerScan360

set_option maxRecDepth 4096 in
theorem outerScan358 : scan edge (search edge 4) 5 [] 0 tail358 best = best := by
  have cons : tail358 = 2437 :: tail359 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates358_checked]
  change scan edge (search edge 4) 5 [] 0 tail359
    (if 4 ≤ rootCandidates358.length then search edge 4 [2437] rootCandidates358 2437 best else best) = best
  rw [if_pos (by decide), rootBranch358]
  exact outerScan359

set_option maxRecDepth 4096 in
theorem outerScan357 : scan edge (search edge 4) 5 [] 0 tail357 best = best := by
  have cons : tail357 = 2423 :: tail358 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates357_checked]
  change scan edge (search edge 4) 5 [] 0 tail358
    (if 4 ≤ rootCandidates357.length then search edge 4 [2423] rootCandidates357 2423 best else best) = best
  rw [if_pos (by decide), rootBranch357]
  exact outerScan358

end ProjectEulerSolutions.PrimePairUpper.Search
