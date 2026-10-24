import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch107
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch108

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan364 : scan edge (search edge 4) 5 [] 0 tail364 best = best := by
  have cons : tail364 = 2477 :: tail365 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates364_checked]
  change scan edge (search edge 4) 5 [] 0 tail365
    (if 4 ≤ rootCandidates364.length then search edge 4 [2477] rootCandidates364 2477 best else best) = best
  rw [if_pos (by decide), rootBranch364]
  exact outerScan365

set_option maxRecDepth 4096 in
theorem outerScan363 : scan edge (search edge 4) 5 [] 0 tail363 best = best := by
  have cons : tail363 = 2473 :: tail364 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates363_checked]
  change scan edge (search edge 4) 5 [] 0 tail364
    (if 4 ≤ rootCandidates363.length then search edge 4 [2473] rootCandidates363 2473 best else best) = best
  rw [if_pos (by decide), rootBranch363]
  exact outerScan364

set_option maxRecDepth 4096 in
theorem outerScan362 : scan edge (search edge 4) 5 [] 0 tail362 best = best := by
  have cons : tail362 = 2467 :: tail363 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates362_checked]
  change scan edge (search edge 4) 5 [] 0 tail363
    (if 4 ≤ rootCandidates362.length then search edge 4 [2467] rootCandidates362 2467 best else best) = best
  rw [if_pos (by decide), rootBranch362]
  exact outerScan363

set_option maxRecDepth 4096 in
theorem outerScan361 : scan edge (search edge 4) 5 [] 0 tail361 best = best := by
  have cons : tail361 = 2459 :: tail362 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates361_checked]
  change scan edge (search edge 4) 5 [] 0 tail362
    (if 4 ≤ rootCandidates361.length then search edge 4 [2459] rootCandidates361 2459 best else best) = best
  rw [if_pos (by decide), rootBranch361]
  exact outerScan362

end ProjectEulerSolutions.PrimePairUpper.Search
