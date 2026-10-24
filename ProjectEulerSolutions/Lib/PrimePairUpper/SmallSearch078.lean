import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch077
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch078

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan486 : scan edge (search edge 4) 5 [] 0 tail486 best = best := by
  have cons : tail486 = 3499 :: tail487 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates486_checked]
  change scan edge (search edge 4) 5 [] 0 tail487
    (if 4 ≤ rootCandidates486.length then search edge 4 [3499] rootCandidates486 3499 best else best) = best
  rw [if_pos (by decide), rootBranch486]
  exact outerScan487

set_option maxRecDepth 4096 in
theorem outerScan485 : scan edge (search edge 4) 5 [] 0 tail485 best = best := by
  have cons : tail485 = 3491 :: tail486 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates485_checked]
  change scan edge (search edge 4) 5 [] 0 tail486
    (if 4 ≤ rootCandidates485.length then search edge 4 [3491] rootCandidates485 3491 best else best) = best
  rw [if_pos (by decide), rootBranch485]
  exact outerScan486

set_option maxRecDepth 4096 in
theorem outerScan484 : scan edge (search edge 4) 5 [] 0 tail484 best = best := by
  have cons : tail484 = 3469 :: tail485 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates484_checked]
  change scan edge (search edge 4) 5 [] 0 tail485
    (if 4 ≤ rootCandidates484.length then search edge 4 [3469] rootCandidates484 3469 best else best) = best
  rw [if_pos (by decide), rootBranch484]
  exact outerScan485

set_option maxRecDepth 4096 in
theorem outerScan483 : scan edge (search edge 4) 5 [] 0 tail483 best = best := by
  have cons : tail483 = 3467 :: tail484 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates483_checked]
  change scan edge (search edge 4) 5 [] 0 tail484
    (if 4 ≤ rootCandidates483.length then search edge 4 [3467] rootCandidates483 3467 best else best) = best
  rw [if_pos (by decide), rootBranch483]
  exact outerScan484

end ProjectEulerSolutions.PrimePairUpper.Search
