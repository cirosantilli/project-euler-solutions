import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch076
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch077

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan491 : scan edge (search edge 4) 5 [] 0 tail491 best = best := by
  have cons : tail491 = 3533 :: tail492 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates491_checked]
  change scan edge (search edge 4) 5 [] 0 tail492
    (if 4 ≤ rootCandidates491.length then search edge 4 [3533] rootCandidates491 3533 best else best) = best
  rw [if_pos (by decide), rootBranch491]
  exact outerScan492

set_option maxRecDepth 4096 in
theorem outerScan490 : scan edge (search edge 4) 5 [] 0 tail490 best = best := by
  have cons : tail490 = 3529 :: tail491 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates490_checked]
  change scan edge (search edge 4) 5 [] 0 tail491
    (if 4 ≤ rootCandidates490.length then search edge 4 [3529] rootCandidates490 3529 best else best) = best
  rw [if_pos (by decide), rootBranch490]
  exact outerScan491

set_option maxRecDepth 4096 in
theorem outerScan489 : scan edge (search edge 4) 5 [] 0 tail489 best = best := by
  have cons : tail489 = 3527 :: tail490 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates489_checked]
  change scan edge (search edge 4) 5 [] 0 tail490
    (if 4 ≤ rootCandidates489.length then search edge 4 [3527] rootCandidates489 3527 best else best) = best
  rw [if_pos (by decide), rootBranch489]
  exact outerScan490

set_option maxRecDepth 4096 in
theorem outerScan488 : scan edge (search edge 4) 5 [] 0 tail488 best = best := by
  have cons : tail488 = 3517 :: tail489 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates488_checked]
  change scan edge (search edge 4) 5 [] 0 tail489
    (if 4 ≤ rootCandidates488.length then search edge 4 [3517] rootCandidates488 3517 best else best) = best
  rw [if_pos (by decide), rootBranch488]
  exact outerScan489

set_option maxRecDepth 4096 in
theorem outerScan487 : scan edge (search edge 4) 5 [] 0 tail487 best = best := by
  have cons : tail487 = 3511 :: tail488 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates487_checked]
  change scan edge (search edge 4) 5 [] 0 tail488
    (if 4 ≤ rootCandidates487.length then search edge 4 [3511] rootCandidates487 3511 best else best) = best
  rw [if_pos (by decide), rootBranch487]
  exact outerScan488

end ProjectEulerSolutions.PrimePairUpper.Search
