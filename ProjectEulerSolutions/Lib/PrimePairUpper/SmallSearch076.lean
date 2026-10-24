import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch075
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch076

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan497 : scan edge (search edge 4) 5 [] 0 tail497 best = best := by
  have cons : tail497 = 3571 :: tail498 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates497_checked]
  change scan edge (search edge 4) 5 [] 0 tail498
    (if 4 ≤ rootCandidates497.length then search edge 4 [3571] rootCandidates497 3571 best else best) = best
  rw [if_pos (by decide), rootBranch497]
  exact outerScan498

set_option maxRecDepth 4096 in
theorem outerScan496 : scan edge (search edge 4) 5 [] 0 tail496 best = best := by
  have cons : tail496 = 3559 :: tail497 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates496_checked]
  change scan edge (search edge 4) 5 [] 0 tail497
    (if 4 ≤ rootCandidates496.length then search edge 4 [3559] rootCandidates496 3559 best else best) = best
  rw [if_pos (by decide), rootBranch496]
  exact outerScan497

set_option maxRecDepth 4096 in
theorem outerScan495 : scan edge (search edge 4) 5 [] 0 tail495 best = best := by
  have cons : tail495 = 3557 :: tail496 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates495_checked]
  change scan edge (search edge 4) 5 [] 0 tail496
    (if 4 ≤ rootCandidates495.length then search edge 4 [3557] rootCandidates495 3557 best else best) = best
  rw [if_pos (by decide), rootBranch495]
  exact outerScan496

set_option maxRecDepth 4096 in
theorem outerScan494 : scan edge (search edge 4) 5 [] 0 tail494 best = best := by
  have cons : tail494 = 3547 :: tail495 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates494_checked]
  change scan edge (search edge 4) 5 [] 0 tail495
    (if 4 ≤ rootCandidates494.length then search edge 4 [3547] rootCandidates494 3547 best else best) = best
  rw [if_pos (by decide), rootBranch494]
  exact outerScan495

set_option maxRecDepth 4096 in
theorem outerScan493 : scan edge (search edge 4) 5 [] 0 tail493 best = best := by
  have cons : tail493 = 3541 :: tail494 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates493_checked]
  change scan edge (search edge 4) 5 [] 0 tail494
    (if 4 ≤ rootCandidates493.length then search edge 4 [3541] rootCandidates493 3541 best else best) = best
  rw [if_pos (by decide), rootBranch493]
  exact outerScan494

set_option maxRecDepth 4096 in
theorem outerScan492 : scan edge (search edge 4) 5 [] 0 tail492 best = best := by
  have cons : tail492 = 3539 :: tail493 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates492_checked]
  change scan edge (search edge 4) 5 [] 0 tail493
    (if 4 ≤ rootCandidates492.length then search edge 4 [3539] rootCandidates492 3539 best else best) = best
  rw [if_pos (by decide), rootBranch492]
  exact outerScan493

end ProjectEulerSolutions.PrimePairUpper.Search
