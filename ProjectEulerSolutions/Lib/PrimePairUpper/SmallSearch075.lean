import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch074
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch075

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan501 : scan edge (search edge 4) 5 [] 0 tail501 best = best := by
  have cons : tail501 = 3607 :: tail502 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates501_checked]
  change scan edge (search edge 4) 5 [] 0 tail502
    (if 4 ≤ rootCandidates501.length then search edge 4 [3607] rootCandidates501 3607 best else best) = best
  rw [if_pos (by decide), rootBranch501]
  exact outerScan502

set_option maxRecDepth 4096 in
theorem outerScan500 : scan edge (search edge 4) 5 [] 0 tail500 best = best := by
  have cons : tail500 = 3593 :: tail501 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates500_checked]
  change scan edge (search edge 4) 5 [] 0 tail501
    (if 4 ≤ rootCandidates500.length then search edge 4 [3593] rootCandidates500 3593 best else best) = best
  rw [if_pos (by decide), rootBranch500]
  exact outerScan501

set_option maxRecDepth 4096 in
theorem outerScan499 : scan edge (search edge 4) 5 [] 0 tail499 best = best := by
  have cons : tail499 = 3583 :: tail500 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates499_checked]
  change scan edge (search edge 4) 5 [] 0 tail500
    (if 4 ≤ rootCandidates499.length then search edge 4 [3583] rootCandidates499 3583 best else best) = best
  rw [if_pos (by decide), rootBranch499]
  exact outerScan500

set_option maxRecDepth 4096 in
theorem outerScan498 : scan edge (search edge 4) 5 [] 0 tail498 best = best := by
  have cons : tail498 = 3581 :: tail499 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates498_checked]
  change scan edge (search edge 4) 5 [] 0 tail499
    (if 4 ≤ rootCandidates498.length then search edge 4 [3581] rootCandidates498 3581 best else best) = best
  rw [if_pos (by decide), rootBranch498]
  exact outerScan499

end ProjectEulerSolutions.PrimePairUpper.Search
