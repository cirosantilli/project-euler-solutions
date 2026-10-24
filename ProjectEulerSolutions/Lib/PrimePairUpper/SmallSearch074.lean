import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch073
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch074

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan507 : scan edge (search edge 4) 5 [] 0 tail507 best = best := by
  have cons : tail507 = 3643 :: tail508 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates507_checked]
  change scan edge (search edge 4) 5 [] 0 tail508
    (if 4 ≤ rootCandidates507.length then search edge 4 [3643] rootCandidates507 3643 best else best) = best
  rw [if_pos (by decide), rootBranch507]
  exact outerScan508

set_option maxRecDepth 4096 in
theorem outerScan506 : scan edge (search edge 4) 5 [] 0 tail506 best = best := by
  have cons : tail506 = 3637 :: tail507 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates506_checked]
  change scan edge (search edge 4) 5 [] 0 tail507
    (if 4 ≤ rootCandidates506.length then search edge 4 [3637] rootCandidates506 3637 best else best) = best
  rw [if_pos (by decide), rootBranch506]
  exact outerScan507

set_option maxRecDepth 4096 in
theorem outerScan505 : scan edge (search edge 4) 5 [] 0 tail505 best = best := by
  have cons : tail505 = 3631 :: tail506 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates505_checked]
  change scan edge (search edge 4) 5 [] 0 tail506
    (if 4 ≤ rootCandidates505.length then search edge 4 [3631] rootCandidates505 3631 best else best) = best
  rw [if_pos (by decide), rootBranch505]
  exact outerScan506

set_option maxRecDepth 4096 in
theorem outerScan504 : scan edge (search edge 4) 5 [] 0 tail504 best = best := by
  have cons : tail504 = 3623 :: tail505 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates504_checked]
  change scan edge (search edge 4) 5 [] 0 tail505
    (if 4 ≤ rootCandidates504.length then search edge 4 [3623] rootCandidates504 3623 best else best) = best
  rw [if_pos (by decide), rootBranch504]
  exact outerScan505

set_option maxRecDepth 4096 in
theorem outerScan503 : scan edge (search edge 4) 5 [] 0 tail503 best = best := by
  have cons : tail503 = 3617 :: tail504 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates503_checked]
  change scan edge (search edge 4) 5 [] 0 tail504
    (if 4 ≤ rootCandidates503.length then search edge 4 [3617] rootCandidates503 3617 best else best) = best
  rw [if_pos (by decide), rootBranch503]
  exact outerScan504

set_option maxRecDepth 4096 in
theorem outerScan502 : scan edge (search edge 4) 5 [] 0 tail502 best = best := by
  have cons : tail502 = 3613 :: tail503 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates502_checked]
  change scan edge (search edge 4) 5 [] 0 tail503
    (if 4 ≤ rootCandidates502.length then search edge 4 [3613] rootCandidates502 3613 best else best) = best
  rw [if_pos (by decide), rootBranch502]
  exact outerScan503

end ProjectEulerSolutions.PrimePairUpper.Search
