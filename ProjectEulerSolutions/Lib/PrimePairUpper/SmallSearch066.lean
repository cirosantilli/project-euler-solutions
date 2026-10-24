import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch065
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch066

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan550 : scan edge (search edge 4) 5 [] 0 tail550 best = best := by
  have cons : tail550 = 4007 :: tail551 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates550_checked]
  change scan edge (search edge 4) 5 [] 0 tail551
    (if 4 ≤ rootCandidates550.length then search edge 4 [4007] rootCandidates550 4007 best else best) = best
  rw [if_pos (by decide), rootBranch550]
  exact outerScan551

set_option maxRecDepth 4096 in
theorem outerScan549 : scan edge (search edge 4) 5 [] 0 tail549 best = best := by
  have cons : tail549 = 4003 :: tail550 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates549_checked]
  change scan edge (search edge 4) 5 [] 0 tail550
    (if 4 ≤ rootCandidates549.length then search edge 4 [4003] rootCandidates549 4003 best else best) = best
  rw [if_pos (by decide), rootBranch549]
  exact outerScan550

set_option maxRecDepth 4096 in
theorem outerScan548 : scan edge (search edge 4) 5 [] 0 tail548 best = best := by
  have cons : tail548 = 4001 :: tail549 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates548_checked]
  change scan edge (search edge 4) 5 [] 0 tail549
    (if 4 ≤ rootCandidates548.length then search edge 4 [4001] rootCandidates548 4001 best else best) = best
  rw [if_pos (by decide), rootBranch548]
  exact outerScan549

set_option maxRecDepth 4096 in
theorem outerScan547 : scan edge (search edge 4) 5 [] 0 tail547 best = best := by
  have cons : tail547 = 3989 :: tail548 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates547_checked]
  change scan edge (search edge 4) 5 [] 0 tail548
    (if 4 ≤ rootCandidates547.length then search edge 4 [3989] rootCandidates547 3989 best else best) = best
  rw [if_pos (by decide), rootBranch547]
  exact outerScan548

set_option maxRecDepth 4096 in
theorem outerScan546 : scan edge (search edge 4) 5 [] 0 tail546 best = best := by
  have cons : tail546 = 3967 :: tail547 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates546_checked]
  change scan edge (search edge 4) 5 [] 0 tail547
    (if 4 ≤ rootCandidates546.length then search edge 4 [3967] rootCandidates546 3967 best else best) = best
  rw [if_pos (by decide), rootBranch546]
  exact outerScan547

end ProjectEulerSolutions.PrimePairUpper.Search
