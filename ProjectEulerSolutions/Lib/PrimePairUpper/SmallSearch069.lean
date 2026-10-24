import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch068
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch069

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan535 : scan edge (search edge 4) 5 [] 0 tail535 best = best := by
  have cons : tail535 = 3881 :: tail536 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates535_checked]
  change scan edge (search edge 4) 5 [] 0 tail536
    (if 4 ≤ rootCandidates535.length then search edge 4 [3881] rootCandidates535 3881 best else best) = best
  rw [if_pos (by decide), rootBranch535]
  exact outerScan536

set_option maxRecDepth 4096 in
theorem outerScan534 : scan edge (search edge 4) 5 [] 0 tail534 best = best := by
  have cons : tail534 = 3877 :: tail535 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates534_checked]
  change scan edge (search edge 4) 5 [] 0 tail535
    (if 4 ≤ rootCandidates534.length then search edge 4 [3877] rootCandidates534 3877 best else best) = best
  rw [if_pos (by decide), rootBranch534]
  exact outerScan535

set_option maxRecDepth 4096 in
theorem outerScan533 : scan edge (search edge 4) 5 [] 0 tail533 best = best := by
  have cons : tail533 = 3863 :: tail534 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates533_checked]
  change scan edge (search edge 4) 5 [] 0 tail534
    (if 4 ≤ rootCandidates533.length then search edge 4 [3863] rootCandidates533 3863 best else best) = best
  rw [if_pos (by decide), rootBranch533]
  exact outerScan534

set_option maxRecDepth 4096 in
theorem outerScan532 : scan edge (search edge 4) 5 [] 0 tail532 best = best := by
  have cons : tail532 = 3853 :: tail533 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates532_checked]
  change scan edge (search edge 4) 5 [] 0 tail533
    (if 4 ≤ rootCandidates532.length then search edge 4 [3853] rootCandidates532 3853 best else best) = best
  rw [if_pos (by decide), rootBranch532]
  exact outerScan533

set_option maxRecDepth 4096 in
theorem outerScan531 : scan edge (search edge 4) 5 [] 0 tail531 best = best := by
  have cons : tail531 = 3851 :: tail532 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates531_checked]
  change scan edge (search edge 4) 5 [] 0 tail532
    (if 4 ≤ rootCandidates531.length then search edge 4 [3851] rootCandidates531 3851 best else best) = best
  rw [if_pos (by decide), rootBranch531]
  exact outerScan532

end ProjectEulerSolutions.PrimePairUpper.Search
