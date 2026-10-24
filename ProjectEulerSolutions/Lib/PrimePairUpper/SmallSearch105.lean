import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch104
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch105

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan375 : scan edge (search edge 4) 5 [] 0 tail375 best = best := by
  have cons : tail375 = 2593 :: tail376 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates375_checked]
  change scan edge (search edge 4) 5 [] 0 tail376
    (if 4 ≤ rootCandidates375.length then search edge 4 [2593] rootCandidates375 2593 best else best) = best
  rw [if_pos (by decide), rootBranch375]
  exact outerScan376

set_option maxRecDepth 4096 in
theorem outerScan374 : scan edge (search edge 4) 5 [] 0 tail374 best = best := by
  have cons : tail374 = 2591 :: tail375 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates374_checked]
  change scan edge (search edge 4) 5 [] 0 tail375
    (if 4 ≤ rootCandidates374.length then search edge 4 [2591] rootCandidates374 2591 best else best) = best
  rw [if_pos (by decide), rootBranch374]
  exact outerScan375

set_option maxRecDepth 4096 in
theorem outerScan373 : scan edge (search edge 4) 5 [] 0 tail373 best = best := by
  have cons : tail373 = 2579 :: tail374 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates373_checked]
  change scan edge (search edge 4) 5 [] 0 tail374
    (if 4 ≤ rootCandidates373.length then search edge 4 [2579] rootCandidates373 2579 best else best) = best
  rw [if_pos (by decide), rootBranch373]
  exact outerScan374

set_option maxRecDepth 4096 in
theorem outerScan372 : scan edge (search edge 4) 5 [] 0 tail372 best = best := by
  have cons : tail372 = 2557 :: tail373 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates372_checked]
  change scan edge (search edge 4) 5 [] 0 tail373
    (if 4 ≤ rootCandidates372.length then search edge 4 [2557] rootCandidates372 2557 best else best) = best
  rw [if_pos (by decide), rootBranch372]
  exact outerScan373

end ProjectEulerSolutions.PrimePairUpper.Search
