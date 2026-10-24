import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch089
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch090

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan433 : scan edge (search edge 4) 5 [] 0 tail433 best = best := by
  have cons : tail433 = 3041 :: tail434 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates433_checked]
  change scan edge (search edge 4) 5 [] 0 tail434
    (if 4 ≤ rootCandidates433.length then search edge 4 [3041] rootCandidates433 3041 best else best) = best
  rw [if_pos (by decide), rootBranch433]
  exact outerScan434

set_option maxRecDepth 4096 in
theorem outerScan432 : scan edge (search edge 4) 5 [] 0 tail432 best = best := by
  have cons : tail432 = 3037 :: tail433 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates432_checked]
  change scan edge (search edge 4) 5 [] 0 tail433
    (if 4 ≤ rootCandidates432.length then search edge 4 [3037] rootCandidates432 3037 best else best) = best
  rw [if_pos (by decide), rootBranch432]
  exact outerScan433

set_option maxRecDepth 4096 in
theorem outerScan431 : scan edge (search edge 4) 5 [] 0 tail431 best = best := by
  have cons : tail431 = 3023 :: tail432 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates431_checked]
  change scan edge (search edge 4) 5 [] 0 tail432
    (if 4 ≤ rootCandidates431.length then search edge 4 [3023] rootCandidates431 3023 best else best) = best
  rw [if_pos (by decide), rootBranch431]
  exact outerScan432

set_option maxRecDepth 4096 in
theorem outerScan430 : scan edge (search edge 4) 5 [] 0 tail430 best = best := by
  have cons : tail430 = 3019 :: tail431 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates430_checked]
  change scan edge (search edge 4) 5 [] 0 tail431
    (if 4 ≤ rootCandidates430.length then search edge 4 [3019] rootCandidates430 3019 best else best) = best
  rw [if_pos (by decide), rootBranch430]
  exact outerScan431

end ProjectEulerSolutions.PrimePairUpper.Search
