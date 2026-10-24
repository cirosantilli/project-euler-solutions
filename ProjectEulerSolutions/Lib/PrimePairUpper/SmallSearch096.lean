import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch095
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch096

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan408 : scan edge (search edge 4) 5 [] 0 tail408 best = best := by
  have cons : tail408 = 2833 :: tail409 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates408_checked]
  change scan edge (search edge 4) 5 [] 0 tail409
    (if 4 ≤ rootCandidates408.length then search edge 4 [2833] rootCandidates408 2833 best else best) = best
  rw [if_pos (by decide), rootBranch408]
  exact outerScan409

set_option maxRecDepth 4096 in
theorem outerScan407 : scan edge (search edge 4) 5 [] 0 tail407 best = best := by
  have cons : tail407 = 2819 :: tail408 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates407_checked]
  change scan edge (search edge 4) 5 [] 0 tail408
    (if 4 ≤ rootCandidates407.length then search edge 4 [2819] rootCandidates407 2819 best else best) = best
  rw [if_pos (by decide), rootBranch407]
  exact outerScan408

set_option maxRecDepth 4096 in
theorem outerScan406 : scan edge (search edge 4) 5 [] 0 tail406 best = best := by
  have cons : tail406 = 2803 :: tail407 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates406_checked]
  change scan edge (search edge 4) 5 [] 0 tail407
    (if 4 ≤ rootCandidates406.length then search edge 4 [2803] rootCandidates406 2803 best else best) = best
  rw [if_pos (by decide), rootBranch406]
  exact outerScan407

end ProjectEulerSolutions.PrimePairUpper.Search
