import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch209
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch210

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan82 : scan edge (search edge 4) 5 [] 0 tail82 best = best := by
  have cons : tail82 = 439 :: tail83 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates82_checked]
  change scan edge (search edge 4) 5 [] 0 tail83
    (if 4 ≤ rootCandidates82.length then search edge 4 [439] rootCandidates82 439 best else best) = best
  rw [if_pos (by decide), rootBranch82]
  exact outerScan83

set_option maxRecDepth 4096 in
theorem outerScan81 : scan edge (search edge 4) 5 [] 0 tail81 best = best := by
  have cons : tail81 = 433 :: tail82 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates81_checked]
  change scan edge (search edge 4) 5 [] 0 tail82
    (if 4 ≤ rootCandidates81.length then search edge 4 [433] rootCandidates81 433 best else best) = best
  rw [if_pos (by decide), rootBranch81]
  exact outerScan82

end ProjectEulerSolutions.PrimePairUpper.Search
