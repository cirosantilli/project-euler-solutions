import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch196
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch197

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan108 : scan edge (search edge 4) 5 [] 0 tail108 best = best := by
  have cons : tail108 = 607 :: tail109 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates108_checked]
  change scan edge (search edge 4) 5 [] 0 tail109
    (if 4 ≤ rootCandidates108.length then search edge 4 [607] rootCandidates108 607 best else best) = best
  rw [if_pos (by decide), rootBranch108]
  exact outerScan109

set_option maxRecDepth 4096 in
theorem outerScan107 : scan edge (search edge 4) 5 [] 0 tail107 best = best := by
  have cons : tail107 = 601 :: tail108 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates107_checked]
  change scan edge (search edge 4) 5 [] 0 tail108
    (if 4 ≤ rootCandidates107.length then search edge 4 [601] rootCandidates107 601 best else best) = best
  rw [if_pos (by decide), rootBranch107]
  exact outerScan108

end ProjectEulerSolutions.PrimePairUpper.Search
