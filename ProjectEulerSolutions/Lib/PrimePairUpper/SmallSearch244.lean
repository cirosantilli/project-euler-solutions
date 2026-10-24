import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch243
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch244

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan23 : scan edge (search edge 4) 5 [] 0 tail23 best = best := by
  have cons : tail23 = 101 :: tail24 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates23_checked]
  change scan edge (search edge 4) 5 [] 0 tail24
    (if 4 ≤ rootCandidates23.length then search edge 4 [101] rootCandidates23 101 best else best) = best
  rw [if_pos (by decide), rootBranch23]
  exact outerScan24

end ProjectEulerSolutions.PrimePairUpper.Search
