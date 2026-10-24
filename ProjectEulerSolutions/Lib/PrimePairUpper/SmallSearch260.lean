import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch259
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch260

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan7 : scan edge (search edge 4) 5 [] 0 tail7 best = best := by
  have cons : tail7 = 29 :: tail8 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates7_checked]
  change scan edge (search edge 4) 5 [] 0 tail8
    (if 4 ≤ rootCandidates7.length then search edge 4 [29] rootCandidates7 29 best else best) = best
  rw [if_pos (by decide), rootBranch7]
  exact outerScan8

end ProjectEulerSolutions.PrimePairUpper.Search
