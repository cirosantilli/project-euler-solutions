import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch255
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch256

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan11 : scan edge (search edge 4) 5 [] 0 tail11 best = best := by
  have cons : tail11 = 43 :: tail12 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates11_checked]
  change scan edge (search edge 4) 5 [] 0 tail12
    (if 4 ≤ rootCandidates11.length then search edge 4 [43] rootCandidates11 43 best else best) = best
  rw [if_pos (by decide), rootBranch11]
  exact outerScan12

end ProjectEulerSolutions.PrimePairUpper.Search
