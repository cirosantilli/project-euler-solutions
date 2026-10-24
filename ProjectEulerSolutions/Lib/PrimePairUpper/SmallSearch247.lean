import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch246
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch247

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan20 : scan edge (search edge 4) 5 [] 0 tail20 best = best := by
  have cons : tail20 = 83 :: tail21 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates20_checked]
  change scan edge (search edge 4) 5 [] 0 tail21
    (if 4 ≤ rootCandidates20.length then search edge 4 [83] rootCandidates20 83 best else best) = best
  rw [if_pos (by decide), rootBranch20]
  exact outerScan21

end ProjectEulerSolutions.PrimePairUpper.Search
