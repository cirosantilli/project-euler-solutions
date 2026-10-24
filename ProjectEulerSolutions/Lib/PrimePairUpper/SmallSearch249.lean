import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch248
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch249

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan18 : scan edge (search edge 4) 5 [] 0 tail18 best = best := by
  have cons : tail18 = 73 :: tail19 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates18_checked]
  change scan edge (search edge 4) 5 [] 0 tail19
    (if 4 ≤ rootCandidates18.length then search edge 4 [73] rootCandidates18 73 best else best) = best
  rw [if_pos (by decide), rootBranch18]
  exact outerScan19

end ProjectEulerSolutions.PrimePairUpper.Search
