import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch245
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch246

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan21 : scan edge (search edge 4) 5 [] 0 tail21 best = best := by
  have cons : tail21 = 89 :: tail22 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates21_checked]
  change scan edge (search edge 4) 5 [] 0 tail22
    (if 4 ≤ rootCandidates21.length then search edge 4 [89] rootCandidates21 89 best else best) = best
  rw [if_pos (by decide), rootBranch21]
  exact outerScan22

end ProjectEulerSolutions.PrimePairUpper.Search
