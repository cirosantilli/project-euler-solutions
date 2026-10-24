import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch254
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch255

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan12 : scan edge (search edge 4) 5 [] 0 tail12 best = best := by
  have cons : tail12 = 47 :: tail13 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates12_checked]
  change scan edge (search edge 4) 5 [] 0 tail13
    (if 4 ≤ rootCandidates12.length then search edge 4 [47] rootCandidates12 47 best else best) = best
  rw [if_pos (by decide), rootBranch12]
  exact outerScan13

end ProjectEulerSolutions.PrimePairUpper.Search
