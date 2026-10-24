import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch226
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch227

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan50 : scan edge (search edge 4) 5 [] 0 tail50 best = best := by
  have cons : tail50 = 241 :: tail51 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates50_checked]
  change scan edge (search edge 4) 5 [] 0 tail51
    (if 4 ≤ rootCandidates50.length then search edge 4 [241] rootCandidates50 241 best else best) = best
  rw [if_pos (by decide), rootBranch50]
  exact outerScan51

end ProjectEulerSolutions.PrimePairUpper.Search
