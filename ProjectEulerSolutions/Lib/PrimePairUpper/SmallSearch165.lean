import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch164
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch165

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan184 : scan edge (search edge 4) 5 [] 0 tail184 best = best := by
  have cons : tail184 = 1117 :: tail185 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates184_checked]
  change scan edge (search edge 4) 5 [] 0 tail185
    (if 4 ≤ rootCandidates184.length then search edge 4 [1117] rootCandidates184 1117 best else best) = best
  rw [if_pos (by decide), rootBranch184]
  exact outerScan185

set_option maxRecDepth 4096 in
theorem outerScan183 : scan edge (search edge 4) 5 [] 0 tail183 best = best := by
  have cons : tail183 = 1109 :: tail184 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates183_checked]
  change scan edge (search edge 4) 5 [] 0 tail184
    (if 4 ≤ rootCandidates183.length then search edge 4 [1109] rootCandidates183 1109 best else best) = best
  rw [if_pos (by decide), rootBranch183]
  exact outerScan184

end ProjectEulerSolutions.PrimePairUpper.Search
