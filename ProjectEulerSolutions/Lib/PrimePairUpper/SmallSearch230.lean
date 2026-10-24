import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch229
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch230

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan47 : scan edge (search edge 4) 5 [] 0 tail47 best = best := by
  have cons : tail47 = 229 :: tail48 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates47_checked]
  change scan edge (search edge 4) 5 [] 0 tail48
    (if 4 ≤ rootCandidates47.length then search edge 4 [229] rootCandidates47 229 best else best) = best
  rw [if_pos (by decide), rootBranch47]
  exact outerScan48

set_option maxRecDepth 4096 in
theorem outerScan46 : scan edge (search edge 4) 5 [] 0 tail46 best = best := by
  have cons : tail46 = 227 :: tail47 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates46_checked]
  change scan edge (search edge 4) 5 [] 0 tail47
    (if 4 ≤ rootCandidates46.length then search edge 4 [227] rootCandidates46 227 best else best) = best
  rw [if_pos (by decide), rootBranch46]
  exact outerScan47

end ProjectEulerSolutions.PrimePairUpper.Search
