import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch212
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch213

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan76 : scan edge (search edge 4) 5 [] 0 tail76 best = best := by
  have cons : tail76 = 401 :: tail77 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates76_checked]
  change scan edge (search edge 4) 5 [] 0 tail77
    (if 4 ≤ rootCandidates76.length then search edge 4 [401] rootCandidates76 401 best else best) = best
  rw [if_pos (by decide), rootBranch76]
  exact outerScan77

set_option maxRecDepth 4096 in
theorem outerScan75 : scan edge (search edge 4) 5 [] 0 tail75 best = best := by
  have cons : tail75 = 397 :: tail76 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates75_checked]
  change scan edge (search edge 4) 5 [] 0 tail76
    (if 4 ≤ rootCandidates75.length then search edge 4 [397] rootCandidates75 397 best else best) = best
  rw [if_pos (by decide), rootBranch75]
  exact outerScan76

end ProjectEulerSolutions.PrimePairUpper.Search
