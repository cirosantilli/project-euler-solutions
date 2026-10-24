import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch145
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch146

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan239 : scan edge (search edge 4) 5 [] 0 tail239 best = best := by
  have cons : tail239 = 1531 :: tail240 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates239_checked]
  change scan edge (search edge 4) 5 [] 0 tail240
    (if 4 ≤ rootCandidates239.length then search edge 4 [1531] rootCandidates239 1531 best else best) = best
  rw [if_pos (by decide), rootBranch239]
  exact outerScan240

set_option maxRecDepth 4096 in
theorem outerScan238 : scan edge (search edge 4) 5 [] 0 tail238 best = best := by
  have cons : tail238 = 1523 :: tail239 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates238_checked]
  change scan edge (search edge 4) 5 [] 0 tail239
    (if 4 ≤ rootCandidates238.length then search edge 4 [1523] rootCandidates238 1523 best else best) = best
  rw [if_pos (by decide), rootBranch238]
  exact outerScan239

set_option maxRecDepth 4096 in
theorem outerScan237 : scan edge (search edge 4) 5 [] 0 tail237 best = best := by
  have cons : tail237 = 1511 :: tail238 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates237_checked]
  change scan edge (search edge 4) 5 [] 0 tail238
    (if 4 ≤ rootCandidates237.length then search edge 4 [1511] rootCandidates237 1511 best else best) = best
  rw [if_pos (by decide), rootBranch237]
  exact outerScan238

end ProjectEulerSolutions.PrimePairUpper.Search
