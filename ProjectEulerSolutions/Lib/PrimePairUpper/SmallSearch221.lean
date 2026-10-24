import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch220
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch221

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan60 : scan edge (search edge 4) 5 [] 0 tail60 best = best := by
  have cons : tail60 = 307 :: tail61 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates60_checked]
  change scan edge (search edge 4) 5 [] 0 tail61
    (if 4 ≤ rootCandidates60.length then search edge 4 [307] rootCandidates60 307 best else best) = best
  rw [if_pos (by decide), rootBranch60]
  exact outerScan61

set_option maxRecDepth 4096 in
theorem outerScan59 : scan edge (search edge 4) 5 [] 0 tail59 best = best := by
  have cons : tail59 = 293 :: tail60 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates59_checked]
  change scan edge (search edge 4) 5 [] 0 tail60
    (if 4 ≤ rootCandidates59.length then search edge 4 [293] rootCandidates59 293 best else best) = best
  rw [if_pos (by decide), rootBranch59]
  exact outerScan60

end ProjectEulerSolutions.PrimePairUpper.Search
