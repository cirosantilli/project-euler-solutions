import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch219
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch220

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan62 : scan edge (search edge 4) 5 [] 0 tail62 best = best := by
  have cons : tail62 = 313 :: tail63 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates62_checked]
  change scan edge (search edge 4) 5 [] 0 tail63
    (if 4 ≤ rootCandidates62.length then search edge 4 [313] rootCandidates62 313 best else best) = best
  rw [if_pos (by decide), rootBranch62]
  exact outerScan63

set_option maxRecDepth 4096 in
theorem outerScan61 : scan edge (search edge 4) 5 [] 0 tail61 best = best := by
  have cons : tail61 = 311 :: tail62 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates61_checked]
  change scan edge (search edge 4) 5 [] 0 tail62
    (if 4 ≤ rootCandidates61.length then search edge 4 [311] rootCandidates61 311 best else best) = best
  rw [if_pos (by decide), rootBranch61]
  exact outerScan62

end ProjectEulerSolutions.PrimePairUpper.Search
