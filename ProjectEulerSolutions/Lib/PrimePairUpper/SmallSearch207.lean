import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch206
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch207

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan88 : scan edge (search edge 4) 5 [] 0 tail88 best = best := by
  have cons : tail88 = 467 :: tail89 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates88_checked]
  change scan edge (search edge 4) 5 [] 0 tail89
    (if 4 ≤ rootCandidates88.length then search edge 4 [467] rootCandidates88 467 best else best) = best
  rw [if_pos (by decide), rootBranch88]
  exact outerScan89

set_option maxRecDepth 4096 in
theorem outerScan87 : scan edge (search edge 4) 5 [] 0 tail87 best = best := by
  have cons : tail87 = 463 :: tail88 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates87_checked]
  change scan edge (search edge 4) 5 [] 0 tail88
    (if 4 ≤ rootCandidates87.length then search edge 4 [463] rootCandidates87 463 best else best) = best
  rw [if_pos (by decide), rootBranch87]
  exact outerScan88

end ProjectEulerSolutions.PrimePairUpper.Search
