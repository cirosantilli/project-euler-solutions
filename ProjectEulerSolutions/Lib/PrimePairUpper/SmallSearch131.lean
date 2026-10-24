import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch130
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch131

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan281 : scan edge (search edge 4) 5 [] 0 tail281 best = best := by
  have cons : tail281 = 1861 :: tail282 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates281_checked]
  change scan edge (search edge 4) 5 [] 0 tail282
    (if 4 ≤ rootCandidates281.length then search edge 4 [1861] rootCandidates281 1861 best else best) = best
  rw [if_pos (by decide), rootBranch281]
  exact outerScan282

set_option maxRecDepth 4096 in
theorem outerScan280 : scan edge (search edge 4) 5 [] 0 tail280 best = best := by
  have cons : tail280 = 1847 :: tail281 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates280_checked]
  change scan edge (search edge 4) 5 [] 0 tail281
    (if 4 ≤ rootCandidates280.length then search edge 4 [1847] rootCandidates280 1847 best else best) = best
  rw [if_pos (by decide), rootBranch280]
  exact outerScan281

set_option maxRecDepth 4096 in
theorem outerScan279 : scan edge (search edge 4) 5 [] 0 tail279 best = best := by
  have cons : tail279 = 1831 :: tail280 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates279_checked]
  change scan edge (search edge 4) 5 [] 0 tail280
    (if 4 ≤ rootCandidates279.length then search edge 4 [1831] rootCandidates279 1831 best else best) = best
  rw [if_pos (by decide), rootBranch279]
  exact outerScan280

end ProjectEulerSolutions.PrimePairUpper.Search
