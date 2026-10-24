import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch135
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch136

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan267 : scan edge (search edge 4) 5 [] 0 tail267 best = best := by
  have cons : tail267 = 1733 :: tail268 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates267_checked]
  change scan edge (search edge 4) 5 [] 0 tail268
    (if 4 ≤ rootCandidates267.length then search edge 4 [1733] rootCandidates267 1733 best else best) = best
  rw [if_pos (by decide), rootBranch267]
  exact outerScan268

set_option maxRecDepth 4096 in
theorem outerScan266 : scan edge (search edge 4) 5 [] 0 tail266 best = best := by
  have cons : tail266 = 1723 :: tail267 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates266_checked]
  change scan edge (search edge 4) 5 [] 0 tail267
    (if 4 ≤ rootCandidates266.length then search edge 4 [1723] rootCandidates266 1723 best else best) = best
  rw [if_pos (by decide), rootBranch266]
  exact outerScan267

set_option maxRecDepth 4096 in
theorem outerScan265 : scan edge (search edge 4) 5 [] 0 tail265 best = best := by
  have cons : tail265 = 1721 :: tail266 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates265_checked]
  change scan edge (search edge 4) 5 [] 0 tail266
    (if 4 ≤ rootCandidates265.length then search edge 4 [1721] rootCandidates265 1721 best else best) = best
  rw [if_pos (by decide), rootBranch265]
  exact outerScan266

end ProjectEulerSolutions.PrimePairUpper.Search
