import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch139
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch140

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan255 : scan edge (search edge 4) 5 [] 0 tail255 best = best := by
  have cons : tail255 = 1627 :: tail256 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates255_checked]
  change scan edge (search edge 4) 5 [] 0 tail256
    (if 4 ≤ rootCandidates255.length then search edge 4 [1627] rootCandidates255 1627 best else best) = best
  rw [if_pos (by decide), rootBranch255]
  exact outerScan256

set_option maxRecDepth 4096 in
theorem outerScan254 : scan edge (search edge 4) 5 [] 0 tail254 best = best := by
  have cons : tail254 = 1621 :: tail255 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates254_checked]
  change scan edge (search edge 4) 5 [] 0 tail255
    (if 4 ≤ rootCandidates254.length then search edge 4 [1621] rootCandidates254 1621 best else best) = best
  rw [if_pos (by decide), rootBranch254]
  exact outerScan255

set_option maxRecDepth 4096 in
theorem outerScan253 : scan edge (search edge 4) 5 [] 0 tail253 best = best := by
  have cons : tail253 = 1619 :: tail254 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates253_checked]
  change scan edge (search edge 4) 5 [] 0 tail254
    (if 4 ≤ rootCandidates253.length then search edge 4 [1619] rootCandidates253 1619 best else best) = best
  rw [if_pos (by decide), rootBranch253]
  exact outerScan254

end ProjectEulerSolutions.PrimePairUpper.Search
