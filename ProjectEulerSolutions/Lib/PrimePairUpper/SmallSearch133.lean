import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch132
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch133

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan275 : scan edge (search edge 4) 5 [] 0 tail275 best = best := by
  have cons : tail275 = 1789 :: tail276 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates275_checked]
  change scan edge (search edge 4) 5 [] 0 tail276
    (if 4 ≤ rootCandidates275.length then search edge 4 [1789] rootCandidates275 1789 best else best) = best
  rw [if_pos (by decide), rootBranch275]
  exact outerScan276

set_option maxRecDepth 4096 in
theorem outerScan274 : scan edge (search edge 4) 5 [] 0 tail274 best = best := by
  have cons : tail274 = 1787 :: tail275 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates274_checked]
  change scan edge (search edge 4) 5 [] 0 tail275
    (if 4 ≤ rootCandidates274.length then search edge 4 [1787] rootCandidates274 1787 best else best) = best
  rw [if_pos (by decide), rootBranch274]
  exact outerScan275

set_option maxRecDepth 4096 in
theorem outerScan273 : scan edge (search edge 4) 5 [] 0 tail273 best = best := by
  have cons : tail273 = 1783 :: tail274 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates273_checked]
  change scan edge (search edge 4) 5 [] 0 tail274
    (if 4 ≤ rootCandidates273.length then search edge 4 [1783] rootCandidates273 1783 best else best) = best
  rw [if_pos (by decide), rootBranch273]
  exact outerScan274

end ProjectEulerSolutions.PrimePairUpper.Search
