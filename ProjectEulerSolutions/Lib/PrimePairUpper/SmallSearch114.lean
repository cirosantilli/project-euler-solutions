import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch113
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch114

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan342 : scan edge (search edge 4) 5 [] 0 tail342 best = best := by
  have cons : tail342 = 2333 :: tail343 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates342_checked]
  change scan edge (search edge 4) 5 [] 0 tail343
    (if 4 ≤ rootCandidates342.length then search edge 4 [2333] rootCandidates342 2333 best else best) = best
  rw [if_pos (by decide), rootBranch342]
  exact outerScan343

set_option maxRecDepth 4096 in
theorem outerScan341 : scan edge (search edge 4) 5 [] 0 tail341 best = best := by
  have cons : tail341 = 2311 :: tail342 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates341_checked]
  change scan edge (search edge 4) 5 [] 0 tail342
    (if 4 ≤ rootCandidates341.length then search edge 4 [2311] rootCandidates341 2311 best else best) = best
  rw [if_pos (by decide), rootBranch341]
  exact outerScan342

set_option maxRecDepth 4096 in
theorem outerScan340 : scan edge (search edge 4) 5 [] 0 tail340 best = best := by
  have cons : tail340 = 2309 :: tail341 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates340_checked]
  change scan edge (search edge 4) 5 [] 0 tail341
    (if 4 ≤ rootCandidates340.length then search edge 4 [2309] rootCandidates340 2309 best else best) = best
  rw [if_pos (by decide), rootBranch340]
  exact outerScan341

set_option maxRecDepth 4096 in
theorem outerScan339 : scan edge (search edge 4) 5 [] 0 tail339 best = best := by
  have cons : tail339 = 2297 :: tail340 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates339_checked]
  change scan edge (search edge 4) 5 [] 0 tail340
    (if 4 ≤ rootCandidates339.length then search edge 4 [2297] rootCandidates339 2297 best else best) = best
  rw [if_pos (by decide), rootBranch339]
  exact outerScan340

end ProjectEulerSolutions.PrimePairUpper.Search
