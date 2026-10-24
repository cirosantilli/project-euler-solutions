import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch115
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch116

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan334 : scan edge (search edge 4) 5 [] 0 tail334 best = best := by
  have cons : tail334 = 2269 :: tail335 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates334_checked]
  change scan edge (search edge 4) 5 [] 0 tail335
    (if 4 ≤ rootCandidates334.length then search edge 4 [2269] rootCandidates334 2269 best else best) = best
  rw [if_pos (by decide), rootBranch334]
  exact outerScan335

set_option maxRecDepth 4096 in
theorem outerScan333 : scan edge (search edge 4) 5 [] 0 tail333 best = best := by
  have cons : tail333 = 2267 :: tail334 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates333_checked]
  change scan edge (search edge 4) 5 [] 0 tail334
    (if 4 ≤ rootCandidates333.length then search edge 4 [2267] rootCandidates333 2267 best else best) = best
  rw [if_pos (by decide), rootBranch333]
  exact outerScan334

set_option maxRecDepth 4096 in
theorem outerScan332 : scan edge (search edge 4) 5 [] 0 tail332 best = best := by
  have cons : tail332 = 2251 :: tail333 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates332_checked]
  change scan edge (search edge 4) 5 [] 0 tail333
    (if 4 ≤ rootCandidates332.length then search edge 4 [2251] rootCandidates332 2251 best else best) = best
  rw [if_pos (by decide), rootBranch332]
  exact outerScan333

end ProjectEulerSolutions.PrimePairUpper.Search
