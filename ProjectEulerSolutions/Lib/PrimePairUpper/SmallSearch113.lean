import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch112
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch113

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan345 : scan edge (search edge 4) 5 [] 0 tail345 best = best := by
  have cons : tail345 = 2347 :: tail346 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates345_checked]
  change scan edge (search edge 4) 5 [] 0 tail346
    (if 4 ≤ rootCandidates345.length then search edge 4 [2347] rootCandidates345 2347 best else best) = best
  rw [if_pos (by decide), rootBranch345]
  exact outerScan346

set_option maxRecDepth 4096 in
theorem outerScan344 : scan edge (search edge 4) 5 [] 0 tail344 best = best := by
  have cons : tail344 = 2341 :: tail345 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates344_checked]
  change scan edge (search edge 4) 5 [] 0 tail345
    (if 4 ≤ rootCandidates344.length then search edge 4 [2341] rootCandidates344 2341 best else best) = best
  rw [if_pos (by decide), rootBranch344]
  exact outerScan345

set_option maxRecDepth 4096 in
theorem outerScan343 : scan edge (search edge 4) 5 [] 0 tail343 best = best := by
  have cons : tail343 = 2339 :: tail344 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates343_checked]
  change scan edge (search edge 4) 5 [] 0 tail344
    (if 4 ≤ rootCandidates343.length then search edge 4 [2339] rootCandidates343 2339 best else best) = best
  rw [if_pos (by decide), rootBranch343]
  exact outerScan344

end ProjectEulerSolutions.PrimePairUpper.Search
