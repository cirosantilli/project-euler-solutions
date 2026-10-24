import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch111
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch112

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan349 : scan edge (search edge 4) 5 [] 0 tail349 best = best := by
  have cons : tail349 = 2377 :: tail350 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates349_checked]
  change scan edge (search edge 4) 5 [] 0 tail350
    (if 4 ≤ rootCandidates349.length then search edge 4 [2377] rootCandidates349 2377 best else best) = best
  rw [if_pos (by decide), rootBranch349]
  exact outerScan350

set_option maxRecDepth 4096 in
theorem outerScan348 : scan edge (search edge 4) 5 [] 0 tail348 best = best := by
  have cons : tail348 = 2371 :: tail349 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates348_checked]
  change scan edge (search edge 4) 5 [] 0 tail349
    (if 4 ≤ rootCandidates348.length then search edge 4 [2371] rootCandidates348 2371 best else best) = best
  rw [if_pos (by decide), rootBranch348]
  exact outerScan349

set_option maxRecDepth 4096 in
theorem outerScan347 : scan edge (search edge 4) 5 [] 0 tail347 best = best := by
  have cons : tail347 = 2357 :: tail348 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates347_checked]
  change scan edge (search edge 4) 5 [] 0 tail348
    (if 4 ≤ rootCandidates347.length then search edge 4 [2357] rootCandidates347 2357 best else best) = best
  rw [if_pos (by decide), rootBranch347]
  exact outerScan348

set_option maxRecDepth 4096 in
theorem outerScan346 : scan edge (search edge 4) 5 [] 0 tail346 best = best := by
  have cons : tail346 = 2351 :: tail347 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates346_checked]
  change scan edge (search edge 4) 5 [] 0 tail347
    (if 4 ≤ rootCandidates346.length then search edge 4 [2351] rootCandidates346 2351 best else best) = best
  rw [if_pos (by decide), rootBranch346]
  exact outerScan347

end ProjectEulerSolutions.PrimePairUpper.Search
