import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch114
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch115

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan338 : scan edge (search edge 4) 5 [] 0 tail338 best = best := by
  have cons : tail338 = 2293 :: tail339 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates338_checked]
  change scan edge (search edge 4) 5 [] 0 tail339
    (if 4 ≤ rootCandidates338.length then search edge 4 [2293] rootCandidates338 2293 best else best) = best
  rw [if_pos (by decide), rootBranch338]
  exact outerScan339

set_option maxRecDepth 4096 in
theorem outerScan337 : scan edge (search edge 4) 5 [] 0 tail337 best = best := by
  have cons : tail337 = 2287 :: tail338 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates337_checked]
  change scan edge (search edge 4) 5 [] 0 tail338
    (if 4 ≤ rootCandidates337.length then search edge 4 [2287] rootCandidates337 2287 best else best) = best
  rw [if_pos (by decide), rootBranch337]
  exact outerScan338

set_option maxRecDepth 4096 in
theorem outerScan336 : scan edge (search edge 4) 5 [] 0 tail336 best = best := by
  have cons : tail336 = 2281 :: tail337 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates336_checked]
  change scan edge (search edge 4) 5 [] 0 tail337
    (if 4 ≤ rootCandidates336.length then search edge 4 [2281] rootCandidates336 2281 best else best) = best
  rw [if_pos (by decide), rootBranch336]
  exact outerScan337

set_option maxRecDepth 4096 in
theorem outerScan335 : scan edge (search edge 4) 5 [] 0 tail335 best = best := by
  have cons : tail335 = 2273 :: tail336 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates335_checked]
  change scan edge (search edge 4) 5 [] 0 tail336
    (if 4 ≤ rootCandidates335.length then search edge 4 [2273] rootCandidates335 2273 best else best) = best
  rw [if_pos (by decide), rootBranch335]
  exact outerScan336

end ProjectEulerSolutions.PrimePairUpper.Search
