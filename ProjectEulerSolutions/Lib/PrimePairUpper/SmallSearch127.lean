import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch126
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch127

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan295 : scan edge (search edge 4) 5 [] 0 tail295 best = best := by
  have cons : tail295 = 1973 :: tail296 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates295_checked]
  change scan edge (search edge 4) 5 [] 0 tail296
    (if 4 ≤ rootCandidates295.length then search edge 4 [1973] rootCandidates295 1973 best else best) = best
  rw [if_pos (by decide), rootBranch295]
  exact outerScan296

set_option maxRecDepth 4096 in
theorem outerScan294 : scan edge (search edge 4) 5 [] 0 tail294 best = best := by
  have cons : tail294 = 1951 :: tail295 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates294_checked]
  change scan edge (search edge 4) 5 [] 0 tail295
    (if 4 ≤ rootCandidates294.length then search edge 4 [1951] rootCandidates294 1951 best else best) = best
  rw [if_pos (by decide), rootBranch294]
  exact outerScan295

set_option maxRecDepth 4096 in
theorem outerScan293 : scan edge (search edge 4) 5 [] 0 tail293 best = best := by
  have cons : tail293 = 1949 :: tail294 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates293_checked]
  change scan edge (search edge 4) 5 [] 0 tail294
    (if 4 ≤ rootCandidates293.length then search edge 4 [1949] rootCandidates293 1949 best else best) = best
  rw [if_pos (by decide), rootBranch293]
  exact outerScan294

set_option maxRecDepth 4096 in
theorem outerScan292 : scan edge (search edge 4) 5 [] 0 tail292 best = best := by
  have cons : tail292 = 1933 :: tail293 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates292_checked]
  change scan edge (search edge 4) 5 [] 0 tail293
    (if 4 ≤ rootCandidates292.length then search edge 4 [1933] rootCandidates292 1933 best else best) = best
  rw [if_pos (by decide), rootBranch292]
  exact outerScan293

end ProjectEulerSolutions.PrimePairUpper.Search
