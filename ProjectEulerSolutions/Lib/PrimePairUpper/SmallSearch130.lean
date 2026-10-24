import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch129
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch130

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan284 : scan edge (search edge 4) 5 [] 0 tail284 best = best := by
  have cons : tail284 = 1873 :: tail285 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates284_checked]
  change scan edge (search edge 4) 5 [] 0 tail285
    (if 4 ≤ rootCandidates284.length then search edge 4 [1873] rootCandidates284 1873 best else best) = best
  rw [if_pos (by decide), rootBranch284]
  exact outerScan285

set_option maxRecDepth 4096 in
theorem outerScan283 : scan edge (search edge 4) 5 [] 0 tail283 best = best := by
  have cons : tail283 = 1871 :: tail284 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates283_checked]
  change scan edge (search edge 4) 5 [] 0 tail284
    (if 4 ≤ rootCandidates283.length then search edge 4 [1871] rootCandidates283 1871 best else best) = best
  rw [if_pos (by decide), rootBranch283]
  exact outerScan284

set_option maxRecDepth 4096 in
theorem outerScan282 : scan edge (search edge 4) 5 [] 0 tail282 best = best := by
  have cons : tail282 = 1867 :: tail283 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates282_checked]
  change scan edge (search edge 4) 5 [] 0 tail283
    (if 4 ≤ rootCandidates282.length then search edge 4 [1867] rootCandidates282 1867 best else best) = best
  rw [if_pos (by decide), rootBranch282]
  exact outerScan283

end ProjectEulerSolutions.PrimePairUpper.Search
