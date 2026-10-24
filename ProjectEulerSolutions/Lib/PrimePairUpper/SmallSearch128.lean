import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch127
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch128

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan291 : scan edge (search edge 4) 5 [] 0 tail291 best = best := by
  have cons : tail291 = 1931 :: tail292 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates291_checked]
  change scan edge (search edge 4) 5 [] 0 tail292
    (if 4 ≤ rootCandidates291.length then search edge 4 [1931] rootCandidates291 1931 best else best) = best
  rw [if_pos (by decide), rootBranch291]
  exact outerScan292

set_option maxRecDepth 4096 in
theorem outerScan290 : scan edge (search edge 4) 5 [] 0 tail290 best = best := by
  have cons : tail290 = 1913 :: tail291 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates290_checked]
  change scan edge (search edge 4) 5 [] 0 tail291
    (if 4 ≤ rootCandidates290.length then search edge 4 [1913] rootCandidates290 1913 best else best) = best
  rw [if_pos (by decide), rootBranch290]
  exact outerScan291

set_option maxRecDepth 4096 in
theorem outerScan289 : scan edge (search edge 4) 5 [] 0 tail289 best = best := by
  have cons : tail289 = 1907 :: tail290 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates289_checked]
  change scan edge (search edge 4) 5 [] 0 tail290
    (if 4 ≤ rootCandidates289.length then search edge 4 [1907] rootCandidates289 1907 best else best) = best
  rw [if_pos (by decide), rootBranch289]
  exact outerScan290

set_option maxRecDepth 4096 in
theorem outerScan288 : scan edge (search edge 4) 5 [] 0 tail288 best = best := by
  have cons : tail288 = 1901 :: tail289 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates288_checked]
  change scan edge (search edge 4) 5 [] 0 tail289
    (if 4 ≤ rootCandidates288.length then search edge 4 [1901] rootCandidates288 1901 best else best) = best
  rw [if_pos (by decide), rootBranch288]
  exact outerScan289

end ProjectEulerSolutions.PrimePairUpper.Search
