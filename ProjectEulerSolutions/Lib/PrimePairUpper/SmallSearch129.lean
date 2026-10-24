import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch128
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch129

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan287 : scan edge (search edge 4) 5 [] 0 tail287 best = best := by
  have cons : tail287 = 1889 :: tail288 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates287_checked]
  change scan edge (search edge 4) 5 [] 0 tail288
    (if 4 ≤ rootCandidates287.length then search edge 4 [1889] rootCandidates287 1889 best else best) = best
  rw [if_pos (by decide), rootBranch287]
  exact outerScan288

set_option maxRecDepth 4096 in
theorem outerScan286 : scan edge (search edge 4) 5 [] 0 tail286 best = best := by
  have cons : tail286 = 1879 :: tail287 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates286_checked]
  change scan edge (search edge 4) 5 [] 0 tail287
    (if 4 ≤ rootCandidates286.length then search edge 4 [1879] rootCandidates286 1879 best else best) = best
  rw [if_pos (by decide), rootBranch286]
  exact outerScan287

set_option maxRecDepth 4096 in
theorem outerScan285 : scan edge (search edge 4) 5 [] 0 tail285 best = best := by
  have cons : tail285 = 1877 :: tail286 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates285_checked]
  change scan edge (search edge 4) 5 [] 0 tail286
    (if 4 ≤ rootCandidates285.length then search edge 4 [1877] rootCandidates285 1877 best else best) = best
  rw [if_pos (by decide), rootBranch285]
  exact outerScan286

end ProjectEulerSolutions.PrimePairUpper.Search
