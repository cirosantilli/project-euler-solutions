import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch070
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch071

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan525 : scan edge (search edge 4) 5 [] 0 tail525 best = best := by
  have cons : tail525 = 3797 :: tail526 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates525_checked]
  change scan edge (search edge 4) 5 [] 0 tail526
    (if 4 ≤ rootCandidates525.length then search edge 4 [3797] rootCandidates525 3797 best else best) = best
  rw [if_pos (by decide), rootBranch525]
  exact outerScan526

set_option maxRecDepth 4096 in
theorem outerScan524 : scan edge (search edge 4) 5 [] 0 tail524 best = best := by
  have cons : tail524 = 3793 :: tail525 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates524_checked]
  change scan edge (search edge 4) 5 [] 0 tail525
    (if 4 ≤ rootCandidates524.length then search edge 4 [3793] rootCandidates524 3793 best else best) = best
  rw [if_pos (by decide), rootBranch524]
  exact outerScan525

set_option maxRecDepth 4096 in
theorem outerScan523 : scan edge (search edge 4) 5 [] 0 tail523 best = best := by
  have cons : tail523 = 3779 :: tail524 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates523_checked]
  change scan edge (search edge 4) 5 [] 0 tail524
    (if 4 ≤ rootCandidates523.length then search edge 4 [3779] rootCandidates523 3779 best else best) = best
  rw [if_pos (by decide), rootBranch523]
  exact outerScan524

set_option maxRecDepth 4096 in
theorem outerScan522 : scan edge (search edge 4) 5 [] 0 tail522 best = best := by
  have cons : tail522 = 3769 :: tail523 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates522_checked]
  change scan edge (search edge 4) 5 [] 0 tail523
    (if 4 ≤ rootCandidates522.length then search edge 4 [3769] rootCandidates522 3769 best else best) = best
  rw [if_pos (by decide), rootBranch522]
  exact outerScan523

set_option maxRecDepth 4096 in
theorem outerScan521 : scan edge (search edge 4) 5 [] 0 tail521 best = best := by
  have cons : tail521 = 3767 :: tail522 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates521_checked]
  change scan edge (search edge 4) 5 [] 0 tail522
    (if 4 ≤ rootCandidates521.length then search edge 4 [3767] rootCandidates521 3767 best else best) = best
  rw [if_pos (by decide), rootBranch521]
  exact outerScan522

end ProjectEulerSolutions.PrimePairUpper.Search
