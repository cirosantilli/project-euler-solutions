import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch064
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch065

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan556 : scan edge (search edge 4) 5 [] 0 tail556 best = best := by
  have cons : tail556 = 4051 :: tail557 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates556_checked]
  change scan edge (search edge 4) 5 [] 0 tail557
    (if 4 ≤ rootCandidates556.length then search edge 4 [4051] rootCandidates556 4051 best else best) = best
  rw [if_pos (by decide), rootBranch556]
  exact outerScan557

set_option maxRecDepth 4096 in
theorem outerScan555 : scan edge (search edge 4) 5 [] 0 tail555 best = best := by
  have cons : tail555 = 4049 :: tail556 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates555_checked]
  change scan edge (search edge 4) 5 [] 0 tail556
    (if 4 ≤ rootCandidates555.length then search edge 4 [4049] rootCandidates555 4049 best else best) = best
  rw [if_pos (by decide), rootBranch555]
  exact outerScan556

set_option maxRecDepth 4096 in
theorem outerScan554 : scan edge (search edge 4) 5 [] 0 tail554 best = best := by
  have cons : tail554 = 4027 :: tail555 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates554_checked]
  change scan edge (search edge 4) 5 [] 0 tail555
    (if 4 ≤ rootCandidates554.length then search edge 4 [4027] rootCandidates554 4027 best else best) = best
  rw [if_pos (by decide), rootBranch554]
  exact outerScan555

set_option maxRecDepth 4096 in
theorem outerScan553 : scan edge (search edge 4) 5 [] 0 tail553 best = best := by
  have cons : tail553 = 4021 :: tail554 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates553_checked]
  change scan edge (search edge 4) 5 [] 0 tail554
    (if 4 ≤ rootCandidates553.length then search edge 4 [4021] rootCandidates553 4021 best else best) = best
  rw [if_pos (by decide), rootBranch553]
  exact outerScan554

set_option maxRecDepth 4096 in
theorem outerScan552 : scan edge (search edge 4) 5 [] 0 tail552 best = best := by
  have cons : tail552 = 4019 :: tail553 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates552_checked]
  change scan edge (search edge 4) 5 [] 0 tail553
    (if 4 ≤ rootCandidates552.length then search edge 4 [4019] rootCandidates552 4019 best else best) = best
  rw [if_pos (by decide), rootBranch552]
  exact outerScan553

set_option maxRecDepth 4096 in
theorem outerScan551 : scan edge (search edge 4) 5 [] 0 tail551 best = best := by
  have cons : tail551 = 4013 :: tail552 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates551_checked]
  change scan edge (search edge 4) 5 [] 0 tail552
    (if 4 ≤ rootCandidates551.length then search edge 4 [4013] rootCandidates551 4013 best else best) = best
  rw [if_pos (by decide), rootBranch551]
  exact outerScan552

end ProjectEulerSolutions.PrimePairUpper.Search
