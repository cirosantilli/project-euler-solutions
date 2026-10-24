import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch091
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch092

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan425 : scan edge (search edge 4) 5 [] 0 tail425 best = best := by
  have cons : tail425 = 2969 :: tail426 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates425_checked]
  change scan edge (search edge 4) 5 [] 0 tail426
    (if 4 ≤ rootCandidates425.length then search edge 4 [2969] rootCandidates425 2969 best else best) = best
  rw [if_pos (by decide), rootBranch425]
  exact outerScan426

set_option maxRecDepth 4096 in
theorem outerScan424 : scan edge (search edge 4) 5 [] 0 tail424 best = best := by
  have cons : tail424 = 2963 :: tail425 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates424_checked]
  change scan edge (search edge 4) 5 [] 0 tail425
    (if 4 ≤ rootCandidates424.length then search edge 4 [2963] rootCandidates424 2963 best else best) = best
  rw [if_pos (by decide), rootBranch424]
  exact outerScan425

set_option maxRecDepth 4096 in
theorem outerScan423 : scan edge (search edge 4) 5 [] 0 tail423 best = best := by
  have cons : tail423 = 2957 :: tail424 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates423_checked]
  change scan edge (search edge 4) 5 [] 0 tail424
    (if 4 ≤ rootCandidates423.length then search edge 4 [2957] rootCandidates423 2957 best else best) = best
  rw [if_pos (by decide), rootBranch423]
  exact outerScan424

set_option maxRecDepth 4096 in
theorem outerScan422 : scan edge (search edge 4) 5 [] 0 tail422 best = best := by
  have cons : tail422 = 2953 :: tail423 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates422_checked]
  change scan edge (search edge 4) 5 [] 0 tail423
    (if 4 ≤ rootCandidates422.length then search edge 4 [2953] rootCandidates422 2953 best else best) = best
  rw [if_pos (by decide), rootBranch422]
  exact outerScan423

end ProjectEulerSolutions.PrimePairUpper.Search
