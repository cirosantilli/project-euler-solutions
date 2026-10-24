import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch066
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch067

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan545 : scan edge (search edge 4) 5 [] 0 tail545 best = best := by
  have cons : tail545 = 3947 :: tail546 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates545_checked]
  change scan edge (search edge 4) 5 [] 0 tail546
    (if 4 ≤ rootCandidates545.length then search edge 4 [3947] rootCandidates545 3947 best else best) = best
  rw [if_pos (by decide), rootBranch545]
  exact outerScan546

set_option maxRecDepth 4096 in
theorem outerScan544 : scan edge (search edge 4) 5 [] 0 tail544 best = best := by
  have cons : tail544 = 3943 :: tail545 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates544_checked]
  change scan edge (search edge 4) 5 [] 0 tail545
    (if 4 ≤ rootCandidates544.length then search edge 4 [3943] rootCandidates544 3943 best else best) = best
  rw [if_pos (by decide), rootBranch544]
  exact outerScan545

set_option maxRecDepth 4096 in
theorem outerScan543 : scan edge (search edge 4) 5 [] 0 tail543 best = best := by
  have cons : tail543 = 3931 :: tail544 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates543_checked]
  change scan edge (search edge 4) 5 [] 0 tail544
    (if 4 ≤ rootCandidates543.length then search edge 4 [3931] rootCandidates543 3931 best else best) = best
  rw [if_pos (by decide), rootBranch543]
  exact outerScan544

set_option maxRecDepth 4096 in
theorem outerScan542 : scan edge (search edge 4) 5 [] 0 tail542 best = best := by
  have cons : tail542 = 3929 :: tail543 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates542_checked]
  change scan edge (search edge 4) 5 [] 0 tail543
    (if 4 ≤ rootCandidates542.length then search edge 4 [3929] rootCandidates542 3929 best else best) = best
  rw [if_pos (by decide), rootBranch542]
  exact outerScan543

set_option maxRecDepth 4096 in
theorem outerScan541 : scan edge (search edge 4) 5 [] 0 tail541 best = best := by
  have cons : tail541 = 3923 :: tail542 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates541_checked]
  change scan edge (search edge 4) 5 [] 0 tail542
    (if 4 ≤ rootCandidates541.length then search edge 4 [3923] rootCandidates541 3923 best else best) = best
  rw [if_pos (by decide), rootBranch541]
  exact outerScan542

end ProjectEulerSolutions.PrimePairUpper.Search
