import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch056
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch057

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan606 : scan edge (search edge 4) 5 [] 0 tail606 best = best := by
  have cons : tail606 = 4483 :: tail607 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates606_checked]
  change scan edge (search edge 4) 5 [] 0 tail607
    (if 4 ≤ rootCandidates606.length then search edge 4 [4483] rootCandidates606 4483 best else best) = best
  rw [if_pos (by decide), rootBranch606]
  exact outerScan607

set_option maxRecDepth 4096 in
theorem outerScan605 : scan edge (search edge 4) 5 [] 0 tail605 best = best := by
  have cons : tail605 = 4481 :: tail606 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates605_checked]
  change scan edge (search edge 4) 5 [] 0 tail606
    (if 4 ≤ rootCandidates605.length then search edge 4 [4481] rootCandidates605 4481 best else best) = best
  rw [if_pos (by decide), rootBranch605]
  exact outerScan606

set_option maxRecDepth 4096 in
theorem outerScan604 : scan edge (search edge 4) 5 [] 0 tail604 best = best := by
  have cons : tail604 = 4463 :: tail605 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates604_checked]
  change scan edge (search edge 4) 5 [] 0 tail605
    (if 4 ≤ rootCandidates604.length then search edge 4 [4463] rootCandidates604 4463 best else best) = best
  rw [if_pos (by decide), rootBranch604]
  exact outerScan605

set_option maxRecDepth 4096 in
theorem outerScan603 : scan edge (search edge 4) 5 [] 0 tail603 best = best := by
  have cons : tail603 = 4457 :: tail604 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates603_checked]
  change scan edge (search edge 4) 5 [] 0 tail604
    (if 4 ≤ rootCandidates603.length then search edge 4 [4457] rootCandidates603 4457 best else best) = best
  rw [if_pos (by decide), rootBranch603]
  exact outerScan604

set_option maxRecDepth 4096 in
theorem outerScan602 : scan edge (search edge 4) 5 [] 0 tail602 best = best := by
  have cons : tail602 = 4451 :: tail603 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates602_checked]
  change scan edge (search edge 4) 5 [] 0 tail603
    (if 4 ≤ rootCandidates602.length then search edge 4 [4451] rootCandidates602 4451 best else best) = best
  rw [if_pos (by decide), rootBranch602]
  exact outerScan603

set_option maxRecDepth 4096 in
theorem outerScan601 : scan edge (search edge 4) 5 [] 0 tail601 best = best := by
  have cons : tail601 = 4447 :: tail602 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates601_checked]
  change scan edge (search edge 4) 5 [] 0 tail602
    (if 4 ≤ rootCandidates601.length then search edge 4 [4447] rootCandidates601 4447 best else best) = best
  rw [if_pos (by decide), rootBranch601]
  exact outerScan602

end ProjectEulerSolutions.PrimePairUpper.Search
