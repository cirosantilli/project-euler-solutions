import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch061
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch062

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan574 : scan edge (search edge 4) 5 [] 0 tail574 best = best := by
  have cons : tail574 = 4217 :: tail575 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates574_checked]
  change scan edge (search edge 4) 5 [] 0 tail575
    (if 4 ≤ rootCandidates574.length then search edge 4 [4217] rootCandidates574 4217 best else best) = best
  rw [if_pos (by decide), rootBranch574]
  exact outerScan575

set_option maxRecDepth 4096 in
theorem outerScan573 : scan edge (search edge 4) 5 [] 0 tail573 best = best := by
  have cons : tail573 = 4211 :: tail574 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates573_checked]
  change scan edge (search edge 4) 5 [] 0 tail574
    (if 4 ≤ rootCandidates573.length then search edge 4 [4211] rootCandidates573 4211 best else best) = best
  rw [if_pos (by decide), rootBranch573]
  exact outerScan574

set_option maxRecDepth 4096 in
theorem outerScan572 : scan edge (search edge 4) 5 [] 0 tail572 best = best := by
  have cons : tail572 = 4201 :: tail573 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates572_checked]
  change scan edge (search edge 4) 5 [] 0 tail573
    (if 4 ≤ rootCandidates572.length then search edge 4 [4201] rootCandidates572 4201 best else best) = best
  rw [if_pos (by decide), rootBranch572]
  exact outerScan573

set_option maxRecDepth 4096 in
theorem outerScan571 : scan edge (search edge 4) 5 [] 0 tail571 best = best := by
  have cons : tail571 = 4177 :: tail572 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates571_checked]
  change scan edge (search edge 4) 5 [] 0 tail572
    (if 4 ≤ rootCandidates571.length then search edge 4 [4177] rootCandidates571 4177 best else best) = best
  rw [if_pos (by decide), rootBranch571]
  exact outerScan572

set_option maxRecDepth 4096 in
theorem outerScan570 : scan edge (search edge 4) 5 [] 0 tail570 best = best := by
  have cons : tail570 = 4159 :: tail571 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates570_checked]
  change scan edge (search edge 4) 5 [] 0 tail571
    (if 4 ≤ rootCandidates570.length then search edge 4 [4159] rootCandidates570 4159 best else best) = best
  rw [if_pos (by decide), rootBranch570]
  exact outerScan571

set_option maxRecDepth 4096 in
theorem outerScan569 : scan edge (search edge 4) 5 [] 0 tail569 best = best := by
  have cons : tail569 = 4157 :: tail570 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates569_checked]
  change scan edge (search edge 4) 5 [] 0 tail570
    (if 4 ≤ rootCandidates569.length then search edge 4 [4157] rootCandidates569 4157 best else best) = best
  rw [if_pos (by decide), rootBranch569]
  exact outerScan570

set_option maxRecDepth 4096 in
theorem outerScan568 : scan edge (search edge 4) 5 [] 0 tail568 best = best := by
  have cons : tail568 = 4153 :: tail569 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates568_checked]
  change scan edge (search edge 4) 5 [] 0 tail569
    (if 4 ≤ rootCandidates568.length then search edge 4 [4153] rootCandidates568 4153 best else best) = best
  rw [if_pos (by decide), rootBranch568]
  exact outerScan569

end ProjectEulerSolutions.PrimePairUpper.Search
