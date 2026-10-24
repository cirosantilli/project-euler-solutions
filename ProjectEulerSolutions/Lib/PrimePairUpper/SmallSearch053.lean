import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch052
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch053

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan632 : scan edge (search edge 4) 5 [] 0 tail632 best = best := by
  have cons : tail632 = 4703 :: tail633 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates632_checked]
  change scan edge (search edge 4) 5 [] 0 tail633
    (if 4 ≤ rootCandidates632.length then search edge 4 [4703] rootCandidates632 4703 best else best) = best
  rw [if_pos (by decide), rootBranch632]
  exact outerScan633

set_option maxRecDepth 4096 in
theorem outerScan631 : scan edge (search edge 4) 5 [] 0 tail631 best = best := by
  have cons : tail631 = 4691 :: tail632 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates631_checked]
  change scan edge (search edge 4) 5 [] 0 tail632
    (if 4 ≤ rootCandidates631.length then search edge 4 [4691] rootCandidates631 4691 best else best) = best
  rw [if_pos (by decide), rootBranch631]
  exact outerScan632

set_option maxRecDepth 4096 in
theorem outerScan630 : scan edge (search edge 4) 5 [] 0 tail630 best = best := by
  have cons : tail630 = 4679 :: tail631 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates630_checked]
  change scan edge (search edge 4) 5 [] 0 tail631
    (if 4 ≤ rootCandidates630.length then search edge 4 [4679] rootCandidates630 4679 best else best) = best
  rw [if_pos (by decide), rootBranch630]
  exact outerScan631

set_option maxRecDepth 4096 in
theorem outerScan629 : scan edge (search edge 4) 5 [] 0 tail629 best = best := by
  have cons : tail629 = 4673 :: tail630 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates629_checked]
  change scan edge (search edge 4) 5 [] 0 tail630
    (if 4 ≤ rootCandidates629.length then search edge 4 [4673] rootCandidates629 4673 best else best) = best
  rw [if_pos (by decide), rootBranch629]
  exact outerScan630

set_option maxRecDepth 4096 in
theorem outerScan628 : scan edge (search edge 4) 5 [] 0 tail628 best = best := by
  have cons : tail628 = 4663 :: tail629 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates628_checked]
  change scan edge (search edge 4) 5 [] 0 tail629
    (if 4 ≤ rootCandidates628.length then search edge 4 [4663] rootCandidates628 4663 best else best) = best
  rw [if_pos (by decide), rootBranch628]
  exact outerScan629

set_option maxRecDepth 4096 in
theorem outerScan627 : scan edge (search edge 4) 5 [] 0 tail627 best = best := by
  have cons : tail627 = 4657 :: tail628 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates627_checked]
  change scan edge (search edge 4) 5 [] 0 tail628
    (if 4 ≤ rootCandidates627.length then search edge 4 [4657] rootCandidates627 4657 best else best) = best
  rw [if_pos (by decide), rootBranch627]
  exact outerScan628

set_option maxRecDepth 4096 in
theorem outerScan626 : scan edge (search edge 4) 5 [] 0 tail626 best = best := by
  have cons : tail626 = 4651 :: tail627 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates626_checked]
  change scan edge (search edge 4) 5 [] 0 tail627
    (if 4 ≤ rootCandidates626.length then search edge 4 [4651] rootCandidates626 4651 best else best) = best
  rw [if_pos (by decide), rootBranch626]
  exact outerScan627

set_option maxRecDepth 4096 in
theorem outerScan625 : scan edge (search edge 4) 5 [] 0 tail625 best = best := by
  have cons : tail625 = 4649 :: tail626 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates625_checked]
  change scan edge (search edge 4) 5 [] 0 tail626
    (if 4 ≤ rootCandidates625.length then search edge 4 [4649] rootCandidates625 4649 best else best) = best
  rw [if_pos (by decide), rootBranch625]
  exact outerScan626

end ProjectEulerSolutions.PrimePairUpper.Search
