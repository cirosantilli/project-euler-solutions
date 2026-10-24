import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch050
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch051

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan646 : scan edge (search edge 4) 5 [] 0 tail646 best = best := by
  have cons : tail646 = 4817 :: tail647 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates646_checked]
  change scan edge (search edge 4) 5 [] 0 tail647
    (if 4 ≤ rootCandidates646.length then search edge 4 [4817] rootCandidates646 4817 best else best) = best
  rw [if_pos (by decide), rootBranch646]
  exact outerScan647

set_option maxRecDepth 4096 in
theorem outerScan645 : scan edge (search edge 4) 5 [] 0 tail645 best = best := by
  have cons : tail645 = 4813 :: tail646 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates645_checked]
  change scan edge (search edge 4) 5 [] 0 tail646
    (if 4 ≤ rootCandidates645.length then search edge 4 [4813] rootCandidates645 4813 best else best) = best
  rw [if_pos (by decide), rootBranch645]
  exact outerScan646

set_option maxRecDepth 4096 in
theorem outerScan644 : scan edge (search edge 4) 5 [] 0 tail644 best = best := by
  have cons : tail644 = 4801 :: tail645 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates644_checked]
  change scan edge (search edge 4) 5 [] 0 tail645
    (if 4 ≤ rootCandidates644.length then search edge 4 [4801] rootCandidates644 4801 best else best) = best
  rw [if_pos (by decide), rootBranch644]
  exact outerScan645

set_option maxRecDepth 4096 in
theorem outerScan643 : scan edge (search edge 4) 5 [] 0 tail643 best = best := by
  have cons : tail643 = 4799 :: tail644 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates643_checked]
  change scan edge (search edge 4) 5 [] 0 tail644
    (if 4 ≤ rootCandidates643.length then search edge 4 [4799] rootCandidates643 4799 best else best) = best
  rw [if_pos (by decide), rootBranch643]
  exact outerScan644

set_option maxRecDepth 4096 in
theorem outerScan642 : scan edge (search edge 4) 5 [] 0 tail642 best = best := by
  have cons : tail642 = 4793 :: tail643 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates642_checked]
  change scan edge (search edge 4) 5 [] 0 tail643
    (if 4 ≤ rootCandidates642.length then search edge 4 [4793] rootCandidates642 4793 best else best) = best
  rw [if_pos (by decide), rootBranch642]
  exact outerScan643

set_option maxRecDepth 4096 in
theorem outerScan641 : scan edge (search edge 4) 5 [] 0 tail641 best = best := by
  have cons : tail641 = 4789 :: tail642 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates641_checked]
  change scan edge (search edge 4) 5 [] 0 tail642
    (if 4 ≤ rootCandidates641.length then search edge 4 [4789] rootCandidates641 4789 best else best) = best
  rw [if_pos (by decide), rootBranch641]
  exact outerScan642

set_option maxRecDepth 4096 in
theorem outerScan640 : scan edge (search edge 4) 5 [] 0 tail640 best = best := by
  have cons : tail640 = 4787 :: tail641 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates640_checked]
  change scan edge (search edge 4) 5 [] 0 tail641
    (if 4 ≤ rootCandidates640.length then search edge 4 [4787] rootCandidates640 4787 best else best) = best
  rw [if_pos (by decide), rootBranch640]
  exact outerScan641

end ProjectEulerSolutions.PrimePairUpper.Search
