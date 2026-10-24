import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch035
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch036

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan759 : scan edge (search edge 4) 5 [] 0 tail759 best = best := by
  have cons : tail759 = 5807 :: tail760 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates759_checked]
  change scan edge (search edge 4) 5 [] 0 tail760
    (if 4 ≤ rootCandidates759.length then search edge 4 [5807] rootCandidates759 5807 best else best) = best
  rw [if_pos (by decide), rootBranch759]
  exact outerScan760

set_option maxRecDepth 4096 in
theorem outerScan758 : scan edge (search edge 4) 5 [] 0 tail758 best = best := by
  have cons : tail758 = 5801 :: tail759 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates758_checked]
  change scan edge (search edge 4) 5 [] 0 tail759
    (if 4 ≤ rootCandidates758.length then search edge 4 [5801] rootCandidates758 5801 best else best) = best
  rw [if_pos (by decide), rootBranch758]
  exact outerScan759

set_option maxRecDepth 4096 in
theorem outerScan757 : scan edge (search edge 4) 5 [] 0 tail757 best = best := by
  have cons : tail757 = 5791 :: tail758 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates757_checked]
  change scan edge (search edge 4) 5 [] 0 tail758
    (if 4 ≤ rootCandidates757.length then search edge 4 [5791] rootCandidates757 5791 best else best) = best
  rw [if_pos (by decide), rootBranch757]
  exact outerScan758

set_option maxRecDepth 4096 in
theorem outerScan756 : scan edge (search edge 4) 5 [] 0 tail756 best = best := by
  have cons : tail756 = 5783 :: tail757 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates756_checked]
  change scan edge (search edge 4) 5 [] 0 tail757
    (if 4 ≤ rootCandidates756.length then search edge 4 [5783] rootCandidates756 5783 best else best) = best
  rw [if_pos (by decide), rootBranch756]
  exact outerScan757

set_option maxRecDepth 4096 in
theorem outerScan755 : scan edge (search edge 4) 5 [] 0 tail755 best = best := by
  have cons : tail755 = 5779 :: tail756 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates755_checked]
  change scan edge (search edge 4) 5 [] 0 tail756
    (if 4 ≤ rootCandidates755.length then search edge 4 [5779] rootCandidates755 5779 best else best) = best
  rw [if_pos (by decide), rootBranch755]
  exact outerScan756

set_option maxRecDepth 4096 in
theorem outerScan754 : scan edge (search edge 4) 5 [] 0 tail754 best = best := by
  have cons : tail754 = 5749 :: tail755 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates754_checked]
  change scan edge (search edge 4) 5 [] 0 tail755
    (if 4 ≤ rootCandidates754.length then search edge 4 [5749] rootCandidates754 5749 best else best) = best
  rw [if_pos (by decide), rootBranch754]
  exact outerScan755

set_option maxRecDepth 4096 in
theorem outerScan753 : scan edge (search edge 4) 5 [] 0 tail753 best = best := by
  have cons : tail753 = 5743 :: tail754 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates753_checked]
  change scan edge (search edge 4) 5 [] 0 tail754
    (if 4 ≤ rootCandidates753.length then search edge 4 [5743] rootCandidates753 5743 best else best) = best
  rw [if_pos (by decide), rootBranch753]
  exact outerScan754

set_option maxRecDepth 4096 in
theorem outerScan752 : scan edge (search edge 4) 5 [] 0 tail752 best = best := by
  have cons : tail752 = 5741 :: tail753 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates752_checked]
  change scan edge (search edge 4) 5 [] 0 tail753
    (if 4 ≤ rootCandidates752.length then search edge 4 [5741] rootCandidates752 5741 best else best) = best
  rw [if_pos (by decide), rootBranch752]
  exact outerScan753

end ProjectEulerSolutions.PrimePairUpper.Search
