import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch034
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch035

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan767 : scan edge (search edge 4) 5 [] 0 tail767 best = best := by
  have cons : tail767 = 5857 :: tail768 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates767_checked]
  change scan edge (search edge 4) 5 [] 0 tail768
    (if 4 ≤ rootCandidates767.length then search edge 4 [5857] rootCandidates767 5857 best else best) = best
  rw [if_pos (by decide), rootBranch767]
  exact outerScan768

set_option maxRecDepth 4096 in
theorem outerScan766 : scan edge (search edge 4) 5 [] 0 tail766 best = best := by
  have cons : tail766 = 5851 :: tail767 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates766_checked]
  change scan edge (search edge 4) 5 [] 0 tail767
    (if 4 ≤ rootCandidates766.length then search edge 4 [5851] rootCandidates766 5851 best else best) = best
  rw [if_pos (by decide), rootBranch766]
  exact outerScan767

set_option maxRecDepth 4096 in
theorem outerScan765 : scan edge (search edge 4) 5 [] 0 tail765 best = best := by
  have cons : tail765 = 5849 :: tail766 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates765_checked]
  change scan edge (search edge 4) 5 [] 0 tail766
    (if 4 ≤ rootCandidates765.length then search edge 4 [5849] rootCandidates765 5849 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan766

set_option maxRecDepth 4096 in
theorem outerScan764 : scan edge (search edge 4) 5 [] 0 tail764 best = best := by
  have cons : tail764 = 5843 :: tail765 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates764_checked]
  change scan edge (search edge 4) 5 [] 0 tail765
    (if 4 ≤ rootCandidates764.length then search edge 4 [5843] rootCandidates764 5843 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan765

set_option maxRecDepth 4096 in
theorem outerScan763 : scan edge (search edge 4) 5 [] 0 tail763 best = best := by
  have cons : tail763 = 5839 :: tail764 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates763_checked]
  change scan edge (search edge 4) 5 [] 0 tail764
    (if 4 ≤ rootCandidates763.length then search edge 4 [5839] rootCandidates763 5839 best else best) = best
  rw [if_pos (by decide), rootBranch763]
  exact outerScan764

set_option maxRecDepth 4096 in
theorem outerScan762 : scan edge (search edge 4) 5 [] 0 tail762 best = best := by
  have cons : tail762 = 5827 :: tail763 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates762_checked]
  change scan edge (search edge 4) 5 [] 0 tail763
    (if 4 ≤ rootCandidates762.length then search edge 4 [5827] rootCandidates762 5827 best else best) = best
  rw [if_pos (by decide), rootBranch762]
  exact outerScan763

set_option maxRecDepth 4096 in
theorem outerScan761 : scan edge (search edge 4) 5 [] 0 tail761 best = best := by
  have cons : tail761 = 5821 :: tail762 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates761_checked]
  change scan edge (search edge 4) 5 [] 0 tail762
    (if 4 ≤ rootCandidates761.length then search edge 4 [5821] rootCandidates761 5821 best else best) = best
  rw [if_pos (by decide), rootBranch761]
  exact outerScan762

set_option maxRecDepth 4096 in
theorem outerScan760 : scan edge (search edge 4) 5 [] 0 tail760 best = best := by
  have cons : tail760 = 5813 :: tail761 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates760_checked]
  change scan edge (search edge 4) 5 [] 0 tail761
    (if 4 ≤ rootCandidates760.length then search edge 4 [5813] rootCandidates760 5813 best else best) = best
  rw [if_pos (by decide), rootBranch760]
  exact outerScan761

end ProjectEulerSolutions.PrimePairUpper.Search
