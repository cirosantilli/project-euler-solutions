import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch026
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch027

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan831 : scan edge (search edge 4) 5 [] 0 tail831 best = best := by
  have cons : tail831 = 6397 :: tail832 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates831_checked]
  change scan edge (search edge 4) 5 [] 0 tail832
    (if 4 ≤ rootCandidates831.length then search edge 4 [6397] rootCandidates831 6397 best else best) = best
  rw [if_pos (by decide), rootBranch831]
  exact outerScan832

set_option maxRecDepth 4096 in
theorem outerScan830 : scan edge (search edge 4) 5 [] 0 tail830 best = best := by
  have cons : tail830 = 6389 :: tail831 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates830_checked]
  change scan edge (search edge 4) 5 [] 0 tail831
    (if 4 ≤ rootCandidates830.length then search edge 4 [6389] rootCandidates830 6389 best else best) = best
  rw [if_pos (by decide), rootBranch830]
  exact outerScan831

set_option maxRecDepth 4096 in
theorem outerScan829 : scan edge (search edge 4) 5 [] 0 tail829 best = best := by
  have cons : tail829 = 6379 :: tail830 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates829_checked]
  change scan edge (search edge 4) 5 [] 0 tail830
    (if 4 ≤ rootCandidates829.length then search edge 4 [6379] rootCandidates829 6379 best else best) = best
  rw [if_pos (by decide), rootBranch829]
  exact outerScan830

set_option maxRecDepth 4096 in
theorem outerScan828 : scan edge (search edge 4) 5 [] 0 tail828 best = best := by
  have cons : tail828 = 6373 :: tail829 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates828_checked]
  change scan edge (search edge 4) 5 [] 0 tail829
    (if 4 ≤ rootCandidates828.length then search edge 4 [6373] rootCandidates828 6373 best else best) = best
  rw [if_pos (by decide), rootBranch828]
  exact outerScan829

set_option maxRecDepth 4096 in
theorem outerScan827 : scan edge (search edge 4) 5 [] 0 tail827 best = best := by
  have cons : tail827 = 6367 :: tail828 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates827_checked]
  change scan edge (search edge 4) 5 [] 0 tail828
    (if 4 ≤ rootCandidates827.length then search edge 4 [6367] rootCandidates827 6367 best else best) = best
  rw [if_pos (by decide), rootBranch827]
  exact outerScan828

set_option maxRecDepth 4096 in
theorem outerScan826 : scan edge (search edge 4) 5 [] 0 tail826 best = best := by
  have cons : tail826 = 6361 :: tail827 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates826_checked]
  change scan edge (search edge 4) 5 [] 0 tail827
    (if 4 ≤ rootCandidates826.length then search edge 4 [6361] rootCandidates826 6361 best else best) = best
  rw [if_pos (by decide), rootBranch826]
  exact outerScan827

set_option maxRecDepth 4096 in
theorem outerScan825 : scan edge (search edge 4) 5 [] 0 tail825 best = best := by
  have cons : tail825 = 6359 :: tail826 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates825_checked]
  change scan edge (search edge 4) 5 [] 0 tail826
    (if 4 ≤ rootCandidates825.length then search edge 4 [6359] rootCandidates825 6359 best else best) = best
  rw [if_pos (by decide), rootBranch825]
  exact outerScan826

set_option maxRecDepth 4096 in
theorem outerScan824 : scan edge (search edge 4) 5 [] 0 tail824 best = best := by
  have cons : tail824 = 6353 :: tail825 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates824_checked]
  change scan edge (search edge 4) 5 [] 0 tail825
    (if 4 ≤ rootCandidates824.length then search edge 4 [6353] rootCandidates824 6353 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan825

end ProjectEulerSolutions.PrimePairUpper.Search
