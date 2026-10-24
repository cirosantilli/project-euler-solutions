import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch040
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch041

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan720 : scan edge (search edge 4) 5 [] 0 tail720 best = best := by
  have cons : tail720 = 5477 :: tail721 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates720_checked]
  change scan edge (search edge 4) 5 [] 0 tail721
    (if 4 ≤ rootCandidates720.length then search edge 4 [5477] rootCandidates720 5477 best else best) = best
  rw [if_pos (by decide), rootBranch720]
  exact outerScan721

set_option maxRecDepth 4096 in
theorem outerScan719 : scan edge (search edge 4) 5 [] 0 tail719 best = best := by
  have cons : tail719 = 5471 :: tail720 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates719_checked]
  change scan edge (search edge 4) 5 [] 0 tail720
    (if 4 ≤ rootCandidates719.length then search edge 4 [5471] rootCandidates719 5471 best else best) = best
  rw [if_pos (by decide), rootBranch719]
  exact outerScan720

set_option maxRecDepth 4096 in
theorem outerScan718 : scan edge (search edge 4) 5 [] 0 tail718 best = best := by
  have cons : tail718 = 5449 :: tail719 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates718_checked]
  change scan edge (search edge 4) 5 [] 0 tail719
    (if 4 ≤ rootCandidates718.length then search edge 4 [5449] rootCandidates718 5449 best else best) = best
  rw [if_pos (by decide), rootBranch718]
  exact outerScan719

set_option maxRecDepth 4096 in
theorem outerScan717 : scan edge (search edge 4) 5 [] 0 tail717 best = best := by
  have cons : tail717 = 5443 :: tail718 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates717_checked]
  change scan edge (search edge 4) 5 [] 0 tail718
    (if 4 ≤ rootCandidates717.length then search edge 4 [5443] rootCandidates717 5443 best else best) = best
  rw [if_pos (by decide), rootBranch717]
  exact outerScan718

set_option maxRecDepth 4096 in
theorem outerScan716 : scan edge (search edge 4) 5 [] 0 tail716 best = best := by
  have cons : tail716 = 5441 :: tail717 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates716_checked]
  change scan edge (search edge 4) 5 [] 0 tail717
    (if 4 ≤ rootCandidates716.length then search edge 4 [5441] rootCandidates716 5441 best else best) = best
  rw [if_pos (by decide), rootBranch716]
  exact outerScan717

set_option maxRecDepth 4096 in
theorem outerScan715 : scan edge (search edge 4) 5 [] 0 tail715 best = best := by
  have cons : tail715 = 5437 :: tail716 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates715_checked]
  change scan edge (search edge 4) 5 [] 0 tail716
    (if 4 ≤ rootCandidates715.length then search edge 4 [5437] rootCandidates715 5437 best else best) = best
  rw [if_pos (by decide), rootBranch715]
  exact outerScan716

set_option maxRecDepth 4096 in
theorem outerScan714 : scan edge (search edge 4) 5 [] 0 tail714 best = best := by
  have cons : tail714 = 5431 :: tail715 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates714_checked]
  change scan edge (search edge 4) 5 [] 0 tail715
    (if 4 ≤ rootCandidates714.length then search edge 4 [5431] rootCandidates714 5431 best else best) = best
  rw [if_pos (by decide), rootBranch714]
  exact outerScan715

set_option maxRecDepth 4096 in
theorem outerScan713 : scan edge (search edge 4) 5 [] 0 tail713 best = best := by
  have cons : tail713 = 5419 :: tail714 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates713_checked]
  change scan edge (search edge 4) 5 [] 0 tail714
    (if 4 ≤ rootCandidates713.length then search edge 4 [5419] rootCandidates713 5419 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan714

end ProjectEulerSolutions.PrimePairUpper.Search
