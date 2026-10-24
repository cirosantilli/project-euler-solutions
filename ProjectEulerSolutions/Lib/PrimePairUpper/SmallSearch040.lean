import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch039
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch040

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan728 : scan edge (search edge 4) 5 [] 0 tail728 best = best := by
  have cons : tail728 = 5527 :: tail729 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates728_checked]
  change scan edge (search edge 4) 5 [] 0 tail729
    (if 4 ≤ rootCandidates728.length then search edge 4 [5527] rootCandidates728 5527 best else best) = best
  rw [if_pos (by decide), rootBranch728]
  exact outerScan729

set_option maxRecDepth 4096 in
theorem outerScan727 : scan edge (search edge 4) 5 [] 0 tail727 best = best := by
  have cons : tail727 = 5521 :: tail728 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates727_checked]
  change scan edge (search edge 4) 5 [] 0 tail728
    (if 4 ≤ rootCandidates727.length then search edge 4 [5521] rootCandidates727 5521 best else best) = best
  rw [if_pos (by decide), rootBranch727]
  exact outerScan728

set_option maxRecDepth 4096 in
theorem outerScan726 : scan edge (search edge 4) 5 [] 0 tail726 best = best := by
  have cons : tail726 = 5519 :: tail727 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates726_checked]
  change scan edge (search edge 4) 5 [] 0 tail727
    (if 4 ≤ rootCandidates726.length then search edge 4 [5519] rootCandidates726 5519 best else best) = best
  rw [if_pos (by decide), rootBranch726]
  exact outerScan727

set_option maxRecDepth 4096 in
theorem outerScan725 : scan edge (search edge 4) 5 [] 0 tail725 best = best := by
  have cons : tail725 = 5507 :: tail726 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates725_checked]
  change scan edge (search edge 4) 5 [] 0 tail726
    (if 4 ≤ rootCandidates725.length then search edge 4 [5507] rootCandidates725 5507 best else best) = best
  rw [if_pos (by decide), rootBranch725]
  exact outerScan726

set_option maxRecDepth 4096 in
theorem outerScan724 : scan edge (search edge 4) 5 [] 0 tail724 best = best := by
  have cons : tail724 = 5503 :: tail725 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates724_checked]
  change scan edge (search edge 4) 5 [] 0 tail725
    (if 4 ≤ rootCandidates724.length then search edge 4 [5503] rootCandidates724 5503 best else best) = best
  rw [if_pos (by decide), rootBranch724]
  exact outerScan725

set_option maxRecDepth 4096 in
theorem outerScan723 : scan edge (search edge 4) 5 [] 0 tail723 best = best := by
  have cons : tail723 = 5501 :: tail724 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates723_checked]
  change scan edge (search edge 4) 5 [] 0 tail724
    (if 4 ≤ rootCandidates723.length then search edge 4 [5501] rootCandidates723 5501 best else best) = best
  rw [if_pos (by decide), rootBranch723]
  exact outerScan724

set_option maxRecDepth 4096 in
theorem outerScan722 : scan edge (search edge 4) 5 [] 0 tail722 best = best := by
  have cons : tail722 = 5483 :: tail723 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates722_checked]
  change scan edge (search edge 4) 5 [] 0 tail723
    (if 4 ≤ rootCandidates722.length then search edge 4 [5483] rootCandidates722 5483 best else best) = best
  rw [if_pos (by decide), rootBranch722]
  exact outerScan723

set_option maxRecDepth 4096 in
theorem outerScan721 : scan edge (search edge 4) 5 [] 0 tail721 best = best := by
  have cons : tail721 = 5479 :: tail722 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates721_checked]
  change scan edge (search edge 4) 5 [] 0 tail722
    (if 4 ≤ rootCandidates721.length then search edge 4 [5479] rootCandidates721 5479 best else best) = best
  rw [if_pos (by decide), rootBranch721]
  exact outerScan722

end ProjectEulerSolutions.PrimePairUpper.Search
