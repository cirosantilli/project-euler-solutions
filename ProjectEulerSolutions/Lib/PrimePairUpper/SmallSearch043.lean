import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch042
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch043

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan704 : scan edge (search edge 4) 5 [] 0 tail704 best = best := by
  have cons : tail704 = 5347 :: tail705 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates704_checked]
  change scan edge (search edge 4) 5 [] 0 tail705
    (if 4 ≤ rootCandidates704.length then search edge 4 [5347] rootCandidates704 5347 best else best) = best
  rw [if_pos (by decide), rootBranch704]
  exact outerScan705

set_option maxRecDepth 4096 in
theorem outerScan703 : scan edge (search edge 4) 5 [] 0 tail703 best = best := by
  have cons : tail703 = 5333 :: tail704 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates703_checked]
  change scan edge (search edge 4) 5 [] 0 tail704
    (if 4 ≤ rootCandidates703.length then search edge 4 [5333] rootCandidates703 5333 best else best) = best
  rw [if_pos (by decide), rootBranch703]
  exact outerScan704

set_option maxRecDepth 4096 in
theorem outerScan702 : scan edge (search edge 4) 5 [] 0 tail702 best = best := by
  have cons : tail702 = 5323 :: tail703 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates702_checked]
  change scan edge (search edge 4) 5 [] 0 tail703
    (if 4 ≤ rootCandidates702.length then search edge 4 [5323] rootCandidates702 5323 best else best) = best
  rw [if_pos (by decide), rootBranch702]
  exact outerScan703

set_option maxRecDepth 4096 in
theorem outerScan701 : scan edge (search edge 4) 5 [] 0 tail701 best = best := by
  have cons : tail701 = 5309 :: tail702 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates701_checked]
  change scan edge (search edge 4) 5 [] 0 tail702
    (if 4 ≤ rootCandidates701.length then search edge 4 [5309] rootCandidates701 5309 best else best) = best
  rw [if_pos (by decide), rootBranch701]
  exact outerScan702

set_option maxRecDepth 4096 in
theorem outerScan700 : scan edge (search edge 4) 5 [] 0 tail700 best = best := by
  have cons : tail700 = 5303 :: tail701 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates700_checked]
  change scan edge (search edge 4) 5 [] 0 tail701
    (if 4 ≤ rootCandidates700.length then search edge 4 [5303] rootCandidates700 5303 best else best) = best
  rw [if_pos (by decide), rootBranch700]
  exact outerScan701

set_option maxRecDepth 4096 in
theorem outerScan699 : scan edge (search edge 4) 5 [] 0 tail699 best = best := by
  have cons : tail699 = 5297 :: tail700 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates699_checked]
  change scan edge (search edge 4) 5 [] 0 tail700
    (if 4 ≤ rootCandidates699.length then search edge 4 [5297] rootCandidates699 5297 best else best) = best
  rw [if_pos (by decide), rootBranch699]
  exact outerScan700

set_option maxRecDepth 4096 in
theorem outerScan698 : scan edge (search edge 4) 5 [] 0 tail698 best = best := by
  have cons : tail698 = 5281 :: tail699 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates698_checked]
  change scan edge (search edge 4) 5 [] 0 tail699
    (if 4 ≤ rootCandidates698.length then search edge 4 [5281] rootCandidates698 5281 best else best) = best
  rw [if_pos (by decide), rootBranch698]
  exact outerScan699

end ProjectEulerSolutions.PrimePairUpper.Search
