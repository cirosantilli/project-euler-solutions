import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch098
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch099

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan399 : scan edge (search edge 4) 5 [] 0 tail399 best = best := by
  have cons : tail399 = 2753 :: tail400 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates399_checked]
  change scan edge (search edge 4) 5 [] 0 tail400
    (if 4 ≤ rootCandidates399.length then search edge 4 [2753] rootCandidates399 2753 best else best) = best
  rw [if_pos (by decide), rootBranch399]
  exact outerScan400

set_option maxRecDepth 4096 in
theorem outerScan398 : scan edge (search edge 4) 5 [] 0 tail398 best = best := by
  have cons : tail398 = 2749 :: tail399 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates398_checked]
  change scan edge (search edge 4) 5 [] 0 tail399
    (if 4 ≤ rootCandidates398.length then search edge 4 [2749] rootCandidates398 2749 best else best) = best
  rw [if_pos (by decide), rootBranch398]
  exact outerScan399

set_option maxRecDepth 4096 in
theorem outerScan397 : scan edge (search edge 4) 5 [] 0 tail397 best = best := by
  have cons : tail397 = 2741 :: tail398 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates397_checked]
  change scan edge (search edge 4) 5 [] 0 tail398
    (if 4 ≤ rootCandidates397.length then search edge 4 [2741] rootCandidates397 2741 best else best) = best
  rw [if_pos (by decide), rootBranch397]
  exact outerScan398

set_option maxRecDepth 4096 in
theorem outerScan396 : scan edge (search edge 4) 5 [] 0 tail396 best = best := by
  have cons : tail396 = 2731 :: tail397 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates396_checked]
  change scan edge (search edge 4) 5 [] 0 tail397
    (if 4 ≤ rootCandidates396.length then search edge 4 [2731] rootCandidates396 2731 best else best) = best
  rw [if_pos (by decide), rootBranch396]
  exact outerScan397

end ProjectEulerSolutions.PrimePairUpper.Search
