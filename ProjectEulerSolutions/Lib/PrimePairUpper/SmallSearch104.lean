import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch103
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch104

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan378 : scan edge (search edge 4) 5 [] 0 tail378 best = best := by
  have cons : tail378 = 2621 :: tail379 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates378_checked]
  change scan edge (search edge 4) 5 [] 0 tail379
    (if 4 ≤ rootCandidates378.length then search edge 4 [2621] rootCandidates378 2621 best else best) = best
  rw [if_pos (by decide), rootBranch378]
  exact outerScan379

set_option maxRecDepth 4096 in
theorem outerScan377 : scan edge (search edge 4) 5 [] 0 tail377 best = best := by
  have cons : tail377 = 2617 :: tail378 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates377_checked]
  change scan edge (search edge 4) 5 [] 0 tail378
    (if 4 ≤ rootCandidates377.length then search edge 4 [2617] rootCandidates377 2617 best else best) = best
  rw [if_pos (by decide), rootBranch377]
  exact outerScan378

set_option maxRecDepth 4096 in
theorem outerScan376 : scan edge (search edge 4) 5 [] 0 tail376 best = best := by
  have cons : tail376 = 2609 :: tail377 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates376_checked]
  change scan edge (search edge 4) 5 [] 0 tail377
    (if 4 ≤ rootCandidates376.length then search edge 4 [2609] rootCandidates376 2609 best else best) = best
  rw [if_pos (by decide), rootBranch376]
  exact outerScan377

end ProjectEulerSolutions.PrimePairUpper.Search
