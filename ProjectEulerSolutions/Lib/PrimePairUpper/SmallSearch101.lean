import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch100
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch101

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan390 : scan edge (search edge 4) 5 [] 0 tail390 best = best := by
  have cons : tail390 = 2699 :: tail391 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates390_checked]
  change scan edge (search edge 4) 5 [] 0 tail391
    (if 4 ≤ rootCandidates390.length then search edge 4 [2699] rootCandidates390 2699 best else best) = best
  rw [if_pos (by decide), rootBranch390]
  exact outerScan391

set_option maxRecDepth 4096 in
theorem outerScan389 : scan edge (search edge 4) 5 [] 0 tail389 best = best := by
  have cons : tail389 = 2693 :: tail390 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates389_checked]
  change scan edge (search edge 4) 5 [] 0 tail390
    (if 4 ≤ rootCandidates389.length then search edge 4 [2693] rootCandidates389 2693 best else best) = best
  rw [if_pos (by decide), rootBranch389]
  exact outerScan390

set_option maxRecDepth 4096 in
theorem outerScan388 : scan edge (search edge 4) 5 [] 0 tail388 best = best := by
  have cons : tail388 = 2689 :: tail389 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates388_checked]
  change scan edge (search edge 4) 5 [] 0 tail389
    (if 4 ≤ rootCandidates388.length then search edge 4 [2689] rootCandidates388 2689 best else best) = best
  rw [if_pos (by decide), rootBranch388]
  exact outerScan389

set_option maxRecDepth 4096 in
theorem outerScan387 : scan edge (search edge 4) 5 [] 0 tail387 best = best := by
  have cons : tail387 = 2687 :: tail388 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates387_checked]
  change scan edge (search edge 4) 5 [] 0 tail388
    (if 4 ≤ rootCandidates387.length then search edge 4 [2687] rootCandidates387 2687 best else best) = best
  rw [if_pos (by decide), rootBranch387]
  exact outerScan388

end ProjectEulerSolutions.PrimePairUpper.Search
