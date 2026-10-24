import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch096
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch097

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan405 : scan edge (search edge 4) 5 [] 0 tail405 best = best := by
  have cons : tail405 = 2801 :: tail406 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates405_checked]
  change scan edge (search edge 4) 5 [] 0 tail406
    (if 4 ≤ rootCandidates405.length then search edge 4 [2801] rootCandidates405 2801 best else best) = best
  rw [if_pos (by decide), rootBranch405]
  exact outerScan406

set_option maxRecDepth 4096 in
theorem outerScan404 : scan edge (search edge 4) 5 [] 0 tail404 best = best := by
  have cons : tail404 = 2797 :: tail405 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates404_checked]
  change scan edge (search edge 4) 5 [] 0 tail405
    (if 4 ≤ rootCandidates404.length then search edge 4 [2797] rootCandidates404 2797 best else best) = best
  rw [if_pos (by decide), rootBranch404]
  exact outerScan405

set_option maxRecDepth 4096 in
theorem outerScan403 : scan edge (search edge 4) 5 [] 0 tail403 best = best := by
  have cons : tail403 = 2791 :: tail404 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates403_checked]
  change scan edge (search edge 4) 5 [] 0 tail404
    (if 4 ≤ rootCandidates403.length then search edge 4 [2791] rootCandidates403 2791 best else best) = best
  rw [if_pos (by decide), rootBranch403]
  exact outerScan404

end ProjectEulerSolutions.PrimePairUpper.Search
