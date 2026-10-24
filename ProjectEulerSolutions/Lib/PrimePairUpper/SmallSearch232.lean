import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch231
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch232

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan43 : scan edge (search edge 4) 5 [] 0 tail43 best = best := by
  have cons : tail43 = 199 :: tail44 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates43_checked]
  change scan edge (search edge 4) 5 [] 0 tail44
    (if 4 ≤ rootCandidates43.length then search edge 4 [199] rootCandidates43 199 best else best) = best
  rw [if_pos (by decide), rootBranch43]
  exact outerScan44

set_option maxRecDepth 4096 in
theorem outerScan42 : scan edge (search edge 4) 5 [] 0 tail42 best = best := by
  have cons : tail42 = 197 :: tail43 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates42_checked]
  change scan edge (search edge 4) 5 [] 0 tail43
    (if 4 ≤ rootCandidates42.length then search edge 4 [197] rootCandidates42 197 best else best) = best
  rw [if_pos (by decide), rootBranch42]
  exact outerScan43

end ProjectEulerSolutions.PrimePairUpper.Search
