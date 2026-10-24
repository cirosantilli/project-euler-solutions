import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch157
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch158

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan201 : scan edge (search edge 4) 5 [] 0 tail201 best = best := by
  have cons : tail201 = 1249 :: tail202 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates201_checked]
  change scan edge (search edge 4) 5 [] 0 tail202
    (if 4 ≤ rootCandidates201.length then search edge 4 [1249] rootCandidates201 1249 best else best) = best
  rw [if_pos (by decide), rootBranch201]
  exact outerScan202

set_option maxRecDepth 4096 in
theorem outerScan200 : scan edge (search edge 4) 5 [] 0 tail200 best = best := by
  have cons : tail200 = 1237 :: tail201 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates200_checked]
  change scan edge (search edge 4) 5 [] 0 tail201
    (if 4 ≤ rootCandidates200.length then search edge 4 [1237] rootCandidates200 1237 best else best) = best
  rw [if_pos (by decide), rootBranch200]
  exact outerScan201

set_option maxRecDepth 4096 in
theorem outerScan199 : scan edge (search edge 4) 5 [] 0 tail199 best = best := by
  have cons : tail199 = 1231 :: tail200 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates199_checked]
  change scan edge (search edge 4) 5 [] 0 tail200
    (if 4 ≤ rootCandidates199.length then search edge 4 [1231] rootCandidates199 1231 best else best) = best
  rw [if_pos (by decide), rootBranch199]
  exact outerScan200

end ProjectEulerSolutions.PrimePairUpper.Search
