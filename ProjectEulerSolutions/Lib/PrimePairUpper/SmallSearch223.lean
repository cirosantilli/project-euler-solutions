import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch222
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch223

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan56 : scan edge (search edge 4) 5 [] 0 tail56 best = best := by
  have cons : tail56 = 277 :: tail57 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates56_checked]
  change scan edge (search edge 4) 5 [] 0 tail57
    (if 4 ≤ rootCandidates56.length then search edge 4 [277] rootCandidates56 277 best else best) = best
  rw [if_pos (by decide), rootBranch56]
  exact outerScan57

set_option maxRecDepth 4096 in
theorem outerScan55 : scan edge (search edge 4) 5 [] 0 tail55 best = best := by
  have cons : tail55 = 271 :: tail56 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates55_checked]
  change scan edge (search edge 4) 5 [] 0 tail56
    (if 4 ≤ rootCandidates55.length then search edge 4 [271] rootCandidates55 271 best else best) = best
  rw [if_pos (by decide), rootBranch55]
  exact outerScan56

end ProjectEulerSolutions.PrimePairUpper.Search
