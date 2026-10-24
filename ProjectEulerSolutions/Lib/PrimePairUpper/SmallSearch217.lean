import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch216
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch217

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan68 : scan edge (search edge 4) 5 [] 0 tail68 best = best := by
  have cons : tail68 = 353 :: tail69 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates68_checked]
  change scan edge (search edge 4) 5 [] 0 tail69
    (if 4 ≤ rootCandidates68.length then search edge 4 [353] rootCandidates68 353 best else best) = best
  rw [if_pos (by decide), rootBranch68]
  exact outerScan69

set_option maxRecDepth 4096 in
theorem outerScan67 : scan edge (search edge 4) 5 [] 0 tail67 best = best := by
  have cons : tail67 = 349 :: tail68 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates67_checked]
  change scan edge (search edge 4) 5 [] 0 tail68
    (if 4 ≤ rootCandidates67.length then search edge 4 [349] rootCandidates67 349 best else best) = best
  rw [if_pos (by decide), rootBranch67]
  exact outerScan68

end ProjectEulerSolutions.PrimePairUpper.Search
