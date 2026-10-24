import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch204
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch205

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan92 : scan edge (search edge 4) 5 [] 0 tail92 best = best := by
  have cons : tail92 = 499 :: tail93 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates92_checked]
  change scan edge (search edge 4) 5 [] 0 tail93
    (if 4 ≤ rootCandidates92.length then search edge 4 [499] rootCandidates92 499 best else best) = best
  rw [if_pos (by decide), rootBranch92]
  exact outerScan93

set_option maxRecDepth 4096 in
theorem outerScan91 : scan edge (search edge 4) 5 [] 0 tail91 best = best := by
  have cons : tail91 = 491 :: tail92 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates91_checked]
  change scan edge (search edge 4) 5 [] 0 tail92
    (if 4 ≤ rootCandidates91.length then search edge 4 [491] rootCandidates91 491 best else best) = best
  rw [if_pos (by decide), rootBranch91]
  exact outerScan92

end ProjectEulerSolutions.PrimePairUpper.Search
