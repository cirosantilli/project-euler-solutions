import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch207
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch208

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan86 : scan edge (search edge 4) 5 [] 0 tail86 best = best := by
  have cons : tail86 = 461 :: tail87 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates86_checked]
  change scan edge (search edge 4) 5 [] 0 tail87
    (if 4 ≤ rootCandidates86.length then search edge 4 [461] rootCandidates86 461 best else best) = best
  rw [if_pos (by decide), rootBranch86]
  exact outerScan87

set_option maxRecDepth 4096 in
theorem outerScan85 : scan edge (search edge 4) 5 [] 0 tail85 best = best := by
  have cons : tail85 = 457 :: tail86 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates85_checked]
  change scan edge (search edge 4) 5 [] 0 tail86
    (if 4 ≤ rootCandidates85.length then search edge 4 [457] rootCandidates85 457 best else best) = best
  rw [if_pos (by decide), rootBranch85]
  exact outerScan86

end ProjectEulerSolutions.PrimePairUpper.Search
