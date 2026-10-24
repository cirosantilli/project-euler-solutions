import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch178
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch179

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan150 : scan edge (search edge 4) 5 [] 0 tail150 best = best := by
  have cons : tail150 = 883 :: tail151 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates150_checked]
  change scan edge (search edge 4) 5 [] 0 tail151
    (if 4 ≤ rootCandidates150.length then search edge 4 [883] rootCandidates150 883 best else best) = best
  rw [if_pos (by decide), rootBranch150]
  exact outerScan151

set_option maxRecDepth 4096 in
theorem outerScan149 : scan edge (search edge 4) 5 [] 0 tail149 best = best := by
  have cons : tail149 = 881 :: tail150 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates149_checked]
  change scan edge (search edge 4) 5 [] 0 tail150
    (if 4 ≤ rootCandidates149.length then search edge 4 [881] rootCandidates149 881 best else best) = best
  rw [if_pos (by decide), rootBranch149]
  exact outerScan150

end ProjectEulerSolutions.PrimePairUpper.Search
