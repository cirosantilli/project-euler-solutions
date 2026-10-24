import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch179
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch180

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan148 : scan edge (search edge 4) 5 [] 0 tail148 best = best := by
  have cons : tail148 = 877 :: tail149 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates148_checked]
  change scan edge (search edge 4) 5 [] 0 tail149
    (if 4 ≤ rootCandidates148.length then search edge 4 [877] rootCandidates148 877 best else best) = best
  rw [if_pos (by decide), rootBranch148]
  exact outerScan149

set_option maxRecDepth 4096 in
theorem outerScan147 : scan edge (search edge 4) 5 [] 0 tail147 best = best := by
  have cons : tail147 = 863 :: tail148 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates147_checked]
  change scan edge (search edge 4) 5 [] 0 tail148
    (if 4 ≤ rootCandidates147.length then search edge 4 [863] rootCandidates147 863 best else best) = best
  rw [if_pos (by decide), rootBranch147]
  exact outerScan148

end ProjectEulerSolutions.PrimePairUpper.Search
