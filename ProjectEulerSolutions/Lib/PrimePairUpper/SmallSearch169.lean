import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch168
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch169

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan174 : scan edge (search edge 4) 5 [] 0 tail174 best = best := by
  have cons : tail174 = 1051 :: tail175 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates174_checked]
  change scan edge (search edge 4) 5 [] 0 tail175
    (if 4 ≤ rootCandidates174.length then search edge 4 [1051] rootCandidates174 1051 best else best) = best
  rw [if_pos (by decide), rootBranch174]
  exact outerScan175

set_option maxRecDepth 4096 in
theorem outerScan173 : scan edge (search edge 4) 5 [] 0 tail173 best = best := by
  have cons : tail173 = 1049 :: tail174 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates173_checked]
  change scan edge (search edge 4) 5 [] 0 tail174
    (if 4 ≤ rootCandidates173.length then search edge 4 [1049] rootCandidates173 1049 best else best) = best
  rw [if_pos (by decide), rootBranch173]
  exact outerScan174

end ProjectEulerSolutions.PrimePairUpper.Search
