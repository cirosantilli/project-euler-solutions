import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch163
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch164

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan186 : scan edge (search edge 4) 5 [] 0 tail186 best = best := by
  have cons : tail186 = 1129 :: tail187 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates186_checked]
  change scan edge (search edge 4) 5 [] 0 tail187
    (if 4 ≤ rootCandidates186.length then search edge 4 [1129] rootCandidates186 1129 best else best) = best
  rw [if_pos (by decide), rootBranch186]
  exact outerScan187

set_option maxRecDepth 4096 in
theorem outerScan185 : scan edge (search edge 4) 5 [] 0 tail185 best = best := by
  have cons : tail185 = 1123 :: tail186 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates185_checked]
  change scan edge (search edge 4) 5 [] 0 tail186
    (if 4 ≤ rootCandidates185.length then search edge 4 [1123] rootCandidates185 1123 best else best) = best
  rw [if_pos (by decide), rootBranch185]
  exact outerScan186

end ProjectEulerSolutions.PrimePairUpper.Search
