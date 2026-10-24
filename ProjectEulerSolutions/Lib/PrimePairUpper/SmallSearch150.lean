import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch149
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch150

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan226 : scan edge (search edge 4) 5 [] 0 tail226 best = best := by
  have cons : tail226 = 1447 :: tail227 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates226_checked]
  change scan edge (search edge 4) 5 [] 0 tail227
    (if 4 ≤ rootCandidates226.length then search edge 4 [1447] rootCandidates226 1447 best else best) = best
  rw [if_pos (by decide), rootBranch226]
  exact outerScan227

set_option maxRecDepth 4096 in
theorem outerScan225 : scan edge (search edge 4) 5 [] 0 tail225 best = best := by
  have cons : tail225 = 1439 :: tail226 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates225_checked]
  change scan edge (search edge 4) 5 [] 0 tail226
    (if 4 ≤ rootCandidates225.length then search edge 4 [1439] rootCandidates225 1439 best else best) = best
  rw [if_pos (by decide), rootBranch225]
  exact outerScan226

end ProjectEulerSolutions.PrimePairUpper.Search
