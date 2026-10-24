import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch234
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch235

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan37 : scan edge (search edge 4) 5 [] 0 tail37 best = best := by
  have cons : tail37 = 173 :: tail38 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates37_checked]
  change scan edge (search edge 4) 5 [] 0 tail38
    (if 4 ≤ rootCandidates37.length then search edge 4 [173] rootCandidates37 173 best else best) = best
  rw [if_pos (by decide), rootBranch37]
  exact outerScan38

set_option maxRecDepth 4096 in
theorem outerScan36 : scan edge (search edge 4) 5 [] 0 tail36 best = best := by
  have cons : tail36 = 167 :: tail37 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates36_checked]
  change scan edge (search edge 4) 5 [] 0 tail37
    (if 4 ≤ rootCandidates36.length then search edge 4 [167] rootCandidates36 167 best else best) = best
  rw [if_pos (by decide), rootBranch36]
  exact outerScan37

end ProjectEulerSolutions.PrimePairUpper.Search
