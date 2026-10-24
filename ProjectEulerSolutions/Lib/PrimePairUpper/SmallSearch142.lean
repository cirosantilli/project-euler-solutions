import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch141
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch142

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan249 : scan edge (search edge 4) 5 [] 0 tail249 best = best := by
  have cons : tail249 = 1601 :: tail250 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates249_checked]
  change scan edge (search edge 4) 5 [] 0 tail250
    (if 4 ≤ rootCandidates249.length then search edge 4 [1601] rootCandidates249 1601 best else best) = best
  rw [if_pos (by decide), rootBranch249]
  exact outerScan250

set_option maxRecDepth 4096 in
theorem outerScan248 : scan edge (search edge 4) 5 [] 0 tail248 best = best := by
  have cons : tail248 = 1597 :: tail249 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates248_checked]
  change scan edge (search edge 4) 5 [] 0 tail249
    (if 4 ≤ rootCandidates248.length then search edge 4 [1597] rootCandidates248 1597 best else best) = best
  rw [if_pos (by decide), rootBranch248]
  exact outerScan249

end ProjectEulerSolutions.PrimePairUpper.Search
