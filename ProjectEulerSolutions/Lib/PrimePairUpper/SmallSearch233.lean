import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch232
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch233

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan41 : scan edge (search edge 4) 5 [] 0 tail41 best = best := by
  have cons : tail41 = 193 :: tail42 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates41_checked]
  change scan edge (search edge 4) 5 [] 0 tail42
    (if 4 ≤ rootCandidates41.length then search edge 4 [193] rootCandidates41 193 best else best) = best
  rw [if_pos (by decide), rootBranch41]
  exact outerScan42

set_option maxRecDepth 4096 in
theorem outerScan40 : scan edge (search edge 4) 5 [] 0 tail40 best = best := by
  have cons : tail40 = 191 :: tail41 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates40_checked]
  change scan edge (search edge 4) 5 [] 0 tail41
    (if 4 ≤ rootCandidates40.length then search edge 4 [191] rootCandidates40 191 best else best) = best
  rw [if_pos (by decide), rootBranch40]
  exact outerScan41

end ProjectEulerSolutions.PrimePairUpper.Search
