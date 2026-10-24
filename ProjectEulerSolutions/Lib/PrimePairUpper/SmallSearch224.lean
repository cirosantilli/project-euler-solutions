import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch223
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch224

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan54 : scan edge (search edge 4) 5 [] 0 tail54 best = best := by
  have cons : tail54 = 269 :: tail55 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates54_checked]
  change scan edge (search edge 4) 5 [] 0 tail55
    (if 4 ≤ rootCandidates54.length then search edge 4 [269] rootCandidates54 269 best else best) = best
  rw [if_pos (by decide), rootBranch54]
  exact outerScan55

set_option maxRecDepth 4096 in
theorem outerScan53 : scan edge (search edge 4) 5 [] 0 tail53 best = best := by
  have cons : tail53 = 263 :: tail54 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates53_checked]
  change scan edge (search edge 4) 5 [] 0 tail54
    (if 4 ≤ rootCandidates53.length then search edge 4 [263] rootCandidates53 263 best else best) = best
  rw [if_pos (by decide), rootBranch53]
  exact outerScan54

end ProjectEulerSolutions.PrimePairUpper.Search
