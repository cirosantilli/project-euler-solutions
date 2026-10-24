import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch156
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch157

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan204 : scan edge (search edge 4) 5 [] 0 tail204 best = best := by
  have cons : tail204 = 1279 :: tail205 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates204_checked]
  change scan edge (search edge 4) 5 [] 0 tail205
    (if 4 ≤ rootCandidates204.length then search edge 4 [1279] rootCandidates204 1279 best else best) = best
  rw [if_pos (by decide), rootBranch204]
  exact outerScan205

set_option maxRecDepth 4096 in
theorem outerScan203 : scan edge (search edge 4) 5 [] 0 tail203 best = best := by
  have cons : tail203 = 1277 :: tail204 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates203_checked]
  change scan edge (search edge 4) 5 [] 0 tail204
    (if 4 ≤ rootCandidates203.length then search edge 4 [1277] rootCandidates203 1277 best else best) = best
  rw [if_pos (by decide), rootBranch203]
  exact outerScan204

set_option maxRecDepth 4096 in
theorem outerScan202 : scan edge (search edge 4) 5 [] 0 tail202 best = best := by
  have cons : tail202 = 1259 :: tail203 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates202_checked]
  change scan edge (search edge 4) 5 [] 0 tail203
    (if 4 ≤ rootCandidates202.length then search edge 4 [1259] rootCandidates202 1259 best else best) = best
  rw [if_pos (by decide), rootBranch202]
  exact outerScan203

end ProjectEulerSolutions.PrimePairUpper.Search
