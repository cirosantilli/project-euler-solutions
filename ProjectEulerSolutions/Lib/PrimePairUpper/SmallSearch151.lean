import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch150
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch151

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan224 : scan edge (search edge 4) 5 [] 0 tail224 best = best := by
  have cons : tail224 = 1433 :: tail225 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates224_checked]
  change scan edge (search edge 4) 5 [] 0 tail225
    (if 4 ≤ rootCandidates224.length then search edge 4 [1433] rootCandidates224 1433 best else best) = best
  rw [if_pos (by decide), rootBranch224]
  exact outerScan225

set_option maxRecDepth 4096 in
theorem outerScan223 : scan edge (search edge 4) 5 [] 0 tail223 best = best := by
  have cons : tail223 = 1429 :: tail224 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates223_checked]
  change scan edge (search edge 4) 5 [] 0 tail224
    (if 4 ≤ rootCandidates223.length then search edge 4 [1429] rootCandidates223 1429 best else best) = best
  rw [if_pos (by decide), rootBranch223]
  exact outerScan224

set_option maxRecDepth 4096 in
theorem outerScan222 : scan edge (search edge 4) 5 [] 0 tail222 best = best := by
  have cons : tail222 = 1427 :: tail223 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates222_checked]
  change scan edge (search edge 4) 5 [] 0 tail223
    (if 4 ≤ rootCandidates222.length then search edge 4 [1427] rootCandidates222 1427 best else best) = best
  rw [if_pos (by decide), rootBranch222]
  exact outerScan223

end ProjectEulerSolutions.PrimePairUpper.Search
