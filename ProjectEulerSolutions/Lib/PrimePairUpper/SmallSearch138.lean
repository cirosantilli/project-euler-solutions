import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch137
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch138

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan261 : scan edge (search edge 4) 5 [] 0 tail261 best = best := by
  have cons : tail261 = 1693 :: tail262 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates261_checked]
  change scan edge (search edge 4) 5 [] 0 tail262
    (if 4 ≤ rootCandidates261.length then search edge 4 [1693] rootCandidates261 1693 best else best) = best
  rw [if_pos (by decide), rootBranch261]
  exact outerScan262

set_option maxRecDepth 4096 in
theorem outerScan260 : scan edge (search edge 4) 5 [] 0 tail260 best = best := by
  have cons : tail260 = 1669 :: tail261 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates260_checked]
  change scan edge (search edge 4) 5 [] 0 tail261
    (if 4 ≤ rootCandidates260.length then search edge 4 [1669] rootCandidates260 1669 best else best) = best
  rw [if_pos (by decide), rootBranch260]
  exact outerScan261

set_option maxRecDepth 4096 in
theorem outerScan259 : scan edge (search edge 4) 5 [] 0 tail259 best = best := by
  have cons : tail259 = 1667 :: tail260 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates259_checked]
  change scan edge (search edge 4) 5 [] 0 tail260
    (if 4 ≤ rootCandidates259.length then search edge 4 [1667] rootCandidates259 1667 best else best) = best
  rw [if_pos (by decide), rootBranch259]
  exact outerScan260

end ProjectEulerSolutions.PrimePairUpper.Search
