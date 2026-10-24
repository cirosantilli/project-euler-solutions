import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch116
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch117

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan331 : scan edge (search edge 4) 5 [] 0 tail331 best = best := by
  have cons : tail331 = 2243 :: tail332 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates331_checked]
  change scan edge (search edge 4) 5 [] 0 tail332
    (if 4 ≤ rootCandidates331.length then search edge 4 [2243] rootCandidates331 2243 best else best) = best
  rw [if_pos (by decide), rootBranch331]
  exact outerScan332

set_option maxRecDepth 4096 in
theorem outerScan330 : scan edge (search edge 4) 5 [] 0 tail330 best = best := by
  have cons : tail330 = 2239 :: tail331 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates330_checked]
  change scan edge (search edge 4) 5 [] 0 tail331
    (if 4 ≤ rootCandidates330.length then search edge 4 [2239] rootCandidates330 2239 best else best) = best
  rw [if_pos (by decide), rootBranch330]
  exact outerScan331

set_option maxRecDepth 4096 in
theorem outerScan329 : scan edge (search edge 4) 5 [] 0 tail329 best = best := by
  have cons : tail329 = 2237 :: tail330 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates329_checked]
  change scan edge (search edge 4) 5 [] 0 tail330
    (if 4 ≤ rootCandidates329.length then search edge 4 [2237] rootCandidates329 2237 best else best) = best
  rw [if_pos (by decide), rootBranch329]
  exact outerScan330

end ProjectEulerSolutions.PrimePairUpper.Search
