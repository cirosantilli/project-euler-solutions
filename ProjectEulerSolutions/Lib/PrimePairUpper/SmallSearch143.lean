import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch142
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch143

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan247 : scan edge (search edge 4) 5 [] 0 tail247 best = best := by
  have cons : tail247 = 1583 :: tail248 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates247_checked]
  change scan edge (search edge 4) 5 [] 0 tail248
    (if 4 ≤ rootCandidates247.length then search edge 4 [1583] rootCandidates247 1583 best else best) = best
  rw [if_pos (by decide), rootBranch247]
  exact outerScan248

set_option maxRecDepth 4096 in
theorem outerScan246 : scan edge (search edge 4) 5 [] 0 tail246 best = best := by
  have cons : tail246 = 1579 :: tail247 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates246_checked]
  change scan edge (search edge 4) 5 [] 0 tail247
    (if 4 ≤ rootCandidates246.length then search edge 4 [1579] rootCandidates246 1579 best else best) = best
  rw [if_pos (by decide), rootBranch246]
  exact outerScan247

set_option maxRecDepth 4096 in
theorem outerScan245 : scan edge (search edge 4) 5 [] 0 tail245 best = best := by
  have cons : tail245 = 1571 :: tail246 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates245_checked]
  change scan edge (search edge 4) 5 [] 0 tail246
    (if 4 ≤ rootCandidates245.length then search edge 4 [1571] rootCandidates245 1571 best else best) = best
  rw [if_pos (by decide), rootBranch245]
  exact outerScan246

end ProjectEulerSolutions.PrimePairUpper.Search
