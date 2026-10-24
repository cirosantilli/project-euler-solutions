import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch138
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch139

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan258 : scan edge (search edge 4) 5 [] 0 tail258 best = best := by
  have cons : tail258 = 1663 :: tail259 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates258_checked]
  change scan edge (search edge 4) 5 [] 0 tail259
    (if 4 ≤ rootCandidates258.length then search edge 4 [1663] rootCandidates258 1663 best else best) = best
  rw [if_pos (by decide), rootBranch258]
  exact outerScan259

set_option maxRecDepth 4096 in
theorem outerScan257 : scan edge (search edge 4) 5 [] 0 tail257 best = best := by
  have cons : tail257 = 1657 :: tail258 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates257_checked]
  change scan edge (search edge 4) 5 [] 0 tail258
    (if 4 ≤ rootCandidates257.length then search edge 4 [1657] rootCandidates257 1657 best else best) = best
  rw [if_pos (by decide), rootBranch257]
  exact outerScan258

set_option maxRecDepth 4096 in
theorem outerScan256 : scan edge (search edge 4) 5 [] 0 tail256 best = best := by
  have cons : tail256 = 1637 :: tail257 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates256_checked]
  change scan edge (search edge 4) 5 [] 0 tail257
    (if 4 ≤ rootCandidates256.length then search edge 4 [1637] rootCandidates256 1637 best else best) = best
  rw [if_pos (by decide), rootBranch256]
  exact outerScan257

end ProjectEulerSolutions.PrimePairUpper.Search
