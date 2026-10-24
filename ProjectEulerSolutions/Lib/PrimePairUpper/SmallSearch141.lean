import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch140
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch141

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan252 : scan edge (search edge 4) 5 [] 0 tail252 best = best := by
  have cons : tail252 = 1613 :: tail253 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates252_checked]
  change scan edge (search edge 4) 5 [] 0 tail253
    (if 4 ≤ rootCandidates252.length then search edge 4 [1613] rootCandidates252 1613 best else best) = best
  rw [if_pos (by decide), rootBranch252]
  exact outerScan253

set_option maxRecDepth 4096 in
theorem outerScan251 : scan edge (search edge 4) 5 [] 0 tail251 best = best := by
  have cons : tail251 = 1609 :: tail252 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates251_checked]
  change scan edge (search edge 4) 5 [] 0 tail252
    (if 4 ≤ rootCandidates251.length then search edge 4 [1609] rootCandidates251 1609 best else best) = best
  rw [if_pos (by decide), rootBranch251]
  exact outerScan252

set_option maxRecDepth 4096 in
theorem outerScan250 : scan edge (search edge 4) 5 [] 0 tail250 best = best := by
  have cons : tail250 = 1607 :: tail251 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates250_checked]
  change scan edge (search edge 4) 5 [] 0 tail251
    (if 4 ≤ rootCandidates250.length then search edge 4 [1607] rootCandidates250 1607 best else best) = best
  rw [if_pos (by decide), rootBranch250]
  exact outerScan251

end ProjectEulerSolutions.PrimePairUpper.Search
