import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch143
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch144

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan244 : scan edge (search edge 4) 5 [] 0 tail244 best = best := by
  have cons : tail244 = 1567 :: tail245 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates244_checked]
  change scan edge (search edge 4) 5 [] 0 tail245
    (if 4 ≤ rootCandidates244.length then search edge 4 [1567] rootCandidates244 1567 best else best) = best
  rw [if_pos (by decide), rootBranch244]
  exact outerScan245

set_option maxRecDepth 4096 in
theorem outerScan243 : scan edge (search edge 4) 5 [] 0 tail243 best = best := by
  have cons : tail243 = 1559 :: tail244 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates243_checked]
  change scan edge (search edge 4) 5 [] 0 tail244
    (if 4 ≤ rootCandidates243.length then search edge 4 [1559] rootCandidates243 1559 best else best) = best
  rw [if_pos (by decide), rootBranch243]
  exact outerScan244

set_option maxRecDepth 4096 in
theorem outerScan242 : scan edge (search edge 4) 5 [] 0 tail242 best = best := by
  have cons : tail242 = 1553 :: tail243 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates242_checked]
  change scan edge (search edge 4) 5 [] 0 tail243
    (if 4 ≤ rootCandidates242.length then search edge 4 [1553] rootCandidates242 1553 best else best) = best
  rw [if_pos (by decide), rootBranch242]
  exact outerScan243

end ProjectEulerSolutions.PrimePairUpper.Search
