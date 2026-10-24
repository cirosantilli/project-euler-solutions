import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch152
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch153

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan217 : scan edge (search edge 4) 5 [] 0 tail217 best = best := by
  have cons : tail217 = 1373 :: tail218 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates217_checked]
  change scan edge (search edge 4) 5 [] 0 tail218
    (if 4 ≤ rootCandidates217.length then search edge 4 [1373] rootCandidates217 1373 best else best) = best
  rw [if_pos (by decide), rootBranch217]
  exact outerScan218

set_option maxRecDepth 4096 in
theorem outerScan216 : scan edge (search edge 4) 5 [] 0 tail216 best = best := by
  have cons : tail216 = 1367 :: tail217 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates216_checked]
  change scan edge (search edge 4) 5 [] 0 tail217
    (if 4 ≤ rootCandidates216.length then search edge 4 [1367] rootCandidates216 1367 best else best) = best
  rw [if_pos (by decide), rootBranch216]
  exact outerScan217

set_option maxRecDepth 4096 in
theorem outerScan215 : scan edge (search edge 4) 5 [] 0 tail215 best = best := by
  have cons : tail215 = 1361 :: tail216 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates215_checked]
  change scan edge (search edge 4) 5 [] 0 tail216
    (if 4 ≤ rootCandidates215.length then search edge 4 [1361] rootCandidates215 1361 best else best) = best
  rw [if_pos (by decide), rootBranch215]
  exact outerScan216

end ProjectEulerSolutions.PrimePairUpper.Search
