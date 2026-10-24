import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch146
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch147

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan236 : scan edge (search edge 4) 5 [] 0 tail236 best = best := by
  have cons : tail236 = 1499 :: tail237 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates236_checked]
  change scan edge (search edge 4) 5 [] 0 tail237
    (if 4 ≤ rootCandidates236.length then search edge 4 [1499] rootCandidates236 1499 best else best) = best
  rw [if_pos (by decide), rootBranch236]
  exact outerScan237

set_option maxRecDepth 4096 in
theorem outerScan235 : scan edge (search edge 4) 5 [] 0 tail235 best = best := by
  have cons : tail235 = 1493 :: tail236 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates235_checked]
  change scan edge (search edge 4) 5 [] 0 tail236
    (if 4 ≤ rootCandidates235.length then search edge 4 [1493] rootCandidates235 1493 best else best) = best
  rw [if_pos (by decide), rootBranch235]
  exact outerScan236

set_option maxRecDepth 4096 in
theorem outerScan234 : scan edge (search edge 4) 5 [] 0 tail234 best = best := by
  have cons : tail234 = 1489 :: tail235 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates234_checked]
  change scan edge (search edge 4) 5 [] 0 tail235
    (if 4 ≤ rootCandidates234.length then search edge 4 [1489] rootCandidates234 1489 best else best) = best
  rw [if_pos (by decide), rootBranch234]
  exact outerScan235

set_option maxRecDepth 4096 in
theorem outerScan233 : scan edge (search edge 4) 5 [] 0 tail233 best = best := by
  have cons : tail233 = 1487 :: tail234 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates233_checked]
  change scan edge (search edge 4) 5 [] 0 tail234
    (if 4 ≤ rootCandidates233.length then search edge 4 [1487] rootCandidates233 1487 best else best) = best
  rw [if_pos (by decide), rootBranch233]
  exact outerScan234

end ProjectEulerSolutions.PrimePairUpper.Search
