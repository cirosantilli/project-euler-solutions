import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch136
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch137

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan264 : scan edge (search edge 4) 5 [] 0 tail264 best = best := by
  have cons : tail264 = 1709 :: tail265 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates264_checked]
  change scan edge (search edge 4) 5 [] 0 tail265
    (if 4 ≤ rootCandidates264.length then search edge 4 [1709] rootCandidates264 1709 best else best) = best
  rw [if_pos (by decide), rootBranch264]
  exact outerScan265

set_option maxRecDepth 4096 in
theorem outerScan263 : scan edge (search edge 4) 5 [] 0 tail263 best = best := by
  have cons : tail263 = 1699 :: tail264 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates263_checked]
  change scan edge (search edge 4) 5 [] 0 tail264
    (if 4 ≤ rootCandidates263.length then search edge 4 [1699] rootCandidates263 1699 best else best) = best
  rw [if_pos (by decide), rootBranch263]
  exact outerScan264

set_option maxRecDepth 4096 in
theorem outerScan262 : scan edge (search edge 4) 5 [] 0 tail262 best = best := by
  have cons : tail262 = 1697 :: tail263 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates262_checked]
  change scan edge (search edge 4) 5 [] 0 tail263
    (if 4 ≤ rootCandidates262.length then search edge 4 [1697] rootCandidates262 1697 best else best) = best
  rw [if_pos (by decide), rootBranch262]
  exact outerScan263

end ProjectEulerSolutions.PrimePairUpper.Search
