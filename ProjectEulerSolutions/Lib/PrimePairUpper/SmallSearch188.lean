import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch187
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch188

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan129 : scan edge (search edge 4) 5 [] 0 tail129 best = best := by
  have cons : tail129 = 743 :: tail130 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates129_checked]
  change scan edge (search edge 4) 5 [] 0 tail130
    (if 4 ≤ rootCandidates129.length then search edge 4 [743] rootCandidates129 743 best else best) = best
  rw [if_pos (by decide), rootBranch129]
  exact outerScan130

set_option maxRecDepth 4096 in
theorem outerScan128 : scan edge (search edge 4) 5 [] 0 tail128 best = best := by
  have cons : tail128 = 739 :: tail129 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates128_checked]
  change scan edge (search edge 4) 5 [] 0 tail129
    (if 4 ≤ rootCandidates128.length then search edge 4 [739] rootCandidates128 739 best else best) = best
  rw [if_pos (by decide), rootBranch128]
  exact outerScan129

set_option maxRecDepth 4096 in
theorem outerScan127 : scan edge (search edge 4) 5 [] 0 tail127 best = best := by
  have cons : tail127 = 733 :: tail128 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates127_checked]
  change scan edge (search edge 4) 5 [] 0 tail128
    (if 4 ≤ rootCandidates127.length then search edge 4 [733] rootCandidates127 733 best else best) = best
  rw [if_pos (by decide), rootBranch127]
  exact outerScan128

end ProjectEulerSolutions.PrimePairUpper.Search
