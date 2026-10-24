import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch177
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch178

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan152 : scan edge (search edge 4) 5 [] 0 tail152 best = best := by
  have cons : tail152 = 907 :: tail153 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates152_checked]
  change scan edge (search edge 4) 5 [] 0 tail153
    (if 4 ≤ rootCandidates152.length then search edge 4 [907] rootCandidates152 907 best else best) = best
  rw [if_pos (by decide), rootBranch152]
  exact outerScan153

set_option maxRecDepth 4096 in
theorem outerScan151 : scan edge (search edge 4) 5 [] 0 tail151 best = best := by
  have cons : tail151 = 887 :: tail152 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates151_checked]
  change scan edge (search edge 4) 5 [] 0 tail152
    (if 4 ≤ rootCandidates151.length then search edge 4 [887] rootCandidates151 887 best else best) = best
  rw [if_pos (by decide), rootBranch151]
  exact outerScan152

end ProjectEulerSolutions.PrimePairUpper.Search
