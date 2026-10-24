import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch181
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch182

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan143 : scan edge (search edge 4) 5 [] 0 tail143 best = best := by
  have cons : tail143 = 839 :: tail144 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates143_checked]
  change scan edge (search edge 4) 5 [] 0 tail144
    (if 4 ≤ rootCandidates143.length then search edge 4 [839] rootCandidates143 839 best else best) = best
  rw [if_pos (by decide), rootBranch143]
  exact outerScan144

set_option maxRecDepth 4096 in
theorem outerScan142 : scan edge (search edge 4) 5 [] 0 tail142 best = best := by
  have cons : tail142 = 829 :: tail143 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates142_checked]
  change scan edge (search edge 4) 5 [] 0 tail143
    (if 4 ≤ rootCandidates142.length then search edge 4 [829] rootCandidates142 829 best else best) = best
  rw [if_pos (by decide), rootBranch142]
  exact outerScan143

end ProjectEulerSolutions.PrimePairUpper.Search
