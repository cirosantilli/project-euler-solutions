import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch180
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch181

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan146 : scan edge (search edge 4) 5 [] 0 tail146 best = best := by
  have cons : tail146 = 859 :: tail147 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates146_checked]
  change scan edge (search edge 4) 5 [] 0 tail147
    (if 4 ≤ rootCandidates146.length then search edge 4 [859] rootCandidates146 859 best else best) = best
  rw [if_pos (by decide), rootBranch146]
  exact outerScan147

set_option maxRecDepth 4096 in
theorem outerScan145 : scan edge (search edge 4) 5 [] 0 tail145 best = best := by
  have cons : tail145 = 857 :: tail146 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates145_checked]
  change scan edge (search edge 4) 5 [] 0 tail146
    (if 4 ≤ rootCandidates145.length then search edge 4 [857] rootCandidates145 857 best else best) = best
  rw [if_pos (by decide), rootBranch145]
  exact outerScan146

set_option maxRecDepth 4096 in
theorem outerScan144 : scan edge (search edge 4) 5 [] 0 tail144 best = best := by
  have cons : tail144 = 853 :: tail145 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates144_checked]
  change scan edge (search edge 4) 5 [] 0 tail145
    (if 4 ≤ rootCandidates144.length then search edge 4 [853] rootCandidates144 853 best else best) = best
  rw [if_pos (by decide), rootBranch144]
  exact outerScan145

end ProjectEulerSolutions.PrimePairUpper.Search
