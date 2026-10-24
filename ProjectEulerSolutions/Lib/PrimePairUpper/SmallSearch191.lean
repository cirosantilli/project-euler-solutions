import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch190
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch191

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan122 : scan edge (search edge 4) 5 [] 0 tail122 best = best := by
  have cons : tail122 = 691 :: tail123 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates122_checked]
  change scan edge (search edge 4) 5 [] 0 tail123
    (if 4 ≤ rootCandidates122.length then search edge 4 [691] rootCandidates122 691 best else best) = best
  rw [if_pos (by decide), rootBranch122]
  exact outerScan123

set_option maxRecDepth 4096 in
theorem outerScan121 : scan edge (search edge 4) 5 [] 0 tail121 best = best := by
  have cons : tail121 = 683 :: tail122 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates121_checked]
  change scan edge (search edge 4) 5 [] 0 tail122
    (if 4 ≤ rootCandidates121.length then search edge 4 [683] rootCandidates121 683 best else best) = best
  rw [if_pos (by decide), rootBranch121]
  exact outerScan122

end ProjectEulerSolutions.PrimePairUpper.Search
