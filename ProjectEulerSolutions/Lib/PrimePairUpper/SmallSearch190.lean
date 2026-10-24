import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch189
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch190

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan124 : scan edge (search edge 4) 5 [] 0 tail124 best = best := by
  have cons : tail124 = 709 :: tail125 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates124_checked]
  change scan edge (search edge 4) 5 [] 0 tail125
    (if 4 ≤ rootCandidates124.length then search edge 4 [709] rootCandidates124 709 best else best) = best
  rw [if_pos (by decide), rootBranch124]
  exact outerScan125

set_option maxRecDepth 4096 in
theorem outerScan123 : scan edge (search edge 4) 5 [] 0 tail123 best = best := by
  have cons : tail123 = 701 :: tail124 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates123_checked]
  change scan edge (search edge 4) 5 [] 0 tail124
    (if 4 ≤ rootCandidates123.length then search edge 4 [701] rootCandidates123 701 best else best) = best
  rw [if_pos (by decide), rootBranch123]
  exact outerScan124

end ProjectEulerSolutions.PrimePairUpper.Search
