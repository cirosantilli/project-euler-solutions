import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch175
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch176

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan156 : scan edge (search edge 4) 5 [] 0 tail156 best = best := by
  have cons : tail156 = 937 :: tail157 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates156_checked]
  change scan edge (search edge 4) 5 [] 0 tail157
    (if 4 ≤ rootCandidates156.length then search edge 4 [937] rootCandidates156 937 best else best) = best
  rw [if_pos (by decide), rootBranch156]
  exact outerScan157

set_option maxRecDepth 4096 in
theorem outerScan155 : scan edge (search edge 4) 5 [] 0 tail155 best = best := by
  have cons : tail155 = 929 :: tail156 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates155_checked]
  change scan edge (search edge 4) 5 [] 0 tail156
    (if 4 ≤ rootCandidates155.length then search edge 4 [929] rootCandidates155 929 best else best) = best
  rw [if_pos (by decide), rootBranch155]
  exact outerScan156

end ProjectEulerSolutions.PrimePairUpper.Search
