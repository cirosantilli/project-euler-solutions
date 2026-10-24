import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch176
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch177

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan154 : scan edge (search edge 4) 5 [] 0 tail154 best = best := by
  have cons : tail154 = 919 :: tail155 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates154_checked]
  change scan edge (search edge 4) 5 [] 0 tail155
    (if 4 ≤ rootCandidates154.length then search edge 4 [919] rootCandidates154 919 best else best) = best
  rw [if_pos (by decide), rootBranch154]
  exact outerScan155

set_option maxRecDepth 4096 in
theorem outerScan153 : scan edge (search edge 4) 5 [] 0 tail153 best = best := by
  have cons : tail153 = 911 :: tail154 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates153_checked]
  change scan edge (search edge 4) 5 [] 0 tail154
    (if 4 ≤ rootCandidates153.length then search edge 4 [911] rootCandidates153 911 best else best) = best
  rw [if_pos (by decide), rootBranch153]
  exact outerScan154

end ProjectEulerSolutions.PrimePairUpper.Search
