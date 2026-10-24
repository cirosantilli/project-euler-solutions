import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch133
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch134

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan272 : scan edge (search edge 4) 5 [] 0 tail272 best = best := by
  have cons : tail272 = 1777 :: tail273 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates272_checked]
  change scan edge (search edge 4) 5 [] 0 tail273
    (if 4 ≤ rootCandidates272.length then search edge 4 [1777] rootCandidates272 1777 best else best) = best
  rw [if_pos (by decide), rootBranch272]
  exact outerScan273

set_option maxRecDepth 4096 in
theorem outerScan271 : scan edge (search edge 4) 5 [] 0 tail271 best = best := by
  have cons : tail271 = 1759 :: tail272 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates271_checked]
  change scan edge (search edge 4) 5 [] 0 tail272
    (if 4 ≤ rootCandidates271.length then search edge 4 [1759] rootCandidates271 1759 best else best) = best
  rw [if_pos (by decide), rootBranch271]
  exact outerScan272

set_option maxRecDepth 4096 in
theorem outerScan270 : scan edge (search edge 4) 5 [] 0 tail270 best = best := by
  have cons : tail270 = 1753 :: tail271 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates270_checked]
  change scan edge (search edge 4) 5 [] 0 tail271
    (if 4 ≤ rootCandidates270.length then search edge 4 [1753] rootCandidates270 1753 best else best) = best
  rw [if_pos (by decide), rootBranch270]
  exact outerScan271

end ProjectEulerSolutions.PrimePairUpper.Search
