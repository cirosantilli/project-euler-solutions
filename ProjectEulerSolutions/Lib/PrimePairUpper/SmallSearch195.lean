import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch194
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch195

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan113 : scan edge (search edge 4) 5 [] 0 tail113 best = best := by
  have cons : tail113 = 641 :: tail114 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates113_checked]
  change scan edge (search edge 4) 5 [] 0 tail114
    (if 4 ≤ rootCandidates113.length then search edge 4 [641] rootCandidates113 641 best else best) = best
  rw [if_pos (by decide), rootBranch113]
  exact outerScan114

set_option maxRecDepth 4096 in
theorem outerScan112 : scan edge (search edge 4) 5 [] 0 tail112 best = best := by
  have cons : tail112 = 631 :: tail113 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates112_checked]
  change scan edge (search edge 4) 5 [] 0 tail113
    (if 4 ≤ rootCandidates112.length then search edge 4 [631] rootCandidates112 631 best else best) = best
  rw [if_pos (by decide), rootBranch112]
  exact outerScan113

set_option maxRecDepth 4096 in
theorem outerScan111 : scan edge (search edge 4) 5 [] 0 tail111 best = best := by
  have cons : tail111 = 619 :: tail112 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates111_checked]
  change scan edge (search edge 4) 5 [] 0 tail112
    (if 4 ≤ rootCandidates111.length then search edge 4 [619] rootCandidates111 619 best else best) = best
  rw [if_pos (by decide), rootBranch111]
  exact outerScan112

end ProjectEulerSolutions.PrimePairUpper.Search
