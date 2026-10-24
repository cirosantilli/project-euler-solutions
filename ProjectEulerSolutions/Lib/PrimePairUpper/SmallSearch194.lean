import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch193
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch194

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan116 : scan edge (search edge 4) 5 [] 0 tail116 best = best := by
  have cons : tail116 = 653 :: tail117 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates116_checked]
  change scan edge (search edge 4) 5 [] 0 tail117
    (if 4 ≤ rootCandidates116.length then search edge 4 [653] rootCandidates116 653 best else best) = best
  rw [if_pos (by decide), rootBranch116]
  exact outerScan117

set_option maxRecDepth 4096 in
theorem outerScan115 : scan edge (search edge 4) 5 [] 0 tail115 best = best := by
  have cons : tail115 = 647 :: tail116 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates115_checked]
  change scan edge (search edge 4) 5 [] 0 tail116
    (if 4 ≤ rootCandidates115.length then search edge 4 [647] rootCandidates115 647 best else best) = best
  rw [if_pos (by decide), rootBranch115]
  exact outerScan116

set_option maxRecDepth 4096 in
theorem outerScan114 : scan edge (search edge 4) 5 [] 0 tail114 best = best := by
  have cons : tail114 = 643 :: tail115 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates114_checked]
  change scan edge (search edge 4) 5 [] 0 tail115
    (if 4 ≤ rootCandidates114.length then search edge 4 [643] rootCandidates114 643 best else best) = best
  rw [if_pos (by decide), rootBranch114]
  exact outerScan115

end ProjectEulerSolutions.PrimePairUpper.Search
