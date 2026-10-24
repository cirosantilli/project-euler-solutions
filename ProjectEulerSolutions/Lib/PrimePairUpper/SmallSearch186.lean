import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch185
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch186

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan135 : scan edge (search edge 4) 5 [] 0 tail135 best = best := by
  have cons : tail135 = 787 :: tail136 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates135_checked]
  change scan edge (search edge 4) 5 [] 0 tail136
    (if 4 ≤ rootCandidates135.length then search edge 4 [787] rootCandidates135 787 best else best) = best
  rw [if_pos (by decide), rootBranch135]
  exact outerScan136

set_option maxRecDepth 4096 in
theorem outerScan134 : scan edge (search edge 4) 5 [] 0 tail134 best = best := by
  have cons : tail134 = 773 :: tail135 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates134_checked]
  change scan edge (search edge 4) 5 [] 0 tail135
    (if 4 ≤ rootCandidates134.length then search edge 4 [773] rootCandidates134 773 best else best) = best
  rw [if_pos (by decide), rootBranch134]
  exact outerScan135

set_option maxRecDepth 4096 in
theorem outerScan133 : scan edge (search edge 4) 5 [] 0 tail133 best = best := by
  have cons : tail133 = 769 :: tail134 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates133_checked]
  change scan edge (search edge 4) 5 [] 0 tail134
    (if 4 ≤ rootCandidates133.length then search edge 4 [769] rootCandidates133 769 best else best) = best
  rw [if_pos (by decide), rootBranch133]
  exact outerScan134

end ProjectEulerSolutions.PrimePairUpper.Search
