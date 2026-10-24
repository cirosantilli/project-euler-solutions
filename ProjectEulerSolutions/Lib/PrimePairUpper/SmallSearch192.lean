import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch191
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch192

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan120 : scan edge (search edge 4) 5 [] 0 tail120 best = best := by
  have cons : tail120 = 677 :: tail121 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates120_checked]
  change scan edge (search edge 4) 5 [] 0 tail121
    (if 4 ≤ rootCandidates120.length then search edge 4 [677] rootCandidates120 677 best else best) = best
  rw [if_pos (by decide), rootBranch120]
  exact outerScan121

set_option maxRecDepth 4096 in
theorem outerScan119 : scan edge (search edge 4) 5 [] 0 tail119 best = best := by
  have cons : tail119 = 673 :: tail120 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates119_checked]
  change scan edge (search edge 4) 5 [] 0 tail120
    (if 4 ≤ rootCandidates119.length then search edge 4 [673] rootCandidates119 673 best else best) = best
  rw [if_pos (by decide), rootBranch119]
  exact outerScan120

end ProjectEulerSolutions.PrimePairUpper.Search
