import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch202
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch203

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan96 : scan edge (search edge 4) 5 [] 0 tail96 best = best := by
  have cons : tail96 = 523 :: tail97 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates96_checked]
  change scan edge (search edge 4) 5 [] 0 tail97
    (if 4 ≤ rootCandidates96.length then search edge 4 [523] rootCandidates96 523 best else best) = best
  rw [if_pos (by decide), rootBranch96]
  exact outerScan97

set_option maxRecDepth 4096 in
theorem outerScan95 : scan edge (search edge 4) 5 [] 0 tail95 best = best := by
  have cons : tail95 = 521 :: tail96 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates95_checked]
  change scan edge (search edge 4) 5 [] 0 tail96
    (if 4 ≤ rootCandidates95.length then search edge 4 [521] rootCandidates95 521 best else best) = best
  rw [if_pos (by decide), rootBranch95]
  exact outerScan96

end ProjectEulerSolutions.PrimePairUpper.Search
