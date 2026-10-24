import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch131
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch132

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan278 : scan edge (search edge 4) 5 [] 0 tail278 best = best := by
  have cons : tail278 = 1823 :: tail279 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates278_checked]
  change scan edge (search edge 4) 5 [] 0 tail279
    (if 4 ≤ rootCandidates278.length then search edge 4 [1823] rootCandidates278 1823 best else best) = best
  rw [if_pos (by decide), rootBranch278]
  exact outerScan279

set_option maxRecDepth 4096 in
theorem outerScan277 : scan edge (search edge 4) 5 [] 0 tail277 best = best := by
  have cons : tail277 = 1811 :: tail278 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates277_checked]
  change scan edge (search edge 4) 5 [] 0 tail278
    (if 4 ≤ rootCandidates277.length then search edge 4 [1811] rootCandidates277 1811 best else best) = best
  rw [if_pos (by decide), rootBranch277]
  exact outerScan278

set_option maxRecDepth 4096 in
theorem outerScan276 : scan edge (search edge 4) 5 [] 0 tail276 best = best := by
  have cons : tail276 = 1801 :: tail277 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates276_checked]
  change scan edge (search edge 4) 5 [] 0 tail277
    (if 4 ≤ rootCandidates276.length then search edge 4 [1801] rootCandidates276 1801 best else best) = best
  rw [if_pos (by decide), rootBranch276]
  exact outerScan277

end ProjectEulerSolutions.PrimePairUpper.Search
