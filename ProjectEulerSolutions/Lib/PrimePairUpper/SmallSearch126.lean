import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch125
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch126

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan298 : scan edge (search edge 4) 5 [] 0 tail298 best = best := by
  have cons : tail298 = 1993 :: tail299 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates298_checked]
  change scan edge (search edge 4) 5 [] 0 tail299
    (if 4 ≤ rootCandidates298.length then search edge 4 [1993] rootCandidates298 1993 best else best) = best
  rw [if_pos (by decide), rootBranch298]
  exact outerScan299

set_option maxRecDepth 4096 in
theorem outerScan297 : scan edge (search edge 4) 5 [] 0 tail297 best = best := by
  have cons : tail297 = 1987 :: tail298 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates297_checked]
  change scan edge (search edge 4) 5 [] 0 tail298
    (if 4 ≤ rootCandidates297.length then search edge 4 [1987] rootCandidates297 1987 best else best) = best
  rw [if_pos (by decide), rootBranch297]
  exact outerScan298

set_option maxRecDepth 4096 in
theorem outerScan296 : scan edge (search edge 4) 5 [] 0 tail296 best = best := by
  have cons : tail296 = 1979 :: tail297 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates296_checked]
  change scan edge (search edge 4) 5 [] 0 tail297
    (if 4 ≤ rootCandidates296.length then search edge 4 [1979] rootCandidates296 1979 best else best) = best
  rw [if_pos (by decide), rootBranch296]
  exact outerScan297

end ProjectEulerSolutions.PrimePairUpper.Search
