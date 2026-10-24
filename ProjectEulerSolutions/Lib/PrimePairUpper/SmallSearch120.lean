import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch119
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch120

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan320 : scan edge (search edge 4) 5 [] 0 tail320 best = best := by
  have cons : tail320 = 2141 :: tail321 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates320_checked]
  change scan edge (search edge 4) 5 [] 0 tail321
    (if 4 ≤ rootCandidates320.length then search edge 4 [2141] rootCandidates320 2141 best else best) = best
  rw [if_pos (by decide), rootBranch320]
  exact outerScan321

set_option maxRecDepth 4096 in
theorem outerScan319 : scan edge (search edge 4) 5 [] 0 tail319 best = best := by
  have cons : tail319 = 2137 :: tail320 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates319_checked]
  change scan edge (search edge 4) 5 [] 0 tail320
    (if 4 ≤ rootCandidates319.length then search edge 4 [2137] rootCandidates319 2137 best else best) = best
  rw [if_pos (by decide), rootBranch319]
  exact outerScan320

set_option maxRecDepth 4096 in
theorem outerScan318 : scan edge (search edge 4) 5 [] 0 tail318 best = best := by
  have cons : tail318 = 2131 :: tail319 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates318_checked]
  change scan edge (search edge 4) 5 [] 0 tail319
    (if 4 ≤ rootCandidates318.length then search edge 4 [2131] rootCandidates318 2131 best else best) = best
  rw [if_pos (by decide), rootBranch318]
  exact outerScan319

end ProjectEulerSolutions.PrimePairUpper.Search
