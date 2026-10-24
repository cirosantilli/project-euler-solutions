import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch218
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch219

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan64 : scan edge (search edge 4) 5 [] 0 tail64 best = best := by
  have cons : tail64 = 331 :: tail65 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates64_checked]
  change scan edge (search edge 4) 5 [] 0 tail65
    (if 4 ≤ rootCandidates64.length then search edge 4 [331] rootCandidates64 331 best else best) = best
  rw [if_pos (by decide), rootBranch64]
  exact outerScan65

set_option maxRecDepth 4096 in
theorem outerScan63 : scan edge (search edge 4) 5 [] 0 tail63 best = best := by
  have cons : tail63 = 317 :: tail64 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates63_checked]
  change scan edge (search edge 4) 5 [] 0 tail64
    (if 4 ≤ rootCandidates63.length then search edge 4 [317] rootCandidates63 317 best else best) = best
  rw [if_pos (by decide), rootBranch63]
  exact outerScan64

end ProjectEulerSolutions.PrimePairUpper.Search
