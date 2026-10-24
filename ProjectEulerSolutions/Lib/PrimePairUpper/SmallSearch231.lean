import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch230
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch231

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan45 : scan edge (search edge 4) 5 [] 0 tail45 best = best := by
  have cons : tail45 = 223 :: tail46 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates45_checked]
  change scan edge (search edge 4) 5 [] 0 tail46
    (if 4 ≤ rootCandidates45.length then search edge 4 [223] rootCandidates45 223 best else best) = best
  rw [if_pos (by decide), rootBranch45]
  exact outerScan46

set_option maxRecDepth 4096 in
theorem outerScan44 : scan edge (search edge 4) 5 [] 0 tail44 best = best := by
  have cons : tail44 = 211 :: tail45 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates44_checked]
  change scan edge (search edge 4) 5 [] 0 tail45
    (if 4 ≤ rootCandidates44.length then search edge 4 [211] rootCandidates44 211 best else best) = best
  rw [if_pos (by decide), rootBranch44]
  exact outerScan45

end ProjectEulerSolutions.PrimePairUpper.Search
