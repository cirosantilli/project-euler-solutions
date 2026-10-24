import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch109
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch110

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan356 : scan edge (search edge 4) 5 [] 0 tail356 best = best := by
  have cons : tail356 = 2417 :: tail357 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates356_checked]
  change scan edge (search edge 4) 5 [] 0 tail357
    (if 4 ≤ rootCandidates356.length then search edge 4 [2417] rootCandidates356 2417 best else best) = best
  rw [if_pos (by decide), rootBranch356]
  exact outerScan357

set_option maxRecDepth 4096 in
theorem outerScan355 : scan edge (search edge 4) 5 [] 0 tail355 best = best := by
  have cons : tail355 = 2411 :: tail356 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates355_checked]
  change scan edge (search edge 4) 5 [] 0 tail356
    (if 4 ≤ rootCandidates355.length then search edge 4 [2411] rootCandidates355 2411 best else best) = best
  rw [if_pos (by decide), rootBranch355]
  exact outerScan356

set_option maxRecDepth 4096 in
theorem outerScan354 : scan edge (search edge 4) 5 [] 0 tail354 best = best := by
  have cons : tail354 = 2399 :: tail355 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates354_checked]
  change scan edge (search edge 4) 5 [] 0 tail355
    (if 4 ≤ rootCandidates354.length then search edge 4 [2399] rootCandidates354 2399 best else best) = best
  rw [if_pos (by decide), rootBranch354]
  exact outerScan355

end ProjectEulerSolutions.PrimePairUpper.Search
