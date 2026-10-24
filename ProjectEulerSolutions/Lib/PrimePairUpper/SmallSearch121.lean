import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch120
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch121

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan317 : scan edge (search edge 4) 5 [] 0 tail317 best = best := by
  have cons : tail317 = 2129 :: tail318 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates317_checked]
  change scan edge (search edge 4) 5 [] 0 tail318
    (if 4 ≤ rootCandidates317.length then search edge 4 [2129] rootCandidates317 2129 best else best) = best
  rw [if_pos (by decide), rootBranch317]
  exact outerScan318

set_option maxRecDepth 4096 in
theorem outerScan316 : scan edge (search edge 4) 5 [] 0 tail316 best = best := by
  have cons : tail316 = 2113 :: tail317 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates316_checked]
  change scan edge (search edge 4) 5 [] 0 tail317
    (if 4 ≤ rootCandidates316.length then search edge 4 [2113] rootCandidates316 2113 best else best) = best
  rw [if_pos (by decide), rootBranch316]
  exact outerScan317

set_option maxRecDepth 4096 in
theorem outerScan315 : scan edge (search edge 4) 5 [] 0 tail315 best = best := by
  have cons : tail315 = 2111 :: tail316 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates315_checked]
  change scan edge (search edge 4) 5 [] 0 tail316
    (if 4 ≤ rootCandidates315.length then search edge 4 [2111] rootCandidates315 2111 best else best) = best
  rw [if_pos (by decide), rootBranch315]
  exact outerScan316

end ProjectEulerSolutions.PrimePairUpper.Search
