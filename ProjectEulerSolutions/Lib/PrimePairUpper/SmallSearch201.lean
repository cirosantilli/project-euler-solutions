import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch200
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch201

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan100 : scan edge (search edge 4) 5 [] 0 tail100 best = best := by
  have cons : tail100 = 563 :: tail101 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates100_checked]
  change scan edge (search edge 4) 5 [] 0 tail101
    (if 4 ≤ rootCandidates100.length then search edge 4 [563] rootCandidates100 563 best else best) = best
  rw [if_pos (by decide), rootBranch100]
  exact outerScan101

set_option maxRecDepth 4096 in
theorem outerScan99 : scan edge (search edge 4) 5 [] 0 tail99 best = best := by
  have cons : tail99 = 557 :: tail100 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates99_checked]
  change scan edge (search edge 4) 5 [] 0 tail100
    (if 4 ≤ rootCandidates99.length then search edge 4 [557] rootCandidates99 557 best else best) = best
  rw [if_pos (by decide), rootBranch99]
  exact outerScan100

end ProjectEulerSolutions.PrimePairUpper.Search
