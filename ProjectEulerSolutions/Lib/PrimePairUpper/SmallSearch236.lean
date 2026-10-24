import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch235
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch236

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan35 : scan edge (search edge 4) 5 [] 0 tail35 best = best := by
  have cons : tail35 = 163 :: tail36 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates35_checked]
  change scan edge (search edge 4) 5 [] 0 tail36
    (if 4 ≤ rootCandidates35.length then search edge 4 [163] rootCandidates35 163 best else best) = best
  rw [if_pos (by decide), rootBranch35]
  exact outerScan36

set_option maxRecDepth 4096 in
theorem outerScan34 : scan edge (search edge 4) 5 [] 0 tail34 best = best := by
  have cons : tail34 = 157 :: tail35 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates34_checked]
  change scan edge (search edge 4) 5 [] 0 tail35
    (if 4 ≤ rootCandidates34.length then search edge 4 [157] rootCandidates34 157 best else best) = best
  rw [if_pos (by decide), rootBranch34]
  exact outerScan35

end ProjectEulerSolutions.PrimePairUpper.Search
