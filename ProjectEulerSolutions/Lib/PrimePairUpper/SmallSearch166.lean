import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch165
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch166

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan182 : scan edge (search edge 4) 5 [] 0 tail182 best = best := by
  have cons : tail182 = 1103 :: tail183 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates182_checked]
  change scan edge (search edge 4) 5 [] 0 tail183
    (if 4 ≤ rootCandidates182.length then search edge 4 [1103] rootCandidates182 1103 best else best) = best
  rw [if_pos (by decide), rootBranch182]
  exact outerScan183

set_option maxRecDepth 4096 in
theorem outerScan181 : scan edge (search edge 4) 5 [] 0 tail181 best = best := by
  have cons : tail181 = 1097 :: tail182 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates181_checked]
  change scan edge (search edge 4) 5 [] 0 tail182
    (if 4 ≤ rootCandidates181.length then search edge 4 [1097] rootCandidates181 1097 best else best) = best
  rw [if_pos (by decide), rootBranch181]
  exact outerScan182

set_option maxRecDepth 4096 in
theorem outerScan180 : scan edge (search edge 4) 5 [] 0 tail180 best = best := by
  have cons : tail180 = 1093 :: tail181 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates180_checked]
  change scan edge (search edge 4) 5 [] 0 tail181
    (if 4 ≤ rootCandidates180.length then search edge 4 [1093] rootCandidates180 1093 best else best) = best
  rw [if_pos (by decide), rootBranch180]
  exact outerScan181

end ProjectEulerSolutions.PrimePairUpper.Search
