import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch161
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch162

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan191 : scan edge (search edge 4) 5 [] 0 tail191 best = best := by
  have cons : tail191 = 1181 :: tail192 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates191_checked]
  change scan edge (search edge 4) 5 [] 0 tail192
    (if 4 ≤ rootCandidates191.length then search edge 4 [1181] rootCandidates191 1181 best else best) = best
  rw [if_pos (by decide), rootBranch191]
  exact outerScan192

set_option maxRecDepth 4096 in
theorem outerScan190 : scan edge (search edge 4) 5 [] 0 tail190 best = best := by
  have cons : tail190 = 1171 :: tail191 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates190_checked]
  change scan edge (search edge 4) 5 [] 0 tail191
    (if 4 ≤ rootCandidates190.length then search edge 4 [1171] rootCandidates190 1171 best else best) = best
  rw [if_pos (by decide), rootBranch190]
  exact outerScan191

set_option maxRecDepth 4096 in
theorem outerScan189 : scan edge (search edge 4) 5 [] 0 tail189 best = best := by
  have cons : tail189 = 1163 :: tail190 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates189_checked]
  change scan edge (search edge 4) 5 [] 0 tail190
    (if 4 ≤ rootCandidates189.length then search edge 4 [1163] rootCandidates189 1163 best else best) = best
  rw [if_pos (by decide), rootBranch189]
  exact outerScan190

end ProjectEulerSolutions.PrimePairUpper.Search
