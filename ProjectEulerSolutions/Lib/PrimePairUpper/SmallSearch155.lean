import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch154
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch155

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan211 : scan edge (search edge 4) 5 [] 0 tail211 best = best := by
  have cons : tail211 = 1307 :: tail212 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates211_checked]
  change scan edge (search edge 4) 5 [] 0 tail212
    (if 4 ≤ rootCandidates211.length then search edge 4 [1307] rootCandidates211 1307 best else best) = best
  rw [if_pos (by decide), rootBranch211]
  exact outerScan212

set_option maxRecDepth 4096 in
theorem outerScan210 : scan edge (search edge 4) 5 [] 0 tail210 best = best := by
  have cons : tail210 = 1303 :: tail211 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates210_checked]
  change scan edge (search edge 4) 5 [] 0 tail211
    (if 4 ≤ rootCandidates210.length then search edge 4 [1303] rootCandidates210 1303 best else best) = best
  rw [if_pos (by decide), rootBranch210]
  exact outerScan211

set_option maxRecDepth 4096 in
theorem outerScan209 : scan edge (search edge 4) 5 [] 0 tail209 best = best := by
  have cons : tail209 = 1301 :: tail210 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates209_checked]
  change scan edge (search edge 4) 5 [] 0 tail210
    (if 4 ≤ rootCandidates209.length then search edge 4 [1301] rootCandidates209 1301 best else best) = best
  rw [if_pos (by decide), rootBranch209]
  exact outerScan210

end ProjectEulerSolutions.PrimePairUpper.Search
