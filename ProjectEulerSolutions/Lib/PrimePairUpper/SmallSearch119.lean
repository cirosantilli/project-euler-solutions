import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch118
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch119

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan324 : scan edge (search edge 4) 5 [] 0 tail324 best = best := by
  have cons : tail324 = 2179 :: tail325 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates324_checked]
  change scan edge (search edge 4) 5 [] 0 tail325
    (if 4 ≤ rootCandidates324.length then search edge 4 [2179] rootCandidates324 2179 best else best) = best
  rw [if_pos (by decide), rootBranch324]
  exact outerScan325

set_option maxRecDepth 4096 in
theorem outerScan323 : scan edge (search edge 4) 5 [] 0 tail323 best = best := by
  have cons : tail323 = 2161 :: tail324 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates323_checked]
  change scan edge (search edge 4) 5 [] 0 tail324
    (if 4 ≤ rootCandidates323.length then search edge 4 [2161] rootCandidates323 2161 best else best) = best
  rw [if_pos (by decide), rootBranch323]
  exact outerScan324

set_option maxRecDepth 4096 in
theorem outerScan322 : scan edge (search edge 4) 5 [] 0 tail322 best = best := by
  have cons : tail322 = 2153 :: tail323 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates322_checked]
  change scan edge (search edge 4) 5 [] 0 tail323
    (if 4 ≤ rootCandidates322.length then search edge 4 [2153] rootCandidates322 2153 best else best) = best
  rw [if_pos (by decide), rootBranch322]
  exact outerScan323

set_option maxRecDepth 4096 in
theorem outerScan321 : scan edge (search edge 4) 5 [] 0 tail321 best = best := by
  have cons : tail321 = 2143 :: tail322 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates321_checked]
  change scan edge (search edge 4) 5 [] 0 tail322
    (if 4 ≤ rootCandidates321.length then search edge 4 [2143] rootCandidates321 2143 best else best) = best
  rw [if_pos (by decide), rootBranch321]
  exact outerScan322

end ProjectEulerSolutions.PrimePairUpper.Search
