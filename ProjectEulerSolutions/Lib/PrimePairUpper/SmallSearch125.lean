import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch124
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch125

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan302 : scan edge (search edge 4) 5 [] 0 tail302 best = best := by
  have cons : tail302 = 2011 :: tail303 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates302_checked]
  change scan edge (search edge 4) 5 [] 0 tail303
    (if 4 ≤ rootCandidates302.length then search edge 4 [2011] rootCandidates302 2011 best else best) = best
  rw [if_pos (by decide), rootBranch302]
  exact outerScan303

set_option maxRecDepth 4096 in
theorem outerScan301 : scan edge (search edge 4) 5 [] 0 tail301 best = best := by
  have cons : tail301 = 2003 :: tail302 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates301_checked]
  change scan edge (search edge 4) 5 [] 0 tail302
    (if 4 ≤ rootCandidates301.length then search edge 4 [2003] rootCandidates301 2003 best else best) = best
  rw [if_pos (by decide), rootBranch301]
  exact outerScan302

set_option maxRecDepth 4096 in
theorem outerScan300 : scan edge (search edge 4) 5 [] 0 tail300 best = best := by
  have cons : tail300 = 1999 :: tail301 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates300_checked]
  change scan edge (search edge 4) 5 [] 0 tail301
    (if 4 ≤ rootCandidates300.length then search edge 4 [1999] rootCandidates300 1999 best else best) = best
  rw [if_pos (by decide), rootBranch300]
  exact outerScan301

set_option maxRecDepth 4096 in
theorem outerScan299 : scan edge (search edge 4) 5 [] 0 tail299 best = best := by
  have cons : tail299 = 1997 :: tail300 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates299_checked]
  change scan edge (search edge 4) 5 [] 0 tail300
    (if 4 ≤ rootCandidates299.length then search edge 4 [1997] rootCandidates299 1997 best else best) = best
  rw [if_pos (by decide), rootBranch299]
  exact outerScan300

end ProjectEulerSolutions.PrimePairUpper.Search
