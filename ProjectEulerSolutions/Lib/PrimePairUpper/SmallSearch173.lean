import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch172
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch173

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan164 : scan edge (search edge 4) 5 [] 0 tail164 best = best := by
  have cons : tail164 = 991 :: tail165 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates164_checked]
  change scan edge (search edge 4) 5 [] 0 tail165
    (if 4 ≤ rootCandidates164.length then search edge 4 [991] rootCandidates164 991 best else best) = best
  rw [if_pos (by decide), rootBranch164]
  exact outerScan165

set_option maxRecDepth 4096 in
theorem outerScan163 : scan edge (search edge 4) 5 [] 0 tail163 best = best := by
  have cons : tail163 = 983 :: tail164 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates163_checked]
  change scan edge (search edge 4) 5 [] 0 tail164
    (if 4 ≤ rootCandidates163.length then search edge 4 [983] rootCandidates163 983 best else best) = best
  rw [if_pos (by decide), rootBranch163]
  exact outerScan164

set_option maxRecDepth 4096 in
theorem outerScan162 : scan edge (search edge 4) 5 [] 0 tail162 best = best := by
  have cons : tail162 = 977 :: tail163 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates162_checked]
  change scan edge (search edge 4) 5 [] 0 tail163
    (if 4 ≤ rootCandidates162.length then search edge 4 [977] rootCandidates162 977 best else best) = best
  rw [if_pos (by decide), rootBranch162]
  exact outerScan163

end ProjectEulerSolutions.PrimePairUpper.Search
