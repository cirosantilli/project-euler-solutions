import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch123
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch124

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan306 : scan edge (search edge 4) 5 [] 0 tail306 best = best := by
  have cons : tail306 = 2039 :: tail307 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates306_checked]
  change scan edge (search edge 4) 5 [] 0 tail307
    (if 4 ≤ rootCandidates306.length then search edge 4 [2039] rootCandidates306 2039 best else best) = best
  rw [if_pos (by decide), rootBranch306]
  exact outerScan307

set_option maxRecDepth 4096 in
theorem outerScan305 : scan edge (search edge 4) 5 [] 0 tail305 best = best := by
  have cons : tail305 = 2029 :: tail306 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates305_checked]
  change scan edge (search edge 4) 5 [] 0 tail306
    (if 4 ≤ rootCandidates305.length then search edge 4 [2029] rootCandidates305 2029 best else best) = best
  rw [if_pos (by decide), rootBranch305]
  exact outerScan306

set_option maxRecDepth 4096 in
theorem outerScan304 : scan edge (search edge 4) 5 [] 0 tail304 best = best := by
  have cons : tail304 = 2027 :: tail305 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates304_checked]
  change scan edge (search edge 4) 5 [] 0 tail305
    (if 4 ≤ rootCandidates304.length then search edge 4 [2027] rootCandidates304 2027 best else best) = best
  rw [if_pos (by decide), rootBranch304]
  exact outerScan305

set_option maxRecDepth 4096 in
theorem outerScan303 : scan edge (search edge 4) 5 [] 0 tail303 best = best := by
  have cons : tail303 = 2017 :: tail304 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates303_checked]
  change scan edge (search edge 4) 5 [] 0 tail304
    (if 4 ≤ rootCandidates303.length then search edge 4 [2017] rootCandidates303 2017 best else best) = best
  rw [if_pos (by decide), rootBranch303]
  exact outerScan304

end ProjectEulerSolutions.PrimePairUpper.Search
