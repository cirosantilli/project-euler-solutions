import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch121
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch122

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan314 : scan edge (search edge 4) 5 [] 0 tail314 best = best := by
  have cons : tail314 = 2099 :: tail315 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates314_checked]
  change scan edge (search edge 4) 5 [] 0 tail315
    (if 4 ≤ rootCandidates314.length then search edge 4 [2099] rootCandidates314 2099 best else best) = best
  rw [if_pos (by decide), rootBranch314]
  exact outerScan315

set_option maxRecDepth 4096 in
theorem outerScan313 : scan edge (search edge 4) 5 [] 0 tail313 best = best := by
  have cons : tail313 = 2089 :: tail314 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates313_checked]
  change scan edge (search edge 4) 5 [] 0 tail314
    (if 4 ≤ rootCandidates313.length then search edge 4 [2089] rootCandidates313 2089 best else best) = best
  rw [if_pos (by decide), rootBranch313]
  exact outerScan314

set_option maxRecDepth 4096 in
theorem outerScan312 : scan edge (search edge 4) 5 [] 0 tail312 best = best := by
  have cons : tail312 = 2087 :: tail313 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates312_checked]
  change scan edge (search edge 4) 5 [] 0 tail313
    (if 4 ≤ rootCandidates312.length then search edge 4 [2087] rootCandidates312 2087 best else best) = best
  rw [if_pos (by decide), rootBranch312]
  exact outerScan313

set_option maxRecDepth 4096 in
theorem outerScan311 : scan edge (search edge 4) 5 [] 0 tail311 best = best := by
  have cons : tail311 = 2083 :: tail312 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates311_checked]
  change scan edge (search edge 4) 5 [] 0 tail312
    (if 4 ≤ rootCandidates311.length then search edge 4 [2083] rootCandidates311 2083 best else best) = best
  rw [if_pos (by decide), rootBranch311]
  exact outerScan312

end ProjectEulerSolutions.PrimePairUpper.Search
