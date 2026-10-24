import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch122
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch123

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan310 : scan edge (search edge 4) 5 [] 0 tail310 best = best := by
  have cons : tail310 = 2081 :: tail311 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates310_checked]
  change scan edge (search edge 4) 5 [] 0 tail311
    (if 4 ≤ rootCandidates310.length then search edge 4 [2081] rootCandidates310 2081 best else best) = best
  rw [if_pos (by decide), rootBranch310]
  exact outerScan311

set_option maxRecDepth 4096 in
theorem outerScan309 : scan edge (search edge 4) 5 [] 0 tail309 best = best := by
  have cons : tail309 = 2069 :: tail310 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates309_checked]
  change scan edge (search edge 4) 5 [] 0 tail310
    (if 4 ≤ rootCandidates309.length then search edge 4 [2069] rootCandidates309 2069 best else best) = best
  rw [if_pos (by decide), rootBranch309]
  exact outerScan310

set_option maxRecDepth 4096 in
theorem outerScan308 : scan edge (search edge 4) 5 [] 0 tail308 best = best := by
  have cons : tail308 = 2063 :: tail309 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates308_checked]
  change scan edge (search edge 4) 5 [] 0 tail309
    (if 4 ≤ rootCandidates308.length then search edge 4 [2063] rootCandidates308 2063 best else best) = best
  rw [if_pos (by decide), rootBranch308]
  exact outerScan309

set_option maxRecDepth 4096 in
theorem outerScan307 : scan edge (search edge 4) 5 [] 0 tail307 best = best := by
  have cons : tail307 = 2053 :: tail308 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates307_checked]
  change scan edge (search edge 4) 5 [] 0 tail308
    (if 4 ≤ rootCandidates307.length then search edge 4 [2053] rootCandidates307 2053 best else best) = best
  rw [if_pos (by decide), rootBranch307]
  exact outerScan308

end ProjectEulerSolutions.PrimePairUpper.Search
