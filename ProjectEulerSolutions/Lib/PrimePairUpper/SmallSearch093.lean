import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch092
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch093

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan421 : scan edge (search edge 4) 5 [] 0 tail421 best = best := by
  have cons : tail421 = 2939 :: tail422 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates421_checked]
  change scan edge (search edge 4) 5 [] 0 tail422
    (if 4 ≤ rootCandidates421.length then search edge 4 [2939] rootCandidates421 2939 best else best) = best
  rw [if_pos (by decide), rootBranch421]
  exact outerScan422

set_option maxRecDepth 4096 in
theorem outerScan420 : scan edge (search edge 4) 5 [] 0 tail420 best = best := by
  have cons : tail420 = 2927 :: tail421 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates420_checked]
  change scan edge (search edge 4) 5 [] 0 tail421
    (if 4 ≤ rootCandidates420.length then search edge 4 [2927] rootCandidates420 2927 best else best) = best
  rw [if_pos (by decide), rootBranch420]
  exact outerScan421

set_option maxRecDepth 4096 in
theorem outerScan419 : scan edge (search edge 4) 5 [] 0 tail419 best = best := by
  have cons : tail419 = 2917 :: tail420 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates419_checked]
  change scan edge (search edge 4) 5 [] 0 tail420
    (if 4 ≤ rootCandidates419.length then search edge 4 [2917] rootCandidates419 2917 best else best) = best
  rw [if_pos (by decide), rootBranch419]
  exact outerScan420

set_option maxRecDepth 4096 in
theorem outerScan418 : scan edge (search edge 4) 5 [] 0 tail418 best = best := by
  have cons : tail418 = 2909 :: tail419 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates418_checked]
  change scan edge (search edge 4) 5 [] 0 tail419
    (if 4 ≤ rootCandidates418.length then search edge 4 [2909] rootCandidates418 2909 best else best) = best
  rw [if_pos (by decide), rootBranch418]
  exact outerScan419

set_option maxRecDepth 4096 in
theorem outerScan417 : scan edge (search edge 4) 5 [] 0 tail417 best = best := by
  have cons : tail417 = 2903 :: tail418 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates417_checked]
  change scan edge (search edge 4) 5 [] 0 tail418
    (if 4 ≤ rootCandidates417.length then search edge 4 [2903] rootCandidates417 2903 best else best) = best
  rw [if_pos (by decide), rootBranch417]
  exact outerScan418

end ProjectEulerSolutions.PrimePairUpper.Search
