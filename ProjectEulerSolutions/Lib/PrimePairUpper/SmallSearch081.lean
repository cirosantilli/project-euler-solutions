import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch080
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch081

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan474 : scan edge (search edge 4) 5 [] 0 tail474 best = best := by
  have cons : tail474 = 3389 :: tail475 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates474_checked]
  change scan edge (search edge 4) 5 [] 0 tail475
    (if 4 ≤ rootCandidates474.length then search edge 4 [3389] rootCandidates474 3389 best else best) = best
  rw [if_pos (by decide), rootBranch474]
  exact outerScan475

set_option maxRecDepth 4096 in
theorem outerScan473 : scan edge (search edge 4) 5 [] 0 tail473 best = best := by
  have cons : tail473 = 3373 :: tail474 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates473_checked]
  change scan edge (search edge 4) 5 [] 0 tail474
    (if 4 ≤ rootCandidates473.length then search edge 4 [3373] rootCandidates473 3373 best else best) = best
  rw [if_pos (by decide), rootBranch473]
  exact outerScan474

set_option maxRecDepth 4096 in
theorem outerScan472 : scan edge (search edge 4) 5 [] 0 tail472 best = best := by
  have cons : tail472 = 3371 :: tail473 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates472_checked]
  change scan edge (search edge 4) 5 [] 0 tail473
    (if 4 ≤ rootCandidates472.length then search edge 4 [3371] rootCandidates472 3371 best else best) = best
  rw [if_pos (by decide), rootBranch472]
  exact outerScan473

set_option maxRecDepth 4096 in
theorem outerScan471 : scan edge (search edge 4) 5 [] 0 tail471 best = best := by
  have cons : tail471 = 3361 :: tail472 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates471_checked]
  change scan edge (search edge 4) 5 [] 0 tail472
    (if 4 ≤ rootCandidates471.length then search edge 4 [3361] rootCandidates471 3361 best else best) = best
  rw [if_pos (by decide), rootBranch471]
  exact outerScan472

set_option maxRecDepth 4096 in
theorem outerScan470 : scan edge (search edge 4) 5 [] 0 tail470 best = best := by
  have cons : tail470 = 3359 :: tail471 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates470_checked]
  change scan edge (search edge 4) 5 [] 0 tail471
    (if 4 ≤ rootCandidates470.length then search edge 4 [3359] rootCandidates470 3359 best else best) = best
  rw [if_pos (by decide), rootBranch470]
  exact outerScan471

end ProjectEulerSolutions.PrimePairUpper.Search
