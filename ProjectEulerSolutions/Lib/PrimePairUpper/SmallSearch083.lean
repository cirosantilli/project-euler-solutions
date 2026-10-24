import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch082
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch083

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan465 : scan edge (search edge 4) 5 [] 0 tail465 best = best := by
  have cons : tail465 = 3323 :: tail466 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates465_checked]
  change scan edge (search edge 4) 5 [] 0 tail466
    (if 4 ≤ rootCandidates465.length then search edge 4 [3323] rootCandidates465 3323 best else best) = best
  rw [if_pos (by decide), rootBranch465]
  exact outerScan466

set_option maxRecDepth 4096 in
theorem outerScan464 : scan edge (search edge 4) 5 [] 0 tail464 best = best := by
  have cons : tail464 = 3319 :: tail465 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates464_checked]
  change scan edge (search edge 4) 5 [] 0 tail465
    (if 4 ≤ rootCandidates464.length then search edge 4 [3319] rootCandidates464 3319 best else best) = best
  rw [if_pos (by decide), rootBranch464]
  exact outerScan465

set_option maxRecDepth 4096 in
theorem outerScan463 : scan edge (search edge 4) 5 [] 0 tail463 best = best := by
  have cons : tail463 = 3313 :: tail464 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates463_checked]
  change scan edge (search edge 4) 5 [] 0 tail464
    (if 4 ≤ rootCandidates463.length then search edge 4 [3313] rootCandidates463 3313 best else best) = best
  rw [if_pos (by decide), rootBranch463]
  exact outerScan464

set_option maxRecDepth 4096 in
theorem outerScan462 : scan edge (search edge 4) 5 [] 0 tail462 best = best := by
  have cons : tail462 = 3307 :: tail463 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates462_checked]
  change scan edge (search edge 4) 5 [] 0 tail463
    (if 4 ≤ rootCandidates462.length then search edge 4 [3307] rootCandidates462 3307 best else best) = best
  rw [if_pos (by decide), rootBranch462]
  exact outerScan463

end ProjectEulerSolutions.PrimePairUpper.Search
