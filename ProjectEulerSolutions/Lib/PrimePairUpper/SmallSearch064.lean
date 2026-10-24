import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch063
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch064

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan562 : scan edge (search edge 4) 5 [] 0 tail562 best = best := by
  have cons : tail562 = 4099 :: tail563 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates562_checked]
  change scan edge (search edge 4) 5 [] 0 tail563
    (if 4 ≤ rootCandidates562.length then search edge 4 [4099] rootCandidates562 4099 best else best) = best
  rw [if_pos (by decide), rootBranch562]
  exact outerScan563

set_option maxRecDepth 4096 in
theorem outerScan561 : scan edge (search edge 4) 5 [] 0 tail561 best = best := by
  have cons : tail561 = 4093 :: tail562 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates561_checked]
  change scan edge (search edge 4) 5 [] 0 tail562
    (if 4 ≤ rootCandidates561.length then search edge 4 [4093] rootCandidates561 4093 best else best) = best
  rw [if_pos (by decide), rootBranch561]
  exact outerScan562

set_option maxRecDepth 4096 in
theorem outerScan560 : scan edge (search edge 4) 5 [] 0 tail560 best = best := by
  have cons : tail560 = 4091 :: tail561 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates560_checked]
  change scan edge (search edge 4) 5 [] 0 tail561
    (if 4 ≤ rootCandidates560.length then search edge 4 [4091] rootCandidates560 4091 best else best) = best
  rw [if_pos (by decide), rootBranch560]
  exact outerScan561

set_option maxRecDepth 4096 in
theorem outerScan559 : scan edge (search edge 4) 5 [] 0 tail559 best = best := by
  have cons : tail559 = 4079 :: tail560 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates559_checked]
  change scan edge (search edge 4) 5 [] 0 tail560
    (if 4 ≤ rootCandidates559.length then search edge 4 [4079] rootCandidates559 4079 best else best) = best
  rw [if_pos (by decide), rootBranch559]
  exact outerScan560

set_option maxRecDepth 4096 in
theorem outerScan558 : scan edge (search edge 4) 5 [] 0 tail558 best = best := by
  have cons : tail558 = 4073 :: tail559 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates558_checked]
  change scan edge (search edge 4) 5 [] 0 tail559
    (if 4 ≤ rootCandidates558.length then search edge 4 [4073] rootCandidates558 4073 best else best) = best
  rw [if_pos (by decide), rootBranch558]
  exact outerScan559

set_option maxRecDepth 4096 in
theorem outerScan557 : scan edge (search edge 4) 5 [] 0 tail557 best = best := by
  have cons : tail557 = 4057 :: tail558 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates557_checked]
  change scan edge (search edge 4) 5 [] 0 tail558
    (if 4 ≤ rootCandidates557.length then search edge 4 [4057] rootCandidates557 4057 best else best) = best
  rw [if_pos (by decide), rootBranch557]
  exact outerScan558

end ProjectEulerSolutions.PrimePairUpper.Search
