import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch085
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch086

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan452 : scan edge (search edge 4) 5 [] 0 tail452 best = best := by
  have cons : tail452 = 3217 :: tail453 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates452_checked]
  change scan edge (search edge 4) 5 [] 0 tail453
    (if 4 ≤ rootCandidates452.length then search edge 4 [3217] rootCandidates452 3217 best else best) = best
  rw [if_pos (by decide), rootBranch452]
  exact outerScan453

set_option maxRecDepth 4096 in
theorem outerScan451 : scan edge (search edge 4) 5 [] 0 tail451 best = best := by
  have cons : tail451 = 3209 :: tail452 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates451_checked]
  change scan edge (search edge 4) 5 [] 0 tail452
    (if 4 ≤ rootCandidates451.length then search edge 4 [3209] rootCandidates451 3209 best else best) = best
  rw [if_pos (by decide), rootBranch451]
  exact outerScan452

set_option maxRecDepth 4096 in
theorem outerScan450 : scan edge (search edge 4) 5 [] 0 tail450 best = best := by
  have cons : tail450 = 3203 :: tail451 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates450_checked]
  change scan edge (search edge 4) 5 [] 0 tail451
    (if 4 ≤ rootCandidates450.length then search edge 4 [3203] rootCandidates450 3203 best else best) = best
  rw [if_pos (by decide), rootBranch450]
  exact outerScan451

set_option maxRecDepth 4096 in
theorem outerScan449 : scan edge (search edge 4) 5 [] 0 tail449 best = best := by
  have cons : tail449 = 3191 :: tail450 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates449_checked]
  change scan edge (search edge 4) 5 [] 0 tail450
    (if 4 ≤ rootCandidates449.length then search edge 4 [3191] rootCandidates449 3191 best else best) = best
  rw [if_pos (by decide), rootBranch449]
  exact outerScan450

set_option maxRecDepth 4096 in
theorem outerScan448 : scan edge (search edge 4) 5 [] 0 tail448 best = best := by
  have cons : tail448 = 3187 :: tail449 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates448_checked]
  change scan edge (search edge 4) 5 [] 0 tail449
    (if 4 ≤ rootCandidates448.length then search edge 4 [3187] rootCandidates448 3187 best else best) = best
  rw [if_pos (by decide), rootBranch448]
  exact outerScan449

end ProjectEulerSolutions.PrimePairUpper.Search
