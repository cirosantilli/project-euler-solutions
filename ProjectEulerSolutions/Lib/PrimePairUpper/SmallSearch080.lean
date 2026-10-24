import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch079
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch080

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan478 : scan edge (search edge 4) 5 [] 0 tail478 best = best := by
  have cons : tail478 = 3433 :: tail479 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates478_checked]
  change scan edge (search edge 4) 5 [] 0 tail479
    (if 4 ≤ rootCandidates478.length then search edge 4 [3433] rootCandidates478 3433 best else best) = best
  rw [if_pos (by decide), rootBranch478]
  exact outerScan479

set_option maxRecDepth 4096 in
theorem outerScan477 : scan edge (search edge 4) 5 [] 0 tail477 best = best := by
  have cons : tail477 = 3413 :: tail478 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates477_checked]
  change scan edge (search edge 4) 5 [] 0 tail478
    (if 4 ≤ rootCandidates477.length then search edge 4 [3413] rootCandidates477 3413 best else best) = best
  rw [if_pos (by decide), rootBranch477]
  exact outerScan478

set_option maxRecDepth 4096 in
theorem outerScan476 : scan edge (search edge 4) 5 [] 0 tail476 best = best := by
  have cons : tail476 = 3407 :: tail477 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates476_checked]
  change scan edge (search edge 4) 5 [] 0 tail477
    (if 4 ≤ rootCandidates476.length then search edge 4 [3407] rootCandidates476 3407 best else best) = best
  rw [if_pos (by decide), rootBranch476]
  exact outerScan477

set_option maxRecDepth 4096 in
theorem outerScan475 : scan edge (search edge 4) 5 [] 0 tail475 best = best := by
  have cons : tail475 = 3391 :: tail476 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates475_checked]
  change scan edge (search edge 4) 5 [] 0 tail476
    (if 4 ≤ rootCandidates475.length then search edge 4 [3391] rootCandidates475 3391 best else best) = best
  rw [if_pos (by decide), rootBranch475]
  exact outerScan476

end ProjectEulerSolutions.PrimePairUpper.Search
