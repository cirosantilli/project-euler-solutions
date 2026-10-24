import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch088
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch089

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan438 : scan edge (search edge 4) 5 [] 0 tail438 best = best := by
  have cons : tail438 = 3083 :: tail439 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates438_checked]
  change scan edge (search edge 4) 5 [] 0 tail439
    (if 4 ≤ rootCandidates438.length then search edge 4 [3083] rootCandidates438 3083 best else best) = best
  rw [if_pos (by decide), rootBranch438]
  exact outerScan439

set_option maxRecDepth 4096 in
theorem outerScan437 : scan edge (search edge 4) 5 [] 0 tail437 best = best := by
  have cons : tail437 = 3079 :: tail438 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates437_checked]
  change scan edge (search edge 4) 5 [] 0 tail438
    (if 4 ≤ rootCandidates437.length then search edge 4 [3079] rootCandidates437 3079 best else best) = best
  rw [if_pos (by decide), rootBranch437]
  exact outerScan438

set_option maxRecDepth 4096 in
theorem outerScan436 : scan edge (search edge 4) 5 [] 0 tail436 best = best := by
  have cons : tail436 = 3067 :: tail437 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates436_checked]
  change scan edge (search edge 4) 5 [] 0 tail437
    (if 4 ≤ rootCandidates436.length then search edge 4 [3067] rootCandidates436 3067 best else best) = best
  rw [if_pos (by decide), rootBranch436]
  exact outerScan437

set_option maxRecDepth 4096 in
theorem outerScan435 : scan edge (search edge 4) 5 [] 0 tail435 best = best := by
  have cons : tail435 = 3061 :: tail436 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates435_checked]
  change scan edge (search edge 4) 5 [] 0 tail436
    (if 4 ≤ rootCandidates435.length then search edge 4 [3061] rootCandidates435 3061 best else best) = best
  rw [if_pos (by decide), rootBranch435]
  exact outerScan436

set_option maxRecDepth 4096 in
theorem outerScan434 : scan edge (search edge 4) 5 [] 0 tail434 best = best := by
  have cons : tail434 = 3049 :: tail435 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates434_checked]
  change scan edge (search edge 4) 5 [] 0 tail435
    (if 4 ≤ rootCandidates434.length then search edge 4 [3049] rootCandidates434 3049 best else best) = best
  rw [if_pos (by decide), rootBranch434]
  exact outerScan435

end ProjectEulerSolutions.PrimePairUpper.Search
