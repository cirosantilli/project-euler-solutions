import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch058
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch059

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan593 : scan edge (search edge 4) 5 [] 0 tail593 best = best := by
  have cons : tail593 = 4363 :: tail594 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates593_checked]
  change scan edge (search edge 4) 5 [] 0 tail594
    (if 4 ≤ rootCandidates593.length then search edge 4 [4363] rootCandidates593 4363 best else best) = best
  rw [if_pos (by decide), rootBranch593]
  exact outerScan594

set_option maxRecDepth 4096 in
theorem outerScan592 : scan edge (search edge 4) 5 [] 0 tail592 best = best := by
  have cons : tail592 = 4357 :: tail593 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates592_checked]
  change scan edge (search edge 4) 5 [] 0 tail593
    (if 4 ≤ rootCandidates592.length then search edge 4 [4357] rootCandidates592 4357 best else best) = best
  rw [if_pos (by decide), rootBranch592]
  exact outerScan593

set_option maxRecDepth 4096 in
theorem outerScan591 : scan edge (search edge 4) 5 [] 0 tail591 best = best := by
  have cons : tail591 = 4349 :: tail592 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates591_checked]
  change scan edge (search edge 4) 5 [] 0 tail592
    (if 4 ≤ rootCandidates591.length then search edge 4 [4349] rootCandidates591 4349 best else best) = best
  rw [if_pos (by decide), rootBranch591]
  exact outerScan592

set_option maxRecDepth 4096 in
theorem outerScan590 : scan edge (search edge 4) 5 [] 0 tail590 best = best := by
  have cons : tail590 = 4339 :: tail591 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates590_checked]
  change scan edge (search edge 4) 5 [] 0 tail591
    (if 4 ≤ rootCandidates590.length then search edge 4 [4339] rootCandidates590 4339 best else best) = best
  rw [if_pos (by decide), rootBranch590]
  exact outerScan591

set_option maxRecDepth 4096 in
theorem outerScan589 : scan edge (search edge 4) 5 [] 0 tail589 best = best := by
  have cons : tail589 = 4337 :: tail590 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates589_checked]
  change scan edge (search edge 4) 5 [] 0 tail590
    (if 4 ≤ rootCandidates589.length then search edge 4 [4337] rootCandidates589 4337 best else best) = best
  rw [if_pos (by decide), rootBranch589]
  exact outerScan590

set_option maxRecDepth 4096 in
theorem outerScan588 : scan edge (search edge 4) 5 [] 0 tail588 best = best := by
  have cons : tail588 = 4327 :: tail589 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates588_checked]
  change scan edge (search edge 4) 5 [] 0 tail589
    (if 4 ≤ rootCandidates588.length then search edge 4 [4327] rootCandidates588 4327 best else best) = best
  rw [if_pos (by decide), rootBranch588]
  exact outerScan589

end ProjectEulerSolutions.PrimePairUpper.Search
