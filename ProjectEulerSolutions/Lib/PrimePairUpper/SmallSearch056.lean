import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch055
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch056

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan612 : scan edge (search edge 4) 5 [] 0 tail612 best = best := by
  have cons : tail612 = 4523 :: tail613 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates612_checked]
  change scan edge (search edge 4) 5 [] 0 tail613
    (if 4 ≤ rootCandidates612.length then search edge 4 [4523] rootCandidates612 4523 best else best) = best
  rw [if_pos (by decide), rootBranch612]
  exact outerScan613

set_option maxRecDepth 4096 in
theorem outerScan611 : scan edge (search edge 4) 5 [] 0 tail611 best = best := by
  have cons : tail611 = 4519 :: tail612 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates611_checked]
  change scan edge (search edge 4) 5 [] 0 tail612
    (if 4 ≤ rootCandidates611.length then search edge 4 [4519] rootCandidates611 4519 best else best) = best
  rw [if_pos (by decide), rootBranch611]
  exact outerScan612

set_option maxRecDepth 4096 in
theorem outerScan610 : scan edge (search edge 4) 5 [] 0 tail610 best = best := by
  have cons : tail610 = 4517 :: tail611 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates610_checked]
  change scan edge (search edge 4) 5 [] 0 tail611
    (if 4 ≤ rootCandidates610.length then search edge 4 [4517] rootCandidates610 4517 best else best) = best
  rw [if_pos (by decide), rootBranch610]
  exact outerScan611

set_option maxRecDepth 4096 in
theorem outerScan609 : scan edge (search edge 4) 5 [] 0 tail609 best = best := by
  have cons : tail609 = 4513 :: tail610 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates609_checked]
  change scan edge (search edge 4) 5 [] 0 tail610
    (if 4 ≤ rootCandidates609.length then search edge 4 [4513] rootCandidates609 4513 best else best) = best
  rw [if_pos (by decide), rootBranch609]
  exact outerScan610

set_option maxRecDepth 4096 in
theorem outerScan608 : scan edge (search edge 4) 5 [] 0 tail608 best = best := by
  have cons : tail608 = 4507 :: tail609 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates608_checked]
  change scan edge (search edge 4) 5 [] 0 tail609
    (if 4 ≤ rootCandidates608.length then search edge 4 [4507] rootCandidates608 4507 best else best) = best
  rw [if_pos (by decide), rootBranch608]
  exact outerScan609

set_option maxRecDepth 4096 in
theorem outerScan607 : scan edge (search edge 4) 5 [] 0 tail607 best = best := by
  have cons : tail607 = 4493 :: tail608 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates607_checked]
  change scan edge (search edge 4) 5 [] 0 tail608
    (if 4 ≤ rootCandidates607.length then search edge 4 [4493] rootCandidates607 4493 best else best) = best
  rw [if_pos (by decide), rootBranch607]
  exact outerScan608

end ProjectEulerSolutions.PrimePairUpper.Search
