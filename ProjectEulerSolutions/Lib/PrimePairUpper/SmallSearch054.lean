import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch053
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch054

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan624 : scan edge (search edge 4) 5 [] 0 tail624 best = best := by
  have cons : tail624 = 4643 :: tail625 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates624_checked]
  change scan edge (search edge 4) 5 [] 0 tail625
    (if 4 ≤ rootCandidates624.length then search edge 4 [4643] rootCandidates624 4643 best else best) = best
  rw [if_pos (by decide), rootBranch624]
  exact outerScan625

set_option maxRecDepth 4096 in
theorem outerScan623 : scan edge (search edge 4) 5 [] 0 tail623 best = best := by
  have cons : tail623 = 4639 :: tail624 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates623_checked]
  change scan edge (search edge 4) 5 [] 0 tail624
    (if 4 ≤ rootCandidates623.length then search edge 4 [4639] rootCandidates623 4639 best else best) = best
  rw [if_pos (by decide), rootBranch623]
  exact outerScan624

set_option maxRecDepth 4096 in
theorem outerScan622 : scan edge (search edge 4) 5 [] 0 tail622 best = best := by
  have cons : tail622 = 4637 :: tail623 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates622_checked]
  change scan edge (search edge 4) 5 [] 0 tail623
    (if 4 ≤ rootCandidates622.length then search edge 4 [4637] rootCandidates622 4637 best else best) = best
  rw [if_pos (by decide), rootBranch622]
  exact outerScan623

set_option maxRecDepth 4096 in
theorem outerScan621 : scan edge (search edge 4) 5 [] 0 tail621 best = best := by
  have cons : tail621 = 4621 :: tail622 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates621_checked]
  change scan edge (search edge 4) 5 [] 0 tail622
    (if 4 ≤ rootCandidates621.length then search edge 4 [4621] rootCandidates621 4621 best else best) = best
  rw [if_pos (by decide), rootBranch621]
  exact outerScan622

set_option maxRecDepth 4096 in
theorem outerScan620 : scan edge (search edge 4) 5 [] 0 tail620 best = best := by
  have cons : tail620 = 4603 :: tail621 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates620_checked]
  change scan edge (search edge 4) 5 [] 0 tail621
    (if 4 ≤ rootCandidates620.length then search edge 4 [4603] rootCandidates620 4603 best else best) = best
  rw [if_pos (by decide), rootBranch620]
  exact outerScan621

set_option maxRecDepth 4096 in
theorem outerScan619 : scan edge (search edge 4) 5 [] 0 tail619 best = best := by
  have cons : tail619 = 4597 :: tail620 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates619_checked]
  change scan edge (search edge 4) 5 [] 0 tail620
    (if 4 ≤ rootCandidates619.length then search edge 4 [4597] rootCandidates619 4597 best else best) = best
  rw [if_pos (by decide), rootBranch619]
  exact outerScan620

end ProjectEulerSolutions.PrimePairUpper.Search
