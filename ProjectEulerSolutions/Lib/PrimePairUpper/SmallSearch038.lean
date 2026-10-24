import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch037
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch038

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan743 : scan edge (search edge 4) 5 [] 0 tail743 best = best := by
  have cons : tail743 = 5659 :: tail744 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates743_checked]
  change scan edge (search edge 4) 5 [] 0 tail744
    (if 4 ≤ rootCandidates743.length then search edge 4 [5659] rootCandidates743 5659 best else best) = best
  rw [if_pos (by decide), rootBranch743]
  exact outerScan744

set_option maxRecDepth 4096 in
theorem outerScan742 : scan edge (search edge 4) 5 [] 0 tail742 best = best := by
  have cons : tail742 = 5657 :: tail743 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates742_checked]
  change scan edge (search edge 4) 5 [] 0 tail743
    (if 4 ≤ rootCandidates742.length then search edge 4 [5657] rootCandidates742 5657 best else best) = best
  rw [if_pos (by decide), rootBranch742]
  exact outerScan743

set_option maxRecDepth 4096 in
theorem outerScan741 : scan edge (search edge 4) 5 [] 0 tail741 best = best := by
  have cons : tail741 = 5653 :: tail742 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates741_checked]
  change scan edge (search edge 4) 5 [] 0 tail742
    (if 4 ≤ rootCandidates741.length then search edge 4 [5653] rootCandidates741 5653 best else best) = best
  rw [if_pos (by decide), rootBranch741]
  exact outerScan742

set_option maxRecDepth 4096 in
theorem outerScan740 : scan edge (search edge 4) 5 [] 0 tail740 best = best := by
  have cons : tail740 = 5651 :: tail741 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates740_checked]
  change scan edge (search edge 4) 5 [] 0 tail741
    (if 4 ≤ rootCandidates740.length then search edge 4 [5651] rootCandidates740 5651 best else best) = best
  rw [if_pos (by decide), rootBranch740]
  exact outerScan741

set_option maxRecDepth 4096 in
theorem outerScan739 : scan edge (search edge 4) 5 [] 0 tail739 best = best := by
  have cons : tail739 = 5647 :: tail740 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates739_checked]
  change scan edge (search edge 4) 5 [] 0 tail740
    (if 4 ≤ rootCandidates739.length then search edge 4 [5647] rootCandidates739 5647 best else best) = best
  rw [if_pos (by decide), rootBranch739]
  exact outerScan740

set_option maxRecDepth 4096 in
theorem outerScan738 : scan edge (search edge 4) 5 [] 0 tail738 best = best := by
  have cons : tail738 = 5641 :: tail739 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates738_checked]
  change scan edge (search edge 4) 5 [] 0 tail739
    (if 4 ≤ rootCandidates738.length then search edge 4 [5641] rootCandidates738 5641 best else best) = best
  rw [if_pos (by decide), rootBranch738]
  exact outerScan739

set_option maxRecDepth 4096 in
theorem outerScan737 : scan edge (search edge 4) 5 [] 0 tail737 best = best := by
  have cons : tail737 = 5639 :: tail738 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates737_checked]
  change scan edge (search edge 4) 5 [] 0 tail738
    (if 4 ≤ rootCandidates737.length then search edge 4 [5639] rootCandidates737 5639 best else best) = best
  rw [if_pos (by decide), rootBranch737]
  exact outerScan738

end ProjectEulerSolutions.PrimePairUpper.Search
