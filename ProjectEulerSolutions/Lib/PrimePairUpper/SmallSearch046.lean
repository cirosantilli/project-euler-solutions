import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch045
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch046

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan681 : scan edge (search edge 4) 5 [] 0 tail681 best = best := by
  have cons : tail681 = 5113 :: tail682 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates681_checked]
  change scan edge (search edge 4) 5 [] 0 tail682
    (if 4 ≤ rootCandidates681.length then search edge 4 [5113] rootCandidates681 5113 best else best) = best
  rw [if_pos (by decide), rootBranch681]
  exact outerScan682

set_option maxRecDepth 4096 in
theorem outerScan680 : scan edge (search edge 4) 5 [] 0 tail680 best = best := by
  have cons : tail680 = 5107 :: tail681 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates680_checked]
  change scan edge (search edge 4) 5 [] 0 tail681
    (if 4 ≤ rootCandidates680.length then search edge 4 [5107] rootCandidates680 5107 best else best) = best
  rw [if_pos (by decide), rootBranch680]
  exact outerScan681

set_option maxRecDepth 4096 in
theorem outerScan679 : scan edge (search edge 4) 5 [] 0 tail679 best = best := by
  have cons : tail679 = 5101 :: tail680 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates679_checked]
  change scan edge (search edge 4) 5 [] 0 tail680
    (if 4 ≤ rootCandidates679.length then search edge 4 [5101] rootCandidates679 5101 best else best) = best
  rw [if_pos (by decide), rootBranch679]
  exact outerScan680

set_option maxRecDepth 4096 in
theorem outerScan678 : scan edge (search edge 4) 5 [] 0 tail678 best = best := by
  have cons : tail678 = 5099 :: tail679 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates678_checked]
  change scan edge (search edge 4) 5 [] 0 tail679
    (if 4 ≤ rootCandidates678.length then search edge 4 [5099] rootCandidates678 5099 best else best) = best
  rw [if_pos (by decide), rootBranch678]
  exact outerScan679

set_option maxRecDepth 4096 in
theorem outerScan677 : scan edge (search edge 4) 5 [] 0 tail677 best = best := by
  have cons : tail677 = 5087 :: tail678 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates677_checked]
  change scan edge (search edge 4) 5 [] 0 tail678
    (if 4 ≤ rootCandidates677.length then search edge 4 [5087] rootCandidates677 5087 best else best) = best
  rw [if_pos (by decide), rootBranch677]
  exact outerScan678

set_option maxRecDepth 4096 in
theorem outerScan676 : scan edge (search edge 4) 5 [] 0 tail676 best = best := by
  have cons : tail676 = 5081 :: tail677 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates676_checked]
  change scan edge (search edge 4) 5 [] 0 tail677
    (if 4 ≤ rootCandidates676.length then search edge 4 [5081] rootCandidates676 5081 best else best) = best
  rw [if_pos (by decide), rootBranch676]
  exact outerScan677

set_option maxRecDepth 4096 in
theorem outerScan675 : scan edge (search edge 4) 5 [] 0 tail675 best = best := by
  have cons : tail675 = 5077 :: tail676 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates675_checked]
  change scan edge (search edge 4) 5 [] 0 tail676
    (if 4 ≤ rootCandidates675.length then search edge 4 [5077] rootCandidates675 5077 best else best) = best
  rw [if_pos (by decide), rootBranch675]
  exact outerScan676

end ProjectEulerSolutions.PrimePairUpper.Search
