import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch044
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch045

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan689 : scan edge (search edge 4) 5 [] 0 tail689 best = best := by
  have cons : tail689 = 5197 :: tail690 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates689_checked]
  change scan edge (search edge 4) 5 [] 0 tail690
    (if 4 ≤ rootCandidates689.length then search edge 4 [5197] rootCandidates689 5197 best else best) = best
  rw [if_pos (by decide), rootBranch689]
  exact outerScan690

set_option maxRecDepth 4096 in
theorem outerScan688 : scan edge (search edge 4) 5 [] 0 tail688 best = best := by
  have cons : tail688 = 5189 :: tail689 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates688_checked]
  change scan edge (search edge 4) 5 [] 0 tail689
    (if 4 ≤ rootCandidates688.length then search edge 4 [5189] rootCandidates688 5189 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan689

set_option maxRecDepth 4096 in
theorem outerScan687 : scan edge (search edge 4) 5 [] 0 tail687 best = best := by
  have cons : tail687 = 5179 :: tail688 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates687_checked]
  change scan edge (search edge 4) 5 [] 0 tail688
    (if 4 ≤ rootCandidates687.length then search edge 4 [5179] rootCandidates687 5179 best else best) = best
  rw [if_pos (by decide), rootBranch687]
  exact outerScan688

set_option maxRecDepth 4096 in
theorem outerScan686 : scan edge (search edge 4) 5 [] 0 tail686 best = best := by
  have cons : tail686 = 5171 :: tail687 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates686_checked]
  change scan edge (search edge 4) 5 [] 0 tail687
    (if 4 ≤ rootCandidates686.length then search edge 4 [5171] rootCandidates686 5171 best else best) = best
  rw [if_pos (by decide), rootBranch686]
  exact outerScan687

set_option maxRecDepth 4096 in
theorem outerScan685 : scan edge (search edge 4) 5 [] 0 tail685 best = best := by
  have cons : tail685 = 5167 :: tail686 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates685_checked]
  change scan edge (search edge 4) 5 [] 0 tail686
    (if 4 ≤ rootCandidates685.length then search edge 4 [5167] rootCandidates685 5167 best else best) = best
  rw [if_pos (by decide), rootBranch685]
  exact outerScan686

set_option maxRecDepth 4096 in
theorem outerScan684 : scan edge (search edge 4) 5 [] 0 tail684 best = best := by
  have cons : tail684 = 5153 :: tail685 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates684_checked]
  change scan edge (search edge 4) 5 [] 0 tail685
    (if 4 ≤ rootCandidates684.length then search edge 4 [5153] rootCandidates684 5153 best else best) = best
  rw [if_pos (by decide), rootBranch684]
  exact outerScan685

set_option maxRecDepth 4096 in
theorem outerScan683 : scan edge (search edge 4) 5 [] 0 tail683 best = best := by
  have cons : tail683 = 5147 :: tail684 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates683_checked]
  change scan edge (search edge 4) 5 [] 0 tail684
    (if 4 ≤ rootCandidates683.length then search edge 4 [5147] rootCandidates683 5147 best else best) = best
  rw [if_pos (by decide), rootBranch683]
  exact outerScan684

set_option maxRecDepth 4096 in
theorem outerScan682 : scan edge (search edge 4) 5 [] 0 tail682 best = best := by
  have cons : tail682 = 5119 :: tail683 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates682_checked]
  change scan edge (search edge 4) 5 [] 0 tail683
    (if 4 ≤ rootCandidates682.length then search edge 4 [5119] rootCandidates682 5119 best else best) = best
  rw [if_pos (by decide), rootBranch682]
  exact outerScan683

end ProjectEulerSolutions.PrimePairUpper.Search
