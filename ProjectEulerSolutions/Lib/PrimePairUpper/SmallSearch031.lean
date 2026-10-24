import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch030
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch031

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan799 : scan edge (search edge 4) 5 [] 0 tail799 best = best := by
  have cons : tail799 = 6151 :: tail800 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates799_checked]
  change scan edge (search edge 4) 5 [] 0 tail800
    (if 4 ≤ rootCandidates799.length then search edge 4 [6151] rootCandidates799 6151 best else best) = best
  rw [if_pos (by decide), rootBranch799]
  exact outerScan800

set_option maxRecDepth 4096 in
theorem outerScan798 : scan edge (search edge 4) 5 [] 0 tail798 best = best := by
  have cons : tail798 = 6143 :: tail799 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates798_checked]
  change scan edge (search edge 4) 5 [] 0 tail799
    (if 4 ≤ rootCandidates798.length then search edge 4 [6143] rootCandidates798 6143 best else best) = best
  rw [if_pos (by decide), rootBranch798]
  exact outerScan799

set_option maxRecDepth 4096 in
theorem outerScan797 : scan edge (search edge 4) 5 [] 0 tail797 best = best := by
  have cons : tail797 = 6133 :: tail798 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates797_checked]
  change scan edge (search edge 4) 5 [] 0 tail798
    (if 4 ≤ rootCandidates797.length then search edge 4 [6133] rootCandidates797 6133 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan798

set_option maxRecDepth 4096 in
theorem outerScan796 : scan edge (search edge 4) 5 [] 0 tail796 best = best := by
  have cons : tail796 = 6131 :: tail797 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates796_checked]
  change scan edge (search edge 4) 5 [] 0 tail797
    (if 4 ≤ rootCandidates796.length then search edge 4 [6131] rootCandidates796 6131 best else best) = best
  rw [if_pos (by decide), rootBranch796]
  exact outerScan797

set_option maxRecDepth 4096 in
theorem outerScan795 : scan edge (search edge 4) 5 [] 0 tail795 best = best := by
  have cons : tail795 = 6121 :: tail796 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates795_checked]
  change scan edge (search edge 4) 5 [] 0 tail796
    (if 4 ≤ rootCandidates795.length then search edge 4 [6121] rootCandidates795 6121 best else best) = best
  rw [if_pos (by decide), rootBranch795]
  exact outerScan796

set_option maxRecDepth 4096 in
theorem outerScan794 : scan edge (search edge 4) 5 [] 0 tail794 best = best := by
  have cons : tail794 = 6113 :: tail795 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates794_checked]
  change scan edge (search edge 4) 5 [] 0 tail795
    (if 4 ≤ rootCandidates794.length then search edge 4 [6113] rootCandidates794 6113 best else best) = best
  rw [if_pos (by decide), rootBranch794]
  exact outerScan795

set_option maxRecDepth 4096 in
theorem outerScan793 : scan edge (search edge 4) 5 [] 0 tail793 best = best := by
  have cons : tail793 = 6101 :: tail794 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates793_checked]
  change scan edge (search edge 4) 5 [] 0 tail794
    (if 4 ≤ rootCandidates793.length then search edge 4 [6101] rootCandidates793 6101 best else best) = best
  rw [if_pos (by decide), rootBranch793]
  exact outerScan794

set_option maxRecDepth 4096 in
theorem outerScan792 : scan edge (search edge 4) 5 [] 0 tail792 best = best := by
  have cons : tail792 = 6091 :: tail793 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates792_checked]
  change scan edge (search edge 4) 5 [] 0 tail793
    (if 4 ≤ rootCandidates792.length then search edge 4 [6091] rootCandidates792 6091 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan793

end ProjectEulerSolutions.PrimePairUpper.Search
