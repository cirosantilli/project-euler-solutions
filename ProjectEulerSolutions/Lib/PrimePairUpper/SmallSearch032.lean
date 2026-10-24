import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch031
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch032

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan791 : scan edge (search edge 4) 5 [] 0 tail791 best = best := by
  have cons : tail791 = 6089 :: tail792 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates791_checked]
  change scan edge (search edge 4) 5 [] 0 tail792
    (if 4 ≤ rootCandidates791.length then search edge 4 [6089] rootCandidates791 6089 best else best) = best
  rw [if_pos (by decide), rootBranch791]
  exact outerScan792

set_option maxRecDepth 4096 in
theorem outerScan790 : scan edge (search edge 4) 5 [] 0 tail790 best = best := by
  have cons : tail790 = 6079 :: tail791 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates790_checked]
  change scan edge (search edge 4) 5 [] 0 tail791
    (if 4 ≤ rootCandidates790.length then search edge 4 [6079] rootCandidates790 6079 best else best) = best
  rw [if_pos (by decide), rootBranch790]
  exact outerScan791

set_option maxRecDepth 4096 in
theorem outerScan789 : scan edge (search edge 4) 5 [] 0 tail789 best = best := by
  have cons : tail789 = 6073 :: tail790 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates789_checked]
  change scan edge (search edge 4) 5 [] 0 tail790
    (if 4 ≤ rootCandidates789.length then search edge 4 [6073] rootCandidates789 6073 best else best) = best
  rw [if_pos (by decide), rootBranch789]
  exact outerScan790

set_option maxRecDepth 4096 in
theorem outerScan788 : scan edge (search edge 4) 5 [] 0 tail788 best = best := by
  have cons : tail788 = 6067 :: tail789 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates788_checked]
  change scan edge (search edge 4) 5 [] 0 tail789
    (if 4 ≤ rootCandidates788.length then search edge 4 [6067] rootCandidates788 6067 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan789

set_option maxRecDepth 4096 in
theorem outerScan787 : scan edge (search edge 4) 5 [] 0 tail787 best = best := by
  have cons : tail787 = 6053 :: tail788 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates787_checked]
  change scan edge (search edge 4) 5 [] 0 tail788
    (if 4 ≤ rootCandidates787.length then search edge 4 [6053] rootCandidates787 6053 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan788

set_option maxRecDepth 4096 in
theorem outerScan786 : scan edge (search edge 4) 5 [] 0 tail786 best = best := by
  have cons : tail786 = 6047 :: tail787 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates786_checked]
  change scan edge (search edge 4) 5 [] 0 tail787
    (if 4 ≤ rootCandidates786.length then search edge 4 [6047] rootCandidates786 6047 best else best) = best
  rw [if_pos (by decide), rootBranch786]
  exact outerScan787

set_option maxRecDepth 4096 in
theorem outerScan785 : scan edge (search edge 4) 5 [] 0 tail785 best = best := by
  have cons : tail785 = 6043 :: tail786 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates785_checked]
  change scan edge (search edge 4) 5 [] 0 tail786
    (if 4 ≤ rootCandidates785.length then search edge 4 [6043] rootCandidates785 6043 best else best) = best
  rw [if_pos (by decide), rootBranch785]
  exact outerScan786

set_option maxRecDepth 4096 in
theorem outerScan784 : scan edge (search edge 4) 5 [] 0 tail784 best = best := by
  have cons : tail784 = 6037 :: tail785 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates784_checked]
  change scan edge (search edge 4) 5 [] 0 tail785
    (if 4 ≤ rootCandidates784.length then search edge 4 [6037] rootCandidates784 6037 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan785

end ProjectEulerSolutions.PrimePairUpper.Search
