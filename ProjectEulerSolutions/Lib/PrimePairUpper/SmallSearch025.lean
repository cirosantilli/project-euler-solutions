import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch024
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch025

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan847 : scan edge (search edge 4) 5 [] 0 tail847 best = best := by
  have cons : tail847 = 6571 :: tail848 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates847_checked]
  change scan edge (search edge 4) 5 [] 0 tail848
    (if 4 ≤ rootCandidates847.length then search edge 4 [6571] rootCandidates847 6571 best else best) = best
  rw [if_pos (by decide), rootBranch847]
  exact outerScan848

set_option maxRecDepth 4096 in
theorem outerScan846 : scan edge (search edge 4) 5 [] 0 tail846 best = best := by
  have cons : tail846 = 6569 :: tail847 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates846_checked]
  change scan edge (search edge 4) 5 [] 0 tail847
    (if 4 ≤ rootCandidates846.length then search edge 4 [6569] rootCandidates846 6569 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan847

set_option maxRecDepth 4096 in
theorem outerScan845 : scan edge (search edge 4) 5 [] 0 tail845 best = best := by
  have cons : tail845 = 6563 :: tail846 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates845_checked]
  change scan edge (search edge 4) 5 [] 0 tail846
    (if 4 ≤ rootCandidates845.length then search edge 4 [6563] rootCandidates845 6563 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan846

set_option maxRecDepth 4096 in
theorem outerScan844 : scan edge (search edge 4) 5 [] 0 tail844 best = best := by
  have cons : tail844 = 6553 :: tail845 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates844_checked]
  change scan edge (search edge 4) 5 [] 0 tail845
    (if 4 ≤ rootCandidates844.length then search edge 4 [6553] rootCandidates844 6553 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan845

set_option maxRecDepth 4096 in
theorem outerScan843 : scan edge (search edge 4) 5 [] 0 tail843 best = best := by
  have cons : tail843 = 6551 :: tail844 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates843_checked]
  change scan edge (search edge 4) 5 [] 0 tail844
    (if 4 ≤ rootCandidates843.length then search edge 4 [6551] rootCandidates843 6551 best else best) = best
  rw [if_pos (by decide), rootBranch843]
  exact outerScan844

set_option maxRecDepth 4096 in
theorem outerScan842 : scan edge (search edge 4) 5 [] 0 tail842 best = best := by
  have cons : tail842 = 6547 :: tail843 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates842_checked]
  change scan edge (search edge 4) 5 [] 0 tail843
    (if 4 ≤ rootCandidates842.length then search edge 4 [6547] rootCandidates842 6547 best else best) = best
  rw [if_pos (by decide), rootBranch842]
  exact outerScan843

set_option maxRecDepth 4096 in
theorem outerScan841 : scan edge (search edge 4) 5 [] 0 tail841 best = best := by
  have cons : tail841 = 6529 :: tail842 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates841_checked]
  change scan edge (search edge 4) 5 [] 0 tail842
    (if 4 ≤ rootCandidates841.length then search edge 4 [6529] rootCandidates841 6529 best else best) = best
  rw [if_pos (by decide), rootBranch841]
  exact outerScan842

set_option maxRecDepth 4096 in
theorem outerScan840 : scan edge (search edge 4) 5 [] 0 tail840 best = best := by
  have cons : tail840 = 6521 :: tail841 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates840_checked]
  change scan edge (search edge 4) 5 [] 0 tail841
    (if 4 ≤ rootCandidates840.length then search edge 4 [6521] rootCandidates840 6521 best else best) = best
  rw [if_pos (by decide), rootBranch840]
  exact outerScan841

end ProjectEulerSolutions.PrimePairUpper.Search
