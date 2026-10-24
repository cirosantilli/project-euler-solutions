import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch029
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch030

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan807 : scan edge (search edge 4) 5 [] 0 tail807 best = best := by
  have cons : tail807 = 6221 :: tail808 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates807_checked]
  change scan edge (search edge 4) 5 [] 0 tail808
    (if 4 ≤ rootCandidates807.length then search edge 4 [6221] rootCandidates807 6221 best else best) = best
  rw [if_pos (by decide), rootBranch807]
  exact outerScan808

set_option maxRecDepth 4096 in
theorem outerScan806 : scan edge (search edge 4) 5 [] 0 tail806 best = best := by
  have cons : tail806 = 6217 :: tail807 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates806_checked]
  change scan edge (search edge 4) 5 [] 0 tail807
    (if 4 ≤ rootCandidates806.length then search edge 4 [6217] rootCandidates806 6217 best else best) = best
  rw [if_pos (by decide), rootBranch806]
  exact outerScan807

set_option maxRecDepth 4096 in
theorem outerScan805 : scan edge (search edge 4) 5 [] 0 tail805 best = best := by
  have cons : tail805 = 6211 :: tail806 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates805_checked]
  change scan edge (search edge 4) 5 [] 0 tail806
    (if 4 ≤ rootCandidates805.length then search edge 4 [6211] rootCandidates805 6211 best else best) = best
  rw [if_pos (by decide), rootBranch805]
  exact outerScan806

set_option maxRecDepth 4096 in
theorem outerScan804 : scan edge (search edge 4) 5 [] 0 tail804 best = best := by
  have cons : tail804 = 6203 :: tail805 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates804_checked]
  change scan edge (search edge 4) 5 [] 0 tail805
    (if 4 ≤ rootCandidates804.length then search edge 4 [6203] rootCandidates804 6203 best else best) = best
  rw [if_pos (by decide), rootBranch804]
  exact outerScan805

set_option maxRecDepth 4096 in
theorem outerScan803 : scan edge (search edge 4) 5 [] 0 tail803 best = best := by
  have cons : tail803 = 6199 :: tail804 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates803_checked]
  change scan edge (search edge 4) 5 [] 0 tail804
    (if 4 ≤ rootCandidates803.length then search edge 4 [6199] rootCandidates803 6199 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan804

set_option maxRecDepth 4096 in
theorem outerScan802 : scan edge (search edge 4) 5 [] 0 tail802 best = best := by
  have cons : tail802 = 6197 :: tail803 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates802_checked]
  change scan edge (search edge 4) 5 [] 0 tail803
    (if 4 ≤ rootCandidates802.length then search edge 4 [6197] rootCandidates802 6197 best else best) = best
  rw [if_pos (by decide), rootBranch802]
  exact outerScan803

set_option maxRecDepth 4096 in
theorem outerScan801 : scan edge (search edge 4) 5 [] 0 tail801 best = best := by
  have cons : tail801 = 6173 :: tail802 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates801_checked]
  change scan edge (search edge 4) 5 [] 0 tail802
    (if 4 ≤ rootCandidates801.length then search edge 4 [6173] rootCandidates801 6173 best else best) = best
  rw [if_pos (by decide), rootBranch801]
  exact outerScan802

set_option maxRecDepth 4096 in
theorem outerScan800 : scan edge (search edge 4) 5 [] 0 tail800 best = best := by
  have cons : tail800 = 6163 :: tail801 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates800_checked]
  change scan edge (search edge 4) 5 [] 0 tail801
    (if 4 ≤ rootCandidates800.length then search edge 4 [6163] rootCandidates800 6163 best else best) = best
  rw [if_pos (by decide), rootBranch800]
  exact outerScan801

end ProjectEulerSolutions.PrimePairUpper.Search
