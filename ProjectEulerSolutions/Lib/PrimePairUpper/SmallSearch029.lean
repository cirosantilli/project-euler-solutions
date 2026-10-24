import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch028
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch029

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan815 : scan edge (search edge 4) 5 [] 0 tail815 best = best := by
  have cons : tail815 = 6287 :: tail816 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates815_checked]
  change scan edge (search edge 4) 5 [] 0 tail816
    (if 4 ≤ rootCandidates815.length then search edge 4 [6287] rootCandidates815 6287 best else best) = best
  rw [if_pos (by decide), rootBranch815]
  exact outerScan816

set_option maxRecDepth 4096 in
theorem outerScan814 : scan edge (search edge 4) 5 [] 0 tail814 best = best := by
  have cons : tail814 = 6277 :: tail815 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates814_checked]
  change scan edge (search edge 4) 5 [] 0 tail815
    (if 4 ≤ rootCandidates814.length then search edge 4 [6277] rootCandidates814 6277 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan815

set_option maxRecDepth 4096 in
theorem outerScan813 : scan edge (search edge 4) 5 [] 0 tail813 best = best := by
  have cons : tail813 = 6271 :: tail814 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates813_checked]
  change scan edge (search edge 4) 5 [] 0 tail814
    (if 4 ≤ rootCandidates813.length then search edge 4 [6271] rootCandidates813 6271 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan814

set_option maxRecDepth 4096 in
theorem outerScan812 : scan edge (search edge 4) 5 [] 0 tail812 best = best := by
  have cons : tail812 = 6269 :: tail813 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates812_checked]
  change scan edge (search edge 4) 5 [] 0 tail813
    (if 4 ≤ rootCandidates812.length then search edge 4 [6269] rootCandidates812 6269 best else best) = best
  rw [if_pos (by decide), rootBranch812]
  exact outerScan813

set_option maxRecDepth 4096 in
theorem outerScan811 : scan edge (search edge 4) 5 [] 0 tail811 best = best := by
  have cons : tail811 = 6263 :: tail812 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates811_checked]
  change scan edge (search edge 4) 5 [] 0 tail812
    (if 4 ≤ rootCandidates811.length then search edge 4 [6263] rootCandidates811 6263 best else best) = best
  rw [if_pos (by decide), rootBranch811]
  exact outerScan812

set_option maxRecDepth 4096 in
theorem outerScan810 : scan edge (search edge 4) 5 [] 0 tail810 best = best := by
  have cons : tail810 = 6257 :: tail811 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates810_checked]
  change scan edge (search edge 4) 5 [] 0 tail811
    (if 4 ≤ rootCandidates810.length then search edge 4 [6257] rootCandidates810 6257 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan811

set_option maxRecDepth 4096 in
theorem outerScan809 : scan edge (search edge 4) 5 [] 0 tail809 best = best := by
  have cons : tail809 = 6247 :: tail810 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates809_checked]
  change scan edge (search edge 4) 5 [] 0 tail810
    (if 4 ≤ rootCandidates809.length then search edge 4 [6247] rootCandidates809 6247 best else best) = best
  rw [if_pos (by decide), rootBranch809]
  exact outerScan810

set_option maxRecDepth 4096 in
theorem outerScan808 : scan edge (search edge 4) 5 [] 0 tail808 best = best := by
  have cons : tail808 = 6229 :: tail809 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates808_checked]
  change scan edge (search edge 4) 5 [] 0 tail809
    (if 4 ≤ rootCandidates808.length then search edge 4 [6229] rootCandidates808 6229 best else best) = best
  rw [if_pos (by decide), rootBranch808]
  exact outerScan809

end ProjectEulerSolutions.PrimePairUpper.Search
