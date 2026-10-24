import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch000

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan1047 : scan edge (search edge 4) 5 [] 0 tail1047 best = best := by
  have cons : tail1047 = 8387 :: tail1048 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1047_checked]
  change scan edge (search edge 4) 5 [] 0 tail1048
    (if 4 ≤ rootCandidates1047.length then search edge 4 [8387] rootCandidates1047 8387 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1048

set_option maxRecDepth 4096 in
theorem outerScan1046 : scan edge (search edge 4) 5 [] 0 tail1046 best = best := by
  have cons : tail1046 = 8377 :: tail1047 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1046_checked]
  change scan edge (search edge 4) 5 [] 0 tail1047
    (if 4 ≤ rootCandidates1046.length then search edge 4 [8377] rootCandidates1046 8377 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1047

set_option maxRecDepth 4096 in
theorem outerScan1045 : scan edge (search edge 4) 5 [] 0 tail1045 best = best := by
  have cons : tail1045 = 8369 :: tail1046 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1045_checked]
  change scan edge (search edge 4) 5 [] 0 tail1046
    (if 4 ≤ rootCandidates1045.length then search edge 4 [8369] rootCandidates1045 8369 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1046

set_option maxRecDepth 4096 in
theorem outerScan1044 : scan edge (search edge 4) 5 [] 0 tail1044 best = best := by
  have cons : tail1044 = 8363 :: tail1045 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1044_checked]
  change scan edge (search edge 4) 5 [] 0 tail1045
    (if 4 ≤ rootCandidates1044.length then search edge 4 [8363] rootCandidates1044 8363 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1045

set_option maxRecDepth 4096 in
theorem outerScan1043 : scan edge (search edge 4) 5 [] 0 tail1043 best = best := by
  have cons : tail1043 = 8353 :: tail1044 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1043_checked]
  change scan edge (search edge 4) 5 [] 0 tail1044
    (if 4 ≤ rootCandidates1043.length then search edge 4 [8353] rootCandidates1043 8353 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1044

set_option maxRecDepth 4096 in
theorem outerScan1042 : scan edge (search edge 4) 5 [] 0 tail1042 best = best := by
  have cons : tail1042 = 8329 :: tail1043 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1042_checked]
  change scan edge (search edge 4) 5 [] 0 tail1043
    (if 4 ≤ rootCandidates1042.length then search edge 4 [8329] rootCandidates1042 8329 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1043

set_option maxRecDepth 4096 in
theorem outerScan1041 : scan edge (search edge 4) 5 [] 0 tail1041 best = best := by
  have cons : tail1041 = 8317 :: tail1042 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1041_checked]
  change scan edge (search edge 4) 5 [] 0 tail1042
    (if 4 ≤ rootCandidates1041.length then search edge 4 [8317] rootCandidates1041 8317 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1042

set_option maxRecDepth 4096 in
theorem outerScan1040 : scan edge (search edge 4) 5 [] 0 tail1040 best = best := by
  have cons : tail1040 = 8311 :: tail1041 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1040_checked]
  change scan edge (search edge 4) 5 [] 0 tail1041
    (if 4 ≤ rootCandidates1040.length then search edge 4 [8311] rootCandidates1040 8311 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1041

end ProjectEulerSolutions.PrimePairUpper.Search
