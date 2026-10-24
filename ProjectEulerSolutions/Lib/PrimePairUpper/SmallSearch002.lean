import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch001
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch002

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan1031 : scan edge (search edge 4) 5 [] 0 tail1031 best = best := by
  have cons : tail1031 = 8237 :: tail1032 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1031_checked]
  change scan edge (search edge 4) 5 [] 0 tail1032
    (if 4 ≤ rootCandidates1031.length then search edge 4 [8237] rootCandidates1031 8237 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1032

set_option maxRecDepth 4096 in
theorem outerScan1030 : scan edge (search edge 4) 5 [] 0 tail1030 best = best := by
  have cons : tail1030 = 8233 :: tail1031 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1030_checked]
  change scan edge (search edge 4) 5 [] 0 tail1031
    (if 4 ≤ rootCandidates1030.length then search edge 4 [8233] rootCandidates1030 8233 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1031

set_option maxRecDepth 4096 in
theorem outerScan1029 : scan edge (search edge 4) 5 [] 0 tail1029 best = best := by
  have cons : tail1029 = 8231 :: tail1030 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1029_checked]
  change scan edge (search edge 4) 5 [] 0 tail1030
    (if 4 ≤ rootCandidates1029.length then search edge 4 [8231] rootCandidates1029 8231 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1030

set_option maxRecDepth 4096 in
theorem outerScan1028 : scan edge (search edge 4) 5 [] 0 tail1028 best = best := by
  have cons : tail1028 = 8221 :: tail1029 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1028_checked]
  change scan edge (search edge 4) 5 [] 0 tail1029
    (if 4 ≤ rootCandidates1028.length then search edge 4 [8221] rootCandidates1028 8221 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1029

set_option maxRecDepth 4096 in
theorem outerScan1027 : scan edge (search edge 4) 5 [] 0 tail1027 best = best := by
  have cons : tail1027 = 8219 :: tail1028 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1027_checked]
  change scan edge (search edge 4) 5 [] 0 tail1028
    (if 4 ≤ rootCandidates1027.length then search edge 4 [8219] rootCandidates1027 8219 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1028

set_option maxRecDepth 4096 in
theorem outerScan1026 : scan edge (search edge 4) 5 [] 0 tail1026 best = best := by
  have cons : tail1026 = 8209 :: tail1027 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1026_checked]
  change scan edge (search edge 4) 5 [] 0 tail1027
    (if 4 ≤ rootCandidates1026.length then search edge 4 [8209] rootCandidates1026 8209 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1027

set_option maxRecDepth 4096 in
theorem outerScan1025 : scan edge (search edge 4) 5 [] 0 tail1025 best = best := by
  have cons : tail1025 = 8191 :: tail1026 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1025_checked]
  change scan edge (search edge 4) 5 [] 0 tail1026
    (if 4 ≤ rootCandidates1025.length then search edge 4 [8191] rootCandidates1025 8191 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1026

set_option maxRecDepth 4096 in
theorem outerScan1024 : scan edge (search edge 4) 5 [] 0 tail1024 best = best := by
  have cons : tail1024 = 8179 :: tail1025 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1024_checked]
  change scan edge (search edge 4) 5 [] 0 tail1025
    (if 4 ≤ rootCandidates1024.length then search edge 4 [8179] rootCandidates1024 8179 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1025

end ProjectEulerSolutions.PrimePairUpper.Search
