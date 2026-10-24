import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch000
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch001

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan1039 : scan edge (search edge 4) 5 [] 0 tail1039 best = best := by
  have cons : tail1039 = 8297 :: tail1040 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1039_checked]
  change scan edge (search edge 4) 5 [] 0 tail1040
    (if 4 ≤ rootCandidates1039.length then search edge 4 [8297] rootCandidates1039 8297 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1040

set_option maxRecDepth 4096 in
theorem outerScan1038 : scan edge (search edge 4) 5 [] 0 tail1038 best = best := by
  have cons : tail1038 = 8293 :: tail1039 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1038_checked]
  change scan edge (search edge 4) 5 [] 0 tail1039
    (if 4 ≤ rootCandidates1038.length then search edge 4 [8293] rootCandidates1038 8293 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1039

set_option maxRecDepth 4096 in
theorem outerScan1037 : scan edge (search edge 4) 5 [] 0 tail1037 best = best := by
  have cons : tail1037 = 8291 :: tail1038 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1037_checked]
  change scan edge (search edge 4) 5 [] 0 tail1038
    (if 4 ≤ rootCandidates1037.length then search edge 4 [8291] rootCandidates1037 8291 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1038

set_option maxRecDepth 4096 in
theorem outerScan1036 : scan edge (search edge 4) 5 [] 0 tail1036 best = best := by
  have cons : tail1036 = 8287 :: tail1037 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1036_checked]
  change scan edge (search edge 4) 5 [] 0 tail1037
    (if 4 ≤ rootCandidates1036.length then search edge 4 [8287] rootCandidates1036 8287 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1037

set_option maxRecDepth 4096 in
theorem outerScan1035 : scan edge (search edge 4) 5 [] 0 tail1035 best = best := by
  have cons : tail1035 = 8273 :: tail1036 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1035_checked]
  change scan edge (search edge 4) 5 [] 0 tail1036
    (if 4 ≤ rootCandidates1035.length then search edge 4 [8273] rootCandidates1035 8273 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1036

set_option maxRecDepth 4096 in
theorem outerScan1034 : scan edge (search edge 4) 5 [] 0 tail1034 best = best := by
  have cons : tail1034 = 8269 :: tail1035 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1034_checked]
  change scan edge (search edge 4) 5 [] 0 tail1035
    (if 4 ≤ rootCandidates1034.length then search edge 4 [8269] rootCandidates1034 8269 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1035

set_option maxRecDepth 4096 in
theorem outerScan1033 : scan edge (search edge 4) 5 [] 0 tail1033 best = best := by
  have cons : tail1033 = 8263 :: tail1034 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1033_checked]
  change scan edge (search edge 4) 5 [] 0 tail1034
    (if 4 ≤ rootCandidates1033.length then search edge 4 [8263] rootCandidates1033 8263 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1034

set_option maxRecDepth 4096 in
theorem outerScan1032 : scan edge (search edge 4) 5 [] 0 tail1032 best = best := by
  have cons : tail1032 = 8243 :: tail1033 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1032_checked]
  change scan edge (search edge 4) 5 [] 0 tail1033
    (if 4 ≤ rootCandidates1032.length then search edge 4 [8243] rootCandidates1032 8243 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1033

end ProjectEulerSolutions.PrimePairUpper.Search
