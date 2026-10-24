import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch011
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch012

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan951 : scan edge (search edge 4) 5 [] 0 tail951 best = best := by
  have cons : tail951 = 7529 :: tail952 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates951_checked]
  change scan edge (search edge 4) 5 [] 0 tail952
    (if 4 ≤ rootCandidates951.length then search edge 4 [7529] rootCandidates951 7529 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan952

set_option maxRecDepth 4096 in
theorem outerScan950 : scan edge (search edge 4) 5 [] 0 tail950 best = best := by
  have cons : tail950 = 7523 :: tail951 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates950_checked]
  change scan edge (search edge 4) 5 [] 0 tail951
    (if 4 ≤ rootCandidates950.length then search edge 4 [7523] rootCandidates950 7523 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan951

set_option maxRecDepth 4096 in
theorem outerScan949 : scan edge (search edge 4) 5 [] 0 tail949 best = best := by
  have cons : tail949 = 7517 :: tail950 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates949_checked]
  change scan edge (search edge 4) 5 [] 0 tail950
    (if 4 ≤ rootCandidates949.length then search edge 4 [7517] rootCandidates949 7517 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan950

set_option maxRecDepth 4096 in
theorem outerScan948 : scan edge (search edge 4) 5 [] 0 tail948 best = best := by
  have cons : tail948 = 7507 :: tail949 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates948_checked]
  change scan edge (search edge 4) 5 [] 0 tail949
    (if 4 ≤ rootCandidates948.length then search edge 4 [7507] rootCandidates948 7507 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan949

set_option maxRecDepth 4096 in
theorem outerScan947 : scan edge (search edge 4) 5 [] 0 tail947 best = best := by
  have cons : tail947 = 7499 :: tail948 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates947_checked]
  change scan edge (search edge 4) 5 [] 0 tail948
    (if 4 ≤ rootCandidates947.length then search edge 4 [7499] rootCandidates947 7499 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan948

set_option maxRecDepth 4096 in
theorem outerScan946 : scan edge (search edge 4) 5 [] 0 tail946 best = best := by
  have cons : tail946 = 7489 :: tail947 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates946_checked]
  change scan edge (search edge 4) 5 [] 0 tail947
    (if 4 ≤ rootCandidates946.length then search edge 4 [7489] rootCandidates946 7489 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan947

set_option maxRecDepth 4096 in
theorem outerScan945 : scan edge (search edge 4) 5 [] 0 tail945 best = best := by
  have cons : tail945 = 7487 :: tail946 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates945_checked]
  change scan edge (search edge 4) 5 [] 0 tail946
    (if 4 ≤ rootCandidates945.length then search edge 4 [7487] rootCandidates945 7487 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan946

set_option maxRecDepth 4096 in
theorem outerScan944 : scan edge (search edge 4) 5 [] 0 tail944 best = best := by
  have cons : tail944 = 7481 :: tail945 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates944_checked]
  change scan edge (search edge 4) 5 [] 0 tail945
    (if 4 ≤ rootCandidates944.length then search edge 4 [7481] rootCandidates944 7481 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan945

end ProjectEulerSolutions.PrimePairUpper.Search
