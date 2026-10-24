import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch006
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch007

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan991 : scan edge (search edge 4) 5 [] 0 tail991 best = best := by
  have cons : tail991 = 7873 :: tail992 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates991_checked]
  change scan edge (search edge 4) 5 [] 0 tail992
    (if 4 ≤ rootCandidates991.length then search edge 4 [7873] rootCandidates991 7873 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan992

set_option maxRecDepth 4096 in
theorem outerScan990 : scan edge (search edge 4) 5 [] 0 tail990 best = best := by
  have cons : tail990 = 7867 :: tail991 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates990_checked]
  change scan edge (search edge 4) 5 [] 0 tail991
    (if 4 ≤ rootCandidates990.length then search edge 4 [7867] rootCandidates990 7867 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan991

set_option maxRecDepth 4096 in
theorem outerScan989 : scan edge (search edge 4) 5 [] 0 tail989 best = best := by
  have cons : tail989 = 7853 :: tail990 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates989_checked]
  change scan edge (search edge 4) 5 [] 0 tail990
    (if 4 ≤ rootCandidates989.length then search edge 4 [7853] rootCandidates989 7853 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan990

set_option maxRecDepth 4096 in
theorem outerScan988 : scan edge (search edge 4) 5 [] 0 tail988 best = best := by
  have cons : tail988 = 7841 :: tail989 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates988_checked]
  change scan edge (search edge 4) 5 [] 0 tail989
    (if 4 ≤ rootCandidates988.length then search edge 4 [7841] rootCandidates988 7841 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan989

set_option maxRecDepth 4096 in
theorem outerScan987 : scan edge (search edge 4) 5 [] 0 tail987 best = best := by
  have cons : tail987 = 7829 :: tail988 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates987_checked]
  change scan edge (search edge 4) 5 [] 0 tail988
    (if 4 ≤ rootCandidates987.length then search edge 4 [7829] rootCandidates987 7829 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan988

set_option maxRecDepth 4096 in
theorem outerScan986 : scan edge (search edge 4) 5 [] 0 tail986 best = best := by
  have cons : tail986 = 7823 :: tail987 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates986_checked]
  change scan edge (search edge 4) 5 [] 0 tail987
    (if 4 ≤ rootCandidates986.length then search edge 4 [7823] rootCandidates986 7823 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan987

set_option maxRecDepth 4096 in
theorem outerScan985 : scan edge (search edge 4) 5 [] 0 tail985 best = best := by
  have cons : tail985 = 7817 :: tail986 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates985_checked]
  change scan edge (search edge 4) 5 [] 0 tail986
    (if 4 ≤ rootCandidates985.length then search edge 4 [7817] rootCandidates985 7817 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan986

set_option maxRecDepth 4096 in
theorem outerScan984 : scan edge (search edge 4) 5 [] 0 tail984 best = best := by
  have cons : tail984 = 7793 :: tail985 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates984_checked]
  change scan edge (search edge 4) 5 [] 0 tail985
    (if 4 ≤ rootCandidates984.length then search edge 4 [7793] rootCandidates984 7793 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan985

end ProjectEulerSolutions.PrimePairUpper.Search
