import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch008
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch009

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan975 : scan edge (search edge 4) 5 [] 0 tail975 best = best := by
  have cons : tail975 = 7703 :: tail976 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates975_checked]
  change scan edge (search edge 4) 5 [] 0 tail976
    (if 4 ≤ rootCandidates975.length then search edge 4 [7703] rootCandidates975 7703 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan976

set_option maxRecDepth 4096 in
theorem outerScan974 : scan edge (search edge 4) 5 [] 0 tail974 best = best := by
  have cons : tail974 = 7699 :: tail975 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates974_checked]
  change scan edge (search edge 4) 5 [] 0 tail975
    (if 4 ≤ rootCandidates974.length then search edge 4 [7699] rootCandidates974 7699 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan975

set_option maxRecDepth 4096 in
theorem outerScan973 : scan edge (search edge 4) 5 [] 0 tail973 best = best := by
  have cons : tail973 = 7691 :: tail974 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates973_checked]
  change scan edge (search edge 4) 5 [] 0 tail974
    (if 4 ≤ rootCandidates973.length then search edge 4 [7691] rootCandidates973 7691 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan974

set_option maxRecDepth 4096 in
theorem outerScan972 : scan edge (search edge 4) 5 [] 0 tail972 best = best := by
  have cons : tail972 = 7687 :: tail973 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates972_checked]
  change scan edge (search edge 4) 5 [] 0 tail973
    (if 4 ≤ rootCandidates972.length then search edge 4 [7687] rootCandidates972 7687 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan973

set_option maxRecDepth 4096 in
theorem outerScan971 : scan edge (search edge 4) 5 [] 0 tail971 best = best := by
  have cons : tail971 = 7681 :: tail972 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates971_checked]
  change scan edge (search edge 4) 5 [] 0 tail972
    (if 4 ≤ rootCandidates971.length then search edge 4 [7681] rootCandidates971 7681 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan972

set_option maxRecDepth 4096 in
theorem outerScan970 : scan edge (search edge 4) 5 [] 0 tail970 best = best := by
  have cons : tail970 = 7673 :: tail971 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates970_checked]
  change scan edge (search edge 4) 5 [] 0 tail971
    (if 4 ≤ rootCandidates970.length then search edge 4 [7673] rootCandidates970 7673 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan971

set_option maxRecDepth 4096 in
theorem outerScan969 : scan edge (search edge 4) 5 [] 0 tail969 best = best := by
  have cons : tail969 = 7669 :: tail970 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates969_checked]
  change scan edge (search edge 4) 5 [] 0 tail970
    (if 4 ≤ rootCandidates969.length then search edge 4 [7669] rootCandidates969 7669 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan970

set_option maxRecDepth 4096 in
theorem outerScan968 : scan edge (search edge 4) 5 [] 0 tail968 best = best := by
  have cons : tail968 = 7649 :: tail969 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates968_checked]
  change scan edge (search edge 4) 5 [] 0 tail969
    (if 4 ≤ rootCandidates968.length then search edge 4 [7649] rootCandidates968 7649 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan969

end ProjectEulerSolutions.PrimePairUpper.Search
