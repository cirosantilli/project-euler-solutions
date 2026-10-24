import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch004
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch005

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan1007 : scan edge (search edge 4) 5 [] 0 tail1007 best = best := by
  have cons : tail1007 = 8017 :: tail1008 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1007_checked]
  change scan edge (search edge 4) 5 [] 0 tail1008
    (if 4 ≤ rootCandidates1007.length then search edge 4 [8017] rootCandidates1007 8017 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1008

set_option maxRecDepth 4096 in
theorem outerScan1006 : scan edge (search edge 4) 5 [] 0 tail1006 best = best := by
  have cons : tail1006 = 8011 :: tail1007 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1006_checked]
  change scan edge (search edge 4) 5 [] 0 tail1007
    (if 4 ≤ rootCandidates1006.length then search edge 4 [8011] rootCandidates1006 8011 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1007

set_option maxRecDepth 4096 in
theorem outerScan1005 : scan edge (search edge 4) 5 [] 0 tail1005 best = best := by
  have cons : tail1005 = 8009 :: tail1006 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1005_checked]
  change scan edge (search edge 4) 5 [] 0 tail1006
    (if 4 ≤ rootCandidates1005.length then search edge 4 [8009] rootCandidates1005 8009 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1006

set_option maxRecDepth 4096 in
theorem outerScan1004 : scan edge (search edge 4) 5 [] 0 tail1004 best = best := by
  have cons : tail1004 = 7993 :: tail1005 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1004_checked]
  change scan edge (search edge 4) 5 [] 0 tail1005
    (if 4 ≤ rootCandidates1004.length then search edge 4 [7993] rootCandidates1004 7993 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1005

set_option maxRecDepth 4096 in
theorem outerScan1003 : scan edge (search edge 4) 5 [] 0 tail1003 best = best := by
  have cons : tail1003 = 7963 :: tail1004 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1003_checked]
  change scan edge (search edge 4) 5 [] 0 tail1004
    (if 4 ≤ rootCandidates1003.length then search edge 4 [7963] rootCandidates1003 7963 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1004

set_option maxRecDepth 4096 in
theorem outerScan1002 : scan edge (search edge 4) 5 [] 0 tail1002 best = best := by
  have cons : tail1002 = 7951 :: tail1003 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1002_checked]
  change scan edge (search edge 4) 5 [] 0 tail1003
    (if 4 ≤ rootCandidates1002.length then search edge 4 [7951] rootCandidates1002 7951 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1003

set_option maxRecDepth 4096 in
theorem outerScan1001 : scan edge (search edge 4) 5 [] 0 tail1001 best = best := by
  have cons : tail1001 = 7949 :: tail1002 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1001_checked]
  change scan edge (search edge 4) 5 [] 0 tail1002
    (if 4 ≤ rootCandidates1001.length then search edge 4 [7949] rootCandidates1001 7949 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1002

set_option maxRecDepth 4096 in
theorem outerScan1000 : scan edge (search edge 4) 5 [] 0 tail1000 best = best := by
  have cons : tail1000 = 7937 :: tail1001 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1000_checked]
  change scan edge (search edge 4) 5 [] 0 tail1001
    (if 4 ≤ rootCandidates1000.length then search edge 4 [7937] rootCandidates1000 7937 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1001

end ProjectEulerSolutions.PrimePairUpper.Search
