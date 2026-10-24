import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch020
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch021

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan879 : scan edge (search edge 4) 5 [] 0 tail879 best = best := by
  have cons : tail879 = 6857 :: tail880 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates879_checked]
  change scan edge (search edge 4) 5 [] 0 tail880
    (if 4 ≤ rootCandidates879.length then search edge 4 [6857] rootCandidates879 6857 best else best) = best
  rw [if_pos (by decide), rootBranch879]
  exact outerScan880

set_option maxRecDepth 4096 in
theorem outerScan878 : scan edge (search edge 4) 5 [] 0 tail878 best = best := by
  have cons : tail878 = 6841 :: tail879 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates878_checked]
  change scan edge (search edge 4) 5 [] 0 tail879
    (if 4 ≤ rootCandidates878.length then search edge 4 [6841] rootCandidates878 6841 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan879

set_option maxRecDepth 4096 in
theorem outerScan877 : scan edge (search edge 4) 5 [] 0 tail877 best = best := by
  have cons : tail877 = 6833 :: tail878 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates877_checked]
  change scan edge (search edge 4) 5 [] 0 tail878
    (if 4 ≤ rootCandidates877.length then search edge 4 [6833] rootCandidates877 6833 best else best) = best
  rw [if_pos (by decide), rootBranch877]
  exact outerScan878

set_option maxRecDepth 4096 in
theorem outerScan876 : scan edge (search edge 4) 5 [] 0 tail876 best = best := by
  have cons : tail876 = 6829 :: tail877 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates876_checked]
  change scan edge (search edge 4) 5 [] 0 tail877
    (if 4 ≤ rootCandidates876.length then search edge 4 [6829] rootCandidates876 6829 best else best) = best
  rw [if_pos (by decide), rootBranch876]
  exact outerScan877

set_option maxRecDepth 4096 in
theorem outerScan875 : scan edge (search edge 4) 5 [] 0 tail875 best = best := by
  have cons : tail875 = 6827 :: tail876 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates875_checked]
  change scan edge (search edge 4) 5 [] 0 tail876
    (if 4 ≤ rootCandidates875.length then search edge 4 [6827] rootCandidates875 6827 best else best) = best
  rw [if_pos (by decide), rootBranch875]
  exact outerScan876

set_option maxRecDepth 4096 in
theorem outerScan874 : scan edge (search edge 4) 5 [] 0 tail874 best = best := by
  have cons : tail874 = 6823 :: tail875 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates874_checked]
  change scan edge (search edge 4) 5 [] 0 tail875
    (if 4 ≤ rootCandidates874.length then search edge 4 [6823] rootCandidates874 6823 best else best) = best
  rw [if_pos (by decide), rootBranch874]
  exact outerScan875

set_option maxRecDepth 4096 in
theorem outerScan873 : scan edge (search edge 4) 5 [] 0 tail873 best = best := by
  have cons : tail873 = 6803 :: tail874 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates873_checked]
  change scan edge (search edge 4) 5 [] 0 tail874
    (if 4 ≤ rootCandidates873.length then search edge 4 [6803] rootCandidates873 6803 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan874

set_option maxRecDepth 4096 in
theorem outerScan872 : scan edge (search edge 4) 5 [] 0 tail872 best = best := by
  have cons : tail872 = 6793 :: tail873 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates872_checked]
  change scan edge (search edge 4) 5 [] 0 tail873
    (if 4 ≤ rootCandidates872.length then search edge 4 [6793] rootCandidates872 6793 best else best) = best
  rw [if_pos (by decide), rootBranch872]
  exact outerScan873

end ProjectEulerSolutions.PrimePairUpper.Search
