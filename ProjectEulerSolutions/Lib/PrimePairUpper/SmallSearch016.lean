import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch015
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch016

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan919 : scan edge (search edge 4) 5 [] 0 tail919 best = best := by
  have cons : tail919 = 7213 :: tail920 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates919_checked]
  change scan edge (search edge 4) 5 [] 0 tail920
    (if 4 ≤ rootCandidates919.length then search edge 4 [7213] rootCandidates919 7213 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan920

set_option maxRecDepth 4096 in
theorem outerScan918 : scan edge (search edge 4) 5 [] 0 tail918 best = best := by
  have cons : tail918 = 7211 :: tail919 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates918_checked]
  change scan edge (search edge 4) 5 [] 0 tail919
    (if 4 ≤ rootCandidates918.length then search edge 4 [7211] rootCandidates918 7211 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan919

set_option maxRecDepth 4096 in
theorem outerScan917 : scan edge (search edge 4) 5 [] 0 tail917 best = best := by
  have cons : tail917 = 7207 :: tail918 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates917_checked]
  change scan edge (search edge 4) 5 [] 0 tail918
    (if 4 ≤ rootCandidates917.length then search edge 4 [7207] rootCandidates917 7207 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan918

set_option maxRecDepth 4096 in
theorem outerScan916 : scan edge (search edge 4) 5 [] 0 tail916 best = best := by
  have cons : tail916 = 7193 :: tail917 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates916_checked]
  change scan edge (search edge 4) 5 [] 0 tail917
    (if 4 ≤ rootCandidates916.length then search edge 4 [7193] rootCandidates916 7193 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan917

set_option maxRecDepth 4096 in
theorem outerScan915 : scan edge (search edge 4) 5 [] 0 tail915 best = best := by
  have cons : tail915 = 7187 :: tail916 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates915_checked]
  change scan edge (search edge 4) 5 [] 0 tail916
    (if 4 ≤ rootCandidates915.length then search edge 4 [7187] rootCandidates915 7187 best else best) = best
  rw [if_pos (by decide), rootBranch915]
  exact outerScan916

set_option maxRecDepth 4096 in
theorem outerScan914 : scan edge (search edge 4) 5 [] 0 tail914 best = best := by
  have cons : tail914 = 7177 :: tail915 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates914_checked]
  change scan edge (search edge 4) 5 [] 0 tail915
    (if 4 ≤ rootCandidates914.length then search edge 4 [7177] rootCandidates914 7177 best else best) = best
  rw [if_pos (by decide), rootBranch914]
  exact outerScan915

set_option maxRecDepth 4096 in
theorem outerScan913 : scan edge (search edge 4) 5 [] 0 tail913 best = best := by
  have cons : tail913 = 7159 :: tail914 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates913_checked]
  change scan edge (search edge 4) 5 [] 0 tail914
    (if 4 ≤ rootCandidates913.length then search edge 4 [7159] rootCandidates913 7159 best else best) = best
  rw [if_pos (by decide), rootBranch913]
  exact outerScan914

set_option maxRecDepth 4096 in
theorem outerScan912 : scan edge (search edge 4) 5 [] 0 tail912 best = best := by
  have cons : tail912 = 7151 :: tail913 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates912_checked]
  change scan edge (search edge 4) 5 [] 0 tail913
    (if 4 ≤ rootCandidates912.length then search edge 4 [7151] rootCandidates912 7151 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan913

end ProjectEulerSolutions.PrimePairUpper.Search
