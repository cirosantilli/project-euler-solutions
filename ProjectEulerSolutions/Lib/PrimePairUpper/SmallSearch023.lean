import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch022
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch023

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan863 : scan edge (search edge 4) 5 [] 0 tail863 best = best := by
  have cons : tail863 = 6709 :: tail864 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates863_checked]
  change scan edge (search edge 4) 5 [] 0 tail864
    (if 4 ≤ rootCandidates863.length then search edge 4 [6709] rootCandidates863 6709 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan864

set_option maxRecDepth 4096 in
theorem outerScan862 : scan edge (search edge 4) 5 [] 0 tail862 best = best := by
  have cons : tail862 = 6703 :: tail863 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates862_checked]
  change scan edge (search edge 4) 5 [] 0 tail863
    (if 4 ≤ rootCandidates862.length then search edge 4 [6703] rootCandidates862 6703 best else best) = best
  rw [if_pos (by decide), rootBranch862]
  exact outerScan863

set_option maxRecDepth 4096 in
theorem outerScan861 : scan edge (search edge 4) 5 [] 0 tail861 best = best := by
  have cons : tail861 = 6701 :: tail862 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates861_checked]
  change scan edge (search edge 4) 5 [] 0 tail862
    (if 4 ≤ rootCandidates861.length then search edge 4 [6701] rootCandidates861 6701 best else best) = best
  rw [if_pos (by decide), rootBranch861]
  exact outerScan862

set_option maxRecDepth 4096 in
theorem outerScan860 : scan edge (search edge 4) 5 [] 0 tail860 best = best := by
  have cons : tail860 = 6691 :: tail861 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates860_checked]
  change scan edge (search edge 4) 5 [] 0 tail861
    (if 4 ≤ rootCandidates860.length then search edge 4 [6691] rootCandidates860 6691 best else best) = best
  rw [if_pos (by decide), rootBranch860]
  exact outerScan861

set_option maxRecDepth 4096 in
theorem outerScan859 : scan edge (search edge 4) 5 [] 0 tail859 best = best := by
  have cons : tail859 = 6689 :: tail860 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates859_checked]
  change scan edge (search edge 4) 5 [] 0 tail860
    (if 4 ≤ rootCandidates859.length then search edge 4 [6689] rootCandidates859 6689 best else best) = best
  rw [if_pos (by decide), rootBranch859]
  exact outerScan860

set_option maxRecDepth 4096 in
theorem outerScan858 : scan edge (search edge 4) 5 [] 0 tail858 best = best := by
  have cons : tail858 = 6679 :: tail859 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates858_checked]
  change scan edge (search edge 4) 5 [] 0 tail859
    (if 4 ≤ rootCandidates858.length then search edge 4 [6679] rootCandidates858 6679 best else best) = best
  rw [if_pos (by decide), rootBranch858]
  exact outerScan859

set_option maxRecDepth 4096 in
theorem outerScan857 : scan edge (search edge 4) 5 [] 0 tail857 best = best := by
  have cons : tail857 = 6673 :: tail858 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates857_checked]
  change scan edge (search edge 4) 5 [] 0 tail858
    (if 4 ≤ rootCandidates857.length then search edge 4 [6673] rootCandidates857 6673 best else best) = best
  rw [if_pos (by decide), rootBranch857]
  exact outerScan858

set_option maxRecDepth 4096 in
theorem outerScan856 : scan edge (search edge 4) 5 [] 0 tail856 best = best := by
  have cons : tail856 = 6661 :: tail857 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates856_checked]
  change scan edge (search edge 4) 5 [] 0 tail857
    (if 4 ≤ rootCandidates856.length then search edge 4 [6661] rootCandidates856 6661 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan857

end ProjectEulerSolutions.PrimePairUpper.Search
