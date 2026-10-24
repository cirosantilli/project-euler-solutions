import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch017
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch018

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan903 : scan edge (search edge 4) 5 [] 0 tail903 best = best := by
  have cons : tail903 = 7043 :: tail904 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates903_checked]
  change scan edge (search edge 4) 5 [] 0 tail904
    (if 4 ≤ rootCandidates903.length then search edge 4 [7043] rootCandidates903 7043 best else best) = best
  rw [if_pos (by decide), rootBranch903]
  exact outerScan904

set_option maxRecDepth 4096 in
theorem outerScan902 : scan edge (search edge 4) 5 [] 0 tail902 best = best := by
  have cons : tail902 = 7039 :: tail903 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates902_checked]
  change scan edge (search edge 4) 5 [] 0 tail903
    (if 4 ≤ rootCandidates902.length then search edge 4 [7039] rootCandidates902 7039 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan903

set_option maxRecDepth 4096 in
theorem outerScan901 : scan edge (search edge 4) 5 [] 0 tail901 best = best := by
  have cons : tail901 = 7027 :: tail902 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates901_checked]
  change scan edge (search edge 4) 5 [] 0 tail902
    (if 4 ≤ rootCandidates901.length then search edge 4 [7027] rootCandidates901 7027 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan902

set_option maxRecDepth 4096 in
theorem outerScan900 : scan edge (search edge 4) 5 [] 0 tail900 best = best := by
  have cons : tail900 = 7019 :: tail901 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates900_checked]
  change scan edge (search edge 4) 5 [] 0 tail901
    (if 4 ≤ rootCandidates900.length then search edge 4 [7019] rootCandidates900 7019 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan901

set_option maxRecDepth 4096 in
theorem outerScan899 : scan edge (search edge 4) 5 [] 0 tail899 best = best := by
  have cons : tail899 = 7013 :: tail900 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates899_checked]
  change scan edge (search edge 4) 5 [] 0 tail900
    (if 4 ≤ rootCandidates899.length then search edge 4 [7013] rootCandidates899 7013 best else best) = best
  rw [if_pos (by decide), rootBranch899]
  exact outerScan900

set_option maxRecDepth 4096 in
theorem outerScan898 : scan edge (search edge 4) 5 [] 0 tail898 best = best := by
  have cons : tail898 = 7001 :: tail899 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates898_checked]
  change scan edge (search edge 4) 5 [] 0 tail899
    (if 4 ≤ rootCandidates898.length then search edge 4 [7001] rootCandidates898 7001 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan899

set_option maxRecDepth 4096 in
theorem outerScan897 : scan edge (search edge 4) 5 [] 0 tail897 best = best := by
  have cons : tail897 = 6997 :: tail898 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates897_checked]
  change scan edge (search edge 4) 5 [] 0 tail898
    (if 4 ≤ rootCandidates897.length then search edge 4 [6997] rootCandidates897 6997 best else best) = best
  rw [if_pos (by decide), rootBranch897]
  exact outerScan898

set_option maxRecDepth 4096 in
theorem outerScan896 : scan edge (search edge 4) 5 [] 0 tail896 best = best := by
  have cons : tail896 = 6991 :: tail897 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates896_checked]
  change scan edge (search edge 4) 5 [] 0 tail897
    (if 4 ≤ rootCandidates896.length then search edge 4 [6991] rootCandidates896 6991 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan897

end ProjectEulerSolutions.PrimePairUpper.Search
