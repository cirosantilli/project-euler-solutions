import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates903 : List Nat := [7253, 7559, 7589, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates903_checked : candidates edge 5 0 7043 tail904 best.bound = rootCandidates903 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext q; rfl
  rw [row]
  decide

def innerTail903_4 : List Nat := []
theorem innerScan903_4 : scan edge (search edge 3) 4 [7043] 7043 innerTail903_4 best = best := rfl

def innerTail903_3 : List Nat := 7949 :: innerTail903_4
def innerCandidates903_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates903_3_checked : candidates edge 4 7043 7949 innerTail903_4 best.bound = innerCandidates903_3 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan903_3 : scan edge (search edge 3) 4 [7043] 7043 innerTail903_3 best = best := by
  rw [innerTail903_3, scan, if_neg (by decide), innerCandidates903_3_checked]
  change scan edge (search edge 3) 4 [7043] 7043 innerTail903_4
    (if 3 ≤ innerCandidates903_3.length then search edge 3 [7043, 7949] innerCandidates903_3 14992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan903_4

def innerTail903_2 : List Nat := 7589 :: innerTail903_3
def innerCandidates903_2 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates903_2_checked : candidates edge 4 7043 7589 innerTail903_3 best.bound = innerCandidates903_2 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan903_2 : scan edge (search edge 3) 4 [7043] 7043 innerTail903_2 best = best := by
  rw [innerTail903_2, scan, if_neg (by decide), innerCandidates903_2_checked]
  change scan edge (search edge 3) 4 [7043] 7043 innerTail903_3
    (if 3 ≤ innerCandidates903_2.length then search edge 3 [7043, 7589] innerCandidates903_2 14632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan903_3

def innerTail903_1 : List Nat := 7559 :: innerTail903_2
def innerCandidates903_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates903_1_checked : candidates edge 4 7043 7559 innerTail903_2 best.bound = innerCandidates903_1 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan903_1 : scan edge (search edge 3) 4 [7043] 7043 innerTail903_1 best = best := by
  rw [innerTail903_1, scan, if_neg (by decide), innerCandidates903_1_checked]
  change scan edge (search edge 3) 4 [7043] 7043 innerTail903_2
    (if 3 ≤ innerCandidates903_1.length then search edge 3 [7043, 7559] innerCandidates903_1 14602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan903_2

def innerTail903_0 : List Nat := 7253 :: innerTail903_1
def innerCandidates903_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates903_0_checked : candidates edge 4 7043 7253 innerTail903_1 best.bound = innerCandidates903_0 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan903_0 : scan edge (search edge 3) 4 [7043] 7043 innerTail903_0 best = best := by
  rw [innerTail903_0, scan, if_neg (by decide), innerCandidates903_0_checked]
  change scan edge (search edge 3) 4 [7043] 7043 innerTail903_1
    (if 3 ≤ innerCandidates903_0.length then search edge 3 [7043, 7253] innerCandidates903_0 14296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan903_1

theorem rootBranch903 : search edge 4 [7043] rootCandidates903 7043 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7043] 7043 innerTail903_0 best = best
  exact innerScan903_0

def rootCandidates902 : List Nat := [7177, 7669]
set_option maxRecDepth 4096 in
theorem rootCandidates902_checked : candidates edge 5 0 7039 tail903 best.bound = rootCandidates902 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates901 : List Nat := [7297, 7687, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates901_checked : candidates edge 5 0 7027 tail902 best.bound = rootCandidates901 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates900 : List Nat := [7451, 7703, 7877]
set_option maxRecDepth 4096 in
theorem rootCandidates900_checked : candidates edge 5 0 7019 tail901 best.bound = rootCandidates900 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates899 : List Nat := [7211, 7517, 8093, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates899_checked : candidates edge 5 0 7013 tail900 best.bound = rootCandidates899 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext q; rfl
  rw [row]
  decide

def innerTail899_4 : List Nat := []
theorem innerScan899_4 : scan edge (search edge 3) 4 [7013] 7013 innerTail899_4 best = best := rfl

def innerTail899_3 : List Nat := 8363 :: innerTail899_4
def innerCandidates899_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates899_3_checked : candidates edge 4 7013 8363 innerTail899_4 best.bound = innerCandidates899_3 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan899_3 : scan edge (search edge 3) 4 [7013] 7013 innerTail899_3 best = best := by
  rw [innerTail899_3, scan, if_neg (by decide), innerCandidates899_3_checked]
  change scan edge (search edge 3) 4 [7013] 7013 innerTail899_4
    (if 3 ≤ innerCandidates899_3.length then search edge 3 [7013, 8363] innerCandidates899_3 15376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan899_4

def innerTail899_2 : List Nat := 8093 :: innerTail899_3
def innerCandidates899_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates899_2_checked : candidates edge 4 7013 8093 innerTail899_3 best.bound = innerCandidates899_2 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan899_2 : scan edge (search edge 3) 4 [7013] 7013 innerTail899_2 best = best := by
  rw [innerTail899_2, scan, if_neg (by decide), innerCandidates899_2_checked]
  change scan edge (search edge 3) 4 [7013] 7013 innerTail899_3
    (if 3 ≤ innerCandidates899_2.length then search edge 3 [7013, 8093] innerCandidates899_2 15106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan899_3

def innerTail899_1 : List Nat := 7517 :: innerTail899_2
def innerCandidates899_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates899_1_checked : candidates edge 4 7013 7517 innerTail899_2 best.bound = innerCandidates899_1 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan899_1 : scan edge (search edge 3) 4 [7013] 7013 innerTail899_1 best = best := by
  rw [innerTail899_1, scan, if_neg (by decide), innerCandidates899_1_checked]
  change scan edge (search edge 3) 4 [7013] 7013 innerTail899_2
    (if 3 ≤ innerCandidates899_1.length then search edge 3 [7013, 7517] innerCandidates899_1 14530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan899_2

def innerTail899_0 : List Nat := 7211 :: innerTail899_1
def innerCandidates899_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates899_0_checked : candidates edge 4 7013 7211 innerTail899_1 best.bound = innerCandidates899_0 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan899_0 : scan edge (search edge 3) 4 [7013] 7013 innerTail899_0 best = best := by
  rw [innerTail899_0, scan, if_neg (by decide), innerCandidates899_0_checked]
  change scan edge (search edge 3) 4 [7013] 7013 innerTail899_1
    (if 3 ≤ innerCandidates899_0.length then search edge 3 [7013, 7211] innerCandidates899_0 14224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan899_1

theorem rootBranch899 : search edge 4 [7013] rootCandidates899 7013 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7013] 7013 innerTail899_0 best = best
  exact innerScan899_0

def rootCandidates898 : List Nat := [7559, 7853]
set_option maxRecDepth 4096 in
theorem rootCandidates898_checked : candidates edge 5 0 7001 tail899 best.bound = rootCandidates898 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates897 : List Nat := [7213, 7351, 7507, 7879, 8011]
set_option maxRecDepth 4096 in
theorem rootCandidates897_checked : candidates edge 5 0 6997 tail898 best.bound = rootCandidates897 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext q; rfl
  rw [row]
  decide

def innerTail897_5 : List Nat := []
theorem innerScan897_5 : scan edge (search edge 3) 4 [6997] 6997 innerTail897_5 best = best := rfl

def innerTail897_4 : List Nat := 8011 :: innerTail897_5
def innerCandidates897_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates897_4_checked : candidates edge 4 6997 8011 innerTail897_5 best.bound = innerCandidates897_4 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan897_4 : scan edge (search edge 3) 4 [6997] 6997 innerTail897_4 best = best := by
  rw [innerTail897_4, scan, if_neg (by decide), innerCandidates897_4_checked]
  change scan edge (search edge 3) 4 [6997] 6997 innerTail897_5
    (if 3 ≤ innerCandidates897_4.length then search edge 3 [6997, 8011] innerCandidates897_4 15008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan897_5

def innerTail897_3 : List Nat := 7879 :: innerTail897_4
def innerCandidates897_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates897_3_checked : candidates edge 4 6997 7879 innerTail897_4 best.bound = innerCandidates897_3 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan897_3 : scan edge (search edge 3) 4 [6997] 6997 innerTail897_3 best = best := by
  rw [innerTail897_3, scan, if_neg (by decide), innerCandidates897_3_checked]
  change scan edge (search edge 3) 4 [6997] 6997 innerTail897_4
    (if 3 ≤ innerCandidates897_3.length then search edge 3 [6997, 7879] innerCandidates897_3 14876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan897_4

def innerTail897_2 : List Nat := 7507 :: innerTail897_3
def innerCandidates897_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates897_2_checked : candidates edge 4 6997 7507 innerTail897_3 best.bound = innerCandidates897_2 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan897_2 : scan edge (search edge 3) 4 [6997] 6997 innerTail897_2 best = best := by
  rw [innerTail897_2, scan, if_neg (by decide), innerCandidates897_2_checked]
  change scan edge (search edge 3) 4 [6997] 6997 innerTail897_3
    (if 3 ≤ innerCandidates897_2.length then search edge 3 [6997, 7507] innerCandidates897_2 14504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan897_3

def innerTail897_1 : List Nat := 7351 :: innerTail897_2
def innerCandidates897_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates897_1_checked : candidates edge 4 6997 7351 innerTail897_2 best.bound = innerCandidates897_1 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan897_1 : scan edge (search edge 3) 4 [6997] 6997 innerTail897_1 best = best := by
  rw [innerTail897_1, scan, if_neg (by decide), innerCandidates897_1_checked]
  change scan edge (search edge 3) 4 [6997] 6997 innerTail897_2
    (if 3 ≤ innerCandidates897_1.length then search edge 3 [6997, 7351] innerCandidates897_1 14348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan897_2

def innerTail897_0 : List Nat := 7213 :: innerTail897_1
def innerCandidates897_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates897_0_checked : candidates edge 4 6997 7213 innerTail897_1 best.bound = innerCandidates897_0 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan897_0 : scan edge (search edge 3) 4 [6997] 6997 innerTail897_0 best = best := by
  rw [innerTail897_0, scan, if_neg (by decide), innerCandidates897_0_checked]
  change scan edge (search edge 3) 4 [6997] 6997 innerTail897_1
    (if 3 ≤ innerCandidates897_0.length then search edge 3 [6997, 7213] innerCandidates897_0 14210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan897_1

theorem rootBranch897 : search edge 4 [6997] rootCandidates897 6997 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6997] 6997 innerTail897_0 best = best
  exact innerScan897_0

def rootCandidates896 : List Nat := [8011]
set_option maxRecDepth 4096 in
theorem rootCandidates896_checked : candidates edge 5 0 6991 tail897 best.bound = rootCandidates896 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
