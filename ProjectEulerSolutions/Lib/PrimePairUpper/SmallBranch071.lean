import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates525 : List Nat := [3833, 4049, 5261, 5303, 5393, 6089, 7013, 7019, 7109, 7529, 7937, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates525_checked : candidates edge 5 0 3797 tail526 best.bound = rootCandidates525 := by
  unfold candidates
  have row : edge 3797 = SearchData.row3797.contains := by funext q; rfl
  rw [row]
  decide

def innerTail525_12 : List Nat := []
theorem innerScan525_12 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_12 best = best := rfl

def innerTail525_11 : List Nat := 8273 :: innerTail525_12
def innerCandidates525_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates525_11_checked : candidates edge 4 3797 8273 innerTail525_12 best.bound = innerCandidates525_11 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan525_11 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_11 best = best := by
  rw [innerTail525_11, scan, if_neg (by decide), innerCandidates525_11_checked]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_12
    (if 3 ≤ innerCandidates525_11.length then search edge 3 [3797, 8273] innerCandidates525_11 12070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan525_12

def innerTail525_10 : List Nat := 7937 :: innerTail525_11
def innerCandidates525_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates525_10_checked : candidates edge 4 3797 7937 innerTail525_11 best.bound = innerCandidates525_10 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan525_10 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_10 best = best := by
  rw [innerTail525_10, scan, if_neg (by decide), innerCandidates525_10_checked]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_11
    (if 3 ≤ innerCandidates525_10.length then search edge 3 [3797, 7937] innerCandidates525_10 11734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan525_11

def innerTail525_9 : List Nat := 7529 :: innerTail525_10
def innerCandidates525_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates525_9_checked : candidates edge 4 3797 7529 innerTail525_10 best.bound = innerCandidates525_9 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan525_9 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_9 best = best := by
  rw [innerTail525_9, scan, if_neg (by decide), innerCandidates525_9_checked]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_10
    (if 3 ≤ innerCandidates525_9.length then search edge 3 [3797, 7529] innerCandidates525_9 11326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan525_10

def innerTail525_8 : List Nat := 7109 :: innerTail525_9
def innerCandidates525_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates525_8_checked : candidates edge 4 3797 7109 innerTail525_9 best.bound = innerCandidates525_8 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan525_8 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_8 best = best := by
  rw [innerTail525_8, scan, if_neg (by decide), innerCandidates525_8_checked]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_9
    (if 3 ≤ innerCandidates525_8.length then search edge 3 [3797, 7109] innerCandidates525_8 10906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan525_9

def innerTail525_7 : List Nat := 7019 :: innerTail525_8
def innerCandidates525_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates525_7_checked : candidates edge 4 3797 7019 innerTail525_8 best.bound = innerCandidates525_7 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan525_7 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_7 best = best := by
  rw [innerTail525_7, scan, if_neg (by decide), innerCandidates525_7_checked]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_8
    (if 3 ≤ innerCandidates525_7.length then search edge 3 [3797, 7019] innerCandidates525_7 10816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan525_8

def innerTail525_6 : List Nat := 7013 :: innerTail525_7
def innerCandidates525_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates525_6_checked : candidates edge 4 3797 7013 innerTail525_7 best.bound = innerCandidates525_6 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan525_6 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_6 best = best := by
  rw [innerTail525_6, scan, if_neg (by decide), innerCandidates525_6_checked]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_7
    (if 3 ≤ innerCandidates525_6.length then search edge 3 [3797, 7013] innerCandidates525_6 10810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan525_7

def innerTail525_5 : List Nat := 6089 :: innerTail525_6
def innerCandidates525_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates525_5_checked : candidates edge 4 3797 6089 innerTail525_6 best.bound = innerCandidates525_5 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan525_5 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_5 best = best := by
  rw [innerTail525_5, scan, if_neg (by decide), innerCandidates525_5_checked]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_6
    (if 3 ≤ innerCandidates525_5.length then search edge 3 [3797, 6089] innerCandidates525_5 9886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan525_6

def innerTail525_4 : List Nat := 5393 :: innerTail525_5
def innerCandidates525_4 : List Nat := [7937]
set_option maxRecDepth 4096 in
theorem innerCandidates525_4_checked : candidates edge 4 3797 5393 innerTail525_5 best.bound = innerCandidates525_4 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan525_4 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_4 best = best := by
  rw [innerTail525_4, scan, if_neg (by decide), innerCandidates525_4_checked]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_5
    (if 3 ≤ innerCandidates525_4.length then search edge 3 [3797, 5393] innerCandidates525_4 9190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan525_5

def innerTail525_3 : List Nat := 5303 :: innerTail525_4
def innerCandidates525_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates525_3_checked : candidates edge 4 3797 5303 innerTail525_4 best.bound = innerCandidates525_3 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan525_3 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_3 best = best := by
  rw [innerTail525_3, scan, if_neg (by decide), innerCandidates525_3_checked]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_4
    (if 3 ≤ innerCandidates525_3.length then search edge 3 [3797, 5303] innerCandidates525_3 9100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan525_4

def innerTail525_2 : List Nat := 5261 :: innerTail525_3
def innerCandidates525_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates525_2_checked : candidates edge 4 3797 5261 innerTail525_3 best.bound = innerCandidates525_2 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan525_2 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_2 best = best := by
  rw [innerTail525_2, scan, if_neg (by decide), innerCandidates525_2_checked]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_3
    (if 3 ≤ innerCandidates525_2.length then search edge 3 [3797, 5261] innerCandidates525_2 9058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan525_3

def innerTail525_1 : List Nat := 4049 :: innerTail525_2
def innerCandidates525_1 : List Nat := [5261]
set_option maxRecDepth 4096 in
theorem innerCandidates525_1_checked : candidates edge 4 3797 4049 innerTail525_2 best.bound = innerCandidates525_1 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan525_1 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_1 best = best := by
  rw [innerTail525_1, scan, if_neg (by decide), innerCandidates525_1_checked]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_2
    (if 3 ≤ innerCandidates525_1.length then search edge 3 [3797, 4049] innerCandidates525_1 7846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan525_2

def innerTail525_0 : List Nat := 3833 :: innerTail525_1
def innerCandidates525_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates525_0_checked : candidates edge 4 3797 3833 innerTail525_1 best.bound = innerCandidates525_0 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan525_0 : scan edge (search edge 3) 4 [3797] 3797 innerTail525_0 best = best := by
  rw [innerTail525_0, scan, if_neg (by decide), innerCandidates525_0_checked]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_1
    (if 3 ≤ innerCandidates525_0.length then search edge 3 [3797, 3833] innerCandidates525_0 7630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan525_1

theorem rootBranch525 : search edge 4 [3797] rootCandidates525 3797 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3797] 3797 innerTail525_0 best = best
  exact innerScan525_0

def rootCandidates524 : List Nat := [5179, 5659, 6733, 6907, 6949, 7213, 7873, 8167, 8179]
set_option maxRecDepth 4096 in
theorem rootCandidates524_checked : candidates edge 5 0 3793 tail525 best.bound = rootCandidates524 := by
  unfold candidates
  have row : edge 3793 = SearchData.row3793.contains := by funext q; rfl
  rw [row]
  decide

def innerTail524_9 : List Nat := []
theorem innerScan524_9 : scan edge (search edge 3) 4 [3793] 3793 innerTail524_9 best = best := rfl

def innerTail524_8 : List Nat := 8179 :: innerTail524_9
def innerCandidates524_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates524_8_checked : candidates edge 4 3793 8179 innerTail524_9 best.bound = innerCandidates524_8 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan524_8 : scan edge (search edge 3) 4 [3793] 3793 innerTail524_8 best = best := by
  rw [innerTail524_8, scan, if_neg (by decide), innerCandidates524_8_checked]
  change scan edge (search edge 3) 4 [3793] 3793 innerTail524_9
    (if 3 ≤ innerCandidates524_8.length then search edge 3 [3793, 8179] innerCandidates524_8 11972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan524_9

def innerTail524_7 : List Nat := 8167 :: innerTail524_8
def innerCandidates524_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates524_7_checked : candidates edge 4 3793 8167 innerTail524_8 best.bound = innerCandidates524_7 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan524_7 : scan edge (search edge 3) 4 [3793] 3793 innerTail524_7 best = best := by
  rw [innerTail524_7, scan, if_neg (by decide), innerCandidates524_7_checked]
  change scan edge (search edge 3) 4 [3793] 3793 innerTail524_8
    (if 3 ≤ innerCandidates524_7.length then search edge 3 [3793, 8167] innerCandidates524_7 11960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan524_8

def innerTail524_6 : List Nat := 7873 :: innerTail524_7
def innerCandidates524_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates524_6_checked : candidates edge 4 3793 7873 innerTail524_7 best.bound = innerCandidates524_6 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan524_6 : scan edge (search edge 3) 4 [3793] 3793 innerTail524_6 best = best := by
  rw [innerTail524_6, scan, if_neg (by decide), innerCandidates524_6_checked]
  change scan edge (search edge 3) 4 [3793] 3793 innerTail524_7
    (if 3 ≤ innerCandidates524_6.length then search edge 3 [3793, 7873] innerCandidates524_6 11666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan524_7

def innerTail524_5 : List Nat := 7213 :: innerTail524_6
def innerCandidates524_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates524_5_checked : candidates edge 4 3793 7213 innerTail524_6 best.bound = innerCandidates524_5 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan524_5 : scan edge (search edge 3) 4 [3793] 3793 innerTail524_5 best = best := by
  rw [innerTail524_5, scan, if_neg (by decide), innerCandidates524_5_checked]
  change scan edge (search edge 3) 4 [3793] 3793 innerTail524_6
    (if 3 ≤ innerCandidates524_5.length then search edge 3 [3793, 7213] innerCandidates524_5 11006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan524_6

def innerTail524_4 : List Nat := 6949 :: innerTail524_5
def innerCandidates524_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates524_4_checked : candidates edge 4 3793 6949 innerTail524_5 best.bound = innerCandidates524_4 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan524_4 : scan edge (search edge 3) 4 [3793] 3793 innerTail524_4 best = best := by
  rw [innerTail524_4, scan, if_neg (by decide), innerCandidates524_4_checked]
  change scan edge (search edge 3) 4 [3793] 3793 innerTail524_5
    (if 3 ≤ innerCandidates524_4.length then search edge 3 [3793, 6949] innerCandidates524_4 10742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan524_5

def innerTail524_3 : List Nat := 6907 :: innerTail524_4
def innerCandidates524_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates524_3_checked : candidates edge 4 3793 6907 innerTail524_4 best.bound = innerCandidates524_3 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan524_3 : scan edge (search edge 3) 4 [3793] 3793 innerTail524_3 best = best := by
  rw [innerTail524_3, scan, if_neg (by decide), innerCandidates524_3_checked]
  change scan edge (search edge 3) 4 [3793] 3793 innerTail524_4
    (if 3 ≤ innerCandidates524_3.length then search edge 3 [3793, 6907] innerCandidates524_3 10700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan524_4

def innerTail524_2 : List Nat := 6733 :: innerTail524_3
def innerCandidates524_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates524_2_checked : candidates edge 4 3793 6733 innerTail524_3 best.bound = innerCandidates524_2 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan524_2 : scan edge (search edge 3) 4 [3793] 3793 innerTail524_2 best = best := by
  rw [innerTail524_2, scan, if_neg (by decide), innerCandidates524_2_checked]
  change scan edge (search edge 3) 4 [3793] 3793 innerTail524_3
    (if 3 ≤ innerCandidates524_2.length then search edge 3 [3793, 6733] innerCandidates524_2 10526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan524_3

def innerTail524_1 : List Nat := 5659 :: innerTail524_2
def innerCandidates524_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates524_1_checked : candidates edge 4 3793 5659 innerTail524_2 best.bound = innerCandidates524_1 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan524_1 : scan edge (search edge 3) 4 [3793] 3793 innerTail524_1 best = best := by
  rw [innerTail524_1, scan, if_neg (by decide), innerCandidates524_1_checked]
  change scan edge (search edge 3) 4 [3793] 3793 innerTail524_2
    (if 3 ≤ innerCandidates524_1.length then search edge 3 [3793, 5659] innerCandidates524_1 9452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan524_2

def innerTail524_0 : List Nat := 5179 :: innerTail524_1
def innerCandidates524_0 : List Nat := [7873]
set_option maxRecDepth 4096 in
theorem innerCandidates524_0_checked : candidates edge 4 3793 5179 innerTail524_1 best.bound = innerCandidates524_0 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan524_0 : scan edge (search edge 3) 4 [3793] 3793 innerTail524_0 best = best := by
  rw [innerTail524_0, scan, if_neg (by decide), innerCandidates524_0_checked]
  change scan edge (search edge 3) 4 [3793] 3793 innerTail524_1
    (if 3 ≤ innerCandidates524_0.length then search edge 3 [3793, 5179] innerCandidates524_0 8972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan524_1

theorem rootBranch524 : search edge 4 [3793] rootCandidates524 3793 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3793] 3793 innerTail524_0 best = best
  exact innerScan524_0

def rootCandidates523 : List Nat := [3863, 4241, 4637, 4931, 5099, 5843, 6053, 6299, 6329, 7523, 7577, 7829]
set_option maxRecDepth 4096 in
theorem rootCandidates523_checked : candidates edge 5 0 3779 tail524 best.bound = rootCandidates523 := by
  unfold candidates
  have row : edge 3779 = SearchData.row3779.contains := by funext q; rfl
  rw [row]
  decide

def innerTail523_12 : List Nat := []
theorem innerScan523_12 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_12 best = best := rfl

def innerTail523_11 : List Nat := 7829 :: innerTail523_12
def innerCandidates523_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates523_11_checked : candidates edge 4 3779 7829 innerTail523_12 best.bound = innerCandidates523_11 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan523_11 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_11 best = best := by
  rw [innerTail523_11, scan, if_neg (by decide), innerCandidates523_11_checked]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_12
    (if 3 ≤ innerCandidates523_11.length then search edge 3 [3779, 7829] innerCandidates523_11 11608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan523_12

def innerTail523_10 : List Nat := 7577 :: innerTail523_11
def innerCandidates523_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates523_10_checked : candidates edge 4 3779 7577 innerTail523_11 best.bound = innerCandidates523_10 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan523_10 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_10 best = best := by
  rw [innerTail523_10, scan, if_neg (by decide), innerCandidates523_10_checked]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_11
    (if 3 ≤ innerCandidates523_10.length then search edge 3 [3779, 7577] innerCandidates523_10 11356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan523_11

def innerTail523_9 : List Nat := 7523 :: innerTail523_10
def innerCandidates523_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates523_9_checked : candidates edge 4 3779 7523 innerTail523_10 best.bound = innerCandidates523_9 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan523_9 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_9 best = best := by
  rw [innerTail523_9, scan, if_neg (by decide), innerCandidates523_9_checked]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_10
    (if 3 ≤ innerCandidates523_9.length then search edge 3 [3779, 7523] innerCandidates523_9 11302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan523_10

def innerTail523_8 : List Nat := 6329 :: innerTail523_9
def innerCandidates523_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates523_8_checked : candidates edge 4 3779 6329 innerTail523_9 best.bound = innerCandidates523_8 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan523_8 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_8 best = best := by
  rw [innerTail523_8, scan, if_neg (by decide), innerCandidates523_8_checked]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_9
    (if 3 ≤ innerCandidates523_8.length then search edge 3 [3779, 6329] innerCandidates523_8 10108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan523_9

def innerTail523_7 : List Nat := 6299 :: innerTail523_8
def innerCandidates523_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates523_7_checked : candidates edge 4 3779 6299 innerTail523_8 best.bound = innerCandidates523_7 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan523_7 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_7 best = best := by
  rw [innerTail523_7, scan, if_neg (by decide), innerCandidates523_7_checked]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_8
    (if 3 ≤ innerCandidates523_7.length then search edge 3 [3779, 6299] innerCandidates523_7 10078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan523_8

def innerTail523_6 : List Nat := 6053 :: innerTail523_7
def innerCandidates523_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates523_6_checked : candidates edge 4 3779 6053 innerTail523_7 best.bound = innerCandidates523_6 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan523_6 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_6 best = best := by
  rw [innerTail523_6, scan, if_neg (by decide), innerCandidates523_6_checked]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_7
    (if 3 ≤ innerCandidates523_6.length then search edge 3 [3779, 6053] innerCandidates523_6 9832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan523_7

def innerTail523_5 : List Nat := 5843 :: innerTail523_6
def innerCandidates523_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates523_5_checked : candidates edge 4 3779 5843 innerTail523_6 best.bound = innerCandidates523_5 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan523_5 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_5 best = best := by
  rw [innerTail523_5, scan, if_neg (by decide), innerCandidates523_5_checked]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_6
    (if 3 ≤ innerCandidates523_5.length then search edge 3 [3779, 5843] innerCandidates523_5 9622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan523_6

def innerTail523_4 : List Nat := 5099 :: innerTail523_5
def innerCandidates523_4 : List Nat := [7523]
set_option maxRecDepth 4096 in
theorem innerCandidates523_4_checked : candidates edge 4 3779 5099 innerTail523_5 best.bound = innerCandidates523_4 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan523_4 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_4 best = best := by
  rw [innerTail523_4, scan, if_neg (by decide), innerCandidates523_4_checked]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_5
    (if 3 ≤ innerCandidates523_4.length then search edge 3 [3779, 5099] innerCandidates523_4 8878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan523_5

def innerTail523_3 : List Nat := 4931 :: innerTail523_4
def innerCandidates523_3 : List Nat := [5843]
set_option maxRecDepth 4096 in
theorem innerCandidates523_3_checked : candidates edge 4 3779 4931 innerTail523_4 best.bound = innerCandidates523_3 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan523_3 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_3 best = best := by
  rw [innerTail523_3, scan, if_neg (by decide), innerCandidates523_3_checked]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_4
    (if 3 ≤ innerCandidates523_3.length then search edge 3 [3779, 4931] innerCandidates523_3 8710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan523_4

def innerTail523_2 : List Nat := 4637 :: innerTail523_3
def innerCandidates523_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates523_2_checked : candidates edge 4 3779 4637 innerTail523_3 best.bound = innerCandidates523_2 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan523_2 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_2 best = best := by
  rw [innerTail523_2, scan, if_neg (by decide), innerCandidates523_2_checked]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_3
    (if 3 ≤ innerCandidates523_2.length then search edge 3 [3779, 4637] innerCandidates523_2 8416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan523_3

def innerTail523_1 : List Nat := 4241 :: innerTail523_2
def innerCandidates523_1 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates523_1_checked : candidates edge 4 3779 4241 innerTail523_2 best.bound = innerCandidates523_1 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan523_1 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_1 best = best := by
  rw [innerTail523_1, scan, if_neg (by decide), innerCandidates523_1_checked]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_2
    (if 3 ≤ innerCandidates523_1.length then search edge 3 [3779, 4241] innerCandidates523_1 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan523_2

def innerTail523_0 : List Nat := 3863 :: innerTail523_1
def innerCandidates523_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates523_0_checked : candidates edge 4 3779 3863 innerTail523_1 best.bound = innerCandidates523_0 := by
  unfold candidates
  have row : edge 3863 = SearchData.row3863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan523_0 : scan edge (search edge 3) 4 [3779] 3779 innerTail523_0 best = best := by
  rw [innerTail523_0, scan, if_neg (by decide), innerCandidates523_0_checked]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_1
    (if 3 ≤ innerCandidates523_0.length then search edge 3 [3779, 3863] innerCandidates523_0 7642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan523_1

theorem rootBranch523 : search edge 4 [3779] rootCandidates523 3779 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3779] 3779 innerTail523_0 best = best
  exact innerScan523_0

def rootCandidates522 : List Nat := [4051, 4597, 4603, 5653, 6619, 6949, 7669, 8161]
set_option maxRecDepth 4096 in
theorem rootCandidates522_checked : candidates edge 5 0 3769 tail523 best.bound = rootCandidates522 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext q; rfl
  rw [row]
  decide

def innerTail522_8 : List Nat := []
theorem innerScan522_8 : scan edge (search edge 3) 4 [3769] 3769 innerTail522_8 best = best := rfl

def innerTail522_7 : List Nat := 8161 :: innerTail522_8
def innerCandidates522_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates522_7_checked : candidates edge 4 3769 8161 innerTail522_8 best.bound = innerCandidates522_7 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan522_7 : scan edge (search edge 3) 4 [3769] 3769 innerTail522_7 best = best := by
  rw [innerTail522_7, scan, if_neg (by decide), innerCandidates522_7_checked]
  change scan edge (search edge 3) 4 [3769] 3769 innerTail522_8
    (if 3 ≤ innerCandidates522_7.length then search edge 3 [3769, 8161] innerCandidates522_7 11930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan522_8

def innerTail522_6 : List Nat := 7669 :: innerTail522_7
def innerCandidates522_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates522_6_checked : candidates edge 4 3769 7669 innerTail522_7 best.bound = innerCandidates522_6 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan522_6 : scan edge (search edge 3) 4 [3769] 3769 innerTail522_6 best = best := by
  rw [innerTail522_6, scan, if_neg (by decide), innerCandidates522_6_checked]
  change scan edge (search edge 3) 4 [3769] 3769 innerTail522_7
    (if 3 ≤ innerCandidates522_6.length then search edge 3 [3769, 7669] innerCandidates522_6 11438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan522_7

def innerTail522_5 : List Nat := 6949 :: innerTail522_6
def innerCandidates522_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates522_5_checked : candidates edge 4 3769 6949 innerTail522_6 best.bound = innerCandidates522_5 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan522_5 : scan edge (search edge 3) 4 [3769] 3769 innerTail522_5 best = best := by
  rw [innerTail522_5, scan, if_neg (by decide), innerCandidates522_5_checked]
  change scan edge (search edge 3) 4 [3769] 3769 innerTail522_6
    (if 3 ≤ innerCandidates522_5.length then search edge 3 [3769, 6949] innerCandidates522_5 10718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan522_6

def innerTail522_4 : List Nat := 6619 :: innerTail522_5
def innerCandidates522_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates522_4_checked : candidates edge 4 3769 6619 innerTail522_5 best.bound = innerCandidates522_4 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan522_4 : scan edge (search edge 3) 4 [3769] 3769 innerTail522_4 best = best := by
  rw [innerTail522_4, scan, if_neg (by decide), innerCandidates522_4_checked]
  change scan edge (search edge 3) 4 [3769] 3769 innerTail522_5
    (if 3 ≤ innerCandidates522_4.length then search edge 3 [3769, 6619] innerCandidates522_4 10388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan522_5

def innerTail522_3 : List Nat := 5653 :: innerTail522_4
def innerCandidates522_3 : List Nat := [6619]
set_option maxRecDepth 4096 in
theorem innerCandidates522_3_checked : candidates edge 4 3769 5653 innerTail522_4 best.bound = innerCandidates522_3 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan522_3 : scan edge (search edge 3) 4 [3769] 3769 innerTail522_3 best = best := by
  rw [innerTail522_3, scan, if_neg (by decide), innerCandidates522_3_checked]
  change scan edge (search edge 3) 4 [3769] 3769 innerTail522_4
    (if 3 ≤ innerCandidates522_3.length then search edge 3 [3769, 5653] innerCandidates522_3 9422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan522_4

def innerTail522_2 : List Nat := 4603 :: innerTail522_3
def innerCandidates522_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates522_2_checked : candidates edge 4 3769 4603 innerTail522_3 best.bound = innerCandidates522_2 := by
  unfold candidates
  have row : edge 4603 = SearchData.row4603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan522_2 : scan edge (search edge 3) 4 [3769] 3769 innerTail522_2 best = best := by
  rw [innerTail522_2, scan, if_neg (by decide), innerCandidates522_2_checked]
  change scan edge (search edge 3) 4 [3769] 3769 innerTail522_3
    (if 3 ≤ innerCandidates522_2.length then search edge 3 [3769, 4603] innerCandidates522_2 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan522_3

def innerTail522_1 : List Nat := 4597 :: innerTail522_2
def innerCandidates522_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates522_1_checked : candidates edge 4 3769 4597 innerTail522_2 best.bound = innerCandidates522_1 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan522_1 : scan edge (search edge 3) 4 [3769] 3769 innerTail522_1 best = best := by
  rw [innerTail522_1, scan, if_neg (by decide), innerCandidates522_1_checked]
  change scan edge (search edge 3) 4 [3769] 3769 innerTail522_2
    (if 3 ≤ innerCandidates522_1.length then search edge 3 [3769, 4597] innerCandidates522_1 8366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan522_2

def innerTail522_0 : List Nat := 4051 :: innerTail522_1
def innerCandidates522_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates522_0_checked : candidates edge 4 3769 4051 innerTail522_1 best.bound = innerCandidates522_0 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan522_0 : scan edge (search edge 3) 4 [3769] 3769 innerTail522_0 best = best := by
  rw [innerTail522_0, scan, if_neg (by decide), innerCandidates522_0_checked]
  change scan edge (search edge 3) 4 [3769] 3769 innerTail522_1
    (if 3 ≤ innerCandidates522_0.length then search edge 3 [3769, 4051] innerCandidates522_0 7820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan522_1

theorem rootBranch522 : search edge 4 [3769] rootCandidates522 3769 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3769] 3769 innerTail522_0 best = best
  exact innerScan522_0

def rootCandidates521 : List Nat := [3821, 3947, 4127, 4523, 4937, 5987, 6521, 6701, 6719, 6911, 7001, 7193, 7487, 7547, 7793]
set_option maxRecDepth 4096 in
theorem rootCandidates521_checked : candidates edge 5 0 3767 tail522 best.bound = rootCandidates521 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext q; rfl
  rw [row]
  decide

def innerTail521_15 : List Nat := []
theorem innerScan521_15 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_15 best = best := rfl

def innerTail521_14 : List Nat := 7793 :: innerTail521_15
def innerCandidates521_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates521_14_checked : candidates edge 4 3767 7793 innerTail521_15 best.bound = innerCandidates521_14 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_14 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_14 best = best := by
  rw [innerTail521_14, scan, if_neg (by decide), innerCandidates521_14_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_15
    (if 3 ≤ innerCandidates521_14.length then search edge 3 [3767, 7793] innerCandidates521_14 11560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_15

def innerTail521_13 : List Nat := 7547 :: innerTail521_14
def innerCandidates521_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates521_13_checked : candidates edge 4 3767 7547 innerTail521_14 best.bound = innerCandidates521_13 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_13 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_13 best = best := by
  rw [innerTail521_13, scan, if_neg (by decide), innerCandidates521_13_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_14
    (if 3 ≤ innerCandidates521_13.length then search edge 3 [3767, 7547] innerCandidates521_13 11314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_14

def innerTail521_12 : List Nat := 7487 :: innerTail521_13
def innerCandidates521_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates521_12_checked : candidates edge 4 3767 7487 innerTail521_13 best.bound = innerCandidates521_12 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_12 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_12 best = best := by
  rw [innerTail521_12, scan, if_neg (by decide), innerCandidates521_12_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_13
    (if 3 ≤ innerCandidates521_12.length then search edge 3 [3767, 7487] innerCandidates521_12 11254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_13

def innerTail521_11 : List Nat := 7193 :: innerTail521_12
def innerCandidates521_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates521_11_checked : candidates edge 4 3767 7193 innerTail521_12 best.bound = innerCandidates521_11 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_11 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_11 best = best := by
  rw [innerTail521_11, scan, if_neg (by decide), innerCandidates521_11_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_12
    (if 3 ≤ innerCandidates521_11.length then search edge 3 [3767, 7193] innerCandidates521_11 10960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_12

def innerTail521_10 : List Nat := 7001 :: innerTail521_11
def innerCandidates521_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates521_10_checked : candidates edge 4 3767 7001 innerTail521_11 best.bound = innerCandidates521_10 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_10 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_10 best = best := by
  rw [innerTail521_10, scan, if_neg (by decide), innerCandidates521_10_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_11
    (if 3 ≤ innerCandidates521_10.length then search edge 3 [3767, 7001] innerCandidates521_10 10768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_11

def innerTail521_9 : List Nat := 6911 :: innerTail521_10
def innerCandidates521_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates521_9_checked : candidates edge 4 3767 6911 innerTail521_10 best.bound = innerCandidates521_9 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_9 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_9 best = best := by
  rw [innerTail521_9, scan, if_neg (by decide), innerCandidates521_9_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_10
    (if 3 ≤ innerCandidates521_9.length then search edge 3 [3767, 6911] innerCandidates521_9 10678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_10

def innerTail521_8 : List Nat := 6719 :: innerTail521_9
def innerCandidates521_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates521_8_checked : candidates edge 4 3767 6719 innerTail521_9 best.bound = innerCandidates521_8 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_8 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_8 best = best := by
  rw [innerTail521_8, scan, if_neg (by decide), innerCandidates521_8_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_9
    (if 3 ≤ innerCandidates521_8.length then search edge 3 [3767, 6719] innerCandidates521_8 10486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_9

def innerTail521_7 : List Nat := 6701 :: innerTail521_8
def innerCandidates521_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates521_7_checked : candidates edge 4 3767 6701 innerTail521_8 best.bound = innerCandidates521_7 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_7 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_7 best = best := by
  rw [innerTail521_7, scan, if_neg (by decide), innerCandidates521_7_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_8
    (if 3 ≤ innerCandidates521_7.length then search edge 3 [3767, 6701] innerCandidates521_7 10468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_8

def innerTail521_6 : List Nat := 6521 :: innerTail521_7
def innerCandidates521_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates521_6_checked : candidates edge 4 3767 6521 innerTail521_7 best.bound = innerCandidates521_6 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_6 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_6 best = best := by
  rw [innerTail521_6, scan, if_neg (by decide), innerCandidates521_6_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_7
    (if 3 ≤ innerCandidates521_6.length then search edge 3 [3767, 6521] innerCandidates521_6 10288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_7

def innerTail521_5 : List Nat := 5987 :: innerTail521_6
def innerCandidates521_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates521_5_checked : candidates edge 4 3767 5987 innerTail521_6 best.bound = innerCandidates521_5 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_5 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_5 best = best := by
  rw [innerTail521_5, scan, if_neg (by decide), innerCandidates521_5_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_6
    (if 3 ≤ innerCandidates521_5.length then search edge 3 [3767, 5987] innerCandidates521_5 9754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_6

def innerTail521_4 : List Nat := 4937 :: innerTail521_5
def innerCandidates521_4 : List Nat := [7001]
set_option maxRecDepth 4096 in
theorem innerCandidates521_4_checked : candidates edge 4 3767 4937 innerTail521_5 best.bound = innerCandidates521_4 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_4 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_4 best = best := by
  rw [innerTail521_4, scan, if_neg (by decide), innerCandidates521_4_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_5
    (if 3 ≤ innerCandidates521_4.length then search edge 3 [3767, 4937] innerCandidates521_4 8704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_5

def innerTail521_3 : List Nat := 4523 :: innerTail521_4
def innerCandidates521_3 : List Nat := [6911]
set_option maxRecDepth 4096 in
theorem innerCandidates521_3_checked : candidates edge 4 3767 4523 innerTail521_4 best.bound = innerCandidates521_3 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_3 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_3 best = best := by
  rw [innerTail521_3, scan, if_neg (by decide), innerCandidates521_3_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_4
    (if 3 ≤ innerCandidates521_3.length then search edge 3 [3767, 4523] innerCandidates521_3 8290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_4

def innerTail521_2 : List Nat := 4127 :: innerTail521_3
def innerCandidates521_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates521_2_checked : candidates edge 4 3767 4127 innerTail521_3 best.bound = innerCandidates521_2 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_2 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_2 best = best := by
  rw [innerTail521_2, scan, if_neg (by decide), innerCandidates521_2_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_3
    (if 3 ≤ innerCandidates521_2.length then search edge 3 [3767, 4127] innerCandidates521_2 7894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_3

def innerTail521_1 : List Nat := 3947 :: innerTail521_2
def innerCandidates521_1 : List Nat := [7001]
set_option maxRecDepth 4096 in
theorem innerCandidates521_1_checked : candidates edge 4 3767 3947 innerTail521_2 best.bound = innerCandidates521_1 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_1 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_1 best = best := by
  rw [innerTail521_1, scan, if_neg (by decide), innerCandidates521_1_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_2
    (if 3 ≤ innerCandidates521_1.length then search edge 3 [3767, 3947] innerCandidates521_1 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_2

def innerTail521_0 : List Nat := 3821 :: innerTail521_1
def innerCandidates521_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates521_0_checked : candidates edge 4 3767 3821 innerTail521_1 best.bound = innerCandidates521_0 := by
  unfold candidates
  have row : edge 3821 = SearchData.row3821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan521_0 : scan edge (search edge 3) 4 [3767] 3767 innerTail521_0 best = best := by
  rw [innerTail521_0, scan, if_neg (by decide), innerCandidates521_0_checked]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_1
    (if 3 ≤ innerCandidates521_0.length then search edge 3 [3767, 3821] innerCandidates521_0 7588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan521_1

theorem rootBranch521 : search edge 4 [3767] rootCandidates521 3767 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3767] 3767 innerTail521_0 best = best
  exact innerScan521_0

end ProjectEulerSolutions.PrimePairUpper.Search
