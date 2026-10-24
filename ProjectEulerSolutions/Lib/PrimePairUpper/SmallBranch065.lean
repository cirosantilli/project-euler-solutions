import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates556 : List Nat := [4129, 4657, 5077, 5521, 5701, 6121, 6211, 6841, 7027, 7219, 7549, 7591, 7951, 8263, 8311]
set_option maxRecDepth 4096 in
theorem rootCandidates556_checked : candidates edge 5 0 4051 tail557 best.bound = rootCandidates556 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext q; rfl
  rw [row]
  decide

def innerTail556_15 : List Nat := []
theorem innerScan556_15 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_15 best = best := rfl

def innerTail556_14 : List Nat := 8311 :: innerTail556_15
def innerCandidates556_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_14_checked : candidates edge 4 4051 8311 innerTail556_15 best.bound = innerCandidates556_14 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_14 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_14 best = best := by
  rw [innerTail556_14, scan, if_neg (by decide), innerCandidates556_14_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_15
    (if 3 ≤ innerCandidates556_14.length then search edge 3 [4051, 8311] innerCandidates556_14 12362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_15

def innerTail556_13 : List Nat := 8263 :: innerTail556_14
def innerCandidates556_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_13_checked : candidates edge 4 4051 8263 innerTail556_14 best.bound = innerCandidates556_13 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_13 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_13 best = best := by
  rw [innerTail556_13, scan, if_neg (by decide), innerCandidates556_13_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_14
    (if 3 ≤ innerCandidates556_13.length then search edge 3 [4051, 8263] innerCandidates556_13 12314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_14

def innerTail556_12 : List Nat := 7951 :: innerTail556_13
def innerCandidates556_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_12_checked : candidates edge 4 4051 7951 innerTail556_13 best.bound = innerCandidates556_12 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_12 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_12 best = best := by
  rw [innerTail556_12, scan, if_neg (by decide), innerCandidates556_12_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_13
    (if 3 ≤ innerCandidates556_12.length then search edge 3 [4051, 7951] innerCandidates556_12 12002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_13

def innerTail556_11 : List Nat := 7591 :: innerTail556_12
def innerCandidates556_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_11_checked : candidates edge 4 4051 7591 innerTail556_12 best.bound = innerCandidates556_11 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_11 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_11 best = best := by
  rw [innerTail556_11, scan, if_neg (by decide), innerCandidates556_11_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_12
    (if 3 ≤ innerCandidates556_11.length then search edge 3 [4051, 7591] innerCandidates556_11 11642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_12

def innerTail556_10 : List Nat := 7549 :: innerTail556_11
def innerCandidates556_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_10_checked : candidates edge 4 4051 7549 innerTail556_11 best.bound = innerCandidates556_10 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_10 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_10 best = best := by
  rw [innerTail556_10, scan, if_neg (by decide), innerCandidates556_10_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_11
    (if 3 ≤ innerCandidates556_10.length then search edge 3 [4051, 7549] innerCandidates556_10 11600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_11

def innerTail556_9 : List Nat := 7219 :: innerTail556_10
def innerCandidates556_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_9_checked : candidates edge 4 4051 7219 innerTail556_10 best.bound = innerCandidates556_9 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_9 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_9 best = best := by
  rw [innerTail556_9, scan, if_neg (by decide), innerCandidates556_9_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_10
    (if 3 ≤ innerCandidates556_9.length then search edge 3 [4051, 7219] innerCandidates556_9 11270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_10

def innerTail556_8 : List Nat := 7027 :: innerTail556_9
def innerCandidates556_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_8_checked : candidates edge 4 4051 7027 innerTail556_9 best.bound = innerCandidates556_8 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_8 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_8 best = best := by
  rw [innerTail556_8, scan, if_neg (by decide), innerCandidates556_8_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_9
    (if 3 ≤ innerCandidates556_8.length then search edge 3 [4051, 7027] innerCandidates556_8 11078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_9

def innerTail556_7 : List Nat := 6841 :: innerTail556_8
def innerCandidates556_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_7_checked : candidates edge 4 4051 6841 innerTail556_8 best.bound = innerCandidates556_7 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_7 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_7 best = best := by
  rw [innerTail556_7, scan, if_neg (by decide), innerCandidates556_7_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_8
    (if 3 ≤ innerCandidates556_7.length then search edge 3 [4051, 6841] innerCandidates556_7 10892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_8

def innerTail556_6 : List Nat := 6211 :: innerTail556_7
def innerCandidates556_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_6_checked : candidates edge 4 4051 6211 innerTail556_7 best.bound = innerCandidates556_6 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_6 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_6 best = best := by
  rw [innerTail556_6, scan, if_neg (by decide), innerCandidates556_6_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_7
    (if 3 ≤ innerCandidates556_6.length then search edge 3 [4051, 6211] innerCandidates556_6 10262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_7

def innerTail556_5 : List Nat := 6121 :: innerTail556_6
def innerCandidates556_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_5_checked : candidates edge 4 4051 6121 innerTail556_6 best.bound = innerCandidates556_5 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_5 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_5 best = best := by
  rw [innerTail556_5, scan, if_neg (by decide), innerCandidates556_5_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_6
    (if 3 ≤ innerCandidates556_5.length then search edge 3 [4051, 6121] innerCandidates556_5 10172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_6

def innerTail556_4 : List Nat := 5701 :: innerTail556_5
def innerCandidates556_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_4_checked : candidates edge 4 4051 5701 innerTail556_5 best.bound = innerCandidates556_4 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_4 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_4 best = best := by
  rw [innerTail556_4, scan, if_neg (by decide), innerCandidates556_4_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_5
    (if 3 ≤ innerCandidates556_4.length then search edge 3 [4051, 5701] innerCandidates556_4 9752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_5

def innerTail556_3 : List Nat := 5521 :: innerTail556_4
def innerCandidates556_3 : List Nat := [5701, 7951]
set_option maxRecDepth 4096 in
theorem innerCandidates556_3_checked : candidates edge 4 4051 5521 innerTail556_4 best.bound = innerCandidates556_3 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_3 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_3 best = best := by
  rw [innerTail556_3, scan, if_neg (by decide), innerCandidates556_3_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_4
    (if 3 ≤ innerCandidates556_3.length then search edge 3 [4051, 5521] innerCandidates556_3 9572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_4

def innerTail556_2 : List Nat := 5077 :: innerTail556_3
def innerCandidates556_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_2_checked : candidates edge 4 4051 5077 innerTail556_3 best.bound = innerCandidates556_2 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_2 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_2 best = best := by
  rw [innerTail556_2, scan, if_neg (by decide), innerCandidates556_2_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_3
    (if 3 ≤ innerCandidates556_2.length then search edge 3 [4051, 5077] innerCandidates556_2 9128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_3

def innerTail556_1 : List Nat := 4657 :: innerTail556_2
def innerCandidates556_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates556_1_checked : candidates edge 4 4051 4657 innerTail556_2 best.bound = innerCandidates556_1 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_1 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_1 best = best := by
  rw [innerTail556_1, scan, if_neg (by decide), innerCandidates556_1_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_2
    (if 3 ≤ innerCandidates556_1.length then search edge 3 [4051, 4657] innerCandidates556_1 8708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_2

def innerTail556_0 : List Nat := 4129 :: innerTail556_1
def innerCandidates556_0 : List Nat := [5521]
set_option maxRecDepth 4096 in
theorem innerCandidates556_0_checked : candidates edge 4 4051 4129 innerTail556_1 best.bound = innerCandidates556_0 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan556_0 : scan edge (search edge 3) 4 [4051] 4051 innerTail556_0 best = best := by
  rw [innerTail556_0, scan, if_neg (by decide), innerCandidates556_0_checked]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_1
    (if 3 ≤ innerCandidates556_0.length then search edge 3 [4051, 4129] innerCandidates556_0 8180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan556_1

theorem rootBranch556 : search edge 4 [4051] rootCandidates556 4051 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4051] 4051 innerTail556_0 best = best
  exact innerScan556_0

def rootCandidates555 : List Nat := [4133, 4877, 4919, 5261, 5927, 7649, 7703, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates555_checked : candidates edge 5 0 4049 tail556 best.bound = rootCandidates555 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext q; rfl
  rw [row]
  decide

def innerTail555_8 : List Nat := []
theorem innerScan555_8 : scan edge (search edge 3) 4 [4049] 4049 innerTail555_8 best = best := rfl

def innerTail555_7 : List Nat := 8111 :: innerTail555_8
def innerCandidates555_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates555_7_checked : candidates edge 4 4049 8111 innerTail555_8 best.bound = innerCandidates555_7 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan555_7 : scan edge (search edge 3) 4 [4049] 4049 innerTail555_7 best = best := by
  rw [innerTail555_7, scan, if_neg (by decide), innerCandidates555_7_checked]
  change scan edge (search edge 3) 4 [4049] 4049 innerTail555_8
    (if 3 ≤ innerCandidates555_7.length then search edge 3 [4049, 8111] innerCandidates555_7 12160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan555_8

def innerTail555_6 : List Nat := 7703 :: innerTail555_7
def innerCandidates555_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates555_6_checked : candidates edge 4 4049 7703 innerTail555_7 best.bound = innerCandidates555_6 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan555_6 : scan edge (search edge 3) 4 [4049] 4049 innerTail555_6 best = best := by
  rw [innerTail555_6, scan, if_neg (by decide), innerCandidates555_6_checked]
  change scan edge (search edge 3) 4 [4049] 4049 innerTail555_7
    (if 3 ≤ innerCandidates555_6.length then search edge 3 [4049, 7703] innerCandidates555_6 11752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan555_7

def innerTail555_5 : List Nat := 7649 :: innerTail555_6
def innerCandidates555_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates555_5_checked : candidates edge 4 4049 7649 innerTail555_6 best.bound = innerCandidates555_5 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan555_5 : scan edge (search edge 3) 4 [4049] 4049 innerTail555_5 best = best := by
  rw [innerTail555_5, scan, if_neg (by decide), innerCandidates555_5_checked]
  change scan edge (search edge 3) 4 [4049] 4049 innerTail555_6
    (if 3 ≤ innerCandidates555_5.length then search edge 3 [4049, 7649] innerCandidates555_5 11698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan555_6

def innerTail555_4 : List Nat := 5927 :: innerTail555_5
def innerCandidates555_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates555_4_checked : candidates edge 4 4049 5927 innerTail555_5 best.bound = innerCandidates555_4 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan555_4 : scan edge (search edge 3) 4 [4049] 4049 innerTail555_4 best = best := by
  rw [innerTail555_4, scan, if_neg (by decide), innerCandidates555_4_checked]
  change scan edge (search edge 3) 4 [4049] 4049 innerTail555_5
    (if 3 ≤ innerCandidates555_4.length then search edge 3 [4049, 5927] innerCandidates555_4 9976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan555_5

def innerTail555_3 : List Nat := 5261 :: innerTail555_4
def innerCandidates555_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates555_3_checked : candidates edge 4 4049 5261 innerTail555_4 best.bound = innerCandidates555_3 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan555_3 : scan edge (search edge 3) 4 [4049] 4049 innerTail555_3 best = best := by
  rw [innerTail555_3, scan, if_neg (by decide), innerCandidates555_3_checked]
  change scan edge (search edge 3) 4 [4049] 4049 innerTail555_4
    (if 3 ≤ innerCandidates555_3.length then search edge 3 [4049, 5261] innerCandidates555_3 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan555_4

def innerTail555_2 : List Nat := 4919 :: innerTail555_3
def innerCandidates555_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates555_2_checked : candidates edge 4 4049 4919 innerTail555_3 best.bound = innerCandidates555_2 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan555_2 : scan edge (search edge 3) 4 [4049] 4049 innerTail555_2 best = best := by
  rw [innerTail555_2, scan, if_neg (by decide), innerCandidates555_2_checked]
  change scan edge (search edge 3) 4 [4049] 4049 innerTail555_3
    (if 3 ≤ innerCandidates555_2.length then search edge 3 [4049, 4919] innerCandidates555_2 8968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan555_3

def innerTail555_1 : List Nat := 4877 :: innerTail555_2
def innerCandidates555_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates555_1_checked : candidates edge 4 4049 4877 innerTail555_2 best.bound = innerCandidates555_1 := by
  unfold candidates
  have row : edge 4877 = SearchData.row4877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan555_1 : scan edge (search edge 3) 4 [4049] 4049 innerTail555_1 best = best := by
  rw [innerTail555_1, scan, if_neg (by decide), innerCandidates555_1_checked]
  change scan edge (search edge 3) 4 [4049] 4049 innerTail555_2
    (if 3 ≤ innerCandidates555_1.length then search edge 3 [4049, 4877] innerCandidates555_1 8926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan555_2

def innerTail555_0 : List Nat := 4133 :: innerTail555_1
def innerCandidates555_0 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates555_0_checked : candidates edge 4 4049 4133 innerTail555_1 best.bound = innerCandidates555_0 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan555_0 : scan edge (search edge 3) 4 [4049] 4049 innerTail555_0 best = best := by
  rw [innerTail555_0, scan, if_neg (by decide), innerCandidates555_0_checked]
  change scan edge (search edge 3) 4 [4049] 4049 innerTail555_1
    (if 3 ≤ innerCandidates555_0.length then search edge 3 [4049, 4133] innerCandidates555_0 8182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan555_1

theorem rootBranch555 : search edge 4 [4049] rootCandidates555 4049 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4049] 4049 innerTail555_0 best = best
  exact innerScan555_0

def rootCandidates554 : List Nat := [4783, 4933, 5233, 5413, 5503, 6199, 6871, 6949, 7417]
set_option maxRecDepth 4096 in
theorem rootCandidates554_checked : candidates edge 5 0 4027 tail555 best.bound = rootCandidates554 := by
  unfold candidates
  have row : edge 4027 = SearchData.row4027.contains := by funext q; rfl
  rw [row]
  decide

def innerTail554_9 : List Nat := []
theorem innerScan554_9 : scan edge (search edge 3) 4 [4027] 4027 innerTail554_9 best = best := rfl

def innerTail554_8 : List Nat := 7417 :: innerTail554_9
def innerCandidates554_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates554_8_checked : candidates edge 4 4027 7417 innerTail554_9 best.bound = innerCandidates554_8 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan554_8 : scan edge (search edge 3) 4 [4027] 4027 innerTail554_8 best = best := by
  rw [innerTail554_8, scan, if_neg (by decide), innerCandidates554_8_checked]
  change scan edge (search edge 3) 4 [4027] 4027 innerTail554_9
    (if 3 ≤ innerCandidates554_8.length then search edge 3 [4027, 7417] innerCandidates554_8 11444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan554_9

def innerTail554_7 : List Nat := 6949 :: innerTail554_8
def innerCandidates554_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates554_7_checked : candidates edge 4 4027 6949 innerTail554_8 best.bound = innerCandidates554_7 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan554_7 : scan edge (search edge 3) 4 [4027] 4027 innerTail554_7 best = best := by
  rw [innerTail554_7, scan, if_neg (by decide), innerCandidates554_7_checked]
  change scan edge (search edge 3) 4 [4027] 4027 innerTail554_8
    (if 3 ≤ innerCandidates554_7.length then search edge 3 [4027, 6949] innerCandidates554_7 10976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan554_8

def innerTail554_6 : List Nat := 6871 :: innerTail554_7
def innerCandidates554_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates554_6_checked : candidates edge 4 4027 6871 innerTail554_7 best.bound = innerCandidates554_6 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan554_6 : scan edge (search edge 3) 4 [4027] 4027 innerTail554_6 best = best := by
  rw [innerTail554_6, scan, if_neg (by decide), innerCandidates554_6_checked]
  change scan edge (search edge 3) 4 [4027] 4027 innerTail554_7
    (if 3 ≤ innerCandidates554_6.length then search edge 3 [4027, 6871] innerCandidates554_6 10898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan554_7

def innerTail554_5 : List Nat := 6199 :: innerTail554_6
def innerCandidates554_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates554_5_checked : candidates edge 4 4027 6199 innerTail554_6 best.bound = innerCandidates554_5 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan554_5 : scan edge (search edge 3) 4 [4027] 4027 innerTail554_5 best = best := by
  rw [innerTail554_5, scan, if_neg (by decide), innerCandidates554_5_checked]
  change scan edge (search edge 3) 4 [4027] 4027 innerTail554_6
    (if 3 ≤ innerCandidates554_5.length then search edge 3 [4027, 6199] innerCandidates554_5 10226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan554_6

def innerTail554_4 : List Nat := 5503 :: innerTail554_5
def innerCandidates554_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates554_4_checked : candidates edge 4 4027 5503 innerTail554_5 best.bound = innerCandidates554_4 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan554_4 : scan edge (search edge 3) 4 [4027] 4027 innerTail554_4 best = best := by
  rw [innerTail554_4, scan, if_neg (by decide), innerCandidates554_4_checked]
  change scan edge (search edge 3) 4 [4027] 4027 innerTail554_5
    (if 3 ≤ innerCandidates554_4.length then search edge 3 [4027, 5503] innerCandidates554_4 9530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan554_5

def innerTail554_3 : List Nat := 5413 :: innerTail554_4
def innerCandidates554_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates554_3_checked : candidates edge 4 4027 5413 innerTail554_4 best.bound = innerCandidates554_3 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan554_3 : scan edge (search edge 3) 4 [4027] 4027 innerTail554_3 best = best := by
  rw [innerTail554_3, scan, if_neg (by decide), innerCandidates554_3_checked]
  change scan edge (search edge 3) 4 [4027] 4027 innerTail554_4
    (if 3 ≤ innerCandidates554_3.length then search edge 3 [4027, 5413] innerCandidates554_3 9440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan554_4

def innerTail554_2 : List Nat := 5233 :: innerTail554_3
def innerCandidates554_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates554_2_checked : candidates edge 4 4027 5233 innerTail554_3 best.bound = innerCandidates554_2 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan554_2 : scan edge (search edge 3) 4 [4027] 4027 innerTail554_2 best = best := by
  rw [innerTail554_2, scan, if_neg (by decide), innerCandidates554_2_checked]
  change scan edge (search edge 3) 4 [4027] 4027 innerTail554_3
    (if 3 ≤ innerCandidates554_2.length then search edge 3 [4027, 5233] innerCandidates554_2 9260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan554_3

def innerTail554_1 : List Nat := 4933 :: innerTail554_2
def innerCandidates554_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates554_1_checked : candidates edge 4 4027 4933 innerTail554_2 best.bound = innerCandidates554_1 := by
  unfold candidates
  have row : edge 4933 = SearchData.row4933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan554_1 : scan edge (search edge 3) 4 [4027] 4027 innerTail554_1 best = best := by
  rw [innerTail554_1, scan, if_neg (by decide), innerCandidates554_1_checked]
  change scan edge (search edge 3) 4 [4027] 4027 innerTail554_2
    (if 3 ≤ innerCandidates554_1.length then search edge 3 [4027, 4933] innerCandidates554_1 8960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan554_2

def innerTail554_0 : List Nat := 4783 :: innerTail554_1
def innerCandidates554_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates554_0_checked : candidates edge 4 4027 4783 innerTail554_1 best.bound = innerCandidates554_0 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan554_0 : scan edge (search edge 3) 4 [4027] 4027 innerTail554_0 best = best := by
  rw [innerTail554_0, scan, if_neg (by decide), innerCandidates554_0_checked]
  change scan edge (search edge 3) 4 [4027] 4027 innerTail554_1
    (if 3 ≤ innerCandidates554_0.length then search edge 3 [4027, 4783] innerCandidates554_0 8810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan554_1

theorem rootBranch554 : search edge 4 [4027] rootCandidates554 4027 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4027] 4027 innerTail554_0 best = best
  exact innerScan554_0

def rootCandidates553 : List Nat := [4441, 4903, 5347, 5563, 6733, 7207, 7537]
set_option maxRecDepth 4096 in
theorem rootCandidates553_checked : candidates edge 5 0 4021 tail554 best.bound = rootCandidates553 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext q; rfl
  rw [row]
  decide

def innerTail553_7 : List Nat := []
theorem innerScan553_7 : scan edge (search edge 3) 4 [4021] 4021 innerTail553_7 best = best := rfl

def innerTail553_6 : List Nat := 7537 :: innerTail553_7
def innerCandidates553_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates553_6_checked : candidates edge 4 4021 7537 innerTail553_7 best.bound = innerCandidates553_6 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan553_6 : scan edge (search edge 3) 4 [4021] 4021 innerTail553_6 best = best := by
  rw [innerTail553_6, scan, if_neg (by decide), innerCandidates553_6_checked]
  change scan edge (search edge 3) 4 [4021] 4021 innerTail553_7
    (if 3 ≤ innerCandidates553_6.length then search edge 3 [4021, 7537] innerCandidates553_6 11558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan553_7

def innerTail553_5 : List Nat := 7207 :: innerTail553_6
def innerCandidates553_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates553_5_checked : candidates edge 4 4021 7207 innerTail553_6 best.bound = innerCandidates553_5 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan553_5 : scan edge (search edge 3) 4 [4021] 4021 innerTail553_5 best = best := by
  rw [innerTail553_5, scan, if_neg (by decide), innerCandidates553_5_checked]
  change scan edge (search edge 3) 4 [4021] 4021 innerTail553_6
    (if 3 ≤ innerCandidates553_5.length then search edge 3 [4021, 7207] innerCandidates553_5 11228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan553_6

def innerTail553_4 : List Nat := 6733 :: innerTail553_5
def innerCandidates553_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates553_4_checked : candidates edge 4 4021 6733 innerTail553_5 best.bound = innerCandidates553_4 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan553_4 : scan edge (search edge 3) 4 [4021] 4021 innerTail553_4 best = best := by
  rw [innerTail553_4, scan, if_neg (by decide), innerCandidates553_4_checked]
  change scan edge (search edge 3) 4 [4021] 4021 innerTail553_5
    (if 3 ≤ innerCandidates553_4.length then search edge 3 [4021, 6733] innerCandidates553_4 10754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan553_5

def innerTail553_3 : List Nat := 5563 :: innerTail553_4
def innerCandidates553_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates553_3_checked : candidates edge 4 4021 5563 innerTail553_4 best.bound = innerCandidates553_3 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan553_3 : scan edge (search edge 3) 4 [4021] 4021 innerTail553_3 best = best := by
  rw [innerTail553_3, scan, if_neg (by decide), innerCandidates553_3_checked]
  change scan edge (search edge 3) 4 [4021] 4021 innerTail553_4
    (if 3 ≤ innerCandidates553_3.length then search edge 3 [4021, 5563] innerCandidates553_3 9584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan553_4

def innerTail553_2 : List Nat := 5347 :: innerTail553_3
def innerCandidates553_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates553_2_checked : candidates edge 4 4021 5347 innerTail553_3 best.bound = innerCandidates553_2 := by
  unfold candidates
  have row : edge 5347 = SearchData.row5347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan553_2 : scan edge (search edge 3) 4 [4021] 4021 innerTail553_2 best = best := by
  rw [innerTail553_2, scan, if_neg (by decide), innerCandidates553_2_checked]
  change scan edge (search edge 3) 4 [4021] 4021 innerTail553_3
    (if 3 ≤ innerCandidates553_2.length then search edge 3 [4021, 5347] innerCandidates553_2 9368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan553_3

def innerTail553_1 : List Nat := 4903 :: innerTail553_2
def innerCandidates553_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates553_1_checked : candidates edge 4 4021 4903 innerTail553_2 best.bound = innerCandidates553_1 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan553_1 : scan edge (search edge 3) 4 [4021] 4021 innerTail553_1 best = best := by
  rw [innerTail553_1, scan, if_neg (by decide), innerCandidates553_1_checked]
  change scan edge (search edge 3) 4 [4021] 4021 innerTail553_2
    (if 3 ≤ innerCandidates553_1.length then search edge 3 [4021, 4903] innerCandidates553_1 8924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan553_2

def innerTail553_0 : List Nat := 4441 :: innerTail553_1
def innerCandidates553_0 : List Nat := [5347]
set_option maxRecDepth 4096 in
theorem innerCandidates553_0_checked : candidates edge 4 4021 4441 innerTail553_1 best.bound = innerCandidates553_0 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan553_0 : scan edge (search edge 3) 4 [4021] 4021 innerTail553_0 best = best := by
  rw [innerTail553_0, scan, if_neg (by decide), innerCandidates553_0_checked]
  change scan edge (search edge 3) 4 [4021] 4021 innerTail553_1
    (if 3 ≤ innerCandidates553_0.length then search edge 3 [4021, 4441] innerCandidates553_0 8462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan553_1

theorem rootBranch553 : search edge 4 [4021] rootCandidates553 4021 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4021] 4021 innerTail553_0 best = best
  exact innerScan553_0

def rootCandidates552 : List Nat := [4211, 5471, 6791, 7121, 7211, 7541]
set_option maxRecDepth 4096 in
theorem rootCandidates552_checked : candidates edge 5 0 4019 tail553 best.bound = rootCandidates552 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext q; rfl
  rw [row]
  decide

def innerTail552_6 : List Nat := []
theorem innerScan552_6 : scan edge (search edge 3) 4 [4019] 4019 innerTail552_6 best = best := rfl

def innerTail552_5 : List Nat := 7541 :: innerTail552_6
def innerCandidates552_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates552_5_checked : candidates edge 4 4019 7541 innerTail552_6 best.bound = innerCandidates552_5 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan552_5 : scan edge (search edge 3) 4 [4019] 4019 innerTail552_5 best = best := by
  rw [innerTail552_5, scan, if_neg (by decide), innerCandidates552_5_checked]
  change scan edge (search edge 3) 4 [4019] 4019 innerTail552_6
    (if 3 ≤ innerCandidates552_5.length then search edge 3 [4019, 7541] innerCandidates552_5 11560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan552_6

def innerTail552_4 : List Nat := 7211 :: innerTail552_5
def innerCandidates552_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates552_4_checked : candidates edge 4 4019 7211 innerTail552_5 best.bound = innerCandidates552_4 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan552_4 : scan edge (search edge 3) 4 [4019] 4019 innerTail552_4 best = best := by
  rw [innerTail552_4, scan, if_neg (by decide), innerCandidates552_4_checked]
  change scan edge (search edge 3) 4 [4019] 4019 innerTail552_5
    (if 3 ≤ innerCandidates552_4.length then search edge 3 [4019, 7211] innerCandidates552_4 11230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan552_5

def innerTail552_3 : List Nat := 7121 :: innerTail552_4
def innerCandidates552_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates552_3_checked : candidates edge 4 4019 7121 innerTail552_4 best.bound = innerCandidates552_3 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan552_3 : scan edge (search edge 3) 4 [4019] 4019 innerTail552_3 best = best := by
  rw [innerTail552_3, scan, if_neg (by decide), innerCandidates552_3_checked]
  change scan edge (search edge 3) 4 [4019] 4019 innerTail552_4
    (if 3 ≤ innerCandidates552_3.length then search edge 3 [4019, 7121] innerCandidates552_3 11140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan552_4

def innerTail552_2 : List Nat := 6791 :: innerTail552_3
def innerCandidates552_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates552_2_checked : candidates edge 4 4019 6791 innerTail552_3 best.bound = innerCandidates552_2 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan552_2 : scan edge (search edge 3) 4 [4019] 4019 innerTail552_2 best = best := by
  rw [innerTail552_2, scan, if_neg (by decide), innerCandidates552_2_checked]
  change scan edge (search edge 3) 4 [4019] 4019 innerTail552_3
    (if 3 ≤ innerCandidates552_2.length then search edge 3 [4019, 6791] innerCandidates552_2 10810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan552_3

def innerTail552_1 : List Nat := 5471 :: innerTail552_2
def innerCandidates552_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates552_1_checked : candidates edge 4 4019 5471 innerTail552_2 best.bound = innerCandidates552_1 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan552_1 : scan edge (search edge 3) 4 [4019] 4019 innerTail552_1 best = best := by
  rw [innerTail552_1, scan, if_neg (by decide), innerCandidates552_1_checked]
  change scan edge (search edge 3) 4 [4019] 4019 innerTail552_2
    (if 3 ≤ innerCandidates552_1.length then search edge 3 [4019, 5471] innerCandidates552_1 9490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan552_2

def innerTail552_0 : List Nat := 4211 :: innerTail552_1
def innerCandidates552_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates552_0_checked : candidates edge 4 4019 4211 innerTail552_1 best.bound = innerCandidates552_0 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan552_0 : scan edge (search edge 3) 4 [4019] 4019 innerTail552_0 best = best := by
  rw [innerTail552_0, scan, if_neg (by decide), innerCandidates552_0_checked]
  change scan edge (search edge 3) 4 [4019] 4019 innerTail552_1
    (if 3 ≤ innerCandidates552_0.length then search edge 3 [4019, 4211] innerCandidates552_0 8230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan552_1

theorem rootBranch552 : search edge 4 [4019] rootCandidates552 4019 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4019] 4019 innerTail552_0 best = best
  exact innerScan552_0

def rootCandidates551 : List Nat := [4421, 4643, 4799, 5153, 5573, 5741, 5867, 6659, 7541, 8069, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates551_checked : candidates edge 5 0 4013 tail552 best.bound = rootCandidates551 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext q; rfl
  rw [row]
  decide

def innerTail551_11 : List Nat := []
theorem innerScan551_11 : scan edge (search edge 3) 4 [4013] 4013 innerTail551_11 best = best := rfl

def innerTail551_10 : List Nat := 8387 :: innerTail551_11
def innerCandidates551_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates551_10_checked : candidates edge 4 4013 8387 innerTail551_11 best.bound = innerCandidates551_10 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan551_10 : scan edge (search edge 3) 4 [4013] 4013 innerTail551_10 best = best := by
  rw [innerTail551_10, scan, if_neg (by decide), innerCandidates551_10_checked]
  change scan edge (search edge 3) 4 [4013] 4013 innerTail551_11
    (if 3 ≤ innerCandidates551_10.length then search edge 3 [4013, 8387] innerCandidates551_10 12400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan551_11

def innerTail551_9 : List Nat := 8069 :: innerTail551_10
def innerCandidates551_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates551_9_checked : candidates edge 4 4013 8069 innerTail551_10 best.bound = innerCandidates551_9 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan551_9 : scan edge (search edge 3) 4 [4013] 4013 innerTail551_9 best = best := by
  rw [innerTail551_9, scan, if_neg (by decide), innerCandidates551_9_checked]
  change scan edge (search edge 3) 4 [4013] 4013 innerTail551_10
    (if 3 ≤ innerCandidates551_9.length then search edge 3 [4013, 8069] innerCandidates551_9 12082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan551_10

def innerTail551_8 : List Nat := 7541 :: innerTail551_9
def innerCandidates551_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates551_8_checked : candidates edge 4 4013 7541 innerTail551_9 best.bound = innerCandidates551_8 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan551_8 : scan edge (search edge 3) 4 [4013] 4013 innerTail551_8 best = best := by
  rw [innerTail551_8, scan, if_neg (by decide), innerCandidates551_8_checked]
  change scan edge (search edge 3) 4 [4013] 4013 innerTail551_9
    (if 3 ≤ innerCandidates551_8.length then search edge 3 [4013, 7541] innerCandidates551_8 11554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan551_9

def innerTail551_7 : List Nat := 6659 :: innerTail551_8
def innerCandidates551_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates551_7_checked : candidates edge 4 4013 6659 innerTail551_8 best.bound = innerCandidates551_7 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan551_7 : scan edge (search edge 3) 4 [4013] 4013 innerTail551_7 best = best := by
  rw [innerTail551_7, scan, if_neg (by decide), innerCandidates551_7_checked]
  change scan edge (search edge 3) 4 [4013] 4013 innerTail551_8
    (if 3 ≤ innerCandidates551_7.length then search edge 3 [4013, 6659] innerCandidates551_7 10672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan551_8

def innerTail551_6 : List Nat := 5867 :: innerTail551_7
def innerCandidates551_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates551_6_checked : candidates edge 4 4013 5867 innerTail551_7 best.bound = innerCandidates551_6 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan551_6 : scan edge (search edge 3) 4 [4013] 4013 innerTail551_6 best = best := by
  rw [innerTail551_6, scan, if_neg (by decide), innerCandidates551_6_checked]
  change scan edge (search edge 3) 4 [4013] 4013 innerTail551_7
    (if 3 ≤ innerCandidates551_6.length then search edge 3 [4013, 5867] innerCandidates551_6 9880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan551_7

def innerTail551_5 : List Nat := 5741 :: innerTail551_6
def innerCandidates551_5 : List Nat := [7541]
set_option maxRecDepth 4096 in
theorem innerCandidates551_5_checked : candidates edge 4 4013 5741 innerTail551_6 best.bound = innerCandidates551_5 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan551_5 : scan edge (search edge 3) 4 [4013] 4013 innerTail551_5 best = best := by
  rw [innerTail551_5, scan, if_neg (by decide), innerCandidates551_5_checked]
  change scan edge (search edge 3) 4 [4013] 4013 innerTail551_6
    (if 3 ≤ innerCandidates551_5.length then search edge 3 [4013, 5741] innerCandidates551_5 9754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan551_6

def innerTail551_4 : List Nat := 5573 :: innerTail551_5
def innerCandidates551_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates551_4_checked : candidates edge 4 4013 5573 innerTail551_5 best.bound = innerCandidates551_4 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan551_4 : scan edge (search edge 3) 4 [4013] 4013 innerTail551_4 best = best := by
  rw [innerTail551_4, scan, if_neg (by decide), innerCandidates551_4_checked]
  change scan edge (search edge 3) 4 [4013] 4013 innerTail551_5
    (if 3 ≤ innerCandidates551_4.length then search edge 3 [4013, 5573] innerCandidates551_4 9586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan551_5

def innerTail551_3 : List Nat := 5153 :: innerTail551_4
def innerCandidates551_3 : List Nat := [5867]
set_option maxRecDepth 4096 in
theorem innerCandidates551_3_checked : candidates edge 4 4013 5153 innerTail551_4 best.bound = innerCandidates551_3 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan551_3 : scan edge (search edge 3) 4 [4013] 4013 innerTail551_3 best = best := by
  rw [innerTail551_3, scan, if_neg (by decide), innerCandidates551_3_checked]
  change scan edge (search edge 3) 4 [4013] 4013 innerTail551_4
    (if 3 ≤ innerCandidates551_3.length then search edge 3 [4013, 5153] innerCandidates551_3 9166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan551_4

def innerTail551_2 : List Nat := 4799 :: innerTail551_3
def innerCandidates551_2 : List Nat := [5867]
set_option maxRecDepth 4096 in
theorem innerCandidates551_2_checked : candidates edge 4 4013 4799 innerTail551_3 best.bound = innerCandidates551_2 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan551_2 : scan edge (search edge 3) 4 [4013] 4013 innerTail551_2 best = best := by
  rw [innerTail551_2, scan, if_neg (by decide), innerCandidates551_2_checked]
  change scan edge (search edge 3) 4 [4013] 4013 innerTail551_3
    (if 3 ≤ innerCandidates551_2.length then search edge 3 [4013, 4799] innerCandidates551_2 8812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan551_3

def innerTail551_1 : List Nat := 4643 :: innerTail551_2
def innerCandidates551_1 : List Nat := [5153]
set_option maxRecDepth 4096 in
theorem innerCandidates551_1_checked : candidates edge 4 4013 4643 innerTail551_2 best.bound = innerCandidates551_1 := by
  unfold candidates
  have row : edge 4643 = SearchData.row4643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan551_1 : scan edge (search edge 3) 4 [4013] 4013 innerTail551_1 best = best := by
  rw [innerTail551_1, scan, if_neg (by decide), innerCandidates551_1_checked]
  change scan edge (search edge 3) 4 [4013] 4013 innerTail551_2
    (if 3 ≤ innerCandidates551_1.length then search edge 3 [4013, 4643] innerCandidates551_1 8656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan551_2

def innerTail551_0 : List Nat := 4421 :: innerTail551_1
def innerCandidates551_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates551_0_checked : candidates edge 4 4013 4421 innerTail551_1 best.bound = innerCandidates551_0 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan551_0 : scan edge (search edge 3) 4 [4013] 4013 innerTail551_0 best = best := by
  rw [innerTail551_0, scan, if_neg (by decide), innerCandidates551_0_checked]
  change scan edge (search edge 3) 4 [4013] 4013 innerTail551_1
    (if 3 ≤ innerCandidates551_0.length then search edge 3 [4013, 4421] innerCandidates551_0 8434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan551_1

theorem rootBranch551 : search edge 4 [4013] rootCandidates551 4013 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4013] 4013 innerTail551_0 best = best
  exact innerScan551_0

end ProjectEulerSolutions.PrimePairUpper.Search
