import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates720 : List Nat := [5711, 5927, 6053, 6089, 6311, 6599, 7109, 8039, 8081, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates720_checked : candidates edge 5 0 5477 tail721 best.bound = rootCandidates720 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext q; rfl
  rw [row]
  decide

def innerTail720_10 : List Nat := []
theorem innerScan720_10 : scan edge (search edge 3) 4 [5477] 5477 innerTail720_10 best = best := rfl

def innerTail720_9 : List Nat := 8123 :: innerTail720_10
def innerCandidates720_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates720_9_checked : candidates edge 4 5477 8123 innerTail720_10 best.bound = innerCandidates720_9 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan720_9 : scan edge (search edge 3) 4 [5477] 5477 innerTail720_9 best = best := by
  rw [innerTail720_9, scan, if_neg (by decide), innerCandidates720_9_checked]
  change scan edge (search edge 3) 4 [5477] 5477 innerTail720_10
    (if 3 ≤ innerCandidates720_9.length then search edge 3 [5477, 8123] innerCandidates720_9 13600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan720_10

def innerTail720_8 : List Nat := 8081 :: innerTail720_9
def innerCandidates720_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates720_8_checked : candidates edge 4 5477 8081 innerTail720_9 best.bound = innerCandidates720_8 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan720_8 : scan edge (search edge 3) 4 [5477] 5477 innerTail720_8 best = best := by
  rw [innerTail720_8, scan, if_neg (by decide), innerCandidates720_8_checked]
  change scan edge (search edge 3) 4 [5477] 5477 innerTail720_9
    (if 3 ≤ innerCandidates720_8.length then search edge 3 [5477, 8081] innerCandidates720_8 13558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan720_9

def innerTail720_7 : List Nat := 8039 :: innerTail720_8
def innerCandidates720_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates720_7_checked : candidates edge 4 5477 8039 innerTail720_8 best.bound = innerCandidates720_7 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan720_7 : scan edge (search edge 3) 4 [5477] 5477 innerTail720_7 best = best := by
  rw [innerTail720_7, scan, if_neg (by decide), innerCandidates720_7_checked]
  change scan edge (search edge 3) 4 [5477] 5477 innerTail720_8
    (if 3 ≤ innerCandidates720_7.length then search edge 3 [5477, 8039] innerCandidates720_7 13516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan720_8

def innerTail720_6 : List Nat := 7109 :: innerTail720_7
def innerCandidates720_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates720_6_checked : candidates edge 4 5477 7109 innerTail720_7 best.bound = innerCandidates720_6 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan720_6 : scan edge (search edge 3) 4 [5477] 5477 innerTail720_6 best = best := by
  rw [innerTail720_6, scan, if_neg (by decide), innerCandidates720_6_checked]
  change scan edge (search edge 3) 4 [5477] 5477 innerTail720_7
    (if 3 ≤ innerCandidates720_6.length then search edge 3 [5477, 7109] innerCandidates720_6 12586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan720_7

def innerTail720_5 : List Nat := 6599 :: innerTail720_6
def innerCandidates720_5 : List Nat := [8081]
set_option maxRecDepth 4096 in
theorem innerCandidates720_5_checked : candidates edge 4 5477 6599 innerTail720_6 best.bound = innerCandidates720_5 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan720_5 : scan edge (search edge 3) 4 [5477] 5477 innerTail720_5 best = best := by
  rw [innerTail720_5, scan, if_neg (by decide), innerCandidates720_5_checked]
  change scan edge (search edge 3) 4 [5477] 5477 innerTail720_6
    (if 3 ≤ innerCandidates720_5.length then search edge 3 [5477, 6599] innerCandidates720_5 12076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan720_6

def innerTail720_4 : List Nat := 6311 :: innerTail720_5
def innerCandidates720_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates720_4_checked : candidates edge 4 5477 6311 innerTail720_5 best.bound = innerCandidates720_4 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan720_4 : scan edge (search edge 3) 4 [5477] 5477 innerTail720_4 best = best := by
  rw [innerTail720_4, scan, if_neg (by decide), innerCandidates720_4_checked]
  change scan edge (search edge 3) 4 [5477] 5477 innerTail720_5
    (if 3 ≤ innerCandidates720_4.length then search edge 3 [5477, 6311] innerCandidates720_4 11788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan720_5

def innerTail720_3 : List Nat := 6089 :: innerTail720_4
def innerCandidates720_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates720_3_checked : candidates edge 4 5477 6089 innerTail720_4 best.bound = innerCandidates720_3 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan720_3 : scan edge (search edge 3) 4 [5477] 5477 innerTail720_3 best = best := by
  rw [innerTail720_3, scan, if_neg (by decide), innerCandidates720_3_checked]
  change scan edge (search edge 3) 4 [5477] 5477 innerTail720_4
    (if 3 ≤ innerCandidates720_3.length then search edge 3 [5477, 6089] innerCandidates720_3 11566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan720_4

def innerTail720_2 : List Nat := 6053 :: innerTail720_3
def innerCandidates720_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates720_2_checked : candidates edge 4 5477 6053 innerTail720_3 best.bound = innerCandidates720_2 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan720_2 : scan edge (search edge 3) 4 [5477] 5477 innerTail720_2 best = best := by
  rw [innerTail720_2, scan, if_neg (by decide), innerCandidates720_2_checked]
  change scan edge (search edge 3) 4 [5477] 5477 innerTail720_3
    (if 3 ≤ innerCandidates720_2.length then search edge 3 [5477, 6053] innerCandidates720_2 11530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan720_3

def innerTail720_1 : List Nat := 5927 :: innerTail720_2
def innerCandidates720_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates720_1_checked : candidates edge 4 5477 5927 innerTail720_2 best.bound = innerCandidates720_1 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan720_1 : scan edge (search edge 3) 4 [5477] 5477 innerTail720_1 best = best := by
  rw [innerTail720_1, scan, if_neg (by decide), innerCandidates720_1_checked]
  change scan edge (search edge 3) 4 [5477] 5477 innerTail720_2
    (if 3 ≤ innerCandidates720_1.length then search edge 3 [5477, 5927] innerCandidates720_1 11404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan720_2

def innerTail720_0 : List Nat := 5711 :: innerTail720_1
def innerCandidates720_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates720_0_checked : candidates edge 4 5477 5711 innerTail720_1 best.bound = innerCandidates720_0 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan720_0 : scan edge (search edge 3) 4 [5477] 5477 innerTail720_0 best = best := by
  rw [innerTail720_0, scan, if_neg (by decide), innerCandidates720_0_checked]
  change scan edge (search edge 3) 4 [5477] 5477 innerTail720_1
    (if 3 ≤ innerCandidates720_0.length then search edge 3 [5477, 5711] innerCandidates720_0 11188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan720_1

theorem rootBranch720 : search edge 4 [5477] rootCandidates720 5477 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5477] 5477 innerTail720_0 best = best
  exact innerScan720_0

def rootCandidates719 : List Nat := [5783, 5807, 6197, 6203, 6449, 6491, 6653, 7583, 7841, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates719_checked : candidates edge 5 0 5471 tail720 best.bound = rootCandidates719 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext q; rfl
  rw [row]
  decide

def innerTail719_10 : List Nat := []
theorem innerScan719_10 : scan edge (search edge 3) 4 [5471] 5471 innerTail719_10 best = best := rfl

def innerTail719_9 : List Nat := 8171 :: innerTail719_10
def innerCandidates719_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates719_9_checked : candidates edge 4 5471 8171 innerTail719_10 best.bound = innerCandidates719_9 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan719_9 : scan edge (search edge 3) 4 [5471] 5471 innerTail719_9 best = best := by
  rw [innerTail719_9, scan, if_neg (by decide), innerCandidates719_9_checked]
  change scan edge (search edge 3) 4 [5471] 5471 innerTail719_10
    (if 3 ≤ innerCandidates719_9.length then search edge 3 [5471, 8171] innerCandidates719_9 13642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan719_10

def innerTail719_8 : List Nat := 7841 :: innerTail719_9
def innerCandidates719_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates719_8_checked : candidates edge 4 5471 7841 innerTail719_9 best.bound = innerCandidates719_8 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan719_8 : scan edge (search edge 3) 4 [5471] 5471 innerTail719_8 best = best := by
  rw [innerTail719_8, scan, if_neg (by decide), innerCandidates719_8_checked]
  change scan edge (search edge 3) 4 [5471] 5471 innerTail719_9
    (if 3 ≤ innerCandidates719_8.length then search edge 3 [5471, 7841] innerCandidates719_8 13312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan719_9

def innerTail719_7 : List Nat := 7583 :: innerTail719_8
def innerCandidates719_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates719_7_checked : candidates edge 4 5471 7583 innerTail719_8 best.bound = innerCandidates719_7 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan719_7 : scan edge (search edge 3) 4 [5471] 5471 innerTail719_7 best = best := by
  rw [innerTail719_7, scan, if_neg (by decide), innerCandidates719_7_checked]
  change scan edge (search edge 3) 4 [5471] 5471 innerTail719_8
    (if 3 ≤ innerCandidates719_7.length then search edge 3 [5471, 7583] innerCandidates719_7 13054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan719_8

def innerTail719_6 : List Nat := 6653 :: innerTail719_7
def innerCandidates719_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates719_6_checked : candidates edge 4 5471 6653 innerTail719_7 best.bound = innerCandidates719_6 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan719_6 : scan edge (search edge 3) 4 [5471] 5471 innerTail719_6 best = best := by
  rw [innerTail719_6, scan, if_neg (by decide), innerCandidates719_6_checked]
  change scan edge (search edge 3) 4 [5471] 5471 innerTail719_7
    (if 3 ≤ innerCandidates719_6.length then search edge 3 [5471, 6653] innerCandidates719_6 12124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan719_7

def innerTail719_5 : List Nat := 6491 :: innerTail719_6
def innerCandidates719_5 : List Nat := [7583]
set_option maxRecDepth 4096 in
theorem innerCandidates719_5_checked : candidates edge 4 5471 6491 innerTail719_6 best.bound = innerCandidates719_5 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan719_5 : scan edge (search edge 3) 4 [5471] 5471 innerTail719_5 best = best := by
  rw [innerTail719_5, scan, if_neg (by decide), innerCandidates719_5_checked]
  change scan edge (search edge 3) 4 [5471] 5471 innerTail719_6
    (if 3 ≤ innerCandidates719_5.length then search edge 3 [5471, 6491] innerCandidates719_5 11962 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan719_6

def innerTail719_4 : List Nat := 6449 :: innerTail719_5
def innerCandidates719_4 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates719_4_checked : candidates edge 4 5471 6449 innerTail719_5 best.bound = innerCandidates719_4 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan719_4 : scan edge (search edge 3) 4 [5471] 5471 innerTail719_4 best = best := by
  rw [innerTail719_4, scan, if_neg (by decide), innerCandidates719_4_checked]
  change scan edge (search edge 3) 4 [5471] 5471 innerTail719_5
    (if 3 ≤ innerCandidates719_4.length then search edge 3 [5471, 6449] innerCandidates719_4 11920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan719_5

def innerTail719_3 : List Nat := 6203 :: innerTail719_4
def innerCandidates719_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates719_3_checked : candidates edge 4 5471 6203 innerTail719_4 best.bound = innerCandidates719_3 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan719_3 : scan edge (search edge 3) 4 [5471] 5471 innerTail719_3 best = best := by
  rw [innerTail719_3, scan, if_neg (by decide), innerCandidates719_3_checked]
  change scan edge (search edge 3) 4 [5471] 5471 innerTail719_4
    (if 3 ≤ innerCandidates719_3.length then search edge 3 [5471, 6203] innerCandidates719_3 11674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan719_4

def innerTail719_2 : List Nat := 6197 :: innerTail719_3
def innerCandidates719_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates719_2_checked : candidates edge 4 5471 6197 innerTail719_3 best.bound = innerCandidates719_2 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan719_2 : scan edge (search edge 3) 4 [5471] 5471 innerTail719_2 best = best := by
  rw [innerTail719_2, scan, if_neg (by decide), innerCandidates719_2_checked]
  change scan edge (search edge 3) 4 [5471] 5471 innerTail719_3
    (if 3 ≤ innerCandidates719_2.length then search edge 3 [5471, 6197] innerCandidates719_2 11668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan719_3

def innerTail719_1 : List Nat := 5807 :: innerTail719_2
def innerCandidates719_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates719_1_checked : candidates edge 4 5471 5807 innerTail719_2 best.bound = innerCandidates719_1 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan719_1 : scan edge (search edge 3) 4 [5471] 5471 innerTail719_1 best = best := by
  rw [innerTail719_1, scan, if_neg (by decide), innerCandidates719_1_checked]
  change scan edge (search edge 3) 4 [5471] 5471 innerTail719_2
    (if 3 ≤ innerCandidates719_1.length then search edge 3 [5471, 5807] innerCandidates719_1 11278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan719_2

def innerTail719_0 : List Nat := 5783 :: innerTail719_1
def innerCandidates719_0 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates719_0_checked : candidates edge 4 5471 5783 innerTail719_1 best.bound = innerCandidates719_0 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan719_0 : scan edge (search edge 3) 4 [5471] 5471 innerTail719_0 best = best := by
  rw [innerTail719_0, scan, if_neg (by decide), innerCandidates719_0_checked]
  change scan edge (search edge 3) 4 [5471] 5471 innerTail719_1
    (if 3 ≤ innerCandidates719_0.length then search edge 3 [5471, 5783] innerCandidates719_0 11254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan719_1

theorem rootBranch719 : search edge 4 [5471] rootCandidates719 5471 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5471] 5471 innerTail719_0 best = best
  exact innerScan719_0

def rootCandidates718 : List Nat := [5851, 6073, 7243, 7321, 7507, 7549, 8089]
set_option maxRecDepth 4096 in
theorem rootCandidates718_checked : candidates edge 5 0 5449 tail719 best.bound = rootCandidates718 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext q; rfl
  rw [row]
  decide

def innerTail718_7 : List Nat := []
theorem innerScan718_7 : scan edge (search edge 3) 4 [5449] 5449 innerTail718_7 best = best := rfl

def innerTail718_6 : List Nat := 8089 :: innerTail718_7
def innerCandidates718_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates718_6_checked : candidates edge 4 5449 8089 innerTail718_7 best.bound = innerCandidates718_6 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan718_6 : scan edge (search edge 3) 4 [5449] 5449 innerTail718_6 best = best := by
  rw [innerTail718_6, scan, if_neg (by decide), innerCandidates718_6_checked]
  change scan edge (search edge 3) 4 [5449] 5449 innerTail718_7
    (if 3 ≤ innerCandidates718_6.length then search edge 3 [5449, 8089] innerCandidates718_6 13538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan718_7

def innerTail718_5 : List Nat := 7549 :: innerTail718_6
def innerCandidates718_5 : List Nat := [8089]
set_option maxRecDepth 4096 in
theorem innerCandidates718_5_checked : candidates edge 4 5449 7549 innerTail718_6 best.bound = innerCandidates718_5 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan718_5 : scan edge (search edge 3) 4 [5449] 5449 innerTail718_5 best = best := by
  rw [innerTail718_5, scan, if_neg (by decide), innerCandidates718_5_checked]
  change scan edge (search edge 3) 4 [5449] 5449 innerTail718_6
    (if 3 ≤ innerCandidates718_5.length then search edge 3 [5449, 7549] innerCandidates718_5 12998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan718_6

def innerTail718_4 : List Nat := 7507 :: innerTail718_5
def innerCandidates718_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates718_4_checked : candidates edge 4 5449 7507 innerTail718_5 best.bound = innerCandidates718_4 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan718_4 : scan edge (search edge 3) 4 [5449] 5449 innerTail718_4 best = best := by
  rw [innerTail718_4, scan, if_neg (by decide), innerCandidates718_4_checked]
  change scan edge (search edge 3) 4 [5449] 5449 innerTail718_5
    (if 3 ≤ innerCandidates718_4.length then search edge 3 [5449, 7507] innerCandidates718_4 12956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan718_5

def innerTail718_3 : List Nat := 7321 :: innerTail718_4
def innerCandidates718_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates718_3_checked : candidates edge 4 5449 7321 innerTail718_4 best.bound = innerCandidates718_3 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan718_3 : scan edge (search edge 3) 4 [5449] 5449 innerTail718_3 best = best := by
  rw [innerTail718_3, scan, if_neg (by decide), innerCandidates718_3_checked]
  change scan edge (search edge 3) 4 [5449] 5449 innerTail718_4
    (if 3 ≤ innerCandidates718_3.length then search edge 3 [5449, 7321] innerCandidates718_3 12770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan718_4

def innerTail718_2 : List Nat := 7243 :: innerTail718_3
def innerCandidates718_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates718_2_checked : candidates edge 4 5449 7243 innerTail718_3 best.bound = innerCandidates718_2 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan718_2 : scan edge (search edge 3) 4 [5449] 5449 innerTail718_2 best = best := by
  rw [innerTail718_2, scan, if_neg (by decide), innerCandidates718_2_checked]
  change scan edge (search edge 3) 4 [5449] 5449 innerTail718_3
    (if 3 ≤ innerCandidates718_2.length then search edge 3 [5449, 7243] innerCandidates718_2 12692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan718_3

def innerTail718_1 : List Nat := 6073 :: innerTail718_2
def innerCandidates718_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates718_1_checked : candidates edge 4 5449 6073 innerTail718_2 best.bound = innerCandidates718_1 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan718_1 : scan edge (search edge 3) 4 [5449] 5449 innerTail718_1 best = best := by
  rw [innerTail718_1, scan, if_neg (by decide), innerCandidates718_1_checked]
  change scan edge (search edge 3) 4 [5449] 5449 innerTail718_2
    (if 3 ≤ innerCandidates718_1.length then search edge 3 [5449, 6073] innerCandidates718_1 11522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan718_2

def innerTail718_0 : List Nat := 5851 :: innerTail718_1
def innerCandidates718_0 : List Nat := [7507]
set_option maxRecDepth 4096 in
theorem innerCandidates718_0_checked : candidates edge 4 5449 5851 innerTail718_1 best.bound = innerCandidates718_0 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan718_0 : scan edge (search edge 3) 4 [5449] 5449 innerTail718_0 best = best := by
  rw [innerTail718_0, scan, if_neg (by decide), innerCandidates718_0_checked]
  change scan edge (search edge 3) 4 [5449] 5449 innerTail718_1
    (if 3 ≤ innerCandidates718_0.length then search edge 3 [5449, 5851] innerCandidates718_0 11300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan718_1

theorem rootBranch718 : search edge 4 [5449] rootCandidates718 5449 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5449] 5449 innerTail718_0 best = best
  exact innerScan718_0

def rootCandidates717 : List Nat := [5653, 5791, 5923, 6079, 6379, 6553, 6637, 6841, 7039, 7351, 7393, 7549, 7561, 8101]
set_option maxRecDepth 4096 in
theorem rootCandidates717_checked : candidates edge 5 0 5443 tail718 best.bound = rootCandidates717 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext q; rfl
  rw [row]
  decide

def innerTail717_14 : List Nat := []
theorem innerScan717_14 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_14 best = best := rfl

def innerTail717_13 : List Nat := 8101 :: innerTail717_14
def innerCandidates717_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates717_13_checked : candidates edge 4 5443 8101 innerTail717_14 best.bound = innerCandidates717_13 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan717_13 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_13 best = best := by
  rw [innerTail717_13, scan, if_neg (by decide), innerCandidates717_13_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_14
    (if 3 ≤ innerCandidates717_13.length then search edge 3 [5443, 8101] innerCandidates717_13 13544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan717_14

def innerTail717_12 : List Nat := 7561 :: innerTail717_13
def innerCandidates717_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates717_12_checked : candidates edge 4 5443 7561 innerTail717_13 best.bound = innerCandidates717_12 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan717_12 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_12 best = best := by
  rw [innerTail717_12, scan, if_neg (by decide), innerCandidates717_12_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_13
    (if 3 ≤ innerCandidates717_12.length then search edge 3 [5443, 7561] innerCandidates717_12 13004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan717_13

def innerTail717_11 : List Nat := 7549 :: innerTail717_12
def innerCandidates717_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates717_11_checked : candidates edge 4 5443 7549 innerTail717_12 best.bound = innerCandidates717_11 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan717_11 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_11 best = best := by
  rw [innerTail717_11, scan, if_neg (by decide), innerCandidates717_11_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_12
    (if 3 ≤ innerCandidates717_11.length then search edge 3 [5443, 7549] innerCandidates717_11 12992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan717_12

def innerTail717_10 : List Nat := 7393 :: innerTail717_11
def innerCandidates717_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates717_10_checked : candidates edge 4 5443 7393 innerTail717_11 best.bound = innerCandidates717_10 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan717_10 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_10 best = best := by
  rw [innerTail717_10, scan, if_neg (by decide), innerCandidates717_10_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_11
    (if 3 ≤ innerCandidates717_10.length then search edge 3 [5443, 7393] innerCandidates717_10 12836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan717_11

def innerTail717_9 : List Nat := 7351 :: innerTail717_10
def innerCandidates717_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates717_9_checked : candidates edge 4 5443 7351 innerTail717_10 best.bound = innerCandidates717_9 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan717_9 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_9 best = best := by
  rw [innerTail717_9, scan, if_neg (by decide), innerCandidates717_9_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_10
    (if 3 ≤ innerCandidates717_9.length then search edge 3 [5443, 7351] innerCandidates717_9 12794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan717_10

def innerTail717_8 : List Nat := 7039 :: innerTail717_9
def innerCandidates717_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates717_8_checked : candidates edge 4 5443 7039 innerTail717_9 best.bound = innerCandidates717_8 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan717_8 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_8 best = best := by
  rw [innerTail717_8, scan, if_neg (by decide), innerCandidates717_8_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_9
    (if 3 ≤ innerCandidates717_8.length then search edge 3 [5443, 7039] innerCandidates717_8 12482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan717_9

def innerTail717_7 : List Nat := 6841 :: innerTail717_8
def innerCandidates717_7 : List Nat := [7039]
set_option maxRecDepth 4096 in
theorem innerCandidates717_7_checked : candidates edge 4 5443 6841 innerTail717_8 best.bound = innerCandidates717_7 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan717_7 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_7 best = best := by
  rw [innerTail717_7, scan, if_neg (by decide), innerCandidates717_7_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_8
    (if 3 ≤ innerCandidates717_7.length then search edge 3 [5443, 6841] innerCandidates717_7 12284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan717_8

def innerTail717_6 : List Nat := 6637 :: innerTail717_7
def innerCandidates717_6 : List Nat := [6841, 7039, 7351]
set_option maxRecDepth 4096 in
theorem innerCandidates717_6_checked : candidates edge 4 5443 6637 innerTail717_7 best.bound = innerCandidates717_6 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild717_6 : search edge 3 [5443, 6637] innerCandidates717_6 12080 best = best := by decide

theorem innerScan717_6 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_6 best = best := by
  rw [innerTail717_6, scan, if_neg (by decide), innerCandidates717_6_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_7
    (if 3 ≤ innerCandidates717_6.length then search edge 3 [5443, 6637] innerCandidates717_6 12080 best else best) = best
  rw [if_pos (by decide), innerChild717_6]
  exact innerScan717_7

def innerTail717_5 : List Nat := 6553 :: innerTail717_6
def innerCandidates717_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates717_5_checked : candidates edge 4 5443 6553 innerTail717_6 best.bound = innerCandidates717_5 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan717_5 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_5 best = best := by
  rw [innerTail717_5, scan, if_neg (by decide), innerCandidates717_5_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_6
    (if 3 ≤ innerCandidates717_5.length then search edge 3 [5443, 6553] innerCandidates717_5 11996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan717_6

def innerTail717_4 : List Nat := 6379 :: innerTail717_5
def innerCandidates717_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates717_4_checked : candidates edge 4 5443 6379 innerTail717_5 best.bound = innerCandidates717_4 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan717_4 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_4 best = best := by
  rw [innerTail717_4, scan, if_neg (by decide), innerCandidates717_4_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_5
    (if 3 ≤ innerCandidates717_4.length then search edge 3 [5443, 6379] innerCandidates717_4 11822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan717_5

def innerTail717_3 : List Nat := 6079 :: innerTail717_4
def innerCandidates717_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates717_3_checked : candidates edge 4 5443 6079 innerTail717_4 best.bound = innerCandidates717_3 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan717_3 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_3 best = best := by
  rw [innerTail717_3, scan, if_neg (by decide), innerCandidates717_3_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_4
    (if 3 ≤ innerCandidates717_3.length then search edge 3 [5443, 6079] innerCandidates717_3 11522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan717_4

def innerTail717_2 : List Nat := 5923 :: innerTail717_3
def innerCandidates717_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates717_2_checked : candidates edge 4 5443 5923 innerTail717_3 best.bound = innerCandidates717_2 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan717_2 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_2 best = best := by
  rw [innerTail717_2, scan, if_neg (by decide), innerCandidates717_2_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_3
    (if 3 ≤ innerCandidates717_2.length then search edge 3 [5443, 5923] innerCandidates717_2 11366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan717_3

def innerTail717_1 : List Nat := 5791 :: innerTail717_2
def innerCandidates717_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates717_1_checked : candidates edge 4 5443 5791 innerTail717_2 best.bound = innerCandidates717_1 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan717_1 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_1 best = best := by
  rw [innerTail717_1, scan, if_neg (by decide), innerCandidates717_1_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_2
    (if 3 ≤ innerCandidates717_1.length then search edge 3 [5443, 5791] innerCandidates717_1 11234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan717_2

def innerTail717_0 : List Nat := 5653 :: innerTail717_1
def innerCandidates717_0 : List Nat := [5923, 6079, 7549]
set_option maxRecDepth 4096 in
theorem innerCandidates717_0_checked : candidates edge 4 5443 5653 innerTail717_1 best.bound = innerCandidates717_0 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild717_0 : search edge 3 [5443, 5653] innerCandidates717_0 11096 best = best := by decide

theorem innerScan717_0 : scan edge (search edge 3) 4 [5443] 5443 innerTail717_0 best = best := by
  rw [innerTail717_0, scan, if_neg (by decide), innerCandidates717_0_checked]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_1
    (if 3 ≤ innerCandidates717_0.length then search edge 3 [5443, 5653] innerCandidates717_0 11096 best else best) = best
  rw [if_pos (by decide), innerChild717_0]
  exact innerScan717_1

theorem rootBranch717 : search edge 4 [5443] rootCandidates717 5443 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5443] 5443 innerTail717_0 best = best
  exact innerScan717_0

def rootCandidates716 : List Nat := [5693, 5717, 5783, 5849, 6287, 6959, 7019, 7457, 7547]
set_option maxRecDepth 4096 in
theorem rootCandidates716_checked : candidates edge 5 0 5441 tail717 best.bound = rootCandidates716 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext q; rfl
  rw [row]
  decide

def innerTail716_9 : List Nat := []
theorem innerScan716_9 : scan edge (search edge 3) 4 [5441] 5441 innerTail716_9 best = best := rfl

def innerTail716_8 : List Nat := 7547 :: innerTail716_9
def innerCandidates716_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates716_8_checked : candidates edge 4 5441 7547 innerTail716_9 best.bound = innerCandidates716_8 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan716_8 : scan edge (search edge 3) 4 [5441] 5441 innerTail716_8 best = best := by
  rw [innerTail716_8, scan, if_neg (by decide), innerCandidates716_8_checked]
  change scan edge (search edge 3) 4 [5441] 5441 innerTail716_9
    (if 3 ≤ innerCandidates716_8.length then search edge 3 [5441, 7547] innerCandidates716_8 12988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan716_9

def innerTail716_7 : List Nat := 7457 :: innerTail716_8
def innerCandidates716_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates716_7_checked : candidates edge 4 5441 7457 innerTail716_8 best.bound = innerCandidates716_7 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan716_7 : scan edge (search edge 3) 4 [5441] 5441 innerTail716_7 best = best := by
  rw [innerTail716_7, scan, if_neg (by decide), innerCandidates716_7_checked]
  change scan edge (search edge 3) 4 [5441] 5441 innerTail716_8
    (if 3 ≤ innerCandidates716_7.length then search edge 3 [5441, 7457] innerCandidates716_7 12898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan716_8

def innerTail716_6 : List Nat := 7019 :: innerTail716_7
def innerCandidates716_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates716_6_checked : candidates edge 4 5441 7019 innerTail716_7 best.bound = innerCandidates716_6 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan716_6 : scan edge (search edge 3) 4 [5441] 5441 innerTail716_6 best = best := by
  rw [innerTail716_6, scan, if_neg (by decide), innerCandidates716_6_checked]
  change scan edge (search edge 3) 4 [5441] 5441 innerTail716_7
    (if 3 ≤ innerCandidates716_6.length then search edge 3 [5441, 7019] innerCandidates716_6 12460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan716_7

def innerTail716_5 : List Nat := 6959 :: innerTail716_6
def innerCandidates716_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates716_5_checked : candidates edge 4 5441 6959 innerTail716_6 best.bound = innerCandidates716_5 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan716_5 : scan edge (search edge 3) 4 [5441] 5441 innerTail716_5 best = best := by
  rw [innerTail716_5, scan, if_neg (by decide), innerCandidates716_5_checked]
  change scan edge (search edge 3) 4 [5441] 5441 innerTail716_6
    (if 3 ≤ innerCandidates716_5.length then search edge 3 [5441, 6959] innerCandidates716_5 12400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan716_6

def innerTail716_4 : List Nat := 6287 :: innerTail716_5
def innerCandidates716_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates716_4_checked : candidates edge 4 5441 6287 innerTail716_5 best.bound = innerCandidates716_4 := by
  unfold candidates
  have row : edge 6287 = SearchData.row6287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan716_4 : scan edge (search edge 3) 4 [5441] 5441 innerTail716_4 best = best := by
  rw [innerTail716_4, scan, if_neg (by decide), innerCandidates716_4_checked]
  change scan edge (search edge 3) 4 [5441] 5441 innerTail716_5
    (if 3 ≤ innerCandidates716_4.length then search edge 3 [5441, 6287] innerCandidates716_4 11728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan716_5

def innerTail716_3 : List Nat := 5849 :: innerTail716_4
def innerCandidates716_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates716_3_checked : candidates edge 4 5441 5849 innerTail716_4 best.bound = innerCandidates716_3 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan716_3 : scan edge (search edge 3) 4 [5441] 5441 innerTail716_3 best = best := by
  rw [innerTail716_3, scan, if_neg (by decide), innerCandidates716_3_checked]
  change scan edge (search edge 3) 4 [5441] 5441 innerTail716_4
    (if 3 ≤ innerCandidates716_3.length then search edge 3 [5441, 5849] innerCandidates716_3 11290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan716_4

def innerTail716_2 : List Nat := 5783 :: innerTail716_3
def innerCandidates716_2 : List Nat := [6959]
set_option maxRecDepth 4096 in
theorem innerCandidates716_2_checked : candidates edge 4 5441 5783 innerTail716_3 best.bound = innerCandidates716_2 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan716_2 : scan edge (search edge 3) 4 [5441] 5441 innerTail716_2 best = best := by
  rw [innerTail716_2, scan, if_neg (by decide), innerCandidates716_2_checked]
  change scan edge (search edge 3) 4 [5441] 5441 innerTail716_3
    (if 3 ≤ innerCandidates716_2.length then search edge 3 [5441, 5783] innerCandidates716_2 11224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan716_3

def innerTail716_1 : List Nat := 5717 :: innerTail716_2
def innerCandidates716_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates716_1_checked : candidates edge 4 5441 5717 innerTail716_2 best.bound = innerCandidates716_1 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan716_1 : scan edge (search edge 3) 4 [5441] 5441 innerTail716_1 best = best := by
  rw [innerTail716_1, scan, if_neg (by decide), innerCandidates716_1_checked]
  change scan edge (search edge 3) 4 [5441] 5441 innerTail716_2
    (if 3 ≤ innerCandidates716_1.length then search edge 3 [5441, 5717] innerCandidates716_1 11158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan716_2

def innerTail716_0 : List Nat := 5693 :: innerTail716_1
def innerCandidates716_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates716_0_checked : candidates edge 4 5441 5693 innerTail716_1 best.bound = innerCandidates716_0 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan716_0 : scan edge (search edge 3) 4 [5441] 5441 innerTail716_0 best = best := by
  rw [innerTail716_0, scan, if_neg (by decide), innerCandidates716_0_checked]
  change scan edge (search edge 3) 4 [5441] 5441 innerTail716_1
    (if 3 ≤ innerCandidates716_0.length then search edge 3 [5441, 5693] innerCandidates716_0 11134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan716_1

theorem rootBranch716 : search edge 4 [5441] rootCandidates716 5441 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5441] 5441 innerTail716_0 best = best
  exact innerScan716_0

def rootCandidates715 : List Nat := [5689, 6961, 7159, 8017, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates715_checked : candidates edge 5 0 5437 tail716 best.bound = rootCandidates715 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext q; rfl
  rw [row]
  decide

def innerTail715_5 : List Nat := []
theorem innerScan715_5 : scan edge (search edge 3) 4 [5437] 5437 innerTail715_5 best = best := rfl

def innerTail715_4 : List Nat := 8263 :: innerTail715_5
def innerCandidates715_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates715_4_checked : candidates edge 4 5437 8263 innerTail715_5 best.bound = innerCandidates715_4 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan715_4 : scan edge (search edge 3) 4 [5437] 5437 innerTail715_4 best = best := by
  rw [innerTail715_4, scan, if_neg (by decide), innerCandidates715_4_checked]
  change scan edge (search edge 3) 4 [5437] 5437 innerTail715_5
    (if 3 ≤ innerCandidates715_4.length then search edge 3 [5437, 8263] innerCandidates715_4 13700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan715_5

def innerTail715_3 : List Nat := 8017 :: innerTail715_4
def innerCandidates715_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates715_3_checked : candidates edge 4 5437 8017 innerTail715_4 best.bound = innerCandidates715_3 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan715_3 : scan edge (search edge 3) 4 [5437] 5437 innerTail715_3 best = best := by
  rw [innerTail715_3, scan, if_neg (by decide), innerCandidates715_3_checked]
  change scan edge (search edge 3) 4 [5437] 5437 innerTail715_4
    (if 3 ≤ innerCandidates715_3.length then search edge 3 [5437, 8017] innerCandidates715_3 13454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan715_4

def innerTail715_2 : List Nat := 7159 :: innerTail715_3
def innerCandidates715_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates715_2_checked : candidates edge 4 5437 7159 innerTail715_3 best.bound = innerCandidates715_2 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan715_2 : scan edge (search edge 3) 4 [5437] 5437 innerTail715_2 best = best := by
  rw [innerTail715_2, scan, if_neg (by decide), innerCandidates715_2_checked]
  change scan edge (search edge 3) 4 [5437] 5437 innerTail715_3
    (if 3 ≤ innerCandidates715_2.length then search edge 3 [5437, 7159] innerCandidates715_2 12596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan715_3

def innerTail715_1 : List Nat := 6961 :: innerTail715_2
def innerCandidates715_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates715_1_checked : candidates edge 4 5437 6961 innerTail715_2 best.bound = innerCandidates715_1 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan715_1 : scan edge (search edge 3) 4 [5437] 5437 innerTail715_1 best = best := by
  rw [innerTail715_1, scan, if_neg (by decide), innerCandidates715_1_checked]
  change scan edge (search edge 3) 4 [5437] 5437 innerTail715_2
    (if 3 ≤ innerCandidates715_1.length then search edge 3 [5437, 6961] innerCandidates715_1 12398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan715_2

def innerTail715_0 : List Nat := 5689 :: innerTail715_1
def innerCandidates715_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates715_0_checked : candidates edge 4 5437 5689 innerTail715_1 best.bound = innerCandidates715_0 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan715_0 : scan edge (search edge 3) 4 [5437] 5437 innerTail715_0 best = best := by
  rw [innerTail715_0, scan, if_neg (by decide), innerCandidates715_0_checked]
  change scan edge (search edge 3) 4 [5437] 5437 innerTail715_1
    (if 3 ≤ innerCandidates715_0.length then search edge 3 [5437, 5689] innerCandidates715_0 11126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan715_1

theorem rootBranch715 : search edge 4 [5437] rootCandidates715 5437 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5437] 5437 innerTail715_0 best = best
  exact innerScan715_0

def rootCandidates714 : List Nat := [5641, 5743, 6151, 7243, 7507, 7573, 8167]
set_option maxRecDepth 4096 in
theorem rootCandidates714_checked : candidates edge 5 0 5431 tail715 best.bound = rootCandidates714 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext q; rfl
  rw [row]
  decide

def innerTail714_7 : List Nat := []
theorem innerScan714_7 : scan edge (search edge 3) 4 [5431] 5431 innerTail714_7 best = best := rfl

def innerTail714_6 : List Nat := 8167 :: innerTail714_7
def innerCandidates714_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates714_6_checked : candidates edge 4 5431 8167 innerTail714_7 best.bound = innerCandidates714_6 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan714_6 : scan edge (search edge 3) 4 [5431] 5431 innerTail714_6 best = best := by
  rw [innerTail714_6, scan, if_neg (by decide), innerCandidates714_6_checked]
  change scan edge (search edge 3) 4 [5431] 5431 innerTail714_7
    (if 3 ≤ innerCandidates714_6.length then search edge 3 [5431, 8167] innerCandidates714_6 13598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan714_7

def innerTail714_5 : List Nat := 7573 :: innerTail714_6
def innerCandidates714_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates714_5_checked : candidates edge 4 5431 7573 innerTail714_6 best.bound = innerCandidates714_5 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan714_5 : scan edge (search edge 3) 4 [5431] 5431 innerTail714_5 best = best := by
  rw [innerTail714_5, scan, if_neg (by decide), innerCandidates714_5_checked]
  change scan edge (search edge 3) 4 [5431] 5431 innerTail714_6
    (if 3 ≤ innerCandidates714_5.length then search edge 3 [5431, 7573] innerCandidates714_5 13004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan714_6

def innerTail714_4 : List Nat := 7507 :: innerTail714_5
def innerCandidates714_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates714_4_checked : candidates edge 4 5431 7507 innerTail714_5 best.bound = innerCandidates714_4 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan714_4 : scan edge (search edge 3) 4 [5431] 5431 innerTail714_4 best = best := by
  rw [innerTail714_4, scan, if_neg (by decide), innerCandidates714_4_checked]
  change scan edge (search edge 3) 4 [5431] 5431 innerTail714_5
    (if 3 ≤ innerCandidates714_4.length then search edge 3 [5431, 7507] innerCandidates714_4 12938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan714_5

def innerTail714_3 : List Nat := 7243 :: innerTail714_4
def innerCandidates714_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates714_3_checked : candidates edge 4 5431 7243 innerTail714_4 best.bound = innerCandidates714_3 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan714_3 : scan edge (search edge 3) 4 [5431] 5431 innerTail714_3 best = best := by
  rw [innerTail714_3, scan, if_neg (by decide), innerCandidates714_3_checked]
  change scan edge (search edge 3) 4 [5431] 5431 innerTail714_4
    (if 3 ≤ innerCandidates714_3.length then search edge 3 [5431, 7243] innerCandidates714_3 12674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan714_4

def innerTail714_2 : List Nat := 6151 :: innerTail714_3
def innerCandidates714_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates714_2_checked : candidates edge 4 5431 6151 innerTail714_3 best.bound = innerCandidates714_2 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan714_2 : scan edge (search edge 3) 4 [5431] 5431 innerTail714_2 best = best := by
  rw [innerTail714_2, scan, if_neg (by decide), innerCandidates714_2_checked]
  change scan edge (search edge 3) 4 [5431] 5431 innerTail714_3
    (if 3 ≤ innerCandidates714_2.length then search edge 3 [5431, 6151] innerCandidates714_2 11582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan714_3

def innerTail714_1 : List Nat := 5743 :: innerTail714_2
def innerCandidates714_1 : List Nat := [7507, 7573]
set_option maxRecDepth 4096 in
theorem innerCandidates714_1_checked : candidates edge 4 5431 5743 innerTail714_2 best.bound = innerCandidates714_1 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan714_1 : scan edge (search edge 3) 4 [5431] 5431 innerTail714_1 best = best := by
  rw [innerTail714_1, scan, if_neg (by decide), innerCandidates714_1_checked]
  change scan edge (search edge 3) 4 [5431] 5431 innerTail714_2
    (if 3 ≤ innerCandidates714_1.length then search edge 3 [5431, 5743] innerCandidates714_1 11174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan714_2

def innerTail714_0 : List Nat := 5641 :: innerTail714_1
def innerCandidates714_0 : List Nat := [5743]
set_option maxRecDepth 4096 in
theorem innerCandidates714_0_checked : candidates edge 4 5431 5641 innerTail714_1 best.bound = innerCandidates714_0 := by
  unfold candidates
  have row : edge 5641 = SearchData.row5641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan714_0 : scan edge (search edge 3) 4 [5431] 5431 innerTail714_0 best = best := by
  rw [innerTail714_0, scan, if_neg (by decide), innerCandidates714_0_checked]
  change scan edge (search edge 3) 4 [5431] 5431 innerTail714_1
    (if 3 ≤ innerCandidates714_0.length then search edge 3 [5431, 5641] innerCandidates714_0 11072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan714_1

theorem rootBranch714 : search edge 4 [5431] rootCandidates714 5431 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5431] 5431 innerTail714_0 best = best
  exact innerScan714_0

def rootCandidates713 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem rootCandidates713_checked : candidates edge 5 0 5419 tail714 best.bound = rootCandidates713 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
