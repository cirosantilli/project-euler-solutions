import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates593 : List Nat := [4813, 5653, 6133, 7459, 7621, 7687, 8311]
set_option maxRecDepth 4096 in
theorem rootCandidates593_checked : candidates edge 5 0 4363 tail594 best.bound = rootCandidates593 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext q; rfl
  rw [row]
  decide

def innerTail593_7 : List Nat := []
theorem innerScan593_7 : scan edge (search edge 3) 4 [4363] 4363 innerTail593_7 best = best := rfl

def innerTail593_6 : List Nat := 8311 :: innerTail593_7
def innerCandidates593_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates593_6_checked : candidates edge 4 4363 8311 innerTail593_7 best.bound = innerCandidates593_6 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan593_6 : scan edge (search edge 3) 4 [4363] 4363 innerTail593_6 best = best := by
  rw [innerTail593_6, scan, if_neg (by decide), innerCandidates593_6_checked]
  change scan edge (search edge 3) 4 [4363] 4363 innerTail593_7
    (if 3 ≤ innerCandidates593_6.length then search edge 3 [4363, 8311] innerCandidates593_6 12674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan593_7

def innerTail593_5 : List Nat := 7687 :: innerTail593_6
def innerCandidates593_5 : List Nat := [8311]
set_option maxRecDepth 4096 in
theorem innerCandidates593_5_checked : candidates edge 4 4363 7687 innerTail593_6 best.bound = innerCandidates593_5 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan593_5 : scan edge (search edge 3) 4 [4363] 4363 innerTail593_5 best = best := by
  rw [innerTail593_5, scan, if_neg (by decide), innerCandidates593_5_checked]
  change scan edge (search edge 3) 4 [4363] 4363 innerTail593_6
    (if 3 ≤ innerCandidates593_5.length then search edge 3 [4363, 7687] innerCandidates593_5 12050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan593_6

def innerTail593_4 : List Nat := 7621 :: innerTail593_5
def innerCandidates593_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates593_4_checked : candidates edge 4 4363 7621 innerTail593_5 best.bound = innerCandidates593_4 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan593_4 : scan edge (search edge 3) 4 [4363] 4363 innerTail593_4 best = best := by
  rw [innerTail593_4, scan, if_neg (by decide), innerCandidates593_4_checked]
  change scan edge (search edge 3) 4 [4363] 4363 innerTail593_5
    (if 3 ≤ innerCandidates593_4.length then search edge 3 [4363, 7621] innerCandidates593_4 11984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan593_5

def innerTail593_3 : List Nat := 7459 :: innerTail593_4
def innerCandidates593_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates593_3_checked : candidates edge 4 4363 7459 innerTail593_4 best.bound = innerCandidates593_3 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan593_3 : scan edge (search edge 3) 4 [4363] 4363 innerTail593_3 best = best := by
  rw [innerTail593_3, scan, if_neg (by decide), innerCandidates593_3_checked]
  change scan edge (search edge 3) 4 [4363] 4363 innerTail593_4
    (if 3 ≤ innerCandidates593_3.length then search edge 3 [4363, 7459] innerCandidates593_3 11822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan593_4

def innerTail593_2 : List Nat := 6133 :: innerTail593_3
def innerCandidates593_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates593_2_checked : candidates edge 4 4363 6133 innerTail593_3 best.bound = innerCandidates593_2 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan593_2 : scan edge (search edge 3) 4 [4363] 4363 innerTail593_2 best = best := by
  rw [innerTail593_2, scan, if_neg (by decide), innerCandidates593_2_checked]
  change scan edge (search edge 3) 4 [4363] 4363 innerTail593_3
    (if 3 ≤ innerCandidates593_2.length then search edge 3 [4363, 6133] innerCandidates593_2 10496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan593_3

def innerTail593_1 : List Nat := 5653 :: innerTail593_2
def innerCandidates593_1 : List Nat := [6133]
set_option maxRecDepth 4096 in
theorem innerCandidates593_1_checked : candidates edge 4 4363 5653 innerTail593_2 best.bound = innerCandidates593_1 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan593_1 : scan edge (search edge 3) 4 [4363] 4363 innerTail593_1 best = best := by
  rw [innerTail593_1, scan, if_neg (by decide), innerCandidates593_1_checked]
  change scan edge (search edge 3) 4 [4363] 4363 innerTail593_2
    (if 3 ≤ innerCandidates593_1.length then search edge 3 [4363, 5653] innerCandidates593_1 10016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan593_2

def innerTail593_0 : List Nat := 4813 :: innerTail593_1
def innerCandidates593_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates593_0_checked : candidates edge 4 4363 4813 innerTail593_1 best.bound = innerCandidates593_0 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan593_0 : scan edge (search edge 3) 4 [4363] 4363 innerTail593_0 best = best := by
  rw [innerTail593_0, scan, if_neg (by decide), innerCandidates593_0_checked]
  change scan edge (search edge 3) 4 [4363] 4363 innerTail593_1
    (if 3 ≤ innerCandidates593_0.length then search edge 3 [4363, 4813] innerCandidates593_0 9176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan593_1

theorem rootBranch593 : search edge 4 [4363] rootCandidates593 4363 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4363] 4363 innerTail593_0 best = best
  exact innerScan593_0

def rootCandidates592 : List Nat := [4813, 5101, 5107, 5431, 5449, 5479, 5563, 6619, 6691, 6829, 7297, 7411, 8053, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates592_checked : candidates edge 5 0 4357 tail593 best.bound = rootCandidates592 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext q; rfl
  rw [row]
  decide

def innerTail592_14 : List Nat := []
theorem innerScan592_14 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_14 best = best := rfl

def innerTail592_13 : List Nat := 8209 :: innerTail592_14
def innerCandidates592_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates592_13_checked : candidates edge 4 4357 8209 innerTail592_14 best.bound = innerCandidates592_13 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_13 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_13 best = best := by
  rw [innerTail592_13, scan, if_neg (by decide), innerCandidates592_13_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_14
    (if 3 ≤ innerCandidates592_13.length then search edge 3 [4357, 8209] innerCandidates592_13 12566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_14

def innerTail592_12 : List Nat := 8053 :: innerTail592_13
def innerCandidates592_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates592_12_checked : candidates edge 4 4357 8053 innerTail592_13 best.bound = innerCandidates592_12 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_12 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_12 best = best := by
  rw [innerTail592_12, scan, if_neg (by decide), innerCandidates592_12_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_13
    (if 3 ≤ innerCandidates592_12.length then search edge 3 [4357, 8053] innerCandidates592_12 12410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_13

def innerTail592_11 : List Nat := 7411 :: innerTail592_12
def innerCandidates592_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates592_11_checked : candidates edge 4 4357 7411 innerTail592_12 best.bound = innerCandidates592_11 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_11 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_11 best = best := by
  rw [innerTail592_11, scan, if_neg (by decide), innerCandidates592_11_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_12
    (if 3 ≤ innerCandidates592_11.length then search edge 3 [4357, 7411] innerCandidates592_11 11768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_12

def innerTail592_10 : List Nat := 7297 :: innerTail592_11
def innerCandidates592_10 : List Nat := [8053]
set_option maxRecDepth 4096 in
theorem innerCandidates592_10_checked : candidates edge 4 4357 7297 innerTail592_11 best.bound = innerCandidates592_10 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_10 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_10 best = best := by
  rw [innerTail592_10, scan, if_neg (by decide), innerCandidates592_10_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_11
    (if 3 ≤ innerCandidates592_10.length then search edge 3 [4357, 7297] innerCandidates592_10 11654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_11

def innerTail592_9 : List Nat := 6829 :: innerTail592_10
def innerCandidates592_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates592_9_checked : candidates edge 4 4357 6829 innerTail592_10 best.bound = innerCandidates592_9 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_9 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_9 best = best := by
  rw [innerTail592_9, scan, if_neg (by decide), innerCandidates592_9_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_10
    (if 3 ≤ innerCandidates592_9.length then search edge 3 [4357, 6829] innerCandidates592_9 11186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_10

def innerTail592_8 : List Nat := 6691 :: innerTail592_9
def innerCandidates592_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates592_8_checked : candidates edge 4 4357 6691 innerTail592_9 best.bound = innerCandidates592_8 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_8 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_8 best = best := by
  rw [innerTail592_8, scan, if_neg (by decide), innerCandidates592_8_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_9
    (if 3 ≤ innerCandidates592_8.length then search edge 3 [4357, 6691] innerCandidates592_8 11048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_9

def innerTail592_7 : List Nat := 6619 :: innerTail592_8
def innerCandidates592_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates592_7_checked : candidates edge 4 4357 6619 innerTail592_8 best.bound = innerCandidates592_7 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_7 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_7 best = best := by
  rw [innerTail592_7, scan, if_neg (by decide), innerCandidates592_7_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_8
    (if 3 ≤ innerCandidates592_7.length then search edge 3 [4357, 6619] innerCandidates592_7 10976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_8

def innerTail592_6 : List Nat := 5563 :: innerTail592_7
def innerCandidates592_6 : List Nat := [8053]
set_option maxRecDepth 4096 in
theorem innerCandidates592_6_checked : candidates edge 4 4357 5563 innerTail592_7 best.bound = innerCandidates592_6 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_6 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_6 best = best := by
  rw [innerTail592_6, scan, if_neg (by decide), innerCandidates592_6_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_7
    (if 3 ≤ innerCandidates592_6.length then search edge 3 [4357, 5563] innerCandidates592_6 9920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_7

def innerTail592_5 : List Nat := 5479 :: innerTail592_6
def innerCandidates592_5 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem innerCandidates592_5_checked : candidates edge 4 4357 5479 innerTail592_6 best.bound = innerCandidates592_5 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_5 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_5 best = best := by
  rw [innerTail592_5, scan, if_neg (by decide), innerCandidates592_5_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_6
    (if 3 ≤ innerCandidates592_5.length then search edge 3 [4357, 5479] innerCandidates592_5 9836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_6

def innerTail592_4 : List Nat := 5449 :: innerTail592_5
def innerCandidates592_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates592_4_checked : candidates edge 4 4357 5449 innerTail592_5 best.bound = innerCandidates592_4 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_4 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_4 best = best := by
  rw [innerTail592_4, scan, if_neg (by decide), innerCandidates592_4_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_5
    (if 3 ≤ innerCandidates592_4.length then search edge 3 [4357, 5449] innerCandidates592_4 9806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_5

def innerTail592_3 : List Nat := 5431 :: innerTail592_4
def innerCandidates592_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates592_3_checked : candidates edge 4 4357 5431 innerTail592_4 best.bound = innerCandidates592_3 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_3 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_3 best = best := by
  rw [innerTail592_3, scan, if_neg (by decide), innerCandidates592_3_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_4
    (if 3 ≤ innerCandidates592_3.length then search edge 3 [4357, 5431] innerCandidates592_3 9788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_4

def innerTail592_2 : List Nat := 5107 :: innerTail592_3
def innerCandidates592_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates592_2_checked : candidates edge 4 4357 5107 innerTail592_3 best.bound = innerCandidates592_2 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_2 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_2 best = best := by
  rw [innerTail592_2, scan, if_neg (by decide), innerCandidates592_2_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_3
    (if 3 ≤ innerCandidates592_2.length then search edge 3 [4357, 5107] innerCandidates592_2 9464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_3

def innerTail592_1 : List Nat := 5101 :: innerTail592_2
def innerCandidates592_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates592_1_checked : candidates edge 4 4357 5101 innerTail592_2 best.bound = innerCandidates592_1 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_1 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_1 best = best := by
  rw [innerTail592_1, scan, if_neg (by decide), innerCandidates592_1_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_2
    (if 3 ≤ innerCandidates592_1.length then search edge 3 [4357, 5101] innerCandidates592_1 9458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_2

def innerTail592_0 : List Nat := 4813 :: innerTail592_1
def innerCandidates592_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates592_0_checked : candidates edge 4 4357 4813 innerTail592_1 best.bound = innerCandidates592_0 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan592_0 : scan edge (search edge 3) 4 [4357] 4357 innerTail592_0 best = best := by
  rw [innerTail592_0, scan, if_neg (by decide), innerCandidates592_0_checked]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_1
    (if 3 ≤ innerCandidates592_0.length then search edge 3 [4357, 4813] innerCandidates592_0 9170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan592_1

theorem rootBranch592 : search edge 4 [4357] rootCandidates592 4357 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4357] 4357 innerTail592_0 best = best
  exact innerScan592_0

def rootCandidates591 : List Nat := [4967, 5051, 5297, 5303, 5483, 5531, 5861, 6581, 7019, 7523, 7793, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates591_checked : candidates edge 5 0 4349 tail592 best.bound = rootCandidates591 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext q; rfl
  rw [row]
  decide

def innerTail591_12 : List Nat := []
theorem innerScan591_12 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_12 best = best := rfl

def innerTail591_11 : List Nat := 8093 :: innerTail591_12
def innerCandidates591_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates591_11_checked : candidates edge 4 4349 8093 innerTail591_12 best.bound = innerCandidates591_11 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan591_11 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_11 best = best := by
  rw [innerTail591_11, scan, if_neg (by decide), innerCandidates591_11_checked]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_12
    (if 3 ≤ innerCandidates591_11.length then search edge 3 [4349, 8093] innerCandidates591_11 12442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan591_12

def innerTail591_10 : List Nat := 7793 :: innerTail591_11
def innerCandidates591_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates591_10_checked : candidates edge 4 4349 7793 innerTail591_11 best.bound = innerCandidates591_10 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan591_10 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_10 best = best := by
  rw [innerTail591_10, scan, if_neg (by decide), innerCandidates591_10_checked]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_11
    (if 3 ≤ innerCandidates591_10.length then search edge 3 [4349, 7793] innerCandidates591_10 12142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan591_11

def innerTail591_9 : List Nat := 7523 :: innerTail591_10
def innerCandidates591_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates591_9_checked : candidates edge 4 4349 7523 innerTail591_10 best.bound = innerCandidates591_9 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan591_9 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_9 best = best := by
  rw [innerTail591_9, scan, if_neg (by decide), innerCandidates591_9_checked]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_10
    (if 3 ≤ innerCandidates591_9.length then search edge 3 [4349, 7523] innerCandidates591_9 11872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan591_10

def innerTail591_8 : List Nat := 7019 :: innerTail591_9
def innerCandidates591_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates591_8_checked : candidates edge 4 4349 7019 innerTail591_9 best.bound = innerCandidates591_8 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan591_8 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_8 best = best := by
  rw [innerTail591_8, scan, if_neg (by decide), innerCandidates591_8_checked]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_9
    (if 3 ≤ innerCandidates591_8.length then search edge 3 [4349, 7019] innerCandidates591_8 11368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan591_9

def innerTail591_7 : List Nat := 6581 :: innerTail591_8
def innerCandidates591_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates591_7_checked : candidates edge 4 4349 6581 innerTail591_8 best.bound = innerCandidates591_7 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan591_7 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_7 best = best := by
  rw [innerTail591_7, scan, if_neg (by decide), innerCandidates591_7_checked]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_8
    (if 3 ≤ innerCandidates591_7.length then search edge 3 [4349, 6581] innerCandidates591_7 10930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan591_8

def innerTail591_6 : List Nat := 5861 :: innerTail591_7
def innerCandidates591_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates591_6_checked : candidates edge 4 4349 5861 innerTail591_7 best.bound = innerCandidates591_6 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan591_6 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_6 best = best := by
  rw [innerTail591_6, scan, if_neg (by decide), innerCandidates591_6_checked]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_7
    (if 3 ≤ innerCandidates591_6.length then search edge 3 [4349, 5861] innerCandidates591_6 10210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan591_7

def innerTail591_5 : List Nat := 5531 :: innerTail591_6
def innerCandidates591_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates591_5_checked : candidates edge 4 4349 5531 innerTail591_6 best.bound = innerCandidates591_5 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan591_5 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_5 best = best := by
  rw [innerTail591_5, scan, if_neg (by decide), innerCandidates591_5_checked]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_6
    (if 3 ≤ innerCandidates591_5.length then search edge 3 [4349, 5531] innerCandidates591_5 9880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan591_6

def innerTail591_4 : List Nat := 5483 :: innerTail591_5
def innerCandidates591_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates591_4_checked : candidates edge 4 4349 5483 innerTail591_5 best.bound = innerCandidates591_4 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan591_4 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_4 best = best := by
  rw [innerTail591_4, scan, if_neg (by decide), innerCandidates591_4_checked]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_5
    (if 3 ≤ innerCandidates591_4.length then search edge 3 [4349, 5483] innerCandidates591_4 9832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan591_5

def innerTail591_3 : List Nat := 5303 :: innerTail591_4
def innerCandidates591_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates591_3_checked : candidates edge 4 4349 5303 innerTail591_4 best.bound = innerCandidates591_3 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan591_3 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_3 best = best := by
  rw [innerTail591_3, scan, if_neg (by decide), innerCandidates591_3_checked]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_4
    (if 3 ≤ innerCandidates591_3.length then search edge 3 [4349, 5303] innerCandidates591_3 9652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan591_4

def innerTail591_2 : List Nat := 5297 :: innerTail591_3
def innerCandidates591_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates591_2_checked : candidates edge 4 4349 5297 innerTail591_3 best.bound = innerCandidates591_2 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan591_2 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_2 best = best := by
  rw [innerTail591_2, scan, if_neg (by decide), innerCandidates591_2_checked]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_3
    (if 3 ≤ innerCandidates591_2.length then search edge 3 [4349, 5297] innerCandidates591_2 9646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan591_3

def innerTail591_1 : List Nat := 5051 :: innerTail591_2
def innerCandidates591_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates591_1_checked : candidates edge 4 4349 5051 innerTail591_2 best.bound = innerCandidates591_1 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan591_1 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_1 best = best := by
  rw [innerTail591_1, scan, if_neg (by decide), innerCandidates591_1_checked]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_2
    (if 3 ≤ innerCandidates591_1.length then search edge 3 [4349, 5051] innerCandidates591_1 9400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan591_2

def innerTail591_0 : List Nat := 4967 :: innerTail591_1
def innerCandidates591_0 : List Nat := [5531]
set_option maxRecDepth 4096 in
theorem innerCandidates591_0_checked : candidates edge 4 4349 4967 innerTail591_1 best.bound = innerCandidates591_0 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan591_0 : scan edge (search edge 3) 4 [4349] 4349 innerTail591_0 best = best := by
  rw [innerTail591_0, scan, if_neg (by decide), innerCandidates591_0_checked]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_1
    (if 3 ≤ innerCandidates591_0.length then search edge 3 [4349, 4967] innerCandidates591_0 9316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan591_1

theorem rootBranch591 : search edge 4 [4349] rootCandidates591 4349 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4349] 4349 innerTail591_0 best = best
  exact innerScan591_0

def rootCandidates590 : List Nat := [4567, 4639, 4831, 4903, 5701, 6961, 7603, 7927, 8167, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates590_checked : candidates edge 5 0 4339 tail591 best.bound = rootCandidates590 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext q; rfl
  rw [row]
  decide

def innerTail590_10 : List Nat := []
theorem innerScan590_10 : scan edge (search edge 3) 4 [4339] 4339 innerTail590_10 best = best := rfl

def innerTail590_9 : List Nat := 8317 :: innerTail590_10
def innerCandidates590_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates590_9_checked : candidates edge 4 4339 8317 innerTail590_10 best.bound = innerCandidates590_9 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan590_9 : scan edge (search edge 3) 4 [4339] 4339 innerTail590_9 best = best := by
  rw [innerTail590_9, scan, if_neg (by decide), innerCandidates590_9_checked]
  change scan edge (search edge 3) 4 [4339] 4339 innerTail590_10
    (if 3 ≤ innerCandidates590_9.length then search edge 3 [4339, 8317] innerCandidates590_9 12656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan590_10

def innerTail590_8 : List Nat := 8167 :: innerTail590_9
def innerCandidates590_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates590_8_checked : candidates edge 4 4339 8167 innerTail590_9 best.bound = innerCandidates590_8 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan590_8 : scan edge (search edge 3) 4 [4339] 4339 innerTail590_8 best = best := by
  rw [innerTail590_8, scan, if_neg (by decide), innerCandidates590_8_checked]
  change scan edge (search edge 3) 4 [4339] 4339 innerTail590_9
    (if 3 ≤ innerCandidates590_8.length then search edge 3 [4339, 8167] innerCandidates590_8 12506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan590_9

def innerTail590_7 : List Nat := 7927 :: innerTail590_8
def innerCandidates590_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates590_7_checked : candidates edge 4 4339 7927 innerTail590_8 best.bound = innerCandidates590_7 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan590_7 : scan edge (search edge 3) 4 [4339] 4339 innerTail590_7 best = best := by
  rw [innerTail590_7, scan, if_neg (by decide), innerCandidates590_7_checked]
  change scan edge (search edge 3) 4 [4339] 4339 innerTail590_8
    (if 3 ≤ innerCandidates590_7.length then search edge 3 [4339, 7927] innerCandidates590_7 12266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan590_8

def innerTail590_6 : List Nat := 7603 :: innerTail590_7
def innerCandidates590_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates590_6_checked : candidates edge 4 4339 7603 innerTail590_7 best.bound = innerCandidates590_6 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan590_6 : scan edge (search edge 3) 4 [4339] 4339 innerTail590_6 best = best := by
  rw [innerTail590_6, scan, if_neg (by decide), innerCandidates590_6_checked]
  change scan edge (search edge 3) 4 [4339] 4339 innerTail590_7
    (if 3 ≤ innerCandidates590_6.length then search edge 3 [4339, 7603] innerCandidates590_6 11942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan590_7

def innerTail590_5 : List Nat := 6961 :: innerTail590_6
def innerCandidates590_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates590_5_checked : candidates edge 4 4339 6961 innerTail590_6 best.bound = innerCandidates590_5 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan590_5 : scan edge (search edge 3) 4 [4339] 4339 innerTail590_5 best = best := by
  rw [innerTail590_5, scan, if_neg (by decide), innerCandidates590_5_checked]
  change scan edge (search edge 3) 4 [4339] 4339 innerTail590_6
    (if 3 ≤ innerCandidates590_5.length then search edge 3 [4339, 6961] innerCandidates590_5 11300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan590_6

def innerTail590_4 : List Nat := 5701 :: innerTail590_5
def innerCandidates590_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates590_4_checked : candidates edge 4 4339 5701 innerTail590_5 best.bound = innerCandidates590_4 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan590_4 : scan edge (search edge 3) 4 [4339] 4339 innerTail590_4 best = best := by
  rw [innerTail590_4, scan, if_neg (by decide), innerCandidates590_4_checked]
  change scan edge (search edge 3) 4 [4339] 4339 innerTail590_5
    (if 3 ≤ innerCandidates590_4.length then search edge 3 [4339, 5701] innerCandidates590_4 10040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan590_5

def innerTail590_3 : List Nat := 4903 :: innerTail590_4
def innerCandidates590_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates590_3_checked : candidates edge 4 4339 4903 innerTail590_4 best.bound = innerCandidates590_3 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan590_3 : scan edge (search edge 3) 4 [4339] 4339 innerTail590_3 best = best := by
  rw [innerTail590_3, scan, if_neg (by decide), innerCandidates590_3_checked]
  change scan edge (search edge 3) 4 [4339] 4339 innerTail590_4
    (if 3 ≤ innerCandidates590_3.length then search edge 3 [4339, 4903] innerCandidates590_3 9242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan590_4

def innerTail590_2 : List Nat := 4831 :: innerTail590_3
def innerCandidates590_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates590_2_checked : candidates edge 4 4339 4831 innerTail590_3 best.bound = innerCandidates590_2 := by
  unfold candidates
  have row : edge 4831 = SearchData.row4831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan590_2 : scan edge (search edge 3) 4 [4339] 4339 innerTail590_2 best = best := by
  rw [innerTail590_2, scan, if_neg (by decide), innerCandidates590_2_checked]
  change scan edge (search edge 3) 4 [4339] 4339 innerTail590_3
    (if 3 ≤ innerCandidates590_2.length then search edge 3 [4339, 4831] innerCandidates590_2 9170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan590_3

def innerTail590_1 : List Nat := 4639 :: innerTail590_2
def innerCandidates590_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates590_1_checked : candidates edge 4 4339 4639 innerTail590_2 best.bound = innerCandidates590_1 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan590_1 : scan edge (search edge 3) 4 [4339] 4339 innerTail590_1 best = best := by
  rw [innerTail590_1, scan, if_neg (by decide), innerCandidates590_1_checked]
  change scan edge (search edge 3) 4 [4339] 4339 innerTail590_2
    (if 3 ≤ innerCandidates590_1.length then search edge 3 [4339, 4639] innerCandidates590_1 8978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan590_2

def innerTail590_0 : List Nat := 4567 :: innerTail590_1
def innerCandidates590_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates590_0_checked : candidates edge 4 4339 4567 innerTail590_1 best.bound = innerCandidates590_0 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan590_0 : scan edge (search edge 3) 4 [4339] 4339 innerTail590_0 best = best := by
  rw [innerTail590_0, scan, if_neg (by decide), innerCandidates590_0_checked]
  change scan edge (search edge 3) 4 [4339] 4339 innerTail590_1
    (if 3 ≤ innerCandidates590_0.length then search edge 3 [4339, 4567] innerCandidates590_0 8906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan590_1

theorem rootBranch590 : search edge 4 [4339] rootCandidates590 4339 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4339] 4339 innerTail590_0 best = best
  exact innerScan590_0

def rootCandidates589 : List Nat := [4421, 4523, 4889, 5099, 5237, 5471, 7043, 7547]
set_option maxRecDepth 4096 in
theorem rootCandidates589_checked : candidates edge 5 0 4337 tail590 best.bound = rootCandidates589 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext q; rfl
  rw [row]
  decide

def innerTail589_8 : List Nat := []
theorem innerScan589_8 : scan edge (search edge 3) 4 [4337] 4337 innerTail589_8 best = best := rfl

def innerTail589_7 : List Nat := 7547 :: innerTail589_8
def innerCandidates589_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates589_7_checked : candidates edge 4 4337 7547 innerTail589_8 best.bound = innerCandidates589_7 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan589_7 : scan edge (search edge 3) 4 [4337] 4337 innerTail589_7 best = best := by
  rw [innerTail589_7, scan, if_neg (by decide), innerCandidates589_7_checked]
  change scan edge (search edge 3) 4 [4337] 4337 innerTail589_8
    (if 3 ≤ innerCandidates589_7.length then search edge 3 [4337, 7547] innerCandidates589_7 11884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan589_8

def innerTail589_6 : List Nat := 7043 :: innerTail589_7
def innerCandidates589_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates589_6_checked : candidates edge 4 4337 7043 innerTail589_7 best.bound = innerCandidates589_6 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan589_6 : scan edge (search edge 3) 4 [4337] 4337 innerTail589_6 best = best := by
  rw [innerTail589_6, scan, if_neg (by decide), innerCandidates589_6_checked]
  change scan edge (search edge 3) 4 [4337] 4337 innerTail589_7
    (if 3 ≤ innerCandidates589_6.length then search edge 3 [4337, 7043] innerCandidates589_6 11380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan589_7

def innerTail589_5 : List Nat := 5471 :: innerTail589_6
def innerCandidates589_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates589_5_checked : candidates edge 4 4337 5471 innerTail589_6 best.bound = innerCandidates589_5 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan589_5 : scan edge (search edge 3) 4 [4337] 4337 innerTail589_5 best = best := by
  rw [innerTail589_5, scan, if_neg (by decide), innerCandidates589_5_checked]
  change scan edge (search edge 3) 4 [4337] 4337 innerTail589_6
    (if 3 ≤ innerCandidates589_5.length then search edge 3 [4337, 5471] innerCandidates589_5 9808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan589_6

def innerTail589_4 : List Nat := 5237 :: innerTail589_5
def innerCandidates589_4 : List Nat := [5471, 7043]
set_option maxRecDepth 4096 in
theorem innerCandidates589_4_checked : candidates edge 4 4337 5237 innerTail589_5 best.bound = innerCandidates589_4 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan589_4 : scan edge (search edge 3) 4 [4337] 4337 innerTail589_4 best = best := by
  rw [innerTail589_4, scan, if_neg (by decide), innerCandidates589_4_checked]
  change scan edge (search edge 3) 4 [4337] 4337 innerTail589_5
    (if 3 ≤ innerCandidates589_4.length then search edge 3 [4337, 5237] innerCandidates589_4 9574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan589_5

def innerTail589_3 : List Nat := 5099 :: innerTail589_4
def innerCandidates589_3 : List Nat := [5471]
set_option maxRecDepth 4096 in
theorem innerCandidates589_3_checked : candidates edge 4 4337 5099 innerTail589_4 best.bound = innerCandidates589_3 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan589_3 : scan edge (search edge 3) 4 [4337] 4337 innerTail589_3 best = best := by
  rw [innerTail589_3, scan, if_neg (by decide), innerCandidates589_3_checked]
  change scan edge (search edge 3) 4 [4337] 4337 innerTail589_4
    (if 3 ≤ innerCandidates589_3.length then search edge 3 [4337, 5099] innerCandidates589_3 9436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan589_4

def innerTail589_2 : List Nat := 4889 :: innerTail589_3
def innerCandidates589_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates589_2_checked : candidates edge 4 4337 4889 innerTail589_3 best.bound = innerCandidates589_2 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan589_2 : scan edge (search edge 3) 4 [4337] 4337 innerTail589_2 best = best := by
  rw [innerTail589_2, scan, if_neg (by decide), innerCandidates589_2_checked]
  change scan edge (search edge 3) 4 [4337] 4337 innerTail589_3
    (if 3 ≤ innerCandidates589_2.length then search edge 3 [4337, 4889] innerCandidates589_2 9226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan589_3

def innerTail589_1 : List Nat := 4523 :: innerTail589_2
def innerCandidates589_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates589_1_checked : candidates edge 4 4337 4523 innerTail589_2 best.bound = innerCandidates589_1 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan589_1 : scan edge (search edge 3) 4 [4337] 4337 innerTail589_1 best = best := by
  rw [innerTail589_1, scan, if_neg (by decide), innerCandidates589_1_checked]
  change scan edge (search edge 3) 4 [4337] 4337 innerTail589_2
    (if 3 ≤ innerCandidates589_1.length then search edge 3 [4337, 4523] innerCandidates589_1 8860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan589_2

def innerTail589_0 : List Nat := 4421 :: innerTail589_1
def innerCandidates589_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates589_0_checked : candidates edge 4 4337 4421 innerTail589_1 best.bound = innerCandidates589_0 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan589_0 : scan edge (search edge 3) 4 [4337] 4337 innerTail589_0 best = best := by
  rw [innerTail589_0, scan, if_neg (by decide), innerCandidates589_0_checked]
  change scan edge (search edge 3) 4 [4337] 4337 innerTail589_1
    (if 3 ≤ innerCandidates589_0.length then search edge 3 [4337, 4421] innerCandidates589_0 8758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan589_1

theorem rootBranch589 : search edge 4 [4337] rootCandidates589 4337 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4337] 4337 innerTail589_0 best = best
  exact innerScan589_0

def rootCandidates588 : List Nat := [4519, 4591, 4801, 5107, 5503, 5737, 6481, 6553, 6703, 7639, 7963, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates588_checked : candidates edge 5 0 4327 tail589 best.bound = rootCandidates588 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext q; rfl
  rw [row]
  decide

def innerTail588_12 : List Nat := []
theorem innerScan588_12 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_12 best = best := rfl

def innerTail588_11 : List Nat := 8233 :: innerTail588_12
def innerCandidates588_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates588_11_checked : candidates edge 4 4327 8233 innerTail588_12 best.bound = innerCandidates588_11 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan588_11 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_11 best = best := by
  rw [innerTail588_11, scan, if_neg (by decide), innerCandidates588_11_checked]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_12
    (if 3 ≤ innerCandidates588_11.length then search edge 3 [4327, 8233] innerCandidates588_11 12560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan588_12

def innerTail588_10 : List Nat := 7963 :: innerTail588_11
def innerCandidates588_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates588_10_checked : candidates edge 4 4327 7963 innerTail588_11 best.bound = innerCandidates588_10 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan588_10 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_10 best = best := by
  rw [innerTail588_10, scan, if_neg (by decide), innerCandidates588_10_checked]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_11
    (if 3 ≤ innerCandidates588_10.length then search edge 3 [4327, 7963] innerCandidates588_10 12290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan588_11

def innerTail588_9 : List Nat := 7639 :: innerTail588_10
def innerCandidates588_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates588_9_checked : candidates edge 4 4327 7639 innerTail588_10 best.bound = innerCandidates588_9 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan588_9 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_9 best = best := by
  rw [innerTail588_9, scan, if_neg (by decide), innerCandidates588_9_checked]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_10
    (if 3 ≤ innerCandidates588_9.length then search edge 3 [4327, 7639] innerCandidates588_9 11966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan588_10

def innerTail588_8 : List Nat := 6703 :: innerTail588_9
def innerCandidates588_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates588_8_checked : candidates edge 4 4327 6703 innerTail588_9 best.bound = innerCandidates588_8 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan588_8 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_8 best = best := by
  rw [innerTail588_8, scan, if_neg (by decide), innerCandidates588_8_checked]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_9
    (if 3 ≤ innerCandidates588_8.length then search edge 3 [4327, 6703] innerCandidates588_8 11030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan588_9

def innerTail588_7 : List Nat := 6553 :: innerTail588_8
def innerCandidates588_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates588_7_checked : candidates edge 4 4327 6553 innerTail588_8 best.bound = innerCandidates588_7 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan588_7 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_7 best = best := by
  rw [innerTail588_7, scan, if_neg (by decide), innerCandidates588_7_checked]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_8
    (if 3 ≤ innerCandidates588_7.length then search edge 3 [4327, 6553] innerCandidates588_7 10880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan588_8

def innerTail588_6 : List Nat := 6481 :: innerTail588_7
def innerCandidates588_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates588_6_checked : candidates edge 4 4327 6481 innerTail588_7 best.bound = innerCandidates588_6 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan588_6 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_6 best = best := by
  rw [innerTail588_6, scan, if_neg (by decide), innerCandidates588_6_checked]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_7
    (if 3 ≤ innerCandidates588_6.length then search edge 3 [4327, 6481] innerCandidates588_6 10808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan588_7

def innerTail588_5 : List Nat := 5737 :: innerTail588_6
def innerCandidates588_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates588_5_checked : candidates edge 4 4327 5737 innerTail588_6 best.bound = innerCandidates588_5 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan588_5 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_5 best = best := by
  rw [innerTail588_5, scan, if_neg (by decide), innerCandidates588_5_checked]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_6
    (if 3 ≤ innerCandidates588_5.length then search edge 3 [4327, 5737] innerCandidates588_5 10064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan588_6

def innerTail588_4 : List Nat := 5503 :: innerTail588_5
def innerCandidates588_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates588_4_checked : candidates edge 4 4327 5503 innerTail588_5 best.bound = innerCandidates588_4 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan588_4 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_4 best = best := by
  rw [innerTail588_4, scan, if_neg (by decide), innerCandidates588_4_checked]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_5
    (if 3 ≤ innerCandidates588_4.length then search edge 3 [4327, 5503] innerCandidates588_4 9830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan588_5

def innerTail588_3 : List Nat := 5107 :: innerTail588_4
def innerCandidates588_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates588_3_checked : candidates edge 4 4327 5107 innerTail588_4 best.bound = innerCandidates588_3 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan588_3 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_3 best = best := by
  rw [innerTail588_3, scan, if_neg (by decide), innerCandidates588_3_checked]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_4
    (if 3 ≤ innerCandidates588_3.length then search edge 3 [4327, 5107] innerCandidates588_3 9434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan588_4

def innerTail588_2 : List Nat := 4801 :: innerTail588_3
def innerCandidates588_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates588_2_checked : candidates edge 4 4327 4801 innerTail588_3 best.bound = innerCandidates588_2 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan588_2 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_2 best = best := by
  rw [innerTail588_2, scan, if_neg (by decide), innerCandidates588_2_checked]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_3
    (if 3 ≤ innerCandidates588_2.length then search edge 3 [4327, 4801] innerCandidates588_2 9128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan588_3

def innerTail588_1 : List Nat := 4591 :: innerTail588_2
def innerCandidates588_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates588_1_checked : candidates edge 4 4327 4591 innerTail588_2 best.bound = innerCandidates588_1 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan588_1 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_1 best = best := by
  rw [innerTail588_1, scan, if_neg (by decide), innerCandidates588_1_checked]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_2
    (if 3 ≤ innerCandidates588_1.length then search edge 3 [4327, 4591] innerCandidates588_1 8918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan588_2

def innerTail588_0 : List Nat := 4519 :: innerTail588_1
def innerCandidates588_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates588_0_checked : candidates edge 4 4327 4519 innerTail588_1 best.bound = innerCandidates588_0 := by
  unfold candidates
  have row : edge 4519 = SearchData.row4519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan588_0 : scan edge (search edge 3) 4 [4327] 4327 innerTail588_0 best = best := by
  rw [innerTail588_0, scan, if_neg (by decide), innerCandidates588_0_checked]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_1
    (if 3 ≤ innerCandidates588_0.length then search edge 3 [4327, 4519] innerCandidates588_0 8846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan588_1

theorem rootBranch588 : search edge 4 [4327] rootCandidates588 4327 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4327] 4327 innerTail588_0 best = best
  exact innerScan588_0

end ProjectEulerSolutions.PrimePairUpper.Search
