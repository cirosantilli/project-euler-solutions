import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates269 : List Nat := [1873, 1933, 2053, 2179, 2671, 3691, 3769, 3889, 4099, 4231, 4993, 5569, 5737, 6037, 6367, 6421, 6967, 7129, 7177, 7477, 7639, 8011]
set_option maxRecDepth 4096 in
theorem rootCandidates269_checked : candidates edge 5 0 1747 tail270 best.bound = rootCandidates269 := by
  unfold candidates
  have row : edge 1747 = PrimePairCache.smallRow1747.contains := by funext q; rfl
  rw [row]
  decide

def innerTail269_22 : List Nat := []
theorem innerScan269_22 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_22 best = best := rfl

def innerTail269_21 : List Nat := 8011 :: innerTail269_22
def innerCandidates269_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_21_checked : candidates edge 4 1747 8011 innerTail269_22 best.bound = innerCandidates269_21 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_21 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_21 best = best := by
  rw [innerTail269_21, scan, if_neg (by decide), innerCandidates269_21_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_22
    (if 3 ≤ innerCandidates269_21.length then search edge 3 [1747, 8011] innerCandidates269_21 9758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_22

def innerTail269_20 : List Nat := 7639 :: innerTail269_21
def innerCandidates269_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_20_checked : candidates edge 4 1747 7639 innerTail269_21 best.bound = innerCandidates269_20 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_20 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_20 best = best := by
  rw [innerTail269_20, scan, if_neg (by decide), innerCandidates269_20_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_21
    (if 3 ≤ innerCandidates269_20.length then search edge 3 [1747, 7639] innerCandidates269_20 9386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_21

def innerTail269_19 : List Nat := 7477 :: innerTail269_20
def innerCandidates269_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_19_checked : candidates edge 4 1747 7477 innerTail269_20 best.bound = innerCandidates269_19 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_19 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_19 best = best := by
  rw [innerTail269_19, scan, if_neg (by decide), innerCandidates269_19_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_20
    (if 3 ≤ innerCandidates269_19.length then search edge 3 [1747, 7477] innerCandidates269_19 9224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_20

def innerTail269_18 : List Nat := 7177 :: innerTail269_19
def innerCandidates269_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_18_checked : candidates edge 4 1747 7177 innerTail269_19 best.bound = innerCandidates269_18 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_18 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_18 best = best := by
  rw [innerTail269_18, scan, if_neg (by decide), innerCandidates269_18_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_19
    (if 3 ≤ innerCandidates269_18.length then search edge 3 [1747, 7177] innerCandidates269_18 8924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_19

def innerTail269_17 : List Nat := 7129 :: innerTail269_18
def innerCandidates269_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_17_checked : candidates edge 4 1747 7129 innerTail269_18 best.bound = innerCandidates269_17 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_17 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_17 best = best := by
  rw [innerTail269_17, scan, if_neg (by decide), innerCandidates269_17_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_18
    (if 3 ≤ innerCandidates269_17.length then search edge 3 [1747, 7129] innerCandidates269_17 8876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_18

def innerTail269_16 : List Nat := 6967 :: innerTail269_17
def innerCandidates269_16 : List Nat := [7177]
set_option maxRecDepth 4096 in
theorem innerCandidates269_16_checked : candidates edge 4 1747 6967 innerTail269_17 best.bound = innerCandidates269_16 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_16 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_16 best = best := by
  rw [innerTail269_16, scan, if_neg (by decide), innerCandidates269_16_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_17
    (if 3 ≤ innerCandidates269_16.length then search edge 3 [1747, 6967] innerCandidates269_16 8714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_17

def innerTail269_15 : List Nat := 6421 :: innerTail269_16
def innerCandidates269_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_15_checked : candidates edge 4 1747 6421 innerTail269_16 best.bound = innerCandidates269_15 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_15 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_15 best = best := by
  rw [innerTail269_15, scan, if_neg (by decide), innerCandidates269_15_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_16
    (if 3 ≤ innerCandidates269_15.length then search edge 3 [1747, 6421] innerCandidates269_15 8168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_16

def innerTail269_14 : List Nat := 6367 :: innerTail269_15
def innerCandidates269_14 : List Nat := [7477]
set_option maxRecDepth 4096 in
theorem innerCandidates269_14_checked : candidates edge 4 1747 6367 innerTail269_15 best.bound = innerCandidates269_14 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_14 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_14 best = best := by
  rw [innerTail269_14, scan, if_neg (by decide), innerCandidates269_14_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_15
    (if 3 ≤ innerCandidates269_14.length then search edge 3 [1747, 6367] innerCandidates269_14 8114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_15

def innerTail269_13 : List Nat := 6037 :: innerTail269_14
def innerCandidates269_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_13_checked : candidates edge 4 1747 6037 innerTail269_14 best.bound = innerCandidates269_13 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_13 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_13 best = best := by
  rw [innerTail269_13, scan, if_neg (by decide), innerCandidates269_13_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_14
    (if 3 ≤ innerCandidates269_13.length then search edge 3 [1747, 6037] innerCandidates269_13 7784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_14

def innerTail269_12 : List Nat := 5737 :: innerTail269_13
def innerCandidates269_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_12_checked : candidates edge 4 1747 5737 innerTail269_13 best.bound = innerCandidates269_12 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_12 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_12 best = best := by
  rw [innerTail269_12, scan, if_neg (by decide), innerCandidates269_12_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_13
    (if 3 ≤ innerCandidates269_12.length then search edge 3 [1747, 5737] innerCandidates269_12 7484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_13

def innerTail269_11 : List Nat := 5569 :: innerTail269_12
def innerCandidates269_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_11_checked : candidates edge 4 1747 5569 innerTail269_12 best.bound = innerCandidates269_11 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_11 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_11 best = best := by
  rw [innerTail269_11, scan, if_neg (by decide), innerCandidates269_11_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_12
    (if 3 ≤ innerCandidates269_11.length then search edge 3 [1747, 5569] innerCandidates269_11 7316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_12

def innerTail269_10 : List Nat := 4993 :: innerTail269_11
def innerCandidates269_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_10_checked : candidates edge 4 1747 4993 innerTail269_11 best.bound = innerCandidates269_10 := by
  unfold candidates
  have row : edge 4993 = SearchData.row4993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_10 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_10 best = best := by
  rw [innerTail269_10, scan, if_neg (by decide), innerCandidates269_10_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_11
    (if 3 ≤ innerCandidates269_10.length then search edge 3 [1747, 4993] innerCandidates269_10 6740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_11

def innerTail269_9 : List Nat := 4231 :: innerTail269_10
def innerCandidates269_9 : List Nat := [6037]
set_option maxRecDepth 4096 in
theorem innerCandidates269_9_checked : candidates edge 4 1747 4231 innerTail269_10 best.bound = innerCandidates269_9 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_9 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_9 best = best := by
  rw [innerTail269_9, scan, if_neg (by decide), innerCandidates269_9_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_10
    (if 3 ≤ innerCandidates269_9.length then search edge 3 [1747, 4231] innerCandidates269_9 5978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_10

def innerTail269_8 : List Nat := 4099 :: innerTail269_9
def innerCandidates269_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_8_checked : candidates edge 4 1747 4099 innerTail269_9 best.bound = innerCandidates269_8 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_8 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_8 best = best := by
  rw [innerTail269_8, scan, if_neg (by decide), innerCandidates269_8_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_9
    (if 3 ≤ innerCandidates269_8.length then search edge 3 [1747, 4099] innerCandidates269_8 5846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_9

def innerTail269_7 : List Nat := 3889 :: innerTail269_8
def innerCandidates269_7 : List Nat := [7477]
set_option maxRecDepth 4096 in
theorem innerCandidates269_7_checked : candidates edge 4 1747 3889 innerTail269_8 best.bound = innerCandidates269_7 := by
  unfold candidates
  have row : edge 3889 = SearchData.row3889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_7 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_7 best = best := by
  rw [innerTail269_7, scan, if_neg (by decide), innerCandidates269_7_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_8
    (if 3 ≤ innerCandidates269_7.length then search edge 3 [1747, 3889] innerCandidates269_7 5636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_8

def innerTail269_6 : List Nat := 3769 :: innerTail269_7
def innerCandidates269_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_6_checked : candidates edge 4 1747 3769 innerTail269_7 best.bound = innerCandidates269_6 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_6 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_6 best = best := by
  rw [innerTail269_6, scan, if_neg (by decide), innerCandidates269_6_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_7
    (if 3 ≤ innerCandidates269_6.length then search edge 3 [1747, 3769] innerCandidates269_6 5516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_7

def innerTail269_5 : List Nat := 3691 :: innerTail269_6
def innerCandidates269_5 : List Nat := [7477]
set_option maxRecDepth 4096 in
theorem innerCandidates269_5_checked : candidates edge 4 1747 3691 innerTail269_6 best.bound = innerCandidates269_5 := by
  unfold candidates
  have row : edge 3691 = SearchData.row3691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_5 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_5 best = best := by
  rw [innerTail269_5, scan, if_neg (by decide), innerCandidates269_5_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_6
    (if 3 ≤ innerCandidates269_5.length then search edge 3 [1747, 3691] innerCandidates269_5 5438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_6

def innerTail269_4 : List Nat := 2671 :: innerTail269_5
def innerCandidates269_4 : List Nat := [3769, 4231, 7129]
set_option maxRecDepth 4096 in
theorem innerCandidates269_4_checked : candidates edge 4 1747 2671 innerTail269_5 best.bound = innerCandidates269_4 := by
  unfold candidates
  have row : edge 2671 = SearchData.row2671.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild269_4 : search edge 3 [1747, 2671] innerCandidates269_4 4418 best = best := by decide

theorem innerScan269_4 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_4 best = best := by
  rw [innerTail269_4, scan, if_neg (by decide), innerCandidates269_4_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_5
    (if 3 ≤ innerCandidates269_4.length then search edge 3 [1747, 2671] innerCandidates269_4 4418 best else best) = best
  rw [if_pos (by decide), innerChild269_4]
  exact innerScan269_5

def innerTail269_3 : List Nat := 2179 :: innerTail269_4
def innerCandidates269_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates269_3_checked : candidates edge 4 1747 2179 innerTail269_4 best.bound = innerCandidates269_3 := by
  unfold candidates
  have row : edge 2179 = PrimePairCache.smallRow2179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_3 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_3 best = best := by
  rw [innerTail269_3, scan, if_neg (by decide), innerCandidates269_3_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_4
    (if 3 ≤ innerCandidates269_3.length then search edge 3 [1747, 2179] innerCandidates269_3 3926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_4

def innerTail269_2 : List Nat := 2053 :: innerTail269_3
def innerCandidates269_2 : List Nat := [6421]
set_option maxRecDepth 4096 in
theorem innerCandidates269_2_checked : candidates edge 4 1747 2053 innerTail269_3 best.bound = innerCandidates269_2 := by
  unfold candidates
  have row : edge 2053 = PrimePairCache.smallRow2053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_2 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_2 best = best := by
  rw [innerTail269_2, scan, if_neg (by decide), innerCandidates269_2_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_3
    (if 3 ≤ innerCandidates269_2.length then search edge 3 [1747, 2053] innerCandidates269_2 3800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_3

def innerTail269_1 : List Nat := 1933 :: innerTail269_2
def innerCandidates269_1 : List Nat := [7639]
set_option maxRecDepth 4096 in
theorem innerCandidates269_1_checked : candidates edge 4 1747 1933 innerTail269_2 best.bound = innerCandidates269_1 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_1 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_1 best = best := by
  rw [innerTail269_1, scan, if_neg (by decide), innerCandidates269_1_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_2
    (if 3 ≤ innerCandidates269_1.length then search edge 3 [1747, 1933] innerCandidates269_1 3680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_2

def innerTail269_0 : List Nat := 1873 :: innerTail269_1
def innerCandidates269_0 : List Nat := [5569, 8011]
set_option maxRecDepth 4096 in
theorem innerCandidates269_0_checked : candidates edge 4 1747 1873 innerTail269_1 best.bound = innerCandidates269_0 := by
  unfold candidates
  have row : edge 1873 = PrimePairCache.smallRow1873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan269_0 : scan edge (search edge 3) 4 [1747] 1747 innerTail269_0 best = best := by
  rw [innerTail269_0, scan, if_neg (by decide), innerCandidates269_0_checked]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_1
    (if 3 ≤ innerCandidates269_0.length then search edge 3 [1747, 1873] innerCandidates269_0 3620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan269_1

theorem rootBranch269 : search edge 4 [1747] rootCandidates269 1747 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1747] 1747 innerTail269_0 best = best
  exact innerScan269_0

def rootCandidates268 : List Nat := [1753, 2053, 2203, 2281, 2287, 2347, 2851, 2887, 2953, 2971, 3307, 3313, 3373, 3607, 3793, 3919, 4003, 5227, 5479, 5557, 5581, 6397, 7489, 7573, 7993, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates268_checked : candidates edge 5 0 1741 tail269 best.bound = rootCandidates268 := by
  unfold candidates
  have row : edge 1741 = PrimePairCache.smallRow1741.contains := by funext q; rfl
  rw [row]
  decide

def innerTail268_26 : List Nat := []
theorem innerScan268_26 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_26 best = best := rfl

def innerTail268_25 : List Nat := 8287 :: innerTail268_26
def innerCandidates268_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_25_checked : candidates edge 4 1741 8287 innerTail268_26 best.bound = innerCandidates268_25 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_25 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_25 best = best := by
  rw [innerTail268_25, scan, if_neg (by decide), innerCandidates268_25_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_26
    (if 3 ≤ innerCandidates268_25.length then search edge 3 [1741, 8287] innerCandidates268_25 10028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_26

def innerTail268_24 : List Nat := 7993 :: innerTail268_25
def innerCandidates268_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_24_checked : candidates edge 4 1741 7993 innerTail268_25 best.bound = innerCandidates268_24 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_24 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_24 best = best := by
  rw [innerTail268_24, scan, if_neg (by decide), innerCandidates268_24_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_25
    (if 3 ≤ innerCandidates268_24.length then search edge 3 [1741, 7993] innerCandidates268_24 9734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_25

def innerTail268_23 : List Nat := 7573 :: innerTail268_24
def innerCandidates268_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_23_checked : candidates edge 4 1741 7573 innerTail268_24 best.bound = innerCandidates268_23 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_23 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_23 best = best := by
  rw [innerTail268_23, scan, if_neg (by decide), innerCandidates268_23_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_24
    (if 3 ≤ innerCandidates268_23.length then search edge 3 [1741, 7573] innerCandidates268_23 9314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_24

def innerTail268_22 : List Nat := 7489 :: innerTail268_23
def innerCandidates268_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_22_checked : candidates edge 4 1741 7489 innerTail268_23 best.bound = innerCandidates268_22 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_22 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_22 best = best := by
  rw [innerTail268_22, scan, if_neg (by decide), innerCandidates268_22_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_23
    (if 3 ≤ innerCandidates268_22.length then search edge 3 [1741, 7489] innerCandidates268_22 9230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_23

def innerTail268_21 : List Nat := 6397 :: innerTail268_22
def innerCandidates268_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_21_checked : candidates edge 4 1741 6397 innerTail268_22 best.bound = innerCandidates268_21 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_21 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_21 best = best := by
  rw [innerTail268_21, scan, if_neg (by decide), innerCandidates268_21_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_22
    (if 3 ≤ innerCandidates268_21.length then search edge 3 [1741, 6397] innerCandidates268_21 8138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_22

def innerTail268_20 : List Nat := 5581 :: innerTail268_21
def innerCandidates268_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_20_checked : candidates edge 4 1741 5581 innerTail268_21 best.bound = innerCandidates268_20 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_20 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_20 best = best := by
  rw [innerTail268_20, scan, if_neg (by decide), innerCandidates268_20_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_21
    (if 3 ≤ innerCandidates268_20.length then search edge 3 [1741, 5581] innerCandidates268_20 7322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_21

def innerTail268_19 : List Nat := 5557 :: innerTail268_20
def innerCandidates268_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_19_checked : candidates edge 4 1741 5557 innerTail268_20 best.bound = innerCandidates268_19 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_19 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_19 best = best := by
  rw [innerTail268_19, scan, if_neg (by decide), innerCandidates268_19_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_20
    (if 3 ≤ innerCandidates268_19.length then search edge 3 [1741, 5557] innerCandidates268_19 7298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_20

def innerTail268_18 : List Nat := 5479 :: innerTail268_19
def innerCandidates268_18 : List Nat := [5581]
set_option maxRecDepth 4096 in
theorem innerCandidates268_18_checked : candidates edge 4 1741 5479 innerTail268_19 best.bound = innerCandidates268_18 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_18 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_18 best = best := by
  rw [innerTail268_18, scan, if_neg (by decide), innerCandidates268_18_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_19
    (if 3 ≤ innerCandidates268_18.length then search edge 3 [1741, 5479] innerCandidates268_18 7220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_19

def innerTail268_17 : List Nat := 5227 :: innerTail268_18
def innerCandidates268_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_17_checked : candidates edge 4 1741 5227 innerTail268_18 best.bound = innerCandidates268_17 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_17 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_17 best = best := by
  rw [innerTail268_17, scan, if_neg (by decide), innerCandidates268_17_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_18
    (if 3 ≤ innerCandidates268_17.length then search edge 3 [1741, 5227] innerCandidates268_17 6968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_18

def innerTail268_16 : List Nat := 4003 :: innerTail268_17
def innerCandidates268_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_16_checked : candidates edge 4 1741 4003 innerTail268_17 best.bound = innerCandidates268_16 := by
  unfold candidates
  have row : edge 4003 = SearchData.row4003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_16 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_16 best = best := by
  rw [innerTail268_16, scan, if_neg (by decide), innerCandidates268_16_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_17
    (if 3 ≤ innerCandidates268_16.length then search edge 3 [1741, 4003] innerCandidates268_16 5744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_17

def innerTail268_15 : List Nat := 3919 :: innerTail268_16
def innerCandidates268_15 : List Nat := [6397, 7489]
set_option maxRecDepth 4096 in
theorem innerCandidates268_15_checked : candidates edge 4 1741 3919 innerTail268_16 best.bound = innerCandidates268_15 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_15 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_15 best = best := by
  rw [innerTail268_15, scan, if_neg (by decide), innerCandidates268_15_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_16
    (if 3 ≤ innerCandidates268_15.length then search edge 3 [1741, 3919] innerCandidates268_15 5660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_16

def innerTail268_14 : List Nat := 3793 :: innerTail268_15
def innerCandidates268_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_14_checked : candidates edge 4 1741 3793 innerTail268_15 best.bound = innerCandidates268_14 := by
  unfold candidates
  have row : edge 3793 = SearchData.row3793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_14 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_14 best = best := by
  rw [innerTail268_14, scan, if_neg (by decide), innerCandidates268_14_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_15
    (if 3 ≤ innerCandidates268_14.length then search edge 3 [1741, 3793] innerCandidates268_14 5534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_15

def innerTail268_13 : List Nat := 3607 :: innerTail268_14
def innerCandidates268_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_13_checked : candidates edge 4 1741 3607 innerTail268_14 best.bound = innerCandidates268_13 := by
  unfold candidates
  have row : edge 3607 = SearchData.row3607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_13 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_13 best = best := by
  rw [innerTail268_13, scan, if_neg (by decide), innerCandidates268_13_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_14
    (if 3 ≤ innerCandidates268_13.length then search edge 3 [1741, 3607] innerCandidates268_13 5348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_14

def innerTail268_12 : List Nat := 3373 :: innerTail268_13
def innerCandidates268_12 : List Nat := [5557]
set_option maxRecDepth 4096 in
theorem innerCandidates268_12_checked : candidates edge 4 1741 3373 innerTail268_13 best.bound = innerCandidates268_12 := by
  unfold candidates
  have row : edge 3373 = SearchData.row3373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_12 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_12 best = best := by
  rw [innerTail268_12, scan, if_neg (by decide), innerCandidates268_12_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_13
    (if 3 ≤ innerCandidates268_12.length then search edge 3 [1741, 3373] innerCandidates268_12 5114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_13

def innerTail268_11 : List Nat := 3313 :: innerTail268_12
def innerCandidates268_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_11_checked : candidates edge 4 1741 3313 innerTail268_12 best.bound = innerCandidates268_11 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_11 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_11 best = best := by
  rw [innerTail268_11, scan, if_neg (by decide), innerCandidates268_11_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_12
    (if 3 ≤ innerCandidates268_11.length then search edge 3 [1741, 3313] innerCandidates268_11 5054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_12

def innerTail268_10 : List Nat := 3307 :: innerTail268_11
def innerCandidates268_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_10_checked : candidates edge 4 1741 3307 innerTail268_11 best.bound = innerCandidates268_10 := by
  unfold candidates
  have row : edge 3307 = SearchData.row3307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_10 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_10 best = best := by
  rw [innerTail268_10, scan, if_neg (by decide), innerCandidates268_10_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_11
    (if 3 ≤ innerCandidates268_10.length then search edge 3 [1741, 3307] innerCandidates268_10 5048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_11

def innerTail268_9 : List Nat := 2971 :: innerTail268_10
def innerCandidates268_9 : List Nat := [5557, 7573]
set_option maxRecDepth 4096 in
theorem innerCandidates268_9_checked : candidates edge 4 1741 2971 innerTail268_10 best.bound = innerCandidates268_9 := by
  unfold candidates
  have row : edge 2971 = SearchData.row2971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_9 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_9 best = best := by
  rw [innerTail268_9, scan, if_neg (by decide), innerCandidates268_9_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_10
    (if 3 ≤ innerCandidates268_9.length then search edge 3 [1741, 2971] innerCandidates268_9 4712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_10

def innerTail268_8 : List Nat := 2953 :: innerTail268_9
def innerCandidates268_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_8_checked : candidates edge 4 1741 2953 innerTail268_9 best.bound = innerCandidates268_8 := by
  unfold candidates
  have row : edge 2953 = SearchData.row2953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_8 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_8 best = best := by
  rw [innerTail268_8, scan, if_neg (by decide), innerCandidates268_8_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_9
    (if 3 ≤ innerCandidates268_8.length then search edge 3 [1741, 2953] innerCandidates268_8 4694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_9

def innerTail268_7 : List Nat := 2887 :: innerTail268_8
def innerCandidates268_7 : List Nat := [7573]
set_option maxRecDepth 4096 in
theorem innerCandidates268_7_checked : candidates edge 4 1741 2887 innerTail268_8 best.bound = innerCandidates268_7 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_7 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_7 best = best := by
  rw [innerTail268_7, scan, if_neg (by decide), innerCandidates268_7_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_8
    (if 3 ≤ innerCandidates268_7.length then search edge 3 [1741, 2887] innerCandidates268_7 4628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_8

def innerTail268_6 : List Nat := 2851 :: innerTail268_7
def innerCandidates268_6 : List Nat := [3313]
set_option maxRecDepth 4096 in
theorem innerCandidates268_6_checked : candidates edge 4 1741 2851 innerTail268_7 best.bound = innerCandidates268_6 := by
  unfold candidates
  have row : edge 2851 = SearchData.row2851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_6 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_6 best = best := by
  rw [innerTail268_6, scan, if_neg (by decide), innerCandidates268_6_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_7
    (if 3 ≤ innerCandidates268_6.length then search edge 3 [1741, 2851] innerCandidates268_6 4592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_7

def innerTail268_5 : List Nat := 2347 :: innerTail268_6
def innerCandidates268_5 : List Nat := [2851]
set_option maxRecDepth 4096 in
theorem innerCandidates268_5_checked : candidates edge 4 1741 2347 innerTail268_6 best.bound = innerCandidates268_5 := by
  unfold candidates
  have row : edge 2347 = PrimePairCache.smallRow2347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_5 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_5 best = best := by
  rw [innerTail268_5, scan, if_neg (by decide), innerCandidates268_5_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_6
    (if 3 ≤ innerCandidates268_5.length then search edge 3 [1741, 2347] innerCandidates268_5 4088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_6

def innerTail268_4 : List Nat := 2287 :: innerTail268_5
def innerCandidates268_4 : List Nat := [3607]
set_option maxRecDepth 4096 in
theorem innerCandidates268_4_checked : candidates edge 4 1741 2287 innerTail268_5 best.bound = innerCandidates268_4 := by
  unfold candidates
  have row : edge 2287 = PrimePairCache.smallRow2287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_4 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_4 best = best := by
  rw [innerTail268_4, scan, if_neg (by decide), innerCandidates268_4_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_5
    (if 3 ≤ innerCandidates268_4.length then search edge 3 [1741, 2287] innerCandidates268_4 4028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_5

def innerTail268_3 : List Nat := 2281 :: innerTail268_4
def innerCandidates268_3 : List Nat := [2887]
set_option maxRecDepth 4096 in
theorem innerCandidates268_3_checked : candidates edge 4 1741 2281 innerTail268_4 best.bound = innerCandidates268_3 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_3 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_3 best = best := by
  rw [innerTail268_3, scan, if_neg (by decide), innerCandidates268_3_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_4
    (if 3 ≤ innerCandidates268_3.length then search edge 3 [1741, 2281] innerCandidates268_3 4022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_4

def innerTail268_2 : List Nat := 2203 :: innerTail268_3
def innerCandidates268_2 : List Nat := [2887, 5557]
set_option maxRecDepth 4096 in
theorem innerCandidates268_2_checked : candidates edge 4 1741 2203 innerTail268_3 best.bound = innerCandidates268_2 := by
  unfold candidates
  have row : edge 2203 = PrimePairCache.smallRow2203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_2 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_2 best = best := by
  rw [innerTail268_2, scan, if_neg (by decide), innerCandidates268_2_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_3
    (if 3 ≤ innerCandidates268_2.length then search edge 3 [1741, 2203] innerCandidates268_2 3944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_3

def innerTail268_1 : List Nat := 2053 :: innerTail268_2
def innerCandidates268_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_1_checked : candidates edge 4 1741 2053 innerTail268_2 best.bound = innerCandidates268_1 := by
  unfold candidates
  have row : edge 2053 = PrimePairCache.smallRow2053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_1 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_1 best = best := by
  rw [innerTail268_1, scan, if_neg (by decide), innerCandidates268_1_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_2
    (if 3 ≤ innerCandidates268_1.length then search edge 3 [1741, 2053] innerCandidates268_1 3794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_2

def innerTail268_0 : List Nat := 1753 :: innerTail268_1
def innerCandidates268_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates268_0_checked : candidates edge 4 1741 1753 innerTail268_1 best.bound = innerCandidates268_0 := by
  unfold candidates
  have row : edge 1753 = PrimePairCache.smallRow1753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan268_0 : scan edge (search edge 3) 4 [1741] 1741 innerTail268_0 best = best := by
  rw [innerTail268_0, scan, if_neg (by decide), innerCandidates268_0_checked]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_1
    (if 3 ≤ innerCandidates268_0.length then search edge 3 [1741, 1753] innerCandidates268_0 3494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan268_1

theorem rootBranch268 : search edge 4 [1741] rootCandidates268 1741 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1741] 1741 innerTail268_0 best = best
  exact innerScan268_0

end ProjectEulerSolutions.PrimePairUpper.Search
