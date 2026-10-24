import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates275 : List Nat := [2089, 2377, 2467, 2503, 2521, 2683, 2887, 3853, 3907, 4111, 4243, 4813, 4831, 5023, 5953, 6133, 6343, 6421, 6547, 6661, 7393, 8011]
set_option maxRecDepth 4096 in
theorem rootCandidates275_checked : candidates edge 5 0 1789 tail276 best.bound = rootCandidates275 := by
  unfold candidates
  have row : edge 1789 = PrimePairCache.smallRow1789.contains := by funext q; rfl
  rw [row]
  decide

def innerTail275_22 : List Nat := []
theorem innerScan275_22 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_22 best = best := rfl

def innerTail275_21 : List Nat := 8011 :: innerTail275_22
def innerCandidates275_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_21_checked : candidates edge 4 1789 8011 innerTail275_22 best.bound = innerCandidates275_21 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_21 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_21 best = best := by
  rw [innerTail275_21, scan, if_neg (by decide), innerCandidates275_21_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_22
    (if 3 ≤ innerCandidates275_21.length then search edge 3 [1789, 8011] innerCandidates275_21 9800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_22

def innerTail275_20 : List Nat := 7393 :: innerTail275_21
def innerCandidates275_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_20_checked : candidates edge 4 1789 7393 innerTail275_21 best.bound = innerCandidates275_20 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_20 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_20 best = best := by
  rw [innerTail275_20, scan, if_neg (by decide), innerCandidates275_20_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_21
    (if 3 ≤ innerCandidates275_20.length then search edge 3 [1789, 7393] innerCandidates275_20 9182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_21

def innerTail275_19 : List Nat := 6661 :: innerTail275_20
def innerCandidates275_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_19_checked : candidates edge 4 1789 6661 innerTail275_20 best.bound = innerCandidates275_19 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_19 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_19 best = best := by
  rw [innerTail275_19, scan, if_neg (by decide), innerCandidates275_19_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_20
    (if 3 ≤ innerCandidates275_19.length then search edge 3 [1789, 6661] innerCandidates275_19 8450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_20

def innerTail275_18 : List Nat := 6547 :: innerTail275_19
def innerCandidates275_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_18_checked : candidates edge 4 1789 6547 innerTail275_19 best.bound = innerCandidates275_18 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_18 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_18 best = best := by
  rw [innerTail275_18, scan, if_neg (by decide), innerCandidates275_18_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_19
    (if 3 ≤ innerCandidates275_18.length then search edge 3 [1789, 6547] innerCandidates275_18 8336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_19

def innerTail275_17 : List Nat := 6421 :: innerTail275_18
def innerCandidates275_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_17_checked : candidates edge 4 1789 6421 innerTail275_18 best.bound = innerCandidates275_17 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_17 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_17 best = best := by
  rw [innerTail275_17, scan, if_neg (by decide), innerCandidates275_17_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_18
    (if 3 ≤ innerCandidates275_17.length then search edge 3 [1789, 6421] innerCandidates275_17 8210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_18

def innerTail275_16 : List Nat := 6343 :: innerTail275_17
def innerCandidates275_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_16_checked : candidates edge 4 1789 6343 innerTail275_17 best.bound = innerCandidates275_16 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_16 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_16 best = best := by
  rw [innerTail275_16, scan, if_neg (by decide), innerCandidates275_16_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_17
    (if 3 ≤ innerCandidates275_16.length then search edge 3 [1789, 6343] innerCandidates275_16 8132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_17

def innerTail275_15 : List Nat := 6133 :: innerTail275_16
def innerCandidates275_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_15_checked : candidates edge 4 1789 6133 innerTail275_16 best.bound = innerCandidates275_15 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_15 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_15 best = best := by
  rw [innerTail275_15, scan, if_neg (by decide), innerCandidates275_15_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_16
    (if 3 ≤ innerCandidates275_15.length then search edge 3 [1789, 6133] innerCandidates275_15 7922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_16

def innerTail275_14 : List Nat := 5953 :: innerTail275_15
def innerCandidates275_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_14_checked : candidates edge 4 1789 5953 innerTail275_15 best.bound = innerCandidates275_14 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_14 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_14 best = best := by
  rw [innerTail275_14, scan, if_neg (by decide), innerCandidates275_14_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_15
    (if 3 ≤ innerCandidates275_14.length then search edge 3 [1789, 5953] innerCandidates275_14 7742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_15

def innerTail275_13 : List Nat := 5023 :: innerTail275_14
def innerCandidates275_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_13_checked : candidates edge 4 1789 5023 innerTail275_14 best.bound = innerCandidates275_13 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_13 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_13 best = best := by
  rw [innerTail275_13, scan, if_neg (by decide), innerCandidates275_13_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_14
    (if 3 ≤ innerCandidates275_13.length then search edge 3 [1789, 5023] innerCandidates275_13 6812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_14

def innerTail275_12 : List Nat := 4831 :: innerTail275_13
def innerCandidates275_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_12_checked : candidates edge 4 1789 4831 innerTail275_13 best.bound = innerCandidates275_12 := by
  unfold candidates
  have row : edge 4831 = SearchData.row4831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_12 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_12 best = best := by
  rw [innerTail275_12, scan, if_neg (by decide), innerCandidates275_12_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_13
    (if 3 ≤ innerCandidates275_12.length then search edge 3 [1789, 4831] innerCandidates275_12 6620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_13

def innerTail275_11 : List Nat := 4813 :: innerTail275_12
def innerCandidates275_11 : List Nat := [6661]
set_option maxRecDepth 4096 in
theorem innerCandidates275_11_checked : candidates edge 4 1789 4813 innerTail275_12 best.bound = innerCandidates275_11 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_11 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_11 best = best := by
  rw [innerTail275_11, scan, if_neg (by decide), innerCandidates275_11_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_12
    (if 3 ≤ innerCandidates275_11.length then search edge 3 [1789, 4813] innerCandidates275_11 6602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_12

def innerTail275_10 : List Nat := 4243 :: innerTail275_11
def innerCandidates275_10 : List Nat := [4831]
set_option maxRecDepth 4096 in
theorem innerCandidates275_10_checked : candidates edge 4 1789 4243 innerTail275_11 best.bound = innerCandidates275_10 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_10 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_10 best = best := by
  rw [innerTail275_10, scan, if_neg (by decide), innerCandidates275_10_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_11
    (if 3 ≤ innerCandidates275_10.length then search edge 3 [1789, 4243] innerCandidates275_10 6032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_11

def innerTail275_9 : List Nat := 4111 :: innerTail275_10
def innerCandidates275_9 : List Nat := [6661]
set_option maxRecDepth 4096 in
theorem innerCandidates275_9_checked : candidates edge 4 1789 4111 innerTail275_10 best.bound = innerCandidates275_9 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_9 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_9 best = best := by
  rw [innerTail275_9, scan, if_neg (by decide), innerCandidates275_9_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_10
    (if 3 ≤ innerCandidates275_9.length then search edge 3 [1789, 4111] innerCandidates275_9 5900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_10

def innerTail275_8 : List Nat := 3907 :: innerTail275_9
def innerCandidates275_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_8_checked : candidates edge 4 1789 3907 innerTail275_9 best.bound = innerCandidates275_8 := by
  unfold candidates
  have row : edge 3907 = SearchData.row3907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_8 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_8 best = best := by
  rw [innerTail275_8, scan, if_neg (by decide), innerCandidates275_8_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_9
    (if 3 ≤ innerCandidates275_8.length then search edge 3 [1789, 3907] innerCandidates275_8 5696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_9

def innerTail275_7 : List Nat := 3853 :: innerTail275_8
def innerCandidates275_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_7_checked : candidates edge 4 1789 3853 innerTail275_8 best.bound = innerCandidates275_7 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_7 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_7 best = best := by
  rw [innerTail275_7, scan, if_neg (by decide), innerCandidates275_7_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_8
    (if 3 ≤ innerCandidates275_7.length then search edge 3 [1789, 3853] innerCandidates275_7 5642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_8

def innerTail275_6 : List Nat := 2887 :: innerTail275_7
def innerCandidates275_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_6_checked : candidates edge 4 1789 2887 innerTail275_7 best.bound = innerCandidates275_6 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_6 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_6 best = best := by
  rw [innerTail275_6, scan, if_neg (by decide), innerCandidates275_6_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_7
    (if 3 ≤ innerCandidates275_6.length then search edge 3 [1789, 2887] innerCandidates275_6 4676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_7

def innerTail275_5 : List Nat := 2683 :: innerTail275_6
def innerCandidates275_5 : List Nat := [5023]
set_option maxRecDepth 4096 in
theorem innerCandidates275_5_checked : candidates edge 4 1789 2683 innerTail275_6 best.bound = innerCandidates275_5 := by
  unfold candidates
  have row : edge 2683 = SearchData.row2683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_5 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_5 best = best := by
  rw [innerTail275_5, scan, if_neg (by decide), innerCandidates275_5_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_6
    (if 3 ≤ innerCandidates275_5.length then search edge 3 [1789, 2683] innerCandidates275_5 4472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_6

def innerTail275_4 : List Nat := 2521 :: innerTail275_5
def innerCandidates275_4 : List Nat := [6343]
set_option maxRecDepth 4096 in
theorem innerCandidates275_4_checked : candidates edge 4 1789 2521 innerTail275_5 best.bound = innerCandidates275_4 := by
  unfold candidates
  have row : edge 2521 = PrimePairCache.smallRow2521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_4 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_4 best = best := by
  rw [innerTail275_4, scan, if_neg (by decide), innerCandidates275_4_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_5
    (if 3 ≤ innerCandidates275_4.length then search edge 3 [1789, 2521] innerCandidates275_4 4310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_5

def innerTail275_3 : List Nat := 2503 :: innerTail275_4
def innerCandidates275_3 : List Nat := [6421, 7393]
set_option maxRecDepth 4096 in
theorem innerCandidates275_3_checked : candidates edge 4 1789 2503 innerTail275_4 best.bound = innerCandidates275_3 := by
  unfold candidates
  have row : edge 2503 = PrimePairCache.smallRow2503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_3 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_3 best = best := by
  rw [innerTail275_3, scan, if_neg (by decide), innerCandidates275_3_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_4
    (if 3 ≤ innerCandidates275_3.length then search edge 3 [1789, 2503] innerCandidates275_3 4292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_4

def innerTail275_2 : List Nat := 2467 :: innerTail275_3
def innerCandidates275_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_2_checked : candidates edge 4 1789 2467 innerTail275_3 best.bound = innerCandidates275_2 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_2 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_2 best = best := by
  rw [innerTail275_2, scan, if_neg (by decide), innerCandidates275_2_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_3
    (if 3 ≤ innerCandidates275_2.length then search edge 3 [1789, 2467] innerCandidates275_2 4256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_3

def innerTail275_1 : List Nat := 2377 :: innerTail275_2
def innerCandidates275_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_1_checked : candidates edge 4 1789 2377 innerTail275_2 best.bound = innerCandidates275_1 := by
  unfold candidates
  have row : edge 2377 = PrimePairCache.smallRow2377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_1 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_1 best = best := by
  rw [innerTail275_1, scan, if_neg (by decide), innerCandidates275_1_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_2
    (if 3 ≤ innerCandidates275_1.length then search edge 3 [1789, 2377] innerCandidates275_1 4166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_2

def innerTail275_0 : List Nat := 2089 :: innerTail275_1
def innerCandidates275_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates275_0_checked : candidates edge 4 1789 2089 innerTail275_1 best.bound = innerCandidates275_0 := by
  unfold candidates
  have row : edge 2089 = PrimePairCache.smallRow2089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan275_0 : scan edge (search edge 3) 4 [1789] 1789 innerTail275_0 best = best := by
  rw [innerTail275_0, scan, if_neg (by decide), innerCandidates275_0_checked]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_1
    (if 3 ≤ innerCandidates275_0.length then search edge 3 [1789, 2089] innerCandidates275_0 3878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan275_1

theorem rootBranch275 : search edge 4 [1789] rootCandidates275 1789 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1789] 1789 innerTail275_0 best = best
  exact innerScan275_0

def rootCandidates274 : List Nat := [2447, 3761, 4211, 4271, 4349, 4391, 4919, 4937, 5387, 5591, 5927, 6011, 6269, 6323, 6983, 7547, 7757, 7823, 7901, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates274_checked : candidates edge 5 0 1787 tail275 best.bound = rootCandidates274 := by
  unfold candidates
  have row : edge 1787 = PrimePairCache.smallRow1787.contains := by funext q; rfl
  rw [row]
  decide

def innerTail274_20 : List Nat := []
theorem innerScan274_20 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_20 best = best := rfl

def innerTail274_19 : List Nat := 8387 :: innerTail274_20
def innerCandidates274_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_19_checked : candidates edge 4 1787 8387 innerTail274_20 best.bound = innerCandidates274_19 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_19 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_19 best = best := by
  rw [innerTail274_19, scan, if_neg (by decide), innerCandidates274_19_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_20
    (if 3 ≤ innerCandidates274_19.length then search edge 3 [1787, 8387] innerCandidates274_19 10174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_20

def innerTail274_18 : List Nat := 7901 :: innerTail274_19
def innerCandidates274_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_18_checked : candidates edge 4 1787 7901 innerTail274_19 best.bound = innerCandidates274_18 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_18 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_18 best = best := by
  rw [innerTail274_18, scan, if_neg (by decide), innerCandidates274_18_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_19
    (if 3 ≤ innerCandidates274_18.length then search edge 3 [1787, 7901] innerCandidates274_18 9688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_19

def innerTail274_17 : List Nat := 7823 :: innerTail274_18
def innerCandidates274_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_17_checked : candidates edge 4 1787 7823 innerTail274_18 best.bound = innerCandidates274_17 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_17 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_17 best = best := by
  rw [innerTail274_17, scan, if_neg (by decide), innerCandidates274_17_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_18
    (if 3 ≤ innerCandidates274_17.length then search edge 3 [1787, 7823] innerCandidates274_17 9610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_18

def innerTail274_16 : List Nat := 7757 :: innerTail274_17
def innerCandidates274_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_16_checked : candidates edge 4 1787 7757 innerTail274_17 best.bound = innerCandidates274_16 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_16 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_16 best = best := by
  rw [innerTail274_16, scan, if_neg (by decide), innerCandidates274_16_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_17
    (if 3 ≤ innerCandidates274_16.length then search edge 3 [1787, 7757] innerCandidates274_16 9544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_17

def innerTail274_15 : List Nat := 7547 :: innerTail274_16
def innerCandidates274_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_15_checked : candidates edge 4 1787 7547 innerTail274_16 best.bound = innerCandidates274_15 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_15 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_15 best = best := by
  rw [innerTail274_15, scan, if_neg (by decide), innerCandidates274_15_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_16
    (if 3 ≤ innerCandidates274_15.length then search edge 3 [1787, 7547] innerCandidates274_15 9334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_16

def innerTail274_14 : List Nat := 6983 :: innerTail274_15
def innerCandidates274_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_14_checked : candidates edge 4 1787 6983 innerTail274_15 best.bound = innerCandidates274_14 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_14 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_14 best = best := by
  rw [innerTail274_14, scan, if_neg (by decide), innerCandidates274_14_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_15
    (if 3 ≤ innerCandidates274_14.length then search edge 3 [1787, 6983] innerCandidates274_14 8770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_15

def innerTail274_13 : List Nat := 6323 :: innerTail274_14
def innerCandidates274_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_13_checked : candidates edge 4 1787 6323 innerTail274_14 best.bound = innerCandidates274_13 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_13 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_13 best = best := by
  rw [innerTail274_13, scan, if_neg (by decide), innerCandidates274_13_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_14
    (if 3 ≤ innerCandidates274_13.length then search edge 3 [1787, 6323] innerCandidates274_13 8110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_14

def innerTail274_12 : List Nat := 6269 :: innerTail274_13
def innerCandidates274_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_12_checked : candidates edge 4 1787 6269 innerTail274_13 best.bound = innerCandidates274_12 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_12 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_12 best = best := by
  rw [innerTail274_12, scan, if_neg (by decide), innerCandidates274_12_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_13
    (if 3 ≤ innerCandidates274_12.length then search edge 3 [1787, 6269] innerCandidates274_12 8056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_13

def innerTail274_11 : List Nat := 6011 :: innerTail274_12
def innerCandidates274_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_11_checked : candidates edge 4 1787 6011 innerTail274_12 best.bound = innerCandidates274_11 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_11 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_11 best = best := by
  rw [innerTail274_11, scan, if_neg (by decide), innerCandidates274_11_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_12
    (if 3 ≤ innerCandidates274_11.length then search edge 3 [1787, 6011] innerCandidates274_11 7798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_12

def innerTail274_10 : List Nat := 5927 :: innerTail274_11
def innerCandidates274_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_10_checked : candidates edge 4 1787 5927 innerTail274_11 best.bound = innerCandidates274_10 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_10 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_10 best = best := by
  rw [innerTail274_10, scan, if_neg (by decide), innerCandidates274_10_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_11
    (if 3 ≤ innerCandidates274_10.length then search edge 3 [1787, 5927] innerCandidates274_10 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_11

def innerTail274_9 : List Nat := 5591 :: innerTail274_10
def innerCandidates274_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_9_checked : candidates edge 4 1787 5591 innerTail274_10 best.bound = innerCandidates274_9 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_9 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_9 best = best := by
  rw [innerTail274_9, scan, if_neg (by decide), innerCandidates274_9_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_10
    (if 3 ≤ innerCandidates274_9.length then search edge 3 [1787, 5591] innerCandidates274_9 7378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_10

def innerTail274_8 : List Nat := 5387 :: innerTail274_9
def innerCandidates274_8 : List Nat := [6269]
set_option maxRecDepth 4096 in
theorem innerCandidates274_8_checked : candidates edge 4 1787 5387 innerTail274_9 best.bound = innerCandidates274_8 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_8 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_8 best = best := by
  rw [innerTail274_8, scan, if_neg (by decide), innerCandidates274_8_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_9
    (if 3 ≤ innerCandidates274_8.length then search edge 3 [1787, 5387] innerCandidates274_8 7174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_9

def innerTail274_7 : List Nat := 4937 :: innerTail274_8
def innerCandidates274_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_7_checked : candidates edge 4 1787 4937 innerTail274_8 best.bound = innerCandidates274_7 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_7 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_7 best = best := by
  rw [innerTail274_7, scan, if_neg (by decide), innerCandidates274_7_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_8
    (if 3 ≤ innerCandidates274_7.length then search edge 3 [1787, 4937] innerCandidates274_7 6724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_8

def innerTail274_6 : List Nat := 4919 :: innerTail274_7
def innerCandidates274_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_6_checked : candidates edge 4 1787 4919 innerTail274_7 best.bound = innerCandidates274_6 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_6 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_6 best = best := by
  rw [innerTail274_6, scan, if_neg (by decide), innerCandidates274_6_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_7
    (if 3 ≤ innerCandidates274_6.length then search edge 3 [1787, 4919] innerCandidates274_6 6706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_7

def innerTail274_5 : List Nat := 4391 :: innerTail274_6
def innerCandidates274_5 : List Nat := [7901]
set_option maxRecDepth 4096 in
theorem innerCandidates274_5_checked : candidates edge 4 1787 4391 innerTail274_6 best.bound = innerCandidates274_5 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_5 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_5 best = best := by
  rw [innerTail274_5, scan, if_neg (by decide), innerCandidates274_5_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_6
    (if 3 ≤ innerCandidates274_5.length then search edge 3 [1787, 4391] innerCandidates274_5 6178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_6

def innerTail274_4 : List Nat := 4349 :: innerTail274_5
def innerCandidates274_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_4_checked : candidates edge 4 1787 4349 innerTail274_5 best.bound = innerCandidates274_4 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_4 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_4 best = best := by
  rw [innerTail274_4, scan, if_neg (by decide), innerCandidates274_4_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_5
    (if 3 ≤ innerCandidates274_4.length then search edge 3 [1787, 4349] innerCandidates274_4 6136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_5

def innerTail274_3 : List Nat := 4271 :: innerTail274_4
def innerCandidates274_3 : List Nat := [7823]
set_option maxRecDepth 4096 in
theorem innerCandidates274_3_checked : candidates edge 4 1787 4271 innerTail274_4 best.bound = innerCandidates274_3 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_3 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_3 best = best := by
  rw [innerTail274_3, scan, if_neg (by decide), innerCandidates274_3_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_4
    (if 3 ≤ innerCandidates274_3.length then search edge 3 [1787, 4271] innerCandidates274_3 6058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_4

def innerTail274_2 : List Nat := 4211 :: innerTail274_3
def innerCandidates274_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates274_2_checked : candidates edge 4 1787 4211 innerTail274_3 best.bound = innerCandidates274_2 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_2 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_2 best = best := by
  rw [innerTail274_2, scan, if_neg (by decide), innerCandidates274_2_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_3
    (if 3 ≤ innerCandidates274_2.length then search edge 3 [1787, 4211] innerCandidates274_2 5998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_3

def innerTail274_1 : List Nat := 3761 :: innerTail274_2
def innerCandidates274_1 : List Nat := [4391]
set_option maxRecDepth 4096 in
theorem innerCandidates274_1_checked : candidates edge 4 1787 3761 innerTail274_2 best.bound = innerCandidates274_1 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_1 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_1 best = best := by
  rw [innerTail274_1, scan, if_neg (by decide), innerCandidates274_1_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_2
    (if 3 ≤ innerCandidates274_1.length then search edge 3 [1787, 3761] innerCandidates274_1 5548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_2

def innerTail274_0 : List Nat := 2447 :: innerTail274_1
def innerCandidates274_0 : List Nat := [4937, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates274_0_checked : candidates edge 4 1787 2447 innerTail274_1 best.bound = innerCandidates274_0 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan274_0 : scan edge (search edge 3) 4 [1787] 1787 innerTail274_0 best = best := by
  rw [innerTail274_0, scan, if_neg (by decide), innerCandidates274_0_checked]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_1
    (if 3 ≤ innerCandidates274_0.length then search edge 3 [1787, 2447] innerCandidates274_0 4234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan274_1

theorem rootBranch274 : search edge 4 [1787] rootCandidates274 1787 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1787] 1787 innerTail274_0 best = best
  exact innerScan274_0

def rootCandidates273 : List Nat := [2053, 2707, 2791, 2953, 3643, 3769, 4273, 4639, 5011, 6121, 6211, 6277, 6373, 6823, 7129, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates273_checked : candidates edge 5 0 1783 tail274 best.bound = rootCandidates273 := by
  unfold candidates
  have row : edge 1783 = PrimePairCache.smallRow1783.contains := by funext q; rfl
  rw [row]
  decide

def innerTail273_16 : List Nat := []
theorem innerScan273_16 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_16 best = best := rfl

def innerTail273_15 : List Nat := 8209 :: innerTail273_16
def innerCandidates273_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates273_15_checked : candidates edge 4 1783 8209 innerTail273_16 best.bound = innerCandidates273_15 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_15 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_15 best = best := by
  rw [innerTail273_15, scan, if_neg (by decide), innerCandidates273_15_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_16
    (if 3 ≤ innerCandidates273_15.length then search edge 3 [1783, 8209] innerCandidates273_15 9992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_16

def innerTail273_14 : List Nat := 7129 :: innerTail273_15
def innerCandidates273_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates273_14_checked : candidates edge 4 1783 7129 innerTail273_15 best.bound = innerCandidates273_14 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_14 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_14 best = best := by
  rw [innerTail273_14, scan, if_neg (by decide), innerCandidates273_14_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_15
    (if 3 ≤ innerCandidates273_14.length then search edge 3 [1783, 7129] innerCandidates273_14 8912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_15

def innerTail273_13 : List Nat := 6823 :: innerTail273_14
def innerCandidates273_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates273_13_checked : candidates edge 4 1783 6823 innerTail273_14 best.bound = innerCandidates273_13 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_13 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_13 best = best := by
  rw [innerTail273_13, scan, if_neg (by decide), innerCandidates273_13_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_14
    (if 3 ≤ innerCandidates273_13.length then search edge 3 [1783, 6823] innerCandidates273_13 8606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_14

def innerTail273_12 : List Nat := 6373 :: innerTail273_13
def innerCandidates273_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates273_12_checked : candidates edge 4 1783 6373 innerTail273_13 best.bound = innerCandidates273_12 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_12 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_12 best = best := by
  rw [innerTail273_12, scan, if_neg (by decide), innerCandidates273_12_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_13
    (if 3 ≤ innerCandidates273_12.length then search edge 3 [1783, 6373] innerCandidates273_12 8156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_13

def innerTail273_11 : List Nat := 6277 :: innerTail273_12
def innerCandidates273_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates273_11_checked : candidates edge 4 1783 6277 innerTail273_12 best.bound = innerCandidates273_11 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_11 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_11 best = best := by
  rw [innerTail273_11, scan, if_neg (by decide), innerCandidates273_11_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_12
    (if 3 ≤ innerCandidates273_11.length then search edge 3 [1783, 6277] innerCandidates273_11 8060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_12

def innerTail273_10 : List Nat := 6211 :: innerTail273_11
def innerCandidates273_10 : List Nat := [6277]
set_option maxRecDepth 4096 in
theorem innerCandidates273_10_checked : candidates edge 4 1783 6211 innerTail273_11 best.bound = innerCandidates273_10 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_10 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_10 best = best := by
  rw [innerTail273_10, scan, if_neg (by decide), innerCandidates273_10_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_11
    (if 3 ≤ innerCandidates273_10.length then search edge 3 [1783, 6211] innerCandidates273_10 7994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_11

def innerTail273_9 : List Nat := 6121 :: innerTail273_10
def innerCandidates273_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates273_9_checked : candidates edge 4 1783 6121 innerTail273_10 best.bound = innerCandidates273_9 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_9 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_9 best = best := by
  rw [innerTail273_9, scan, if_neg (by decide), innerCandidates273_9_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_10
    (if 3 ≤ innerCandidates273_9.length then search edge 3 [1783, 6121] innerCandidates273_9 7904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_10

def innerTail273_8 : List Nat := 5011 :: innerTail273_9
def innerCandidates273_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates273_8_checked : candidates edge 4 1783 5011 innerTail273_9 best.bound = innerCandidates273_8 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_8 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_8 best = best := by
  rw [innerTail273_8, scan, if_neg (by decide), innerCandidates273_8_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_9
    (if 3 ≤ innerCandidates273_8.length then search edge 3 [1783, 5011] innerCandidates273_8 6794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_9

def innerTail273_7 : List Nat := 4639 :: innerTail273_8
def innerCandidates273_7 : List Nat := [6121, 6211, 6373]
set_option maxRecDepth 4096 in
theorem innerCandidates273_7_checked : candidates edge 4 1783 4639 innerTail273_8 best.bound = innerCandidates273_7 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild273_7 : search edge 3 [1783, 4639] innerCandidates273_7 6422 best = best := by decide

theorem innerScan273_7 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_7 best = best := by
  rw [innerTail273_7, scan, if_neg (by decide), innerCandidates273_7_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_8
    (if 3 ≤ innerCandidates273_7.length then search edge 3 [1783, 4639] innerCandidates273_7 6422 best else best) = best
  rw [if_pos (by decide), innerChild273_7]
  exact innerScan273_8

def innerTail273_6 : List Nat := 4273 :: innerTail273_7
def innerCandidates273_6 : List Nat := [6373]
set_option maxRecDepth 4096 in
theorem innerCandidates273_6_checked : candidates edge 4 1783 4273 innerTail273_7 best.bound = innerCandidates273_6 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_6 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_6 best = best := by
  rw [innerTail273_6, scan, if_neg (by decide), innerCandidates273_6_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_7
    (if 3 ≤ innerCandidates273_6.length then search edge 3 [1783, 4273] innerCandidates273_6 6056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_7

def innerTail273_5 : List Nat := 3769 :: innerTail273_6
def innerCandidates273_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates273_5_checked : candidates edge 4 1783 3769 innerTail273_6 best.bound = innerCandidates273_5 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_5 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_5 best = best := by
  rw [innerTail273_5, scan, if_neg (by decide), innerCandidates273_5_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_6
    (if 3 ≤ innerCandidates273_5.length then search edge 3 [1783, 3769] innerCandidates273_5 5552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_6

def innerTail273_4 : List Nat := 3643 :: innerTail273_5
def innerCandidates273_4 : List Nat := [4273]
set_option maxRecDepth 4096 in
theorem innerCandidates273_4_checked : candidates edge 4 1783 3643 innerTail273_5 best.bound = innerCandidates273_4 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_4 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_4 best = best := by
  rw [innerTail273_4, scan, if_neg (by decide), innerCandidates273_4_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_5
    (if 3 ≤ innerCandidates273_4.length then search edge 3 [1783, 3643] innerCandidates273_4 5426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_5

def innerTail273_3 : List Nat := 2953 :: innerTail273_4
def innerCandidates273_3 : List Nat := [3643, 4273]
set_option maxRecDepth 4096 in
theorem innerCandidates273_3_checked : candidates edge 4 1783 2953 innerTail273_4 best.bound = innerCandidates273_3 := by
  unfold candidates
  have row : edge 2953 = SearchData.row2953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_3 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_3 best = best := by
  rw [innerTail273_3, scan, if_neg (by decide), innerCandidates273_3_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_4
    (if 3 ≤ innerCandidates273_3.length then search edge 3 [1783, 2953] innerCandidates273_3 4736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_4

def innerTail273_2 : List Nat := 2791 :: innerTail273_3
def innerCandidates273_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates273_2_checked : candidates edge 4 1783 2791 innerTail273_3 best.bound = innerCandidates273_2 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_2 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_2 best = best := by
  rw [innerTail273_2, scan, if_neg (by decide), innerCandidates273_2_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_3
    (if 3 ≤ innerCandidates273_2.length then search edge 3 [1783, 2791] innerCandidates273_2 4574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_3

def innerTail273_1 : List Nat := 2707 :: innerTail273_2
def innerCandidates273_1 : List Nat := [2791, 4273]
set_option maxRecDepth 4096 in
theorem innerCandidates273_1_checked : candidates edge 4 1783 2707 innerTail273_2 best.bound = innerCandidates273_1 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_1 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_1 best = best := by
  rw [innerTail273_1, scan, if_neg (by decide), innerCandidates273_1_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_2
    (if 3 ≤ innerCandidates273_1.length then search edge 3 [1783, 2707] innerCandidates273_1 4490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_2

def innerTail273_0 : List Nat := 2053 :: innerTail273_1
def innerCandidates273_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates273_0_checked : candidates edge 4 1783 2053 innerTail273_1 best.bound = innerCandidates273_0 := by
  unfold candidates
  have row : edge 2053 = PrimePairCache.smallRow2053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan273_0 : scan edge (search edge 3) 4 [1783] 1783 innerTail273_0 best = best := by
  rw [innerTail273_0, scan, if_neg (by decide), innerCandidates273_0_checked]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_1
    (if 3 ≤ innerCandidates273_0.length then search edge 3 [1783, 2053] innerCandidates273_0 3836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan273_1

theorem rootBranch273 : search edge 4 [1783] rootCandidates273 1783 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1783] 1783 innerTail273_0 best = best
  exact innerScan273_0

end ProjectEulerSolutions.PrimePairUpper.Search
