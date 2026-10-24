import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates150 : List Nat := [991, 1783, 1987, 2377, 2749, 3049, 4561, 6043, 6151, 6367, 6571, 6661, 7753, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates150_checked : candidates edge 5 0 883 tail151 best.bound = rootCandidates150 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext q; rfl
  rw [row]
  decide

def innerTail150_14 : List Nat := []
theorem innerScan150_14 : scan edge (search edge 3) 4 [883] 883 innerTail150_14 best = best := rfl

def innerTail150_13 : List Nat := 7993 :: innerTail150_14
def innerCandidates150_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_13_checked : candidates edge 4 883 7993 innerTail150_14 best.bound = innerCandidates150_13 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_13 : scan edge (search edge 3) 4 [883] 883 innerTail150_13 best = best := by
  rw [innerTail150_13, scan, if_neg (by decide), innerCandidates150_13_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_14
    (if 3 ≤ innerCandidates150_13.length then search edge 3 [883, 7993] innerCandidates150_13 8876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_14

def innerTail150_12 : List Nat := 7753 :: innerTail150_13
def innerCandidates150_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_12_checked : candidates edge 4 883 7753 innerTail150_13 best.bound = innerCandidates150_12 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_12 : scan edge (search edge 3) 4 [883] 883 innerTail150_12 best = best := by
  rw [innerTail150_12, scan, if_neg (by decide), innerCandidates150_12_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_13
    (if 3 ≤ innerCandidates150_12.length then search edge 3 [883, 7753] innerCandidates150_12 8636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_13

def innerTail150_11 : List Nat := 6661 :: innerTail150_12
def innerCandidates150_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_11_checked : candidates edge 4 883 6661 innerTail150_12 best.bound = innerCandidates150_11 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_11 : scan edge (search edge 3) 4 [883] 883 innerTail150_11 best = best := by
  rw [innerTail150_11, scan, if_neg (by decide), innerCandidates150_11_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_12
    (if 3 ≤ innerCandidates150_11.length then search edge 3 [883, 6661] innerCandidates150_11 7544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_12

def innerTail150_10 : List Nat := 6571 :: innerTail150_11
def innerCandidates150_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_10_checked : candidates edge 4 883 6571 innerTail150_11 best.bound = innerCandidates150_10 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_10 : scan edge (search edge 3) 4 [883] 883 innerTail150_10 best = best := by
  rw [innerTail150_10, scan, if_neg (by decide), innerCandidates150_10_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_11
    (if 3 ≤ innerCandidates150_10.length then search edge 3 [883, 6571] innerCandidates150_10 7454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_11

def innerTail150_9 : List Nat := 6367 :: innerTail150_10
def innerCandidates150_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_9_checked : candidates edge 4 883 6367 innerTail150_10 best.bound = innerCandidates150_9 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_9 : scan edge (search edge 3) 4 [883] 883 innerTail150_9 best = best := by
  rw [innerTail150_9, scan, if_neg (by decide), innerCandidates150_9_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_10
    (if 3 ≤ innerCandidates150_9.length then search edge 3 [883, 6367] innerCandidates150_9 7250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_10

def innerTail150_8 : List Nat := 6151 :: innerTail150_9
def innerCandidates150_8 : List Nat := [6571]
set_option maxRecDepth 4096 in
theorem innerCandidates150_8_checked : candidates edge 4 883 6151 innerTail150_9 best.bound = innerCandidates150_8 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_8 : scan edge (search edge 3) 4 [883] 883 innerTail150_8 best = best := by
  rw [innerTail150_8, scan, if_neg (by decide), innerCandidates150_8_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_9
    (if 3 ≤ innerCandidates150_8.length then search edge 3 [883, 6151] innerCandidates150_8 7034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_9

def innerTail150_7 : List Nat := 6043 :: innerTail150_8
def innerCandidates150_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_7_checked : candidates edge 4 883 6043 innerTail150_8 best.bound = innerCandidates150_7 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_7 : scan edge (search edge 3) 4 [883] 883 innerTail150_7 best = best := by
  rw [innerTail150_7, scan, if_neg (by decide), innerCandidates150_7_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_8
    (if 3 ≤ innerCandidates150_7.length then search edge 3 [883, 6043] innerCandidates150_7 6926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_8

def innerTail150_6 : List Nat := 4561 :: innerTail150_7
def innerCandidates150_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_6_checked : candidates edge 4 883 4561 innerTail150_7 best.bound = innerCandidates150_6 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_6 : scan edge (search edge 3) 4 [883] 883 innerTail150_6 best = best := by
  rw [innerTail150_6, scan, if_neg (by decide), innerCandidates150_6_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_7
    (if 3 ≤ innerCandidates150_6.length then search edge 3 [883, 4561] innerCandidates150_6 5444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_7

def innerTail150_5 : List Nat := 3049 :: innerTail150_6
def innerCandidates150_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_5_checked : candidates edge 4 883 3049 innerTail150_6 best.bound = innerCandidates150_5 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_5 : scan edge (search edge 3) 4 [883] 883 innerTail150_5 best = best := by
  rw [innerTail150_5, scan, if_neg (by decide), innerCandidates150_5_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_6
    (if 3 ≤ innerCandidates150_5.length then search edge 3 [883, 3049] innerCandidates150_5 3932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_6

def innerTail150_4 : List Nat := 2749 :: innerTail150_5
def innerCandidates150_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_4_checked : candidates edge 4 883 2749 innerTail150_5 best.bound = innerCandidates150_4 := by
  unfold candidates
  have row : edge 2749 = SearchData.row2749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_4 : scan edge (search edge 3) 4 [883] 883 innerTail150_4 best = best := by
  rw [innerTail150_4, scan, if_neg (by decide), innerCandidates150_4_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_5
    (if 3 ≤ innerCandidates150_4.length then search edge 3 [883, 2749] innerCandidates150_4 3632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_5

def innerTail150_3 : List Nat := 2377 :: innerTail150_4
def innerCandidates150_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_3_checked : candidates edge 4 883 2377 innerTail150_4 best.bound = innerCandidates150_3 := by
  unfold candidates
  have row : edge 2377 = PrimePairCache.smallRow2377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_3 : scan edge (search edge 3) 4 [883] 883 innerTail150_3 best = best := by
  rw [innerTail150_3, scan, if_neg (by decide), innerCandidates150_3_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_4
    (if 3 ≤ innerCandidates150_3.length then search edge 3 [883, 2377] innerCandidates150_3 3260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_4

def innerTail150_2 : List Nat := 1987 :: innerTail150_3
def innerCandidates150_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_2_checked : candidates edge 4 883 1987 innerTail150_3 best.bound = innerCandidates150_2 := by
  unfold candidates
  have row : edge 1987 = PrimePairCache.smallRow1987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_2 : scan edge (search edge 3) 4 [883] 883 innerTail150_2 best = best := by
  rw [innerTail150_2, scan, if_neg (by decide), innerCandidates150_2_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_3
    (if 3 ≤ innerCandidates150_2.length then search edge 3 [883, 1987] innerCandidates150_2 2870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_3

def innerTail150_1 : List Nat := 1783 :: innerTail150_2
def innerCandidates150_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_1_checked : candidates edge 4 883 1783 innerTail150_2 best.bound = innerCandidates150_1 := by
  unfold candidates
  have row : edge 1783 = PrimePairCache.smallRow1783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_1 : scan edge (search edge 3) 4 [883] 883 innerTail150_1 best = best := by
  rw [innerTail150_1, scan, if_neg (by decide), innerCandidates150_1_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_2
    (if 3 ≤ innerCandidates150_1.length then search edge 3 [883, 1783] innerCandidates150_1 2666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_2

def innerTail150_0 : List Nat := 991 :: innerTail150_1
def innerCandidates150_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates150_0_checked : candidates edge 4 883 991 innerTail150_1 best.bound = innerCandidates150_0 := by
  unfold candidates
  have row : edge 991 = PrimePairCache.smallRow991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan150_0 : scan edge (search edge 3) 4 [883] 883 innerTail150_0 best = best := by
  rw [innerTail150_0, scan, if_neg (by decide), innerCandidates150_0_checked]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_1
    (if 3 ≤ innerCandidates150_0.length then search edge 3 [883, 991] innerCandidates150_0 1874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan150_1

theorem rootBranch150 : search edge 4 [883] rootCandidates150 883 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [883] 883 innerTail150_0 best = best
  exact innerScan150_0

def rootCandidates149 : List Nat := [953, 983, 1373, 1499, 1667, 1949, 2411, 3011, 3407, 4217, 4229, 4457, 4733, 4931, 5021, 5591, 5801, 6053, 6449, 6473, 6701, 6791, 6827, 6959, 7103, 7253, 7331, 7541, 7607, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates149_checked : candidates edge 5 0 881 tail150 best.bound = rootCandidates149 := by
  unfold candidates
  have row : edge 881 = PrimePairCache.smallRow881.contains := by funext q; rfl
  rw [row]
  decide

def innerTail149_30 : List Nat := []
theorem innerScan149_30 : scan edge (search edge 3) 4 [881] 881 innerTail149_30 best = best := rfl

def innerTail149_29 : List Nat := 8297 :: innerTail149_30
def innerCandidates149_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_29_checked : candidates edge 4 881 8297 innerTail149_30 best.bound = innerCandidates149_29 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_29 : scan edge (search edge 3) 4 [881] 881 innerTail149_29 best = best := by
  rw [innerTail149_29, scan, if_neg (by decide), innerCandidates149_29_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_30
    (if 3 ≤ innerCandidates149_29.length then search edge 3 [881, 8297] innerCandidates149_29 9178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_30

def innerTail149_28 : List Nat := 7607 :: innerTail149_29
def innerCandidates149_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_28_checked : candidates edge 4 881 7607 innerTail149_29 best.bound = innerCandidates149_28 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_28 : scan edge (search edge 3) 4 [881] 881 innerTail149_28 best = best := by
  rw [innerTail149_28, scan, if_neg (by decide), innerCandidates149_28_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_29
    (if 3 ≤ innerCandidates149_28.length then search edge 3 [881, 7607] innerCandidates149_28 8488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_29

def innerTail149_27 : List Nat := 7541 :: innerTail149_28
def innerCandidates149_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_27_checked : candidates edge 4 881 7541 innerTail149_28 best.bound = innerCandidates149_27 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_27 : scan edge (search edge 3) 4 [881] 881 innerTail149_27 best = best := by
  rw [innerTail149_27, scan, if_neg (by decide), innerCandidates149_27_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_28
    (if 3 ≤ innerCandidates149_27.length then search edge 3 [881, 7541] innerCandidates149_27 8422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_28

def innerTail149_26 : List Nat := 7331 :: innerTail149_27
def innerCandidates149_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_26_checked : candidates edge 4 881 7331 innerTail149_27 best.bound = innerCandidates149_26 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_26 : scan edge (search edge 3) 4 [881] 881 innerTail149_26 best = best := by
  rw [innerTail149_26, scan, if_neg (by decide), innerCandidates149_26_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_27
    (if 3 ≤ innerCandidates149_26.length then search edge 3 [881, 7331] innerCandidates149_26 8212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_27

def innerTail149_25 : List Nat := 7253 :: innerTail149_26
def innerCandidates149_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_25_checked : candidates edge 4 881 7253 innerTail149_26 best.bound = innerCandidates149_25 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_25 : scan edge (search edge 3) 4 [881] 881 innerTail149_25 best = best := by
  rw [innerTail149_25, scan, if_neg (by decide), innerCandidates149_25_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_26
    (if 3 ≤ innerCandidates149_25.length then search edge 3 [881, 7253] innerCandidates149_25 8134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_26

def innerTail149_24 : List Nat := 7103 :: innerTail149_25
def innerCandidates149_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_24_checked : candidates edge 4 881 7103 innerTail149_25 best.bound = innerCandidates149_24 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_24 : scan edge (search edge 3) 4 [881] 881 innerTail149_24 best = best := by
  rw [innerTail149_24, scan, if_neg (by decide), innerCandidates149_24_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_25
    (if 3 ≤ innerCandidates149_24.length then search edge 3 [881, 7103] innerCandidates149_24 7984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_25

def innerTail149_23 : List Nat := 6959 :: innerTail149_24
def innerCandidates149_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_23_checked : candidates edge 4 881 6959 innerTail149_24 best.bound = innerCandidates149_23 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_23 : scan edge (search edge 3) 4 [881] 881 innerTail149_23 best = best := by
  rw [innerTail149_23, scan, if_neg (by decide), innerCandidates149_23_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_24
    (if 3 ≤ innerCandidates149_23.length then search edge 3 [881, 6959] innerCandidates149_23 7840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_24

def innerTail149_22 : List Nat := 6827 :: innerTail149_23
def innerCandidates149_22 : List Nat := [7103]
set_option maxRecDepth 4096 in
theorem innerCandidates149_22_checked : candidates edge 4 881 6827 innerTail149_23 best.bound = innerCandidates149_22 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_22 : scan edge (search edge 3) 4 [881] 881 innerTail149_22 best = best := by
  rw [innerTail149_22, scan, if_neg (by decide), innerCandidates149_22_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_23
    (if 3 ≤ innerCandidates149_22.length then search edge 3 [881, 6827] innerCandidates149_22 7708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_23

def innerTail149_21 : List Nat := 6791 :: innerTail149_22
def innerCandidates149_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_21_checked : candidates edge 4 881 6791 innerTail149_22 best.bound = innerCandidates149_21 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_21 : scan edge (search edge 3) 4 [881] 881 innerTail149_21 best = best := by
  rw [innerTail149_21, scan, if_neg (by decide), innerCandidates149_21_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_22
    (if 3 ≤ innerCandidates149_21.length then search edge 3 [881, 6791] innerCandidates149_21 7672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_22

def innerTail149_20 : List Nat := 6701 :: innerTail149_21
def innerCandidates149_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_20_checked : candidates edge 4 881 6701 innerTail149_21 best.bound = innerCandidates149_20 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_20 : scan edge (search edge 3) 4 [881] 881 innerTail149_20 best = best := by
  rw [innerTail149_20, scan, if_neg (by decide), innerCandidates149_20_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_21
    (if 3 ≤ innerCandidates149_20.length then search edge 3 [881, 6701] innerCandidates149_20 7582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_21

def innerTail149_19 : List Nat := 6473 :: innerTail149_20
def innerCandidates149_19 : List Nat := [6701]
set_option maxRecDepth 4096 in
theorem innerCandidates149_19_checked : candidates edge 4 881 6473 innerTail149_20 best.bound = innerCandidates149_19 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_19 : scan edge (search edge 3) 4 [881] 881 innerTail149_19 best = best := by
  rw [innerTail149_19, scan, if_neg (by decide), innerCandidates149_19_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_20
    (if 3 ≤ innerCandidates149_19.length then search edge 3 [881, 6473] innerCandidates149_19 7354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_20

def innerTail149_18 : List Nat := 6449 :: innerTail149_19
def innerCandidates149_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_18_checked : candidates edge 4 881 6449 innerTail149_19 best.bound = innerCandidates149_18 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_18 : scan edge (search edge 3) 4 [881] 881 innerTail149_18 best = best := by
  rw [innerTail149_18, scan, if_neg (by decide), innerCandidates149_18_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_19
    (if 3 ≤ innerCandidates149_18.length then search edge 3 [881, 6449] innerCandidates149_18 7330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_19

def innerTail149_17 : List Nat := 6053 :: innerTail149_18
def innerCandidates149_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_17_checked : candidates edge 4 881 6053 innerTail149_18 best.bound = innerCandidates149_17 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_17 : scan edge (search edge 3) 4 [881] 881 innerTail149_17 best = best := by
  rw [innerTail149_17, scan, if_neg (by decide), innerCandidates149_17_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_18
    (if 3 ≤ innerCandidates149_17.length then search edge 3 [881, 6053] innerCandidates149_17 6934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_18

def innerTail149_16 : List Nat := 5801 :: innerTail149_17
def innerCandidates149_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_16_checked : candidates edge 4 881 5801 innerTail149_17 best.bound = innerCandidates149_16 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_16 : scan edge (search edge 3) 4 [881] 881 innerTail149_16 best = best := by
  rw [innerTail149_16, scan, if_neg (by decide), innerCandidates149_16_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_17
    (if 3 ≤ innerCandidates149_16.length then search edge 3 [881, 5801] innerCandidates149_16 6682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_17

def innerTail149_15 : List Nat := 5591 :: innerTail149_16
def innerCandidates149_15 : List Nat := [6701]
set_option maxRecDepth 4096 in
theorem innerCandidates149_15_checked : candidates edge 4 881 5591 innerTail149_16 best.bound = innerCandidates149_15 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_15 : scan edge (search edge 3) 4 [881] 881 innerTail149_15 best = best := by
  rw [innerTail149_15, scan, if_neg (by decide), innerCandidates149_15_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_16
    (if 3 ≤ innerCandidates149_15.length then search edge 3 [881, 5591] innerCandidates149_15 6472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_16

def innerTail149_14 : List Nat := 5021 :: innerTail149_15
def innerCandidates149_14 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates149_14_checked : candidates edge 4 881 5021 innerTail149_15 best.bound = innerCandidates149_14 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_14 : scan edge (search edge 3) 4 [881] 881 innerTail149_14 best = best := by
  rw [innerTail149_14, scan, if_neg (by decide), innerCandidates149_14_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_15
    (if 3 ≤ innerCandidates149_14.length then search edge 3 [881, 5021] innerCandidates149_14 5902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_15

def innerTail149_13 : List Nat := 4931 :: innerTail149_14
def innerCandidates149_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_13_checked : candidates edge 4 881 4931 innerTail149_14 best.bound = innerCandidates149_13 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_13 : scan edge (search edge 3) 4 [881] 881 innerTail149_13 best = best := by
  rw [innerTail149_13, scan, if_neg (by decide), innerCandidates149_13_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_14
    (if 3 ≤ innerCandidates149_13.length then search edge 3 [881, 4931] innerCandidates149_13 5812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_14

def innerTail149_12 : List Nat := 4733 :: innerTail149_13
def innerCandidates149_12 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates149_12_checked : candidates edge 4 881 4733 innerTail149_13 best.bound = innerCandidates149_12 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_12 : scan edge (search edge 3) 4 [881] 881 innerTail149_12 best = best := by
  rw [innerTail149_12, scan, if_neg (by decide), innerCandidates149_12_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_13
    (if 3 ≤ innerCandidates149_12.length then search edge 3 [881, 4733] innerCandidates149_12 5614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_13

def innerTail149_11 : List Nat := 4457 :: innerTail149_12
def innerCandidates149_11 : List Nat := [5591, 6473, 7331, 7607]
set_option maxRecDepth 4096 in
theorem innerCandidates149_11_checked : candidates edge 4 881 4457 innerTail149_12 best.bound = innerCandidates149_11 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild149_11 : search edge 3 [881, 4457] innerCandidates149_11 5338 best = best := by decide

theorem innerScan149_11 : scan edge (search edge 3) 4 [881] 881 innerTail149_11 best = best := by
  rw [innerTail149_11, scan, if_neg (by decide), innerCandidates149_11_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_12
    (if 3 ≤ innerCandidates149_11.length then search edge 3 [881, 4457] innerCandidates149_11 5338 best else best) = best
  rw [if_pos (by decide), innerChild149_11]
  exact innerScan149_12

def innerTail149_10 : List Nat := 4229 :: innerTail149_11
def innerCandidates149_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_10_checked : candidates edge 4 881 4229 innerTail149_11 best.bound = innerCandidates149_10 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_10 : scan edge (search edge 3) 4 [881] 881 innerTail149_10 best = best := by
  rw [innerTail149_10, scan, if_neg (by decide), innerCandidates149_10_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_11
    (if 3 ≤ innerCandidates149_10.length then search edge 3 [881, 4229] innerCandidates149_10 5110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_11

def innerTail149_9 : List Nat := 4217 :: innerTail149_10
def innerCandidates149_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_9_checked : candidates edge 4 881 4217 innerTail149_10 best.bound = innerCandidates149_9 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_9 : scan edge (search edge 3) 4 [881] 881 innerTail149_9 best = best := by
  rw [innerTail149_9, scan, if_neg (by decide), innerCandidates149_9_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_10
    (if 3 ≤ innerCandidates149_9.length then search edge 3 [881, 4217] innerCandidates149_9 5098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_10

def innerTail149_8 : List Nat := 3407 :: innerTail149_9
def innerCandidates149_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_8_checked : candidates edge 4 881 3407 innerTail149_9 best.bound = innerCandidates149_8 := by
  unfold candidates
  have row : edge 3407 = SearchData.row3407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_8 : scan edge (search edge 3) 4 [881] 881 innerTail149_8 best = best := by
  rw [innerTail149_8, scan, if_neg (by decide), innerCandidates149_8_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_9
    (if 3 ≤ innerCandidates149_8.length then search edge 3 [881, 3407] innerCandidates149_8 4288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_9

def innerTail149_7 : List Nat := 3011 :: innerTail149_8
def innerCandidates149_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_7_checked : candidates edge 4 881 3011 innerTail149_8 best.bound = innerCandidates149_7 := by
  unfold candidates
  have row : edge 3011 = SearchData.row3011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_7 : scan edge (search edge 3) 4 [881] 881 innerTail149_7 best = best := by
  rw [innerTail149_7, scan, if_neg (by decide), innerCandidates149_7_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_8
    (if 3 ≤ innerCandidates149_7.length then search edge 3 [881, 3011] innerCandidates149_7 3892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_8

def innerTail149_6 : List Nat := 2411 :: innerTail149_7
def innerCandidates149_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_6_checked : candidates edge 4 881 2411 innerTail149_7 best.bound = innerCandidates149_6 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_6 : scan edge (search edge 3) 4 [881] 881 innerTail149_6 best = best := by
  rw [innerTail149_6, scan, if_neg (by decide), innerCandidates149_6_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_7
    (if 3 ≤ innerCandidates149_6.length then search edge 3 [881, 2411] innerCandidates149_6 3292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_7

def innerTail149_5 : List Nat := 1949 :: innerTail149_6
def innerCandidates149_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_5_checked : candidates edge 4 881 1949 innerTail149_6 best.bound = innerCandidates149_5 := by
  unfold candidates
  have row : edge 1949 = PrimePairCache.smallRow1949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_5 : scan edge (search edge 3) 4 [881] 881 innerTail149_5 best = best := by
  rw [innerTail149_5, scan, if_neg (by decide), innerCandidates149_5_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_6
    (if 3 ≤ innerCandidates149_5.length then search edge 3 [881, 1949] innerCandidates149_5 2830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_6

def innerTail149_4 : List Nat := 1667 :: innerTail149_5
def innerCandidates149_4 : List Nat := [6701]
set_option maxRecDepth 4096 in
theorem innerCandidates149_4_checked : candidates edge 4 881 1667 innerTail149_5 best.bound = innerCandidates149_4 := by
  unfold candidates
  have row : edge 1667 = PrimePairCache.smallRow1667.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_4 : scan edge (search edge 3) 4 [881] 881 innerTail149_4 best = best := by
  rw [innerTail149_4, scan, if_neg (by decide), innerCandidates149_4_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_5
    (if 3 ≤ innerCandidates149_4.length then search edge 3 [881, 1667] innerCandidates149_4 2548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_5

def innerTail149_3 : List Nat := 1499 :: innerTail149_4
def innerCandidates149_3 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates149_3_checked : candidates edge 4 881 1499 innerTail149_4 best.bound = innerCandidates149_3 := by
  unfold candidates
  have row : edge 1499 = PrimePairCache.smallRow1499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_3 : scan edge (search edge 3) 4 [881] 881 innerTail149_3 best = best := by
  rw [innerTail149_3, scan, if_neg (by decide), innerCandidates149_3_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_4
    (if 3 ≤ innerCandidates149_3.length then search edge 3 [881, 1499] innerCandidates149_3 2380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_4

def innerTail149_2 : List Nat := 1373 :: innerTail149_3
def innerCandidates149_2 : List Nat := [1499, 7541]
set_option maxRecDepth 4096 in
theorem innerCandidates149_2_checked : candidates edge 4 881 1373 innerTail149_3 best.bound = innerCandidates149_2 := by
  unfold candidates
  have row : edge 1373 = PrimePairCache.smallRow1373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_2 : scan edge (search edge 3) 4 [881] 881 innerTail149_2 best = best := by
  rw [innerTail149_2, scan, if_neg (by decide), innerCandidates149_2_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_3
    (if 3 ≤ innerCandidates149_2.length then search edge 3 [881, 1373] innerCandidates149_2 2254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_3

def innerTail149_1 : List Nat := 983 :: innerTail149_2
def innerCandidates149_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_1_checked : candidates edge 4 881 983 innerTail149_2 best.bound = innerCandidates149_1 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_1 : scan edge (search edge 3) 4 [881] 881 innerTail149_1 best = best := by
  rw [innerTail149_1, scan, if_neg (by decide), innerCandidates149_1_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_2
    (if 3 ≤ innerCandidates149_1.length then search edge 3 [881, 983] innerCandidates149_1 1864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_2

def innerTail149_0 : List Nat := 953 :: innerTail149_1
def innerCandidates149_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates149_0_checked : candidates edge 4 881 953 innerTail149_1 best.bound = innerCandidates149_0 := by
  unfold candidates
  have row : edge 953 = PrimePairCache.smallRow953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan149_0 : scan edge (search edge 3) 4 [881] 881 innerTail149_0 best = best := by
  rw [innerTail149_0, scan, if_neg (by decide), innerCandidates149_0_checked]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_1
    (if 3 ≤ innerCandidates149_0.length then search edge 3 [881, 953] innerCandidates149_0 1834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan149_1

theorem rootBranch149 : search edge 4 [881] rootCandidates149 881 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [881] 881 innerTail149_0 best = best
  exact innerScan149_0

end ProjectEulerSolutions.PrimePairUpper.Search
