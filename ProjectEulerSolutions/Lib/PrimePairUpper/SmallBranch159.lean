import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates198 : List Nat := [1607, 2081, 2663, 2963, 3089, 3389, 3881, 4127, 4229, 5399, 5693, 5741, 5861, 6353, 7877, 7907]
set_option maxRecDepth 4096 in
theorem rootCandidates198_checked : candidates edge 5 0 1229 tail199 best.bound = rootCandidates198 := by
  unfold candidates
  have row : edge 1229 = PrimePairCache.smallRow1229.contains := by funext q; rfl
  rw [row]
  decide

def innerTail198_16 : List Nat := []
theorem innerScan198_16 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_16 best = best := rfl

def innerTail198_15 : List Nat := 7907 :: innerTail198_16
def innerCandidates198_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates198_15_checked : candidates edge 4 1229 7907 innerTail198_16 best.bound = innerCandidates198_15 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_15 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_15 best = best := by
  rw [innerTail198_15, scan, if_neg (by decide), innerCandidates198_15_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_16
    (if 3 ≤ innerCandidates198_15.length then search edge 3 [1229, 7907] innerCandidates198_15 9136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_16

def innerTail198_14 : List Nat := 7877 :: innerTail198_15
def innerCandidates198_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates198_14_checked : candidates edge 4 1229 7877 innerTail198_15 best.bound = innerCandidates198_14 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_14 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_14 best = best := by
  rw [innerTail198_14, scan, if_neg (by decide), innerCandidates198_14_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_15
    (if 3 ≤ innerCandidates198_14.length then search edge 3 [1229, 7877] innerCandidates198_14 9106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_15

def innerTail198_13 : List Nat := 6353 :: innerTail198_14
def innerCandidates198_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates198_13_checked : candidates edge 4 1229 6353 innerTail198_14 best.bound = innerCandidates198_13 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_13 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_13 best = best := by
  rw [innerTail198_13, scan, if_neg (by decide), innerCandidates198_13_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_14
    (if 3 ≤ innerCandidates198_13.length then search edge 3 [1229, 6353] innerCandidates198_13 7582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_14

def innerTail198_12 : List Nat := 5861 :: innerTail198_13
def innerCandidates198_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates198_12_checked : candidates edge 4 1229 5861 innerTail198_13 best.bound = innerCandidates198_12 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_12 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_12 best = best := by
  rw [innerTail198_12, scan, if_neg (by decide), innerCandidates198_12_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_13
    (if 3 ≤ innerCandidates198_12.length then search edge 3 [1229, 5861] innerCandidates198_12 7090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_13

def innerTail198_11 : List Nat := 5741 :: innerTail198_12
def innerCandidates198_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates198_11_checked : candidates edge 4 1229 5741 innerTail198_12 best.bound = innerCandidates198_11 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_11 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_11 best = best := by
  rw [innerTail198_11, scan, if_neg (by decide), innerCandidates198_11_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_12
    (if 3 ≤ innerCandidates198_11.length then search edge 3 [1229, 5741] innerCandidates198_11 6970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_12

def innerTail198_10 : List Nat := 5693 :: innerTail198_11
def innerCandidates198_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates198_10_checked : candidates edge 4 1229 5693 innerTail198_11 best.bound = innerCandidates198_10 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_10 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_10 best = best := by
  rw [innerTail198_10, scan, if_neg (by decide), innerCandidates198_10_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_11
    (if 3 ≤ innerCandidates198_10.length then search edge 3 [1229, 5693] innerCandidates198_10 6922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_11

def innerTail198_9 : List Nat := 5399 :: innerTail198_10
def innerCandidates198_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates198_9_checked : candidates edge 4 1229 5399 innerTail198_10 best.bound = innerCandidates198_9 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_9 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_9 best = best := by
  rw [innerTail198_9, scan, if_neg (by decide), innerCandidates198_9_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_10
    (if 3 ≤ innerCandidates198_9.length then search edge 3 [1229, 5399] innerCandidates198_9 6628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_10

def innerTail198_8 : List Nat := 4229 :: innerTail198_9
def innerCandidates198_8 : List Nat := [5399]
set_option maxRecDepth 4096 in
theorem innerCandidates198_8_checked : candidates edge 4 1229 4229 innerTail198_9 best.bound = innerCandidates198_8 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_8 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_8 best = best := by
  rw [innerTail198_8, scan, if_neg (by decide), innerCandidates198_8_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_9
    (if 3 ≤ innerCandidates198_8.length then search edge 3 [1229, 4229] innerCandidates198_8 5458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_9

def innerTail198_7 : List Nat := 4127 :: innerTail198_8
def innerCandidates198_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates198_7_checked : candidates edge 4 1229 4127 innerTail198_8 best.bound = innerCandidates198_7 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_7 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_7 best = best := by
  rw [innerTail198_7, scan, if_neg (by decide), innerCandidates198_7_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_8
    (if 3 ≤ innerCandidates198_7.length then search edge 3 [1229, 4127] innerCandidates198_7 5356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_8

def innerTail198_6 : List Nat := 3881 :: innerTail198_7
def innerCandidates198_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates198_6_checked : candidates edge 4 1229 3881 innerTail198_7 best.bound = innerCandidates198_6 := by
  unfold candidates
  have row : edge 3881 = SearchData.row3881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_6 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_6 best = best := by
  rw [innerTail198_6, scan, if_neg (by decide), innerCandidates198_6_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_7
    (if 3 ≤ innerCandidates198_6.length then search edge 3 [1229, 3881] innerCandidates198_6 5110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_7

def innerTail198_5 : List Nat := 3389 :: innerTail198_6
def innerCandidates198_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates198_5_checked : candidates edge 4 1229 3389 innerTail198_6 best.bound = innerCandidates198_5 := by
  unfold candidates
  have row : edge 3389 = SearchData.row3389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_5 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_5 best = best := by
  rw [innerTail198_5, scan, if_neg (by decide), innerCandidates198_5_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_6
    (if 3 ≤ innerCandidates198_5.length then search edge 3 [1229, 3389] innerCandidates198_5 4618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_6

def innerTail198_4 : List Nat := 3089 :: innerTail198_5
def innerCandidates198_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates198_4_checked : candidates edge 4 1229 3089 innerTail198_5 best.bound = innerCandidates198_4 := by
  unfold candidates
  have row : edge 3089 = SearchData.row3089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_4 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_4 best = best := by
  rw [innerTail198_4, scan, if_neg (by decide), innerCandidates198_4_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_5
    (if 3 ≤ innerCandidates198_4.length then search edge 3 [1229, 3089] innerCandidates198_4 4318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_5

def innerTail198_3 : List Nat := 2963 :: innerTail198_4
def innerCandidates198_3 : List Nat := [5693, 5861]
set_option maxRecDepth 4096 in
theorem innerCandidates198_3_checked : candidates edge 4 1229 2963 innerTail198_4 best.bound = innerCandidates198_3 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_3 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_3 best = best := by
  rw [innerTail198_3, scan, if_neg (by decide), innerCandidates198_3_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_4
    (if 3 ≤ innerCandidates198_3.length then search edge 3 [1229, 2963] innerCandidates198_3 4192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_4

def innerTail198_2 : List Nat := 2663 :: innerTail198_3
def innerCandidates198_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates198_2_checked : candidates edge 4 1229 2663 innerTail198_3 best.bound = innerCandidates198_2 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_2 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_2 best = best := by
  rw [innerTail198_2, scan, if_neg (by decide), innerCandidates198_2_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_3
    (if 3 ≤ innerCandidates198_2.length then search edge 3 [1229, 2663] innerCandidates198_2 3892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_3

def innerTail198_1 : List Nat := 2081 :: innerTail198_2
def innerCandidates198_1 : List Nat := [4127]
set_option maxRecDepth 4096 in
theorem innerCandidates198_1_checked : candidates edge 4 1229 2081 innerTail198_2 best.bound = innerCandidates198_1 := by
  unfold candidates
  have row : edge 2081 = PrimePairCache.smallRow2081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_1 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_1 best = best := by
  rw [innerTail198_1, scan, if_neg (by decide), innerCandidates198_1_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_2
    (if 3 ≤ innerCandidates198_1.length then search edge 3 [1229, 2081] innerCandidates198_1 3310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_2

def innerTail198_0 : List Nat := 1607 :: innerTail198_1
def innerCandidates198_0 : List Nat := [7877]
set_option maxRecDepth 4096 in
theorem innerCandidates198_0_checked : candidates edge 4 1229 1607 innerTail198_1 best.bound = innerCandidates198_0 := by
  unfold candidates
  have row : edge 1607 = PrimePairCache.smallRow1607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan198_0 : scan edge (search edge 3) 4 [1229] 1229 innerTail198_0 best = best := by
  rw [innerTail198_0, scan, if_neg (by decide), innerCandidates198_0_checked]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_1
    (if 3 ≤ innerCandidates198_0.length then search edge 3 [1229, 1607] innerCandidates198_0 2836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan198_1

theorem rootBranch198 : search edge 4 [1229] rootCandidates198 1229 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1229] 1229 innerTail198_0 best = best
  exact innerScan198_0

def rootCandidates197 : List Nat := [1283, 1433, 1493, 1697, 1931, 2141, 2543, 2579, 2621, 2687, 2777, 3323, 3467, 4517, 4793, 5381, 5897, 5939, 6113, 6131, 6701, 6761, 7523, 7559, 7703, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates197_checked : candidates edge 5 0 1223 tail198 best.bound = rootCandidates197 := by
  unfold candidates
  have row : edge 1223 = PrimePairCache.smallRow1223.contains := by funext q; rfl
  rw [row]
  decide

def innerTail197_26 : List Nat := []
theorem innerScan197_26 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_26 best = best := rfl

def innerTail197_25 : List Nat := 8297 :: innerTail197_26
def innerCandidates197_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_25_checked : candidates edge 4 1223 8297 innerTail197_26 best.bound = innerCandidates197_25 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_25 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_25 best = best := by
  rw [innerTail197_25, scan, if_neg (by decide), innerCandidates197_25_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_26
    (if 3 ≤ innerCandidates197_25.length then search edge 3 [1223, 8297] innerCandidates197_25 9520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_26

def innerTail197_24 : List Nat := 7703 :: innerTail197_25
def innerCandidates197_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_24_checked : candidates edge 4 1223 7703 innerTail197_25 best.bound = innerCandidates197_24 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_24 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_24 best = best := by
  rw [innerTail197_24, scan, if_neg (by decide), innerCandidates197_24_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_25
    (if 3 ≤ innerCandidates197_24.length then search edge 3 [1223, 7703] innerCandidates197_24 8926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_25

def innerTail197_23 : List Nat := 7559 :: innerTail197_24
def innerCandidates197_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_23_checked : candidates edge 4 1223 7559 innerTail197_24 best.bound = innerCandidates197_23 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_23 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_23 best = best := by
  rw [innerTail197_23, scan, if_neg (by decide), innerCandidates197_23_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_24
    (if 3 ≤ innerCandidates197_23.length then search edge 3 [1223, 7559] innerCandidates197_23 8782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_24

def innerTail197_22 : List Nat := 7523 :: innerTail197_23
def innerCandidates197_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_22_checked : candidates edge 4 1223 7523 innerTail197_23 best.bound = innerCandidates197_22 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_22 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_22 best = best := by
  rw [innerTail197_22, scan, if_neg (by decide), innerCandidates197_22_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_23
    (if 3 ≤ innerCandidates197_22.length then search edge 3 [1223, 7523] innerCandidates197_22 8746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_23

def innerTail197_21 : List Nat := 6761 :: innerTail197_22
def innerCandidates197_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_21_checked : candidates edge 4 1223 6761 innerTail197_22 best.bound = innerCandidates197_21 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_21 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_21 best = best := by
  rw [innerTail197_21, scan, if_neg (by decide), innerCandidates197_21_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_22
    (if 3 ≤ innerCandidates197_21.length then search edge 3 [1223, 6761] innerCandidates197_21 7984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_22

def innerTail197_20 : List Nat := 6701 :: innerTail197_21
def innerCandidates197_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_20_checked : candidates edge 4 1223 6701 innerTail197_21 best.bound = innerCandidates197_20 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_20 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_20 best = best := by
  rw [innerTail197_20, scan, if_neg (by decide), innerCandidates197_20_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_21
    (if 3 ≤ innerCandidates197_20.length then search edge 3 [1223, 6701] innerCandidates197_20 7924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_21

def innerTail197_19 : List Nat := 6131 :: innerTail197_20
def innerCandidates197_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_19_checked : candidates edge 4 1223 6131 innerTail197_20 best.bound = innerCandidates197_19 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_19 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_19 best = best := by
  rw [innerTail197_19, scan, if_neg (by decide), innerCandidates197_19_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_20
    (if 3 ≤ innerCandidates197_19.length then search edge 3 [1223, 6131] innerCandidates197_19 7354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_20

def innerTail197_18 : List Nat := 6113 :: innerTail197_19
def innerCandidates197_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_18_checked : candidates edge 4 1223 6113 innerTail197_19 best.bound = innerCandidates197_18 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_18 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_18 best = best := by
  rw [innerTail197_18, scan, if_neg (by decide), innerCandidates197_18_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_19
    (if 3 ≤ innerCandidates197_18.length then search edge 3 [1223, 6113] innerCandidates197_18 7336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_19

def innerTail197_17 : List Nat := 5939 :: innerTail197_18
def innerCandidates197_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_17_checked : candidates edge 4 1223 5939 innerTail197_18 best.bound = innerCandidates197_17 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_17 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_17 best = best := by
  rw [innerTail197_17, scan, if_neg (by decide), innerCandidates197_17_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_18
    (if 3 ≤ innerCandidates197_17.length then search edge 3 [1223, 5939] innerCandidates197_17 7162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_18

def innerTail197_16 : List Nat := 5897 :: innerTail197_17
def innerCandidates197_16 : List Nat := [6131]
set_option maxRecDepth 4096 in
theorem innerCandidates197_16_checked : candidates edge 4 1223 5897 innerTail197_17 best.bound = innerCandidates197_16 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_16 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_16 best = best := by
  rw [innerTail197_16, scan, if_neg (by decide), innerCandidates197_16_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_17
    (if 3 ≤ innerCandidates197_16.length then search edge 3 [1223, 5897] innerCandidates197_16 7120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_17

def innerTail197_15 : List Nat := 5381 :: innerTail197_16
def innerCandidates197_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_15_checked : candidates edge 4 1223 5381 innerTail197_16 best.bound = innerCandidates197_15 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_15 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_15 best = best := by
  rw [innerTail197_15, scan, if_neg (by decide), innerCandidates197_15_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_16
    (if 3 ≤ innerCandidates197_15.length then search edge 3 [1223, 5381] innerCandidates197_15 6604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_16

def innerTail197_14 : List Nat := 4793 :: innerTail197_15
def innerCandidates197_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_14_checked : candidates edge 4 1223 4793 innerTail197_15 best.bound = innerCandidates197_14 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_14 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_14 best = best := by
  rw [innerTail197_14, scan, if_neg (by decide), innerCandidates197_14_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_15
    (if 3 ≤ innerCandidates197_14.length then search edge 3 [1223, 4793] innerCandidates197_14 6016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_15

def innerTail197_13 : List Nat := 4517 :: innerTail197_14
def innerCandidates197_13 : List Nat := [7523]
set_option maxRecDepth 4096 in
theorem innerCandidates197_13_checked : candidates edge 4 1223 4517 innerTail197_14 best.bound = innerCandidates197_13 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_13 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_13 best = best := by
  rw [innerTail197_13, scan, if_neg (by decide), innerCandidates197_13_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_14
    (if 3 ≤ innerCandidates197_13.length then search edge 3 [1223, 4517] innerCandidates197_13 5740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_14

def innerTail197_12 : List Nat := 3467 :: innerTail197_13
def innerCandidates197_12 : List Nat := [6113]
set_option maxRecDepth 4096 in
theorem innerCandidates197_12_checked : candidates edge 4 1223 3467 innerTail197_13 best.bound = innerCandidates197_12 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_12 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_12 best = best := by
  rw [innerTail197_12, scan, if_neg (by decide), innerCandidates197_12_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_13
    (if 3 ≤ innerCandidates197_12.length then search edge 3 [1223, 3467] innerCandidates197_12 4690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_13

def innerTail197_11 : List Nat := 3323 :: innerTail197_12
def innerCandidates197_11 : List Nat := [5381]
set_option maxRecDepth 4096 in
theorem innerCandidates197_11_checked : candidates edge 4 1223 3323 innerTail197_12 best.bound = innerCandidates197_11 := by
  unfold candidates
  have row : edge 3323 = SearchData.row3323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_11 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_11 best = best := by
  rw [innerTail197_11, scan, if_neg (by decide), innerCandidates197_11_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_12
    (if 3 ≤ innerCandidates197_11.length then search edge 3 [1223, 3323] innerCandidates197_11 4546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_12

def innerTail197_10 : List Nat := 2777 :: innerTail197_11
def innerCandidates197_10 : List Nat := [4793]
set_option maxRecDepth 4096 in
theorem innerCandidates197_10_checked : candidates edge 4 1223 2777 innerTail197_11 best.bound = innerCandidates197_10 := by
  unfold candidates
  have row : edge 2777 = SearchData.row2777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_10 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_10 best = best := by
  rw [innerTail197_10, scan, if_neg (by decide), innerCandidates197_10_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_11
    (if 3 ≤ innerCandidates197_10.length then search edge 3 [1223, 2777] innerCandidates197_10 4000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_11

def innerTail197_9 : List Nat := 2687 :: innerTail197_10
def innerCandidates197_9 : List Nat := [5897, 7703]
set_option maxRecDepth 4096 in
theorem innerCandidates197_9_checked : candidates edge 4 1223 2687 innerTail197_10 best.bound = innerCandidates197_9 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_9 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_9 best = best := by
  rw [innerTail197_9, scan, if_neg (by decide), innerCandidates197_9_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_10
    (if 3 ≤ innerCandidates197_9.length then search edge 3 [1223, 2687] innerCandidates197_9 3910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_10

def innerTail197_8 : List Nat := 2621 :: innerTail197_9
def innerCandidates197_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_8_checked : candidates edge 4 1223 2621 innerTail197_9 best.bound = innerCandidates197_8 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_8 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_8 best = best := by
  rw [innerTail197_8, scan, if_neg (by decide), innerCandidates197_8_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_9
    (if 3 ≤ innerCandidates197_8.length then search edge 3 [1223, 2621] innerCandidates197_8 3844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_9

def innerTail197_7 : List Nat := 2579 :: innerTail197_8
def innerCandidates197_7 : List Nat := [6131]
set_option maxRecDepth 4096 in
theorem innerCandidates197_7_checked : candidates edge 4 1223 2579 innerTail197_8 best.bound = innerCandidates197_7 := by
  unfold candidates
  have row : edge 2579 = PrimePairCache.smallRow2579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_7 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_7 best = best := by
  rw [innerTail197_7, scan, if_neg (by decide), innerCandidates197_7_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_8
    (if 3 ≤ innerCandidates197_7.length then search edge 3 [1223, 2579] innerCandidates197_7 3802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_8

def innerTail197_6 : List Nat := 2543 :: innerTail197_7
def innerCandidates197_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_6_checked : candidates edge 4 1223 2543 innerTail197_7 best.bound = innerCandidates197_6 := by
  unfold candidates
  have row : edge 2543 = PrimePairCache.smallRow2543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_6 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_6 best = best := by
  rw [innerTail197_6, scan, if_neg (by decide), innerCandidates197_6_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_7
    (if 3 ≤ innerCandidates197_6.length then search edge 3 [1223, 2543] innerCandidates197_6 3766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_7

def innerTail197_5 : List Nat := 2141 :: innerTail197_6
def innerCandidates197_5 : List Nat := [2621]
set_option maxRecDepth 4096 in
theorem innerCandidates197_5_checked : candidates edge 4 1223 2141 innerTail197_6 best.bound = innerCandidates197_5 := by
  unfold candidates
  have row : edge 2141 = PrimePairCache.smallRow2141.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_5 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_5 best = best := by
  rw [innerTail197_5, scan, if_neg (by decide), innerCandidates197_5_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_6
    (if 3 ≤ innerCandidates197_5.length then search edge 3 [1223, 2141] innerCandidates197_5 3364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_6

def innerTail197_4 : List Nat := 1931 :: innerTail197_5
def innerCandidates197_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_4_checked : candidates edge 4 1223 1931 innerTail197_5 best.bound = innerCandidates197_4 := by
  unfold candidates
  have row : edge 1931 = PrimePairCache.smallRow1931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_4 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_4 best = best := by
  rw [innerTail197_4, scan, if_neg (by decide), innerCandidates197_4_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_5
    (if 3 ≤ innerCandidates197_4.length then search edge 3 [1223, 1931] innerCandidates197_4 3154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_5

def innerTail197_3 : List Nat := 1697 :: innerTail197_4
def innerCandidates197_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_3_checked : candidates edge 4 1223 1697 innerTail197_4 best.bound = innerCandidates197_3 := by
  unfold candidates
  have row : edge 1697 = PrimePairCache.smallRow1697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_3 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_3 best = best := by
  rw [innerTail197_3, scan, if_neg (by decide), innerCandidates197_3_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_4
    (if 3 ≤ innerCandidates197_3.length then search edge 3 [1223, 1697] innerCandidates197_3 2920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_4

def innerTail197_2 : List Nat := 1493 :: innerTail197_3
def innerCandidates197_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_2_checked : candidates edge 4 1223 1493 innerTail197_3 best.bound = innerCandidates197_2 := by
  unfold candidates
  have row : edge 1493 = PrimePairCache.smallRow1493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_2 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_2 best = best := by
  rw [innerTail197_2, scan, if_neg (by decide), innerCandidates197_2_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_3
    (if 3 ≤ innerCandidates197_2.length then search edge 3 [1223, 1493] innerCandidates197_2 2716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_3

def innerTail197_1 : List Nat := 1433 :: innerTail197_2
def innerCandidates197_1 : List Nat := [3323, 6761, 8297]
set_option maxRecDepth 4096 in
theorem innerCandidates197_1_checked : candidates edge 4 1223 1433 innerTail197_2 best.bound = innerCandidates197_1 := by
  unfold candidates
  have row : edge 1433 = PrimePairCache.smallRow1433.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild197_1 : search edge 3 [1223, 1433] innerCandidates197_1 2656 best = best := by decide

theorem innerScan197_1 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_1 best = best := by
  rw [innerTail197_1, scan, if_neg (by decide), innerCandidates197_1_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_2
    (if 3 ≤ innerCandidates197_1.length then search edge 3 [1223, 1433] innerCandidates197_1 2656 best else best) = best
  rw [if_pos (by decide), innerChild197_1]
  exact innerScan197_2

def innerTail197_0 : List Nat := 1283 :: innerTail197_1
def innerCandidates197_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates197_0_checked : candidates edge 4 1223 1283 innerTail197_1 best.bound = innerCandidates197_0 := by
  unfold candidates
  have row : edge 1283 = PrimePairCache.smallRow1283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan197_0 : scan edge (search edge 3) 4 [1223] 1223 innerTail197_0 best = best := by
  rw [innerTail197_0, scan, if_neg (by decide), innerCandidates197_0_checked]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_1
    (if 3 ≤ innerCandidates197_0.length then search edge 3 [1223, 1283] innerCandidates197_0 2506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan197_1

theorem rootBranch197 : search edge 4 [1223] rootCandidates197 1223 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1223] 1223 innerTail197_0 best = best
  exact innerScan197_0

end ProjectEulerSolutions.PrimePairUpper.Search
