import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates214 : List Nat := [1429, 1471, 1723, 2293, 2551, 3109, 3361, 3739, 3769, 3853, 4357, 4621, 5701, 6163, 6217, 6301, 6619, 7129, 7351, 7717, 7963, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates214_checked : candidates edge 5 0 1327 tail215 best.bound = rootCandidates214 := by
  unfold candidates
  have row : edge 1327 = PrimePairCache.smallRow1327.contains := by funext q; rfl
  rw [row]
  decide

def innerTail214_22 : List Nat := []
theorem innerScan214_22 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_22 best = best := rfl

def innerTail214_21 : List Nat := 8263 :: innerTail214_22
def innerCandidates214_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_21_checked : candidates edge 4 1327 8263 innerTail214_22 best.bound = innerCandidates214_21 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_21 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_21 best = best := by
  rw [innerTail214_21, scan, if_neg (by decide), innerCandidates214_21_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_22
    (if 3 ≤ innerCandidates214_21.length then search edge 3 [1327, 8263] innerCandidates214_21 9590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_22

def innerTail214_20 : List Nat := 7963 :: innerTail214_21
def innerCandidates214_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_20_checked : candidates edge 4 1327 7963 innerTail214_21 best.bound = innerCandidates214_20 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_20 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_20 best = best := by
  rw [innerTail214_20, scan, if_neg (by decide), innerCandidates214_20_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_21
    (if 3 ≤ innerCandidates214_20.length then search edge 3 [1327, 7963] innerCandidates214_20 9290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_21

def innerTail214_19 : List Nat := 7717 :: innerTail214_20
def innerCandidates214_19 : List Nat := [7963]
set_option maxRecDepth 4096 in
theorem innerCandidates214_19_checked : candidates edge 4 1327 7717 innerTail214_20 best.bound = innerCandidates214_19 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_19 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_19 best = best := by
  rw [innerTail214_19, scan, if_neg (by decide), innerCandidates214_19_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_20
    (if 3 ≤ innerCandidates214_19.length then search edge 3 [1327, 7717] innerCandidates214_19 9044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_20

def innerTail214_18 : List Nat := 7351 :: innerTail214_19
def innerCandidates214_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_18_checked : candidates edge 4 1327 7351 innerTail214_19 best.bound = innerCandidates214_18 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_18 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_18 best = best := by
  rw [innerTail214_18, scan, if_neg (by decide), innerCandidates214_18_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_19
    (if 3 ≤ innerCandidates214_18.length then search edge 3 [1327, 7351] innerCandidates214_18 8678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_19

def innerTail214_17 : List Nat := 7129 :: innerTail214_18
def innerCandidates214_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_17_checked : candidates edge 4 1327 7129 innerTail214_18 best.bound = innerCandidates214_17 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_17 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_17 best = best := by
  rw [innerTail214_17, scan, if_neg (by decide), innerCandidates214_17_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_18
    (if 3 ≤ innerCandidates214_17.length then search edge 3 [1327, 7129] innerCandidates214_17 8456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_18

def innerTail214_16 : List Nat := 6619 :: innerTail214_17
def innerCandidates214_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_16_checked : candidates edge 4 1327 6619 innerTail214_17 best.bound = innerCandidates214_16 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_16 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_16 best = best := by
  rw [innerTail214_16, scan, if_neg (by decide), innerCandidates214_16_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_17
    (if 3 ≤ innerCandidates214_16.length then search edge 3 [1327, 6619] innerCandidates214_16 7946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_17

def innerTail214_15 : List Nat := 6301 :: innerTail214_16
def innerCandidates214_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_15_checked : candidates edge 4 1327 6301 innerTail214_16 best.bound = innerCandidates214_15 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_15 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_15 best = best := by
  rw [innerTail214_15, scan, if_neg (by decide), innerCandidates214_15_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_16
    (if 3 ≤ innerCandidates214_15.length then search edge 3 [1327, 6301] innerCandidates214_15 7628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_16

def innerTail214_14 : List Nat := 6217 :: innerTail214_15
def innerCandidates214_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_14_checked : candidates edge 4 1327 6217 innerTail214_15 best.bound = innerCandidates214_14 := by
  unfold candidates
  have row : edge 6217 = SearchData.row6217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_14 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_14 best = best := by
  rw [innerTail214_14, scan, if_neg (by decide), innerCandidates214_14_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_15
    (if 3 ≤ innerCandidates214_14.length then search edge 3 [1327, 6217] innerCandidates214_14 7544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_15

def innerTail214_13 : List Nat := 6163 :: innerTail214_14
def innerCandidates214_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_13_checked : candidates edge 4 1327 6163 innerTail214_14 best.bound = innerCandidates214_13 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_13 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_13 best = best := by
  rw [innerTail214_13, scan, if_neg (by decide), innerCandidates214_13_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_14
    (if 3 ≤ innerCandidates214_13.length then search edge 3 [1327, 6163] innerCandidates214_13 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_14

def innerTail214_12 : List Nat := 5701 :: innerTail214_13
def innerCandidates214_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_12_checked : candidates edge 4 1327 5701 innerTail214_13 best.bound = innerCandidates214_12 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_12 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_12 best = best := by
  rw [innerTail214_12, scan, if_neg (by decide), innerCandidates214_12_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_13
    (if 3 ≤ innerCandidates214_12.length then search edge 3 [1327, 5701] innerCandidates214_12 7028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_13

def innerTail214_11 : List Nat := 4621 :: innerTail214_12
def innerCandidates214_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_11_checked : candidates edge 4 1327 4621 innerTail214_12 best.bound = innerCandidates214_11 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_11 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_11 best = best := by
  rw [innerTail214_11, scan, if_neg (by decide), innerCandidates214_11_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_12
    (if 3 ≤ innerCandidates214_11.length then search edge 3 [1327, 4621] innerCandidates214_11 5948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_12

def innerTail214_10 : List Nat := 4357 :: innerTail214_11
def innerCandidates214_10 : List Nat := [6619]
set_option maxRecDepth 4096 in
theorem innerCandidates214_10_checked : candidates edge 4 1327 4357 innerTail214_11 best.bound = innerCandidates214_10 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_10 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_10 best = best := by
  rw [innerTail214_10, scan, if_neg (by decide), innerCandidates214_10_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_11
    (if 3 ≤ innerCandidates214_10.length then search edge 3 [1327, 4357] innerCandidates214_10 5684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_11

def innerTail214_9 : List Nat := 3853 :: innerTail214_10
def innerCandidates214_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_9_checked : candidates edge 4 1327 3853 innerTail214_10 best.bound = innerCandidates214_9 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_9 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_9 best = best := by
  rw [innerTail214_9, scan, if_neg (by decide), innerCandidates214_9_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_10
    (if 3 ≤ innerCandidates214_9.length then search edge 3 [1327, 3853] innerCandidates214_9 5180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_10

def innerTail214_8 : List Nat := 3769 :: innerTail214_9
def innerCandidates214_8 : List Nat := [6619]
set_option maxRecDepth 4096 in
theorem innerCandidates214_8_checked : candidates edge 4 1327 3769 innerTail214_9 best.bound = innerCandidates214_8 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_8 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_8 best = best := by
  rw [innerTail214_8, scan, if_neg (by decide), innerCandidates214_8_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_9
    (if 3 ≤ innerCandidates214_8.length then search edge 3 [1327, 3769] innerCandidates214_8 5096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_9

def innerTail214_7 : List Nat := 3739 :: innerTail214_8
def innerCandidates214_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_7_checked : candidates edge 4 1327 3739 innerTail214_8 best.bound = innerCandidates214_7 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_7 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_7 best = best := by
  rw [innerTail214_7, scan, if_neg (by decide), innerCandidates214_7_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_8
    (if 3 ≤ innerCandidates214_7.length then search edge 3 [1327, 3739] innerCandidates214_7 5066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_8

def innerTail214_6 : List Nat := 3361 :: innerTail214_7
def innerCandidates214_6 : List Nat := [3739]
set_option maxRecDepth 4096 in
theorem innerCandidates214_6_checked : candidates edge 4 1327 3361 innerTail214_7 best.bound = innerCandidates214_6 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_6 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_6 best = best := by
  rw [innerTail214_6, scan, if_neg (by decide), innerCandidates214_6_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_7
    (if 3 ≤ innerCandidates214_6.length then search edge 3 [1327, 3361] innerCandidates214_6 4688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_7

def innerTail214_5 : List Nat := 3109 :: innerTail214_6
def innerCandidates214_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates214_5_checked : candidates edge 4 1327 3109 innerTail214_6 best.bound = innerCandidates214_5 := by
  unfold candidates
  have row : edge 3109 = SearchData.row3109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_5 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_5 best = best := by
  rw [innerTail214_5, scan, if_neg (by decide), innerCandidates214_5_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_6
    (if 3 ≤ innerCandidates214_5.length then search edge 3 [1327, 3109] innerCandidates214_5 4436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_6

def innerTail214_4 : List Nat := 2551 :: innerTail214_5
def innerCandidates214_4 : List Nat := [6163]
set_option maxRecDepth 4096 in
theorem innerCandidates214_4_checked : candidates edge 4 1327 2551 innerTail214_5 best.bound = innerCandidates214_4 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_4 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_4 best = best := by
  rw [innerTail214_4, scan, if_neg (by decide), innerCandidates214_4_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_5
    (if 3 ≤ innerCandidates214_4.length then search edge 3 [1327, 2551] innerCandidates214_4 3878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_5

def innerTail214_3 : List Nat := 2293 :: innerTail214_4
def innerCandidates214_3 : List Nat := [3853]
set_option maxRecDepth 4096 in
theorem innerCandidates214_3_checked : candidates edge 4 1327 2293 innerTail214_4 best.bound = innerCandidates214_3 := by
  unfold candidates
  have row : edge 2293 = PrimePairCache.smallRow2293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_3 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_3 best = best := by
  rw [innerTail214_3, scan, if_neg (by decide), innerCandidates214_3_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_4
    (if 3 ≤ innerCandidates214_3.length then search edge 3 [1327, 2293] innerCandidates214_3 3620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_4

def innerTail214_2 : List Nat := 1723 :: innerTail214_3
def innerCandidates214_2 : List Nat := [4621]
set_option maxRecDepth 4096 in
theorem innerCandidates214_2_checked : candidates edge 4 1327 1723 innerTail214_3 best.bound = innerCandidates214_2 := by
  unfold candidates
  have row : edge 1723 = PrimePairCache.smallRow1723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_2 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_2 best = best := by
  rw [innerTail214_2, scan, if_neg (by decide), innerCandidates214_2_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_3
    (if 3 ≤ innerCandidates214_2.length then search edge 3 [1327, 1723] innerCandidates214_2 3050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_3

def innerTail214_1 : List Nat := 1471 :: innerTail214_2
def innerCandidates214_1 : List Nat := [6301]
set_option maxRecDepth 4096 in
theorem innerCandidates214_1_checked : candidates edge 4 1327 1471 innerTail214_2 best.bound = innerCandidates214_1 := by
  unfold candidates
  have row : edge 1471 = PrimePairCache.smallRow1471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan214_1 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_1 best = best := by
  rw [innerTail214_1, scan, if_neg (by decide), innerCandidates214_1_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_2
    (if 3 ≤ innerCandidates214_1.length then search edge 3 [1327, 1471] innerCandidates214_1 2798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan214_2

def innerTail214_0 : List Nat := 1429 :: innerTail214_1
def innerCandidates214_0 : List Nat := [1723, 3739, 6619, 7351, 7963]
set_option maxRecDepth 4096 in
theorem innerCandidates214_0_checked : candidates edge 4 1327 1429 innerTail214_1 best.bound = innerCandidates214_0 := by
  unfold candidates
  have row : edge 1429 = PrimePairCache.smallRow1429.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild214_0 : search edge 3 [1327, 1429] innerCandidates214_0 2756 best = best := by decide

theorem innerScan214_0 : scan edge (search edge 3) 4 [1327] 1327 innerTail214_0 best = best := by
  rw [innerTail214_0, scan, if_neg (by decide), innerCandidates214_0_checked]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_1
    (if 3 ≤ innerCandidates214_0.length then search edge 3 [1327, 1429] innerCandidates214_0 2756 best else best) = best
  rw [if_pos (by decide), innerChild214_0]
  exact innerScan214_1

theorem rootBranch214 : search edge 4 [1327] rootCandidates214 1327 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1327] 1327 innerTail214_0 best = best
  exact innerScan214_0

def rootCandidates213 : List Nat := [1381, 2767, 2803, 2917, 3001, 4261, 4273, 4447, 4507, 5581, 6607, 7537, 7759, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates213_checked : candidates edge 5 0 1321 tail214 best.bound = rootCandidates213 := by
  unfold candidates
  have row : edge 1321 = PrimePairCache.smallRow1321.contains := by funext q; rfl
  rw [row]
  decide

def innerTail213_14 : List Nat := []
theorem innerScan213_14 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_14 best = best := rfl

def innerTail213_13 : List Nat := 8263 :: innerTail213_14
def innerCandidates213_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates213_13_checked : candidates edge 4 1321 8263 innerTail213_14 best.bound = innerCandidates213_13 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_13 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_13 best = best := by
  rw [innerTail213_13, scan, if_neg (by decide), innerCandidates213_13_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_14
    (if 3 ≤ innerCandidates213_13.length then search edge 3 [1321, 8263] innerCandidates213_13 9584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_14

def innerTail213_12 : List Nat := 7759 :: innerTail213_13
def innerCandidates213_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates213_12_checked : candidates edge 4 1321 7759 innerTail213_13 best.bound = innerCandidates213_12 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_12 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_12 best = best := by
  rw [innerTail213_12, scan, if_neg (by decide), innerCandidates213_12_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_13
    (if 3 ≤ innerCandidates213_12.length then search edge 3 [1321, 7759] innerCandidates213_12 9080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_13

def innerTail213_11 : List Nat := 7537 :: innerTail213_12
def innerCandidates213_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates213_11_checked : candidates edge 4 1321 7537 innerTail213_12 best.bound = innerCandidates213_11 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_11 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_11 best = best := by
  rw [innerTail213_11, scan, if_neg (by decide), innerCandidates213_11_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_12
    (if 3 ≤ innerCandidates213_11.length then search edge 3 [1321, 7537] innerCandidates213_11 8858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_12

def innerTail213_10 : List Nat := 6607 :: innerTail213_11
def innerCandidates213_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates213_10_checked : candidates edge 4 1321 6607 innerTail213_11 best.bound = innerCandidates213_10 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_10 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_10 best = best := by
  rw [innerTail213_10, scan, if_neg (by decide), innerCandidates213_10_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_11
    (if 3 ≤ innerCandidates213_10.length then search edge 3 [1321, 6607] innerCandidates213_10 7928 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_11

def innerTail213_9 : List Nat := 5581 :: innerTail213_10
def innerCandidates213_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates213_9_checked : candidates edge 4 1321 5581 innerTail213_10 best.bound = innerCandidates213_9 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_9 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_9 best = best := by
  rw [innerTail213_9, scan, if_neg (by decide), innerCandidates213_9_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_10
    (if 3 ≤ innerCandidates213_9.length then search edge 3 [1321, 5581] innerCandidates213_9 6902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_10

def innerTail213_8 : List Nat := 4507 :: innerTail213_9
def innerCandidates213_8 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates213_8_checked : candidates edge 4 1321 4507 innerTail213_9 best.bound = innerCandidates213_8 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_8 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_8 best = best := by
  rw [innerTail213_8, scan, if_neg (by decide), innerCandidates213_8_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_9
    (if 3 ≤ innerCandidates213_8.length then search edge 3 [1321, 4507] innerCandidates213_8 5828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_9

def innerTail213_7 : List Nat := 4447 :: innerTail213_8
def innerCandidates213_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates213_7_checked : candidates edge 4 1321 4447 innerTail213_8 best.bound = innerCandidates213_7 := by
  unfold candidates
  have row : edge 4447 = SearchData.row4447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_7 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_7 best = best := by
  rw [innerTail213_7, scan, if_neg (by decide), innerCandidates213_7_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_8
    (if 3 ≤ innerCandidates213_7.length then search edge 3 [1321, 4447] innerCandidates213_7 5768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_8

def innerTail213_6 : List Nat := 4273 :: innerTail213_7
def innerCandidates213_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates213_6_checked : candidates edge 4 1321 4273 innerTail213_7 best.bound = innerCandidates213_6 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_6 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_6 best = best := by
  rw [innerTail213_6, scan, if_neg (by decide), innerCandidates213_6_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_7
    (if 3 ≤ innerCandidates213_6.length then search edge 3 [1321, 4273] innerCandidates213_6 5594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_7

def innerTail213_5 : List Nat := 4261 :: innerTail213_6
def innerCandidates213_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates213_5_checked : candidates edge 4 1321 4261 innerTail213_6 best.bound = innerCandidates213_5 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_5 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_5 best = best := by
  rw [innerTail213_5, scan, if_neg (by decide), innerCandidates213_5_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_6
    (if 3 ≤ innerCandidates213_5.length then search edge 3 [1321, 4261] innerCandidates213_5 5582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_6

def innerTail213_4 : List Nat := 3001 :: innerTail213_5
def innerCandidates213_4 : List Nat := [6607]
set_option maxRecDepth 4096 in
theorem innerCandidates213_4_checked : candidates edge 4 1321 3001 innerTail213_5 best.bound = innerCandidates213_4 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_4 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_4 best = best := by
  rw [innerTail213_4, scan, if_neg (by decide), innerCandidates213_4_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_5
    (if 3 ≤ innerCandidates213_4.length then search edge 3 [1321, 3001] innerCandidates213_4 4322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_5

def innerTail213_3 : List Nat := 2917 :: innerTail213_4
def innerCandidates213_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates213_3_checked : candidates edge 4 1321 2917 innerTail213_4 best.bound = innerCandidates213_3 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_3 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_3 best = best := by
  rw [innerTail213_3, scan, if_neg (by decide), innerCandidates213_3_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_4
    (if 3 ≤ innerCandidates213_3.length then search edge 3 [1321, 2917] innerCandidates213_3 4238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_4

def innerTail213_2 : List Nat := 2803 :: innerTail213_3
def innerCandidates213_2 : List Nat := [4507]
set_option maxRecDepth 4096 in
theorem innerCandidates213_2_checked : candidates edge 4 1321 2803 innerTail213_3 best.bound = innerCandidates213_2 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_2 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_2 best = best := by
  rw [innerTail213_2, scan, if_neg (by decide), innerCandidates213_2_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_3
    (if 3 ≤ innerCandidates213_2.length then search edge 3 [1321, 2803] innerCandidates213_2 4124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_3

def innerTail213_1 : List Nat := 2767 :: innerTail213_2
def innerCandidates213_1 : List Nat := [7537]
set_option maxRecDepth 4096 in
theorem innerCandidates213_1_checked : candidates edge 4 1321 2767 innerTail213_2 best.bound = innerCandidates213_1 := by
  unfold candidates
  have row : edge 2767 = SearchData.row2767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_1 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_1 best = best := by
  rw [innerTail213_1, scan, if_neg (by decide), innerCandidates213_1_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_2
    (if 3 ≤ innerCandidates213_1.length then search edge 3 [1321, 2767] innerCandidates213_1 4088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_2

def innerTail213_0 : List Nat := 1381 :: innerTail213_1
def innerCandidates213_0 : List Nat := [2767]
set_option maxRecDepth 4096 in
theorem innerCandidates213_0_checked : candidates edge 4 1321 1381 innerTail213_1 best.bound = innerCandidates213_0 := by
  unfold candidates
  have row : edge 1381 = PrimePairCache.smallRow1381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan213_0 : scan edge (search edge 3) 4 [1321] 1321 innerTail213_0 best = best := by
  rw [innerTail213_0, scan, if_neg (by decide), innerCandidates213_0_checked]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_1
    (if 3 ≤ innerCandidates213_0.length then search edge 3 [1321, 1381] innerCandidates213_0 2702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan213_1

theorem rootBranch213 : search edge 4 [1321] rootCandidates213 1321 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1321] 1321 innerTail213_0 best = best
  exact innerScan213_0

def rootCandidates212 : List Nat := [1361, 1613, 1637, 1901, 1979, 2243, 2789, 3407, 3533, 3623, 3863, 4007, 4133, 4253, 4283, 5843, 5981, 6971, 7649, 8039, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates212_checked : candidates edge 5 0 1319 tail213 best.bound = rootCandidates212 := by
  unfold candidates
  have row : edge 1319 = PrimePairCache.smallRow1319.contains := by funext q; rfl
  rw [row]
  decide

def innerTail212_21 : List Nat := []
theorem innerScan212_21 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_21 best = best := rfl

def innerTail212_20 : List Nat := 8363 :: innerTail212_21
def innerCandidates212_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_20_checked : candidates edge 4 1319 8363 innerTail212_21 best.bound = innerCandidates212_20 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_20 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_20 best = best := by
  rw [innerTail212_20, scan, if_neg (by decide), innerCandidates212_20_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_21
    (if 3 ≤ innerCandidates212_20.length then search edge 3 [1319, 8363] innerCandidates212_20 9682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_21

def innerTail212_19 : List Nat := 8039 :: innerTail212_20
def innerCandidates212_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_19_checked : candidates edge 4 1319 8039 innerTail212_20 best.bound = innerCandidates212_19 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_19 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_19 best = best := by
  rw [innerTail212_19, scan, if_neg (by decide), innerCandidates212_19_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_20
    (if 3 ≤ innerCandidates212_19.length then search edge 3 [1319, 8039] innerCandidates212_19 9358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_20

def innerTail212_18 : List Nat := 7649 :: innerTail212_19
def innerCandidates212_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_18_checked : candidates edge 4 1319 7649 innerTail212_19 best.bound = innerCandidates212_18 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_18 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_18 best = best := by
  rw [innerTail212_18, scan, if_neg (by decide), innerCandidates212_18_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_19
    (if 3 ≤ innerCandidates212_18.length then search edge 3 [1319, 7649] innerCandidates212_18 8968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_19

def innerTail212_17 : List Nat := 6971 :: innerTail212_18
def innerCandidates212_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_17_checked : candidates edge 4 1319 6971 innerTail212_18 best.bound = innerCandidates212_17 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_17 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_17 best = best := by
  rw [innerTail212_17, scan, if_neg (by decide), innerCandidates212_17_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_18
    (if 3 ≤ innerCandidates212_17.length then search edge 3 [1319, 6971] innerCandidates212_17 8290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_18

def innerTail212_16 : List Nat := 5981 :: innerTail212_17
def innerCandidates212_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_16_checked : candidates edge 4 1319 5981 innerTail212_17 best.bound = innerCandidates212_16 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_16 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_16 best = best := by
  rw [innerTail212_16, scan, if_neg (by decide), innerCandidates212_16_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_17
    (if 3 ≤ innerCandidates212_16.length then search edge 3 [1319, 5981] innerCandidates212_16 7300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_17

def innerTail212_15 : List Nat := 5843 :: innerTail212_16
def innerCandidates212_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_15_checked : candidates edge 4 1319 5843 innerTail212_16 best.bound = innerCandidates212_15 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_15 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_15 best = best := by
  rw [innerTail212_15, scan, if_neg (by decide), innerCandidates212_15_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_16
    (if 3 ≤ innerCandidates212_15.length then search edge 3 [1319, 5843] innerCandidates212_15 7162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_16

def innerTail212_14 : List Nat := 4283 :: innerTail212_15
def innerCandidates212_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_14_checked : candidates edge 4 1319 4283 innerTail212_15 best.bound = innerCandidates212_14 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_14 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_14 best = best := by
  rw [innerTail212_14, scan, if_neg (by decide), innerCandidates212_14_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_15
    (if 3 ≤ innerCandidates212_14.length then search edge 3 [1319, 4283] innerCandidates212_14 5602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_15

def innerTail212_13 : List Nat := 4253 :: innerTail212_14
def innerCandidates212_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_13_checked : candidates edge 4 1319 4253 innerTail212_14 best.bound = innerCandidates212_13 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_13 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_13 best = best := by
  rw [innerTail212_13, scan, if_neg (by decide), innerCandidates212_13_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_14
    (if 3 ≤ innerCandidates212_13.length then search edge 3 [1319, 4253] innerCandidates212_13 5572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_14

def innerTail212_12 : List Nat := 4133 :: innerTail212_13
def innerCandidates212_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_12_checked : candidates edge 4 1319 4133 innerTail212_13 best.bound = innerCandidates212_12 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_12 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_12 best = best := by
  rw [innerTail212_12, scan, if_neg (by decide), innerCandidates212_12_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_13
    (if 3 ≤ innerCandidates212_12.length then search edge 3 [1319, 4133] innerCandidates212_12 5452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_13

def innerTail212_11 : List Nat := 4007 :: innerTail212_12
def innerCandidates212_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_11_checked : candidates edge 4 1319 4007 innerTail212_12 best.bound = innerCandidates212_11 := by
  unfold candidates
  have row : edge 4007 = SearchData.row4007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_11 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_11 best = best := by
  rw [innerTail212_11, scan, if_neg (by decide), innerCandidates212_11_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_12
    (if 3 ≤ innerCandidates212_11.length then search edge 3 [1319, 4007] innerCandidates212_11 5326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_12

def innerTail212_10 : List Nat := 3863 :: innerTail212_11
def innerCandidates212_10 : List Nat := [4007]
set_option maxRecDepth 4096 in
theorem innerCandidates212_10_checked : candidates edge 4 1319 3863 innerTail212_11 best.bound = innerCandidates212_10 := by
  unfold candidates
  have row : edge 3863 = SearchData.row3863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_10 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_10 best = best := by
  rw [innerTail212_10, scan, if_neg (by decide), innerCandidates212_10_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_11
    (if 3 ≤ innerCandidates212_10.length then search edge 3 [1319, 3863] innerCandidates212_10 5182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_11

def innerTail212_9 : List Nat := 3623 :: innerTail212_10
def innerCandidates212_9 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates212_9_checked : candidates edge 4 1319 3623 innerTail212_10 best.bound = innerCandidates212_9 := by
  unfold candidates
  have row : edge 3623 = SearchData.row3623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_9 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_9 best = best := by
  rw [innerTail212_9, scan, if_neg (by decide), innerCandidates212_9_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_10
    (if 3 ≤ innerCandidates212_9.length then search edge 3 [1319, 3623] innerCandidates212_9 4942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_10

def innerTail212_8 : List Nat := 3533 :: innerTail212_9
def innerCandidates212_8 : List Nat := [4133]
set_option maxRecDepth 4096 in
theorem innerCandidates212_8_checked : candidates edge 4 1319 3533 innerTail212_9 best.bound = innerCandidates212_8 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_8 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_8 best = best := by
  rw [innerTail212_8, scan, if_neg (by decide), innerCandidates212_8_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_9
    (if 3 ≤ innerCandidates212_8.length then search edge 3 [1319, 3533] innerCandidates212_8 4852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_9

def innerTail212_7 : List Nat := 3407 :: innerTail212_8
def innerCandidates212_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_7_checked : candidates edge 4 1319 3407 innerTail212_8 best.bound = innerCandidates212_7 := by
  unfold candidates
  have row : edge 3407 = SearchData.row3407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_7 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_7 best = best := by
  rw [innerTail212_7, scan, if_neg (by decide), innerCandidates212_7_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_8
    (if 3 ≤ innerCandidates212_7.length then search edge 3 [1319, 3407] innerCandidates212_7 4726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_8

def innerTail212_6 : List Nat := 2789 :: innerTail212_7
def innerCandidates212_6 : List Nat := [4007, 4133]
set_option maxRecDepth 4096 in
theorem innerCandidates212_6_checked : candidates edge 4 1319 2789 innerTail212_7 best.bound = innerCandidates212_6 := by
  unfold candidates
  have row : edge 2789 = SearchData.row2789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_6 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_6 best = best := by
  rw [innerTail212_6, scan, if_neg (by decide), innerCandidates212_6_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_7
    (if 3 ≤ innerCandidates212_6.length then search edge 3 [1319, 2789] innerCandidates212_6 4108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_7

def innerTail212_5 : List Nat := 2243 :: innerTail212_6
def innerCandidates212_5 : List Nat := [2789, 4133]
set_option maxRecDepth 4096 in
theorem innerCandidates212_5_checked : candidates edge 4 1319 2243 innerTail212_6 best.bound = innerCandidates212_5 := by
  unfold candidates
  have row : edge 2243 = PrimePairCache.smallRow2243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_5 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_5 best = best := by
  rw [innerTail212_5, scan, if_neg (by decide), innerCandidates212_5_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_6
    (if 3 ≤ innerCandidates212_5.length then search edge 3 [1319, 2243] innerCandidates212_5 3562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_6

def innerTail212_4 : List Nat := 1979 :: innerTail212_5
def innerCandidates212_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_4_checked : candidates edge 4 1319 1979 innerTail212_5 best.bound = innerCandidates212_4 := by
  unfold candidates
  have row : edge 1979 = PrimePairCache.smallRow1979.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_4 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_4 best = best := by
  rw [innerTail212_4, scan, if_neg (by decide), innerCandidates212_4_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_5
    (if 3 ≤ innerCandidates212_4.length then search edge 3 [1319, 1979] innerCandidates212_4 3298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_5

def innerTail212_3 : List Nat := 1901 :: innerTail212_4
def innerCandidates212_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_3_checked : candidates edge 4 1319 1901 innerTail212_4 best.bound = innerCandidates212_3 := by
  unfold candidates
  have row : edge 1901 = PrimePairCache.smallRow1901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_3 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_3 best = best := by
  rw [innerTail212_3, scan, if_neg (by decide), innerCandidates212_3_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_4
    (if 3 ≤ innerCandidates212_3.length then search edge 3 [1319, 1901] innerCandidates212_3 3220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_4

def innerTail212_2 : List Nat := 1637 :: innerTail212_3
def innerCandidates212_2 : List Nat := [8039, 8363]
set_option maxRecDepth 4096 in
theorem innerCandidates212_2_checked : candidates edge 4 1319 1637 innerTail212_3 best.bound = innerCandidates212_2 := by
  unfold candidates
  have row : edge 1637 = PrimePairCache.smallRow1637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_2 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_2 best = best := by
  rw [innerTail212_2, scan, if_neg (by decide), innerCandidates212_2_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_3
    (if 3 ≤ innerCandidates212_2.length then search edge 3 [1319, 1637] innerCandidates212_2 2956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_3

def innerTail212_1 : List Nat := 1613 :: innerTail212_2
def innerCandidates212_1 : List Nat := [1637, 1979, 4133, 6971]
set_option maxRecDepth 4096 in
theorem innerCandidates212_1_checked : candidates edge 4 1319 1613 innerTail212_2 best.bound = innerCandidates212_1 := by
  unfold candidates
  have row : edge 1613 = PrimePairCache.smallRow1613.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild212_1 : search edge 3 [1319, 1613] innerCandidates212_1 2932 best = best := by decide

theorem innerScan212_1 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_1 best = best := by
  rw [innerTail212_1, scan, if_neg (by decide), innerCandidates212_1_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_2
    (if 3 ≤ innerCandidates212_1.length then search edge 3 [1319, 1613] innerCandidates212_1 2932 best else best) = best
  rw [if_pos (by decide), innerChild212_1]
  exact innerScan212_2

def innerTail212_0 : List Nat := 1361 :: innerTail212_1
def innerCandidates212_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates212_0_checked : candidates edge 4 1319 1361 innerTail212_1 best.bound = innerCandidates212_0 := by
  unfold candidates
  have row : edge 1361 = PrimePairCache.smallRow1361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan212_0 : scan edge (search edge 3) 4 [1319] 1319 innerTail212_0 best = best := by
  rw [innerTail212_0, scan, if_neg (by decide), innerCandidates212_0_checked]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_1
    (if 3 ≤ innerCandidates212_0.length then search edge 3 [1319, 1361] innerCandidates212_0 2680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan212_1

theorem rootBranch212 : search edge 4 [1319] rootCandidates212 1319 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1319] 1319 innerTail212_0 best = best
  exact innerScan212_0

end ProjectEulerSolutions.PrimePairUpper.Search
