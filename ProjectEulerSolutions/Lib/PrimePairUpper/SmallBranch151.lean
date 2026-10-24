import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates224 : List Nat := [2663, 2957, 2999, 3089, 3323, 4157, 4283, 4637, 4643, 5393, 5501, 5903, 6029, 6299, 6737, 6761, 7607, 7793, 7937, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates224_checked : candidates edge 5 0 1433 tail225 best.bound = rootCandidates224 := by
  unfold candidates
  have row : edge 1433 = PrimePairCache.smallRow1433.contains := by funext q; rfl
  rw [row]
  decide

def innerTail224_20 : List Nat := []
theorem innerScan224_20 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_20 best = best := rfl

def innerTail224_19 : List Nat := 8297 :: innerTail224_20
def innerCandidates224_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_19_checked : candidates edge 4 1433 8297 innerTail224_20 best.bound = innerCandidates224_19 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_19 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_19 best = best := by
  rw [innerTail224_19, scan, if_neg (by decide), innerCandidates224_19_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_20
    (if 3 ≤ innerCandidates224_19.length then search edge 3 [1433, 8297] innerCandidates224_19 9730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_20

def innerTail224_18 : List Nat := 7937 :: innerTail224_19
def innerCandidates224_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_18_checked : candidates edge 4 1433 7937 innerTail224_19 best.bound = innerCandidates224_18 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_18 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_18 best = best := by
  rw [innerTail224_18, scan, if_neg (by decide), innerCandidates224_18_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_19
    (if 3 ≤ innerCandidates224_18.length then search edge 3 [1433, 7937] innerCandidates224_18 9370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_19

def innerTail224_17 : List Nat := 7793 :: innerTail224_18
def innerCandidates224_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_17_checked : candidates edge 4 1433 7793 innerTail224_18 best.bound = innerCandidates224_17 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_17 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_17 best = best := by
  rw [innerTail224_17, scan, if_neg (by decide), innerCandidates224_17_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_18
    (if 3 ≤ innerCandidates224_17.length then search edge 3 [1433, 7793] innerCandidates224_17 9226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_18

def innerTail224_16 : List Nat := 7607 :: innerTail224_17
def innerCandidates224_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_16_checked : candidates edge 4 1433 7607 innerTail224_17 best.bound = innerCandidates224_16 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_16 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_16 best = best := by
  rw [innerTail224_16, scan, if_neg (by decide), innerCandidates224_16_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_17
    (if 3 ≤ innerCandidates224_16.length then search edge 3 [1433, 7607] innerCandidates224_16 9040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_17

def innerTail224_15 : List Nat := 6761 :: innerTail224_16
def innerCandidates224_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_15_checked : candidates edge 4 1433 6761 innerTail224_16 best.bound = innerCandidates224_15 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_15 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_15 best = best := by
  rw [innerTail224_15, scan, if_neg (by decide), innerCandidates224_15_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_16
    (if 3 ≤ innerCandidates224_15.length then search edge 3 [1433, 6761] innerCandidates224_15 8194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_16

def innerTail224_14 : List Nat := 6737 :: innerTail224_15
def innerCandidates224_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_14_checked : candidates edge 4 1433 6737 innerTail224_15 best.bound = innerCandidates224_14 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_14 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_14 best = best := by
  rw [innerTail224_14, scan, if_neg (by decide), innerCandidates224_14_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_15
    (if 3 ≤ innerCandidates224_14.length then search edge 3 [1433, 6737] innerCandidates224_14 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_15

def innerTail224_13 : List Nat := 6299 :: innerTail224_14
def innerCandidates224_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_13_checked : candidates edge 4 1433 6299 innerTail224_14 best.bound = innerCandidates224_13 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_13 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_13 best = best := by
  rw [innerTail224_13, scan, if_neg (by decide), innerCandidates224_13_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_14
    (if 3 ≤ innerCandidates224_13.length then search edge 3 [1433, 6299] innerCandidates224_13 7732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_14

def innerTail224_12 : List Nat := 6029 :: innerTail224_13
def innerCandidates224_12 : List Nat := [7607]
set_option maxRecDepth 4096 in
theorem innerCandidates224_12_checked : candidates edge 4 1433 6029 innerTail224_13 best.bound = innerCandidates224_12 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_12 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_12 best = best := by
  rw [innerTail224_12, scan, if_neg (by decide), innerCandidates224_12_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_13
    (if 3 ≤ innerCandidates224_12.length then search edge 3 [1433, 6029] innerCandidates224_12 7462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_13

def innerTail224_11 : List Nat := 5903 :: innerTail224_12
def innerCandidates224_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_11_checked : candidates edge 4 1433 5903 innerTail224_12 best.bound = innerCandidates224_11 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_11 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_11 best = best := by
  rw [innerTail224_11, scan, if_neg (by decide), innerCandidates224_11_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_12
    (if 3 ≤ innerCandidates224_11.length then search edge 3 [1433, 5903] innerCandidates224_11 7336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_12

def innerTail224_10 : List Nat := 5501 :: innerTail224_11
def innerCandidates224_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_10_checked : candidates edge 4 1433 5501 innerTail224_11 best.bound = innerCandidates224_10 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_10 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_10 best = best := by
  rw [innerTail224_10, scan, if_neg (by decide), innerCandidates224_10_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_11
    (if 3 ≤ innerCandidates224_10.length then search edge 3 [1433, 5501] innerCandidates224_10 6934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_11

def innerTail224_9 : List Nat := 5393 :: innerTail224_10
def innerCandidates224_9 : List Nat := [7937]
set_option maxRecDepth 4096 in
theorem innerCandidates224_9_checked : candidates edge 4 1433 5393 innerTail224_10 best.bound = innerCandidates224_9 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_9 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_9 best = best := by
  rw [innerTail224_9, scan, if_neg (by decide), innerCandidates224_9_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_10
    (if 3 ≤ innerCandidates224_9.length then search edge 3 [1433, 5393] innerCandidates224_9 6826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_10

def innerTail224_8 : List Nat := 4643 :: innerTail224_9
def innerCandidates224_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_8_checked : candidates edge 4 1433 4643 innerTail224_9 best.bound = innerCandidates224_8 := by
  unfold candidates
  have row : edge 4643 = SearchData.row4643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_8 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_8 best = best := by
  rw [innerTail224_8, scan, if_neg (by decide), innerCandidates224_8_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_9
    (if 3 ≤ innerCandidates224_8.length then search edge 3 [1433, 4643] innerCandidates224_8 6076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_9

def innerTail224_7 : List Nat := 4637 :: innerTail224_8
def innerCandidates224_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_7_checked : candidates edge 4 1433 4637 innerTail224_8 best.bound = innerCandidates224_7 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_7 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_7 best = best := by
  rw [innerTail224_7, scan, if_neg (by decide), innerCandidates224_7_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_8
    (if 3 ≤ innerCandidates224_7.length then search edge 3 [1433, 4637] innerCandidates224_7 6070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_8

def innerTail224_6 : List Nat := 4283 :: innerTail224_7
def innerCandidates224_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_6_checked : candidates edge 4 1433 4283 innerTail224_7 best.bound = innerCandidates224_6 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_6 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_6 best = best := by
  rw [innerTail224_6, scan, if_neg (by decide), innerCandidates224_6_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_7
    (if 3 ≤ innerCandidates224_6.length then search edge 3 [1433, 4283] innerCandidates224_6 5716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_7

def innerTail224_5 : List Nat := 4157 :: innerTail224_6
def innerCandidates224_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_5_checked : candidates edge 4 1433 4157 innerTail224_6 best.bound = innerCandidates224_5 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_5 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_5 best = best := by
  rw [innerTail224_5, scan, if_neg (by decide), innerCandidates224_5_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_6
    (if 3 ≤ innerCandidates224_5.length then search edge 3 [1433, 4157] innerCandidates224_5 5590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_6

def innerTail224_4 : List Nat := 3323 :: innerTail224_5
def innerCandidates224_4 : List Nat := [5393]
set_option maxRecDepth 4096 in
theorem innerCandidates224_4_checked : candidates edge 4 1433 3323 innerTail224_5 best.bound = innerCandidates224_4 := by
  unfold candidates
  have row : edge 3323 = SearchData.row3323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_4 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_4 best = best := by
  rw [innerTail224_4, scan, if_neg (by decide), innerCandidates224_4_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_5
    (if 3 ≤ innerCandidates224_4.length then search edge 3 [1433, 3323] innerCandidates224_4 4756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_5

def innerTail224_3 : List Nat := 3089 :: innerTail224_4
def innerCandidates224_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_3_checked : candidates edge 4 1433 3089 innerTail224_4 best.bound = innerCandidates224_3 := by
  unfold candidates
  have row : edge 3089 = SearchData.row3089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_3 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_3 best = best := by
  rw [innerTail224_3, scan, if_neg (by decide), innerCandidates224_3_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_4
    (if 3 ≤ innerCandidates224_3.length then search edge 3 [1433, 3089] innerCandidates224_3 4522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_4

def innerTail224_2 : List Nat := 2999 :: innerTail224_3
def innerCandidates224_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_2_checked : candidates edge 4 1433 2999 innerTail224_3 best.bound = innerCandidates224_2 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_2 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_2 best = best := by
  rw [innerTail224_2, scan, if_neg (by decide), innerCandidates224_2_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_3
    (if 3 ≤ innerCandidates224_2.length then search edge 3 [1433, 2999] innerCandidates224_2 4432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_3

def innerTail224_1 : List Nat := 2957 :: innerTail224_2
def innerCandidates224_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates224_1_checked : candidates edge 4 1433 2957 innerTail224_2 best.bound = innerCandidates224_1 := by
  unfold candidates
  have row : edge 2957 = SearchData.row2957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_1 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_1 best = best := by
  rw [innerTail224_1, scan, if_neg (by decide), innerCandidates224_1_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_2
    (if 3 ≤ innerCandidates224_1.length then search edge 3 [1433, 2957] innerCandidates224_1 4390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_2

def innerTail224_0 : List Nat := 2663 :: innerTail224_1
def innerCandidates224_0 : List Nat := [7607]
set_option maxRecDepth 4096 in
theorem innerCandidates224_0_checked : candidates edge 4 1433 2663 innerTail224_1 best.bound = innerCandidates224_0 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan224_0 : scan edge (search edge 3) 4 [1433] 1433 innerTail224_0 best = best := by
  rw [innerTail224_0, scan, if_neg (by decide), innerCandidates224_0_checked]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_1
    (if 3 ≤ innerCandidates224_0.length then search edge 3 [1433, 2663] innerCandidates224_0 4096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan224_1

theorem rootBranch224 : search edge 4 [1433] rootCandidates224 1433 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1433] 1433 innerTail224_0 best = best
  exact innerScan224_0

def rootCandidates223 : List Nat := [1723, 1831, 2221, 3313, 3691, 3739, 3919, 4957, 6211, 6343, 6427, 6553, 6619, 6967, 7351, 7477, 7963, 8101]
set_option maxRecDepth 4096 in
theorem rootCandidates223_checked : candidates edge 5 0 1429 tail224 best.bound = rootCandidates223 := by
  unfold candidates
  have row : edge 1429 = PrimePairCache.smallRow1429.contains := by funext q; rfl
  rw [row]
  decide

def innerTail223_18 : List Nat := []
theorem innerScan223_18 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_18 best = best := rfl

def innerTail223_17 : List Nat := 8101 :: innerTail223_18
def innerCandidates223_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_17_checked : candidates edge 4 1429 8101 innerTail223_18 best.bound = innerCandidates223_17 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_17 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_17 best = best := by
  rw [innerTail223_17, scan, if_neg (by decide), innerCandidates223_17_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_18
    (if 3 ≤ innerCandidates223_17.length then search edge 3 [1429, 8101] innerCandidates223_17 9530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_18

def innerTail223_16 : List Nat := 7963 :: innerTail223_17
def innerCandidates223_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_16_checked : candidates edge 4 1429 7963 innerTail223_17 best.bound = innerCandidates223_16 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_16 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_16 best = best := by
  rw [innerTail223_16, scan, if_neg (by decide), innerCandidates223_16_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_17
    (if 3 ≤ innerCandidates223_16.length then search edge 3 [1429, 7963] innerCandidates223_16 9392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_17

def innerTail223_15 : List Nat := 7477 :: innerTail223_16
def innerCandidates223_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_15_checked : candidates edge 4 1429 7477 innerTail223_16 best.bound = innerCandidates223_15 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_15 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_15 best = best := by
  rw [innerTail223_15, scan, if_neg (by decide), innerCandidates223_15_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_16
    (if 3 ≤ innerCandidates223_15.length then search edge 3 [1429, 7477] innerCandidates223_15 8906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_16

def innerTail223_14 : List Nat := 7351 :: innerTail223_15
def innerCandidates223_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_14_checked : candidates edge 4 1429 7351 innerTail223_15 best.bound = innerCandidates223_14 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_14 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_14 best = best := by
  rw [innerTail223_14, scan, if_neg (by decide), innerCandidates223_14_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_15
    (if 3 ≤ innerCandidates223_14.length then search edge 3 [1429, 7351] innerCandidates223_14 8780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_15

def innerTail223_13 : List Nat := 6967 :: innerTail223_14
def innerCandidates223_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_13_checked : candidates edge 4 1429 6967 innerTail223_14 best.bound = innerCandidates223_13 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_13 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_13 best = best := by
  rw [innerTail223_13, scan, if_neg (by decide), innerCandidates223_13_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_14
    (if 3 ≤ innerCandidates223_13.length then search edge 3 [1429, 6967] innerCandidates223_13 8396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_14

def innerTail223_12 : List Nat := 6619 :: innerTail223_13
def innerCandidates223_12 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates223_12_checked : candidates edge 4 1429 6619 innerTail223_13 best.bound = innerCandidates223_12 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_12 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_12 best = best := by
  rw [innerTail223_12, scan, if_neg (by decide), innerCandidates223_12_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_13
    (if 3 ≤ innerCandidates223_12.length then search edge 3 [1429, 6619] innerCandidates223_12 8048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_13

def innerTail223_11 : List Nat := 6553 :: innerTail223_12
def innerCandidates223_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_11_checked : candidates edge 4 1429 6553 innerTail223_12 best.bound = innerCandidates223_11 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_11 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_11 best = best := by
  rw [innerTail223_11, scan, if_neg (by decide), innerCandidates223_11_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_12
    (if 3 ≤ innerCandidates223_11.length then search edge 3 [1429, 6553] innerCandidates223_11 7982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_12

def innerTail223_10 : List Nat := 6427 :: innerTail223_11
def innerCandidates223_10 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates223_10_checked : candidates edge 4 1429 6427 innerTail223_11 best.bound = innerCandidates223_10 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_10 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_10 best = best := by
  rw [innerTail223_10, scan, if_neg (by decide), innerCandidates223_10_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_11
    (if 3 ≤ innerCandidates223_10.length then search edge 3 [1429, 6427] innerCandidates223_10 7856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_11

def innerTail223_9 : List Nat := 6343 :: innerTail223_10
def innerCandidates223_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_9_checked : candidates edge 4 1429 6343 innerTail223_10 best.bound = innerCandidates223_9 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_9 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_9 best = best := by
  rw [innerTail223_9, scan, if_neg (by decide), innerCandidates223_9_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_10
    (if 3 ≤ innerCandidates223_9.length then search edge 3 [1429, 6343] innerCandidates223_9 7772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_10

def innerTail223_8 : List Nat := 6211 :: innerTail223_9
def innerCandidates223_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_8_checked : candidates edge 4 1429 6211 innerTail223_9 best.bound = innerCandidates223_8 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_8 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_8 best = best := by
  rw [innerTail223_8, scan, if_neg (by decide), innerCandidates223_8_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_9
    (if 3 ≤ innerCandidates223_8.length then search edge 3 [1429, 6211] innerCandidates223_8 7640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_9

def innerTail223_7 : List Nat := 4957 :: innerTail223_8
def innerCandidates223_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_7_checked : candidates edge 4 1429 4957 innerTail223_8 best.bound = innerCandidates223_7 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_7 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_7 best = best := by
  rw [innerTail223_7, scan, if_neg (by decide), innerCandidates223_7_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_8
    (if 3 ≤ innerCandidates223_7.length then search edge 3 [1429, 4957] innerCandidates223_7 6386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_8

def innerTail223_6 : List Nat := 3919 :: innerTail223_7
def innerCandidates223_6 : List Nat := [4957]
set_option maxRecDepth 4096 in
theorem innerCandidates223_6_checked : candidates edge 4 1429 3919 innerTail223_7 best.bound = innerCandidates223_6 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_6 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_6 best = best := by
  rw [innerTail223_6, scan, if_neg (by decide), innerCandidates223_6_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_7
    (if 3 ≤ innerCandidates223_6.length then search edge 3 [1429, 3919] innerCandidates223_6 5348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_7

def innerTail223_5 : List Nat := 3739 :: innerTail223_6
def innerCandidates223_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_5_checked : candidates edge 4 1429 3739 innerTail223_6 best.bound = innerCandidates223_5 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_5 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_5 best = best := by
  rw [innerTail223_5, scan, if_neg (by decide), innerCandidates223_5_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_6
    (if 3 ≤ innerCandidates223_5.length then search edge 3 [1429, 3739] innerCandidates223_5 5168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_6

def innerTail223_4 : List Nat := 3691 :: innerTail223_5
def innerCandidates223_4 : List Nat := [7477]
set_option maxRecDepth 4096 in
theorem innerCandidates223_4_checked : candidates edge 4 1429 3691 innerTail223_5 best.bound = innerCandidates223_4 := by
  unfold candidates
  have row : edge 3691 = SearchData.row3691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_4 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_4 best = best := by
  rw [innerTail223_4, scan, if_neg (by decide), innerCandidates223_4_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_5
    (if 3 ≤ innerCandidates223_4.length then search edge 3 [1429, 3691] innerCandidates223_4 5120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_5

def innerTail223_3 : List Nat := 3313 :: innerTail223_4
def innerCandidates223_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_3_checked : candidates edge 4 1429 3313 innerTail223_4 best.bound = innerCandidates223_3 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_3 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_3 best = best := by
  rw [innerTail223_3, scan, if_neg (by decide), innerCandidates223_3_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_4
    (if 3 ≤ innerCandidates223_3.length then search edge 3 [1429, 3313] innerCandidates223_3 4742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_4

def innerTail223_2 : List Nat := 2221 :: innerTail223_3
def innerCandidates223_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_2_checked : candidates edge 4 1429 2221 innerTail223_3 best.bound = innerCandidates223_2 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_2 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_2 best = best := by
  rw [innerTail223_2, scan, if_neg (by decide), innerCandidates223_2_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_3
    (if 3 ≤ innerCandidates223_2.length then search edge 3 [1429, 2221] innerCandidates223_2 3650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_3

def innerTail223_1 : List Nat := 1831 :: innerTail223_2
def innerCandidates223_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates223_1_checked : candidates edge 4 1429 1831 innerTail223_2 best.bound = innerCandidates223_1 := by
  unfold candidates
  have row : edge 1831 = PrimePairCache.smallRow1831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_1 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_1 best = best := by
  rw [innerTail223_1, scan, if_neg (by decide), innerCandidates223_1_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_2
    (if 3 ≤ innerCandidates223_1.length then search edge 3 [1429, 1831] innerCandidates223_1 3260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_2

def innerTail223_0 : List Nat := 1723 :: innerTail223_1
def innerCandidates223_0 : List Nat := [1831]
set_option maxRecDepth 4096 in
theorem innerCandidates223_0_checked : candidates edge 4 1429 1723 innerTail223_1 best.bound = innerCandidates223_0 := by
  unfold candidates
  have row : edge 1723 = PrimePairCache.smallRow1723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan223_0 : scan edge (search edge 3) 4 [1429] 1429 innerTail223_0 best = best := by
  rw [innerTail223_0, scan, if_neg (by decide), innerCandidates223_0_checked]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_1
    (if 3 ≤ innerCandidates223_0.length then search edge 3 [1429, 1723] innerCandidates223_0 3152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan223_1

theorem rootBranch223 : search edge 4 [1429] rootCandidates223 1429 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1429] 1429 innerTail223_0 best = best
  exact innerScan223_0

def rootCandidates222 : List Nat := [1931, 2339, 2777, 3011, 3359, 3767, 3989, 4019, 4157, 4451, 5297, 5477, 5501, 5843, 7451, 7937, 8111, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates222_checked : candidates edge 5 0 1427 tail223 best.bound = rootCandidates222 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext q; rfl
  rw [row]
  decide

def innerTail222_18 : List Nat := []
theorem innerScan222_18 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_18 best = best := rfl

def innerTail222_17 : List Nat := 8273 :: innerTail222_18
def innerCandidates222_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates222_17_checked : candidates edge 4 1427 8273 innerTail222_18 best.bound = innerCandidates222_17 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_17 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_17 best = best := by
  rw [innerTail222_17, scan, if_neg (by decide), innerCandidates222_17_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_18
    (if 3 ≤ innerCandidates222_17.length then search edge 3 [1427, 8273] innerCandidates222_17 9700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_18

def innerTail222_16 : List Nat := 8111 :: innerTail222_17
def innerCandidates222_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates222_16_checked : candidates edge 4 1427 8111 innerTail222_17 best.bound = innerCandidates222_16 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_16 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_16 best = best := by
  rw [innerTail222_16, scan, if_neg (by decide), innerCandidates222_16_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_17
    (if 3 ≤ innerCandidates222_16.length then search edge 3 [1427, 8111] innerCandidates222_16 9538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_17

def innerTail222_15 : List Nat := 7937 :: innerTail222_16
def innerCandidates222_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates222_15_checked : candidates edge 4 1427 7937 innerTail222_16 best.bound = innerCandidates222_15 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_15 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_15 best = best := by
  rw [innerTail222_15, scan, if_neg (by decide), innerCandidates222_15_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_16
    (if 3 ≤ innerCandidates222_15.length then search edge 3 [1427, 7937] innerCandidates222_15 9364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_16

def innerTail222_14 : List Nat := 7451 :: innerTail222_15
def innerCandidates222_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates222_14_checked : candidates edge 4 1427 7451 innerTail222_15 best.bound = innerCandidates222_14 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_14 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_14 best = best := by
  rw [innerTail222_14, scan, if_neg (by decide), innerCandidates222_14_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_15
    (if 3 ≤ innerCandidates222_14.length then search edge 3 [1427, 7451] innerCandidates222_14 8878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_15

def innerTail222_13 : List Nat := 5843 :: innerTail222_14
def innerCandidates222_13 : List Nat := [7937]
set_option maxRecDepth 4096 in
theorem innerCandidates222_13_checked : candidates edge 4 1427 5843 innerTail222_14 best.bound = innerCandidates222_13 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_13 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_13 best = best := by
  rw [innerTail222_13, scan, if_neg (by decide), innerCandidates222_13_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_14
    (if 3 ≤ innerCandidates222_13.length then search edge 3 [1427, 5843] innerCandidates222_13 7270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_14

def innerTail222_12 : List Nat := 5501 :: innerTail222_13
def innerCandidates222_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates222_12_checked : candidates edge 4 1427 5501 innerTail222_13 best.bound = innerCandidates222_12 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_12 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_12 best = best := by
  rw [innerTail222_12, scan, if_neg (by decide), innerCandidates222_12_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_13
    (if 3 ≤ innerCandidates222_12.length then search edge 3 [1427, 5501] innerCandidates222_12 6928 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_13

def innerTail222_11 : List Nat := 5477 :: innerTail222_12
def innerCandidates222_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates222_11_checked : candidates edge 4 1427 5477 innerTail222_12 best.bound = innerCandidates222_11 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_11 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_11 best = best := by
  rw [innerTail222_11, scan, if_neg (by decide), innerCandidates222_11_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_12
    (if 3 ≤ innerCandidates222_11.length then search edge 3 [1427, 5477] innerCandidates222_11 6904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_12

def innerTail222_10 : List Nat := 5297 :: innerTail222_11
def innerCandidates222_10 : List Nat := [5843]
set_option maxRecDepth 4096 in
theorem innerCandidates222_10_checked : candidates edge 4 1427 5297 innerTail222_11 best.bound = innerCandidates222_10 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_10 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_10 best = best := by
  rw [innerTail222_10, scan, if_neg (by decide), innerCandidates222_10_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_11
    (if 3 ≤ innerCandidates222_10.length then search edge 3 [1427, 5297] innerCandidates222_10 6724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_11

def innerTail222_9 : List Nat := 4451 :: innerTail222_10
def innerCandidates222_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates222_9_checked : candidates edge 4 1427 4451 innerTail222_10 best.bound = innerCandidates222_9 := by
  unfold candidates
  have row : edge 4451 = SearchData.row4451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_9 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_9 best = best := by
  rw [innerTail222_9, scan, if_neg (by decide), innerCandidates222_9_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_10
    (if 3 ≤ innerCandidates222_9.length then search edge 3 [1427, 4451] innerCandidates222_9 5878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_10

def innerTail222_8 : List Nat := 4157 :: innerTail222_9
def innerCandidates222_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates222_8_checked : candidates edge 4 1427 4157 innerTail222_9 best.bound = innerCandidates222_8 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_8 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_8 best = best := by
  rw [innerTail222_8, scan, if_neg (by decide), innerCandidates222_8_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_9
    (if 3 ≤ innerCandidates222_8.length then search edge 3 [1427, 4157] innerCandidates222_8 5584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_9

def innerTail222_7 : List Nat := 4019 :: innerTail222_8
def innerCandidates222_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates222_7_checked : candidates edge 4 1427 4019 innerTail222_8 best.bound = innerCandidates222_7 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_7 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_7 best = best := by
  rw [innerTail222_7, scan, if_neg (by decide), innerCandidates222_7_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_8
    (if 3 ≤ innerCandidates222_7.length then search edge 3 [1427, 4019] innerCandidates222_7 5446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_8

def innerTail222_6 : List Nat := 3989 :: innerTail222_7
def innerCandidates222_6 : List Nat := [4157]
set_option maxRecDepth 4096 in
theorem innerCandidates222_6_checked : candidates edge 4 1427 3989 innerTail222_7 best.bound = innerCandidates222_6 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_6 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_6 best = best := by
  rw [innerTail222_6, scan, if_neg (by decide), innerCandidates222_6_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_7
    (if 3 ≤ innerCandidates222_6.length then search edge 3 [1427, 3989] innerCandidates222_6 5416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_7

def innerTail222_5 : List Nat := 3767 :: innerTail222_6
def innerCandidates222_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates222_5_checked : candidates edge 4 1427 3767 innerTail222_6 best.bound = innerCandidates222_5 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_5 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_5 best = best := by
  rw [innerTail222_5, scan, if_neg (by decide), innerCandidates222_5_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_6
    (if 3 ≤ innerCandidates222_5.length then search edge 3 [1427, 3767] innerCandidates222_5 5194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_6

def innerTail222_4 : List Nat := 3359 :: innerTail222_5
def innerCandidates222_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates222_4_checked : candidates edge 4 1427 3359 innerTail222_5 best.bound = innerCandidates222_4 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_4 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_4 best = best := by
  rw [innerTail222_4, scan, if_neg (by decide), innerCandidates222_4_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_5
    (if 3 ≤ innerCandidates222_4.length then search edge 3 [1427, 3359] innerCandidates222_4 4786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_5

def innerTail222_3 : List Nat := 3011 :: innerTail222_4
def innerCandidates222_3 : List Nat := [7937]
set_option maxRecDepth 4096 in
theorem innerCandidates222_3_checked : candidates edge 4 1427 3011 innerTail222_4 best.bound = innerCandidates222_3 := by
  unfold candidates
  have row : edge 3011 = SearchData.row3011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_3 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_3 best = best := by
  rw [innerTail222_3, scan, if_neg (by decide), innerCandidates222_3_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_4
    (if 3 ≤ innerCandidates222_3.length then search edge 3 [1427, 3011] innerCandidates222_3 4438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_4

def innerTail222_2 : List Nat := 2777 :: innerTail222_3
def innerCandidates222_2 : List Nat := [5477]
set_option maxRecDepth 4096 in
theorem innerCandidates222_2_checked : candidates edge 4 1427 2777 innerTail222_3 best.bound = innerCandidates222_2 := by
  unfold candidates
  have row : edge 2777 = SearchData.row2777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_2 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_2 best = best := by
  rw [innerTail222_2, scan, if_neg (by decide), innerCandidates222_2_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_3
    (if 3 ≤ innerCandidates222_2.length then search edge 3 [1427, 2777] innerCandidates222_2 4204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_3

def innerTail222_1 : List Nat := 2339 :: innerTail222_2
def innerCandidates222_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates222_1_checked : candidates edge 4 1427 2339 innerTail222_2 best.bound = innerCandidates222_1 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_1 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_1 best = best := by
  rw [innerTail222_1, scan, if_neg (by decide), innerCandidates222_1_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_2
    (if 3 ≤ innerCandidates222_1.length then search edge 3 [1427, 2339] innerCandidates222_1 3766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_2

def innerTail222_0 : List Nat := 1931 :: innerTail222_1
def innerCandidates222_0 : List Nat := [5297, 5477]
set_option maxRecDepth 4096 in
theorem innerCandidates222_0_checked : candidates edge 4 1427 1931 innerTail222_1 best.bound = innerCandidates222_0 := by
  unfold candidates
  have row : edge 1931 = PrimePairCache.smallRow1931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan222_0 : scan edge (search edge 3) 4 [1427] 1427 innerTail222_0 best = best := by
  rw [innerTail222_0, scan, if_neg (by decide), innerCandidates222_0_checked]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_1
    (if 3 ≤ innerCandidates222_0.length then search edge 3 [1427, 1931] innerCandidates222_0 3358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan222_1

theorem rootBranch222 : search edge 4 [1427] rootCandidates222 1427 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1427] 1427 innerTail222_0 best = best
  exact innerScan222_0

end ProjectEulerSolutions.PrimePairUpper.Search
