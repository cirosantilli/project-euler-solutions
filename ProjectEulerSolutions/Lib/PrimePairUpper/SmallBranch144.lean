import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates244 : List Nat := [1879, 2161, 2311, 2539, 2551, 2593, 3919, 4441, 4507, 4567, 4597, 5431, 5641, 6277, 6481, 7027, 7759, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates244_checked : candidates edge 5 0 1567 tail245 best.bound = rootCandidates244 := by
  unfold candidates
  have row : edge 1567 = PrimePairCache.smallRow1567.contains := by funext q; rfl
  rw [row]
  decide

def innerTail244_18 : List Nat := []
theorem innerScan244_18 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_18 best = best := rfl

def innerTail244_17 : List Nat := 8293 :: innerTail244_18
def innerCandidates244_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates244_17_checked : candidates edge 4 1567 8293 innerTail244_18 best.bound = innerCandidates244_17 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_17 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_17 best = best := by
  rw [innerTail244_17, scan, if_neg (by decide), innerCandidates244_17_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_18
    (if 3 ≤ innerCandidates244_17.length then search edge 3 [1567, 8293] innerCandidates244_17 9860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_18

def innerTail244_16 : List Nat := 7759 :: innerTail244_17
def innerCandidates244_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates244_16_checked : candidates edge 4 1567 7759 innerTail244_17 best.bound = innerCandidates244_16 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_16 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_16 best = best := by
  rw [innerTail244_16, scan, if_neg (by decide), innerCandidates244_16_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_17
    (if 3 ≤ innerCandidates244_16.length then search edge 3 [1567, 7759] innerCandidates244_16 9326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_17

def innerTail244_15 : List Nat := 7027 :: innerTail244_16
def innerCandidates244_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates244_15_checked : candidates edge 4 1567 7027 innerTail244_16 best.bound = innerCandidates244_15 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_15 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_15 best = best := by
  rw [innerTail244_15, scan, if_neg (by decide), innerCandidates244_15_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_16
    (if 3 ≤ innerCandidates244_15.length then search edge 3 [1567, 7027] innerCandidates244_15 8594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_16

def innerTail244_14 : List Nat := 6481 :: innerTail244_15
def innerCandidates244_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates244_14_checked : candidates edge 4 1567 6481 innerTail244_15 best.bound = innerCandidates244_14 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_14 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_14 best = best := by
  rw [innerTail244_14, scan, if_neg (by decide), innerCandidates244_14_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_15
    (if 3 ≤ innerCandidates244_14.length then search edge 3 [1567, 6481] innerCandidates244_14 8048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_15

def innerTail244_13 : List Nat := 6277 :: innerTail244_14
def innerCandidates244_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates244_13_checked : candidates edge 4 1567 6277 innerTail244_14 best.bound = innerCandidates244_13 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_13 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_13 best = best := by
  rw [innerTail244_13, scan, if_neg (by decide), innerCandidates244_13_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_14
    (if 3 ≤ innerCandidates244_13.length then search edge 3 [1567, 6277] innerCandidates244_13 7844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_14

def innerTail244_12 : List Nat := 5641 :: innerTail244_13
def innerCandidates244_12 : List Nat := [7027, 7759]
set_option maxRecDepth 4096 in
theorem innerCandidates244_12_checked : candidates edge 4 1567 5641 innerTail244_13 best.bound = innerCandidates244_12 := by
  unfold candidates
  have row : edge 5641 = SearchData.row5641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_12 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_12 best = best := by
  rw [innerTail244_12, scan, if_neg (by decide), innerCandidates244_12_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_13
    (if 3 ≤ innerCandidates244_12.length then search edge 3 [1567, 5641] innerCandidates244_12 7208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_13

def innerTail244_11 : List Nat := 5431 :: innerTail244_12
def innerCandidates244_11 : List Nat := [5641]
set_option maxRecDepth 4096 in
theorem innerCandidates244_11_checked : candidates edge 4 1567 5431 innerTail244_12 best.bound = innerCandidates244_11 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_11 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_11 best = best := by
  rw [innerTail244_11, scan, if_neg (by decide), innerCandidates244_11_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_12
    (if 3 ≤ innerCandidates244_11.length then search edge 3 [1567, 5431] innerCandidates244_11 6998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_12

def innerTail244_10 : List Nat := 4597 :: innerTail244_11
def innerCandidates244_10 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates244_10_checked : candidates edge 4 1567 4597 innerTail244_11 best.bound = innerCandidates244_10 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_10 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_10 best = best := by
  rw [innerTail244_10, scan, if_neg (by decide), innerCandidates244_10_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_11
    (if 3 ≤ innerCandidates244_10.length then search edge 3 [1567, 4597] innerCandidates244_10 6164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_11

def innerTail244_9 : List Nat := 4567 :: innerTail244_10
def innerCandidates244_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates244_9_checked : candidates edge 4 1567 4567 innerTail244_10 best.bound = innerCandidates244_9 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_9 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_9 best = best := by
  rw [innerTail244_9, scan, if_neg (by decide), innerCandidates244_9_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_10
    (if 3 ≤ innerCandidates244_9.length then search edge 3 [1567, 4567] innerCandidates244_9 6134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_10

def innerTail244_8 : List Nat := 4507 :: innerTail244_9
def innerCandidates244_8 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates244_8_checked : candidates edge 4 1567 4507 innerTail244_9 best.bound = innerCandidates244_8 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_8 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_8 best = best := by
  rw [innerTail244_8, scan, if_neg (by decide), innerCandidates244_8_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_9
    (if 3 ≤ innerCandidates244_8.length then search edge 3 [1567, 4507] innerCandidates244_8 6074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_9

def innerTail244_7 : List Nat := 4441 :: innerTail244_8
def innerCandidates244_7 : List Nat := [6481, 7759]
set_option maxRecDepth 4096 in
theorem innerCandidates244_7_checked : candidates edge 4 1567 4441 innerTail244_8 best.bound = innerCandidates244_7 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_7 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_7 best = best := by
  rw [innerTail244_7, scan, if_neg (by decide), innerCandidates244_7_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_8
    (if 3 ≤ innerCandidates244_7.length then search edge 3 [1567, 4441] innerCandidates244_7 6008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_8

def innerTail244_6 : List Nat := 3919 :: innerTail244_7
def innerCandidates244_6 : List Nat := [4567]
set_option maxRecDepth 4096 in
theorem innerCandidates244_6_checked : candidates edge 4 1567 3919 innerTail244_7 best.bound = innerCandidates244_6 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_6 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_6 best = best := by
  rw [innerTail244_6, scan, if_neg (by decide), innerCandidates244_6_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_7
    (if 3 ≤ innerCandidates244_6.length then search edge 3 [1567, 3919] innerCandidates244_6 5486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_7

def innerTail244_5 : List Nat := 2593 :: innerTail244_6
def innerCandidates244_5 : List Nat := [7027, 8293]
set_option maxRecDepth 4096 in
theorem innerCandidates244_5_checked : candidates edge 4 1567 2593 innerTail244_6 best.bound = innerCandidates244_5 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_5 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_5 best = best := by
  rw [innerTail244_5, scan, if_neg (by decide), innerCandidates244_5_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_6
    (if 3 ≤ innerCandidates244_5.length then search edge 3 [1567, 2593] innerCandidates244_5 4160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_6

def innerTail244_4 : List Nat := 2551 :: innerTail244_5
def innerCandidates244_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates244_4_checked : candidates edge 4 1567 2551 innerTail244_5 best.bound = innerCandidates244_4 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_4 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_4 best = best := by
  rw [innerTail244_4, scan, if_neg (by decide), innerCandidates244_4_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_5
    (if 3 ≤ innerCandidates244_4.length then search edge 3 [1567, 2551] innerCandidates244_4 4118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_5

def innerTail244_3 : List Nat := 2539 :: innerTail244_4
def innerCandidates244_3 : List Nat := [4441]
set_option maxRecDepth 4096 in
theorem innerCandidates244_3_checked : candidates edge 4 1567 2539 innerTail244_4 best.bound = innerCandidates244_3 := by
  unfold candidates
  have row : edge 2539 = PrimePairCache.smallRow2539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_3 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_3 best = best := by
  rw [innerTail244_3, scan, if_neg (by decide), innerCandidates244_3_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_4
    (if 3 ≤ innerCandidates244_3.length then search edge 3 [1567, 2539] innerCandidates244_3 4106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_4

def innerTail244_2 : List Nat := 2311 :: innerTail244_3
def innerCandidates244_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates244_2_checked : candidates edge 4 1567 2311 innerTail244_3 best.bound = innerCandidates244_2 := by
  unfold candidates
  have row : edge 2311 = PrimePairCache.smallRow2311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_2 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_2 best = best := by
  rw [innerTail244_2, scan, if_neg (by decide), innerCandidates244_2_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_3
    (if 3 ≤ innerCandidates244_2.length then search edge 3 [1567, 2311] innerCandidates244_2 3878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_3

def innerTail244_1 : List Nat := 2161 :: innerTail244_2
def innerCandidates244_1 : List Nat := [4597, 7027, 7759]
set_option maxRecDepth 4096 in
theorem innerCandidates244_1_checked : candidates edge 4 1567 2161 innerTail244_2 best.bound = innerCandidates244_1 := by
  unfold candidates
  have row : edge 2161 = PrimePairCache.smallRow2161.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild244_1 : search edge 3 [1567, 2161] innerCandidates244_1 3728 best = best := by decide

theorem innerScan244_1 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_1 best = best := by
  rw [innerTail244_1, scan, if_neg (by decide), innerCandidates244_1_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_2
    (if 3 ≤ innerCandidates244_1.length then search edge 3 [1567, 2161] innerCandidates244_1 3728 best else best) = best
  rw [if_pos (by decide), innerChild244_1]
  exact innerScan244_2

def innerTail244_0 : List Nat := 1879 :: innerTail244_1
def innerCandidates244_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates244_0_checked : candidates edge 4 1567 1879 innerTail244_1 best.bound = innerCandidates244_0 := by
  unfold candidates
  have row : edge 1879 = PrimePairCache.smallRow1879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan244_0 : scan edge (search edge 3) 4 [1567] 1567 innerTail244_0 best = best := by
  rw [innerTail244_0, scan, if_neg (by decide), innerCandidates244_0_checked]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_1
    (if 3 ≤ innerCandidates244_0.length then search edge 3 [1567, 1879] innerCandidates244_0 3446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan244_1

theorem rootBranch244 : search edge 4 [1567] rootCandidates244 1567 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1567] 1567 innerTail244_0 best = best
  exact innerScan244_0

def rootCandidates243 : List Nat := [1721, 1787, 2477, 2693, 3191, 3593, 3911, 5441, 5507, 5861, 5897, 6299, 7079, 7559, 8171, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates243_checked : candidates edge 5 0 1559 tail244 best.bound = rootCandidates243 := by
  unfold candidates
  have row : edge 1559 = PrimePairCache.smallRow1559.contains := by funext q; rfl
  rw [row]
  decide

def innerTail243_16 : List Nat := []
theorem innerScan243_16 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_16 best = best := rfl

def innerTail243_15 : List Nat := 8237 :: innerTail243_16
def innerCandidates243_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_15_checked : candidates edge 4 1559 8237 innerTail243_16 best.bound = innerCandidates243_15 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_15 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_15 best = best := by
  rw [innerTail243_15, scan, if_neg (by decide), innerCandidates243_15_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_16
    (if 3 ≤ innerCandidates243_15.length then search edge 3 [1559, 8237] innerCandidates243_15 9796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_16

def innerTail243_14 : List Nat := 8171 :: innerTail243_15
def innerCandidates243_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_14_checked : candidates edge 4 1559 8171 innerTail243_15 best.bound = innerCandidates243_14 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_14 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_14 best = best := by
  rw [innerTail243_14, scan, if_neg (by decide), innerCandidates243_14_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_15
    (if 3 ≤ innerCandidates243_14.length then search edge 3 [1559, 8171] innerCandidates243_14 9730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_15

def innerTail243_13 : List Nat := 7559 :: innerTail243_14
def innerCandidates243_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_13_checked : candidates edge 4 1559 7559 innerTail243_14 best.bound = innerCandidates243_13 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_13 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_13 best = best := by
  rw [innerTail243_13, scan, if_neg (by decide), innerCandidates243_13_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_14
    (if 3 ≤ innerCandidates243_13.length then search edge 3 [1559, 7559] innerCandidates243_13 9118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_14

def innerTail243_12 : List Nat := 7079 :: innerTail243_13
def innerCandidates243_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_12_checked : candidates edge 4 1559 7079 innerTail243_13 best.bound = innerCandidates243_12 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_12 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_12 best = best := by
  rw [innerTail243_12, scan, if_neg (by decide), innerCandidates243_12_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_13
    (if 3 ≤ innerCandidates243_12.length then search edge 3 [1559, 7079] innerCandidates243_12 8638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_13

def innerTail243_11 : List Nat := 6299 :: innerTail243_12
def innerCandidates243_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_11_checked : candidates edge 4 1559 6299 innerTail243_12 best.bound = innerCandidates243_11 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_11 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_11 best = best := by
  rw [innerTail243_11, scan, if_neg (by decide), innerCandidates243_11_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_12
    (if 3 ≤ innerCandidates243_11.length then search edge 3 [1559, 6299] innerCandidates243_11 7858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_12

def innerTail243_10 : List Nat := 5897 :: innerTail243_11
def innerCandidates243_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_10_checked : candidates edge 4 1559 5897 innerTail243_11 best.bound = innerCandidates243_10 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_10 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_10 best = best := by
  rw [innerTail243_10, scan, if_neg (by decide), innerCandidates243_10_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_11
    (if 3 ≤ innerCandidates243_10.length then search edge 3 [1559, 5897] innerCandidates243_10 7456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_11

def innerTail243_9 : List Nat := 5861 :: innerTail243_10
def innerCandidates243_9 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates243_9_checked : candidates edge 4 1559 5861 innerTail243_10 best.bound = innerCandidates243_9 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_9 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_9 best = best := by
  rw [innerTail243_9, scan, if_neg (by decide), innerCandidates243_9_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_10
    (if 3 ≤ innerCandidates243_9.length then search edge 3 [1559, 5861] innerCandidates243_9 7420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_10

def innerTail243_8 : List Nat := 5507 :: innerTail243_9
def innerCandidates243_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_8_checked : candidates edge 4 1559 5507 innerTail243_9 best.bound = innerCandidates243_8 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_8 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_8 best = best := by
  rw [innerTail243_8, scan, if_neg (by decide), innerCandidates243_8_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_9
    (if 3 ≤ innerCandidates243_8.length then search edge 3 [1559, 5507] innerCandidates243_8 7066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_9

def innerTail243_7 : List Nat := 5441 :: innerTail243_8
def innerCandidates243_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_7_checked : candidates edge 4 1559 5441 innerTail243_8 best.bound = innerCandidates243_7 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_7 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_7 best = best := by
  rw [innerTail243_7, scan, if_neg (by decide), innerCandidates243_7_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_8
    (if 3 ≤ innerCandidates243_7.length then search edge 3 [1559, 5441] innerCandidates243_7 7000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_8

def innerTail243_6 : List Nat := 3911 :: innerTail243_7
def innerCandidates243_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_6_checked : candidates edge 4 1559 3911 innerTail243_7 best.bound = innerCandidates243_6 := by
  unfold candidates
  have row : edge 3911 = SearchData.row3911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_6 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_6 best = best := by
  rw [innerTail243_6, scan, if_neg (by decide), innerCandidates243_6_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_7
    (if 3 ≤ innerCandidates243_6.length then search edge 3 [1559, 3911] innerCandidates243_6 5470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_7

def innerTail243_5 : List Nat := 3593 :: innerTail243_6
def innerCandidates243_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_5_checked : candidates edge 4 1559 3593 innerTail243_6 best.bound = innerCandidates243_5 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_5 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_5 best = best := by
  rw [innerTail243_5, scan, if_neg (by decide), innerCandidates243_5_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_6
    (if 3 ≤ innerCandidates243_5.length then search edge 3 [1559, 3593] innerCandidates243_5 5152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_6

def innerTail243_4 : List Nat := 3191 :: innerTail243_5
def innerCandidates243_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_4_checked : candidates edge 4 1559 3191 innerTail243_5 best.bound = innerCandidates243_4 := by
  unfold candidates
  have row : edge 3191 = SearchData.row3191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_4 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_4 best = best := by
  rw [innerTail243_4, scan, if_neg (by decide), innerCandidates243_4_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_5
    (if 3 ≤ innerCandidates243_4.length then search edge 3 [1559, 3191] innerCandidates243_4 4750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_5

def innerTail243_3 : List Nat := 2693 :: innerTail243_4
def innerCandidates243_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_3_checked : candidates edge 4 1559 2693 innerTail243_4 best.bound = innerCandidates243_3 := by
  unfold candidates
  have row : edge 2693 = SearchData.row2693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_3 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_3 best = best := by
  rw [innerTail243_3, scan, if_neg (by decide), innerCandidates243_3_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_4
    (if 3 ≤ innerCandidates243_3.length then search edge 3 [1559, 2693] innerCandidates243_3 4252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_4

def innerTail243_2 : List Nat := 2477 :: innerTail243_3
def innerCandidates243_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_2_checked : candidates edge 4 1559 2477 innerTail243_3 best.bound = innerCandidates243_2 := by
  unfold candidates
  have row : edge 2477 = PrimePairCache.smallRow2477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_2 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_2 best = best := by
  rw [innerTail243_2, scan, if_neg (by decide), innerCandidates243_2_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_3
    (if 3 ≤ innerCandidates243_2.length then search edge 3 [1559, 2477] innerCandidates243_2 4036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_3

def innerTail243_1 : List Nat := 1787 :: innerTail243_2
def innerCandidates243_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_1_checked : candidates edge 4 1559 1787 innerTail243_2 best.bound = innerCandidates243_1 := by
  unfold candidates
  have row : edge 1787 = PrimePairCache.smallRow1787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_1 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_1 best = best := by
  rw [innerTail243_1, scan, if_neg (by decide), innerCandidates243_1_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_2
    (if 3 ≤ innerCandidates243_1.length then search edge 3 [1559, 1787] innerCandidates243_1 3346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_2

def innerTail243_0 : List Nat := 1721 :: innerTail243_1
def innerCandidates243_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates243_0_checked : candidates edge 4 1559 1721 innerTail243_1 best.bound = innerCandidates243_0 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan243_0 : scan edge (search edge 3) 4 [1559] 1559 innerTail243_0 best = best := by
  rw [innerTail243_0, scan, if_neg (by decide), innerCandidates243_0_checked]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_1
    (if 3 ≤ innerCandidates243_0.length then search edge 3 [1559, 1721] innerCandidates243_0 3280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan243_1

theorem rootBranch243 : search edge 4 [1559] rootCandidates243 1559 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1559] 1559 innerTail243_0 best = best
  exact innerScan243_0

def rootCandidates242 : List Nat := [1583, 1613, 1697, 1811, 2087, 2393, 2549, 2687, 4049, 6203, 6221, 6947, 7013, 7211, 7757, 7883, 8111, 8147]
set_option maxRecDepth 4096 in
theorem rootCandidates242_checked : candidates edge 5 0 1553 tail243 best.bound = rootCandidates242 := by
  unfold candidates
  have row : edge 1553 = PrimePairCache.smallRow1553.contains := by funext q; rfl
  rw [row]
  decide

def innerTail242_18 : List Nat := []
theorem innerScan242_18 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_18 best = best := rfl

def innerTail242_17 : List Nat := 8147 :: innerTail242_18
def innerCandidates242_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates242_17_checked : candidates edge 4 1553 8147 innerTail242_18 best.bound = innerCandidates242_17 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_17 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_17 best = best := by
  rw [innerTail242_17, scan, if_neg (by decide), innerCandidates242_17_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_18
    (if 3 ≤ innerCandidates242_17.length then search edge 3 [1553, 8147] innerCandidates242_17 9700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_18

def innerTail242_16 : List Nat := 8111 :: innerTail242_17
def innerCandidates242_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates242_16_checked : candidates edge 4 1553 8111 innerTail242_17 best.bound = innerCandidates242_16 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_16 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_16 best = best := by
  rw [innerTail242_16, scan, if_neg (by decide), innerCandidates242_16_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_17
    (if 3 ≤ innerCandidates242_16.length then search edge 3 [1553, 8111] innerCandidates242_16 9664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_17

def innerTail242_15 : List Nat := 7883 :: innerTail242_16
def innerCandidates242_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates242_15_checked : candidates edge 4 1553 7883 innerTail242_16 best.bound = innerCandidates242_15 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_15 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_15 best = best := by
  rw [innerTail242_15, scan, if_neg (by decide), innerCandidates242_15_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_16
    (if 3 ≤ innerCandidates242_15.length then search edge 3 [1553, 7883] innerCandidates242_15 9436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_16

def innerTail242_14 : List Nat := 7757 :: innerTail242_15
def innerCandidates242_14 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem innerCandidates242_14_checked : candidates edge 4 1553 7757 innerTail242_15 best.bound = innerCandidates242_14 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_14 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_14 best = best := by
  rw [innerTail242_14, scan, if_neg (by decide), innerCandidates242_14_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_15
    (if 3 ≤ innerCandidates242_14.length then search edge 3 [1553, 7757] innerCandidates242_14 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_15

def innerTail242_13 : List Nat := 7211 :: innerTail242_14
def innerCandidates242_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates242_13_checked : candidates edge 4 1553 7211 innerTail242_14 best.bound = innerCandidates242_13 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_13 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_13 best = best := by
  rw [innerTail242_13, scan, if_neg (by decide), innerCandidates242_13_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_14
    (if 3 ≤ innerCandidates242_13.length then search edge 3 [1553, 7211] innerCandidates242_13 8764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_14

def innerTail242_12 : List Nat := 7013 :: innerTail242_13
def innerCandidates242_12 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates242_12_checked : candidates edge 4 1553 7013 innerTail242_13 best.bound = innerCandidates242_12 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_12 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_12 best = best := by
  rw [innerTail242_12, scan, if_neg (by decide), innerCandidates242_12_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_13
    (if 3 ≤ innerCandidates242_12.length then search edge 3 [1553, 7013] innerCandidates242_12 8566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_13

def innerTail242_11 : List Nat := 6947 :: innerTail242_12
def innerCandidates242_11 : List Nat := [7013]
set_option maxRecDepth 4096 in
theorem innerCandidates242_11_checked : candidates edge 4 1553 6947 innerTail242_12 best.bound = innerCandidates242_11 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_11 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_11 best = best := by
  rw [innerTail242_11, scan, if_neg (by decide), innerCandidates242_11_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_12
    (if 3 ≤ innerCandidates242_11.length then search edge 3 [1553, 6947] innerCandidates242_11 8500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_12

def innerTail242_10 : List Nat := 6221 :: innerTail242_11
def innerCandidates242_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates242_10_checked : candidates edge 4 1553 6221 innerTail242_11 best.bound = innerCandidates242_10 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_10 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_10 best = best := by
  rw [innerTail242_10, scan, if_neg (by decide), innerCandidates242_10_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_11
    (if 3 ≤ innerCandidates242_10.length then search edge 3 [1553, 6221] innerCandidates242_10 7774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_11

def innerTail242_9 : List Nat := 6203 :: innerTail242_10
def innerCandidates242_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates242_9_checked : candidates edge 4 1553 6203 innerTail242_10 best.bound = innerCandidates242_9 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_9 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_9 best = best := by
  rw [innerTail242_9, scan, if_neg (by decide), innerCandidates242_9_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_10
    (if 3 ≤ innerCandidates242_9.length then search edge 3 [1553, 6203] innerCandidates242_9 7756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_10

def innerTail242_8 : List Nat := 4049 :: innerTail242_9
def innerCandidates242_8 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates242_8_checked : candidates edge 4 1553 4049 innerTail242_9 best.bound = innerCandidates242_8 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_8 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_8 best = best := by
  rw [innerTail242_8, scan, if_neg (by decide), innerCandidates242_8_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_9
    (if 3 ≤ innerCandidates242_8.length then search edge 3 [1553, 4049] innerCandidates242_8 5602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_9

def innerTail242_7 : List Nat := 2687 :: innerTail242_8
def innerCandidates242_7 : List Nat := [6221]
set_option maxRecDepth 4096 in
theorem innerCandidates242_7_checked : candidates edge 4 1553 2687 innerTail242_8 best.bound = innerCandidates242_7 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_7 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_7 best = best := by
  rw [innerTail242_7, scan, if_neg (by decide), innerCandidates242_7_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_8
    (if 3 ≤ innerCandidates242_7.length then search edge 3 [1553, 2687] innerCandidates242_7 4240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_8

def innerTail242_6 : List Nat := 2549 :: innerTail242_7
def innerCandidates242_6 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates242_6_checked : candidates edge 4 1553 2549 innerTail242_7 best.bound = innerCandidates242_6 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_6 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_6 best = best := by
  rw [innerTail242_6, scan, if_neg (by decide), innerCandidates242_6_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_7
    (if 3 ≤ innerCandidates242_6.length then search edge 3 [1553, 2549] innerCandidates242_6 4102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_7

def innerTail242_5 : List Nat := 2393 :: innerTail242_6
def innerCandidates242_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates242_5_checked : candidates edge 4 1553 2393 innerTail242_6 best.bound = innerCandidates242_5 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_5 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_5 best = best := by
  rw [innerTail242_5, scan, if_neg (by decide), innerCandidates242_5_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_6
    (if 3 ≤ innerCandidates242_5.length then search edge 3 [1553, 2393] innerCandidates242_5 3946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_6

def innerTail242_4 : List Nat := 2087 :: innerTail242_5
def innerCandidates242_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates242_4_checked : candidates edge 4 1553 2087 innerTail242_5 best.bound = innerCandidates242_4 := by
  unfold candidates
  have row : edge 2087 = PrimePairCache.smallRow2087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_4 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_4 best = best := by
  rw [innerTail242_4, scan, if_neg (by decide), innerCandidates242_4_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_5
    (if 3 ≤ innerCandidates242_4.length then search edge 3 [1553, 2087] innerCandidates242_4 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_5

def innerTail242_3 : List Nat := 1811 :: innerTail242_4
def innerCandidates242_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates242_3_checked : candidates edge 4 1553 1811 innerTail242_4 best.bound = innerCandidates242_3 := by
  unfold candidates
  have row : edge 1811 = PrimePairCache.smallRow1811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_3 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_3 best = best := by
  rw [innerTail242_3, scan, if_neg (by decide), innerCandidates242_3_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_4
    (if 3 ≤ innerCandidates242_3.length then search edge 3 [1553, 1811] innerCandidates242_3 3364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_4

def innerTail242_2 : List Nat := 1697 :: innerTail242_3
def innerCandidates242_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates242_2_checked : candidates edge 4 1553 1697 innerTail242_3 best.bound = innerCandidates242_2 := by
  unfold candidates
  have row : edge 1697 = PrimePairCache.smallRow1697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_2 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_2 best = best := by
  rw [innerTail242_2, scan, if_neg (by decide), innerCandidates242_2_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_3
    (if 3 ≤ innerCandidates242_2.length then search edge 3 [1553, 1697] innerCandidates242_2 3250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_3

def innerTail242_1 : List Nat := 1613 :: innerTail242_2
def innerCandidates242_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates242_1_checked : candidates edge 4 1553 1613 innerTail242_2 best.bound = innerCandidates242_1 := by
  unfold candidates
  have row : edge 1613 = PrimePairCache.smallRow1613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_1 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_1 best = best := by
  rw [innerTail242_1, scan, if_neg (by decide), innerCandidates242_1_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_2
    (if 3 ≤ innerCandidates242_1.length then search edge 3 [1553, 1613] innerCandidates242_1 3166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_2

def innerTail242_0 : List Nat := 1583 :: innerTail242_1
def innerCandidates242_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates242_0_checked : candidates edge 4 1553 1583 innerTail242_1 best.bound = innerCandidates242_0 := by
  unfold candidates
  have row : edge 1583 = PrimePairCache.smallRow1583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan242_0 : scan edge (search edge 3) 4 [1553] 1553 innerTail242_0 best = best := by
  rw [innerTail242_0, scan, if_neg (by decide), innerCandidates242_0_checked]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_1
    (if 3 ≤ innerCandidates242_0.length then search edge 3 [1553, 1583] innerCandidates242_0 3136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan242_1

theorem rootBranch242 : search edge 4 [1553] rootCandidates242 1553 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1553] 1553 innerTail242_0 best = best
  exact innerScan242_0

end ProjectEulerSolutions.PrimePairUpper.Search
