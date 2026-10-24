import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates482 : List Nat := [4051, 4447, 4801, 4969, 5479, 5869, 6037, 6079, 6211, 6337, 6883, 7411, 8011, 8053, 8059]
set_option maxRecDepth 4096 in
theorem rootCandidates482_checked : candidates edge 5 0 3463 tail483 best.bound = rootCandidates482 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext q; rfl
  rw [row]
  decide

def innerTail482_15 : List Nat := []
theorem innerScan482_15 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_15 best = best := rfl

def innerTail482_14 : List Nat := 8059 :: innerTail482_15
def innerCandidates482_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates482_14_checked : candidates edge 4 3463 8059 innerTail482_15 best.bound = innerCandidates482_14 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_14 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_14 best = best := by
  rw [innerTail482_14, scan, if_neg (by decide), innerCandidates482_14_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_15
    (if 3 ≤ innerCandidates482_14.length then search edge 3 [3463, 8059] innerCandidates482_14 11522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_15

def innerTail482_13 : List Nat := 8053 :: innerTail482_14
def innerCandidates482_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates482_13_checked : candidates edge 4 3463 8053 innerTail482_14 best.bound = innerCandidates482_13 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_13 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_13 best = best := by
  rw [innerTail482_13, scan, if_neg (by decide), innerCandidates482_13_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_14
    (if 3 ≤ innerCandidates482_13.length then search edge 3 [3463, 8053] innerCandidates482_13 11516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_14

def innerTail482_12 : List Nat := 8011 :: innerTail482_13
def innerCandidates482_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates482_12_checked : candidates edge 4 3463 8011 innerTail482_13 best.bound = innerCandidates482_12 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_12 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_12 best = best := by
  rw [innerTail482_12, scan, if_neg (by decide), innerCandidates482_12_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_13
    (if 3 ≤ innerCandidates482_12.length then search edge 3 [3463, 8011] innerCandidates482_12 11474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_13

def innerTail482_11 : List Nat := 7411 :: innerTail482_12
def innerCandidates482_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates482_11_checked : candidates edge 4 3463 7411 innerTail482_12 best.bound = innerCandidates482_11 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_11 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_11 best = best := by
  rw [innerTail482_11, scan, if_neg (by decide), innerCandidates482_11_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_12
    (if 3 ≤ innerCandidates482_11.length then search edge 3 [3463, 7411] innerCandidates482_11 10874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_12

def innerTail482_10 : List Nat := 6883 :: innerTail482_11
def innerCandidates482_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates482_10_checked : candidates edge 4 3463 6883 innerTail482_11 best.bound = innerCandidates482_10 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_10 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_10 best = best := by
  rw [innerTail482_10, scan, if_neg (by decide), innerCandidates482_10_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_11
    (if 3 ≤ innerCandidates482_10.length then search edge 3 [3463, 6883] innerCandidates482_10 10346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_11

def innerTail482_9 : List Nat := 6337 :: innerTail482_10
def innerCandidates482_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates482_9_checked : candidates edge 4 3463 6337 innerTail482_10 best.bound = innerCandidates482_9 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_9 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_9 best = best := by
  rw [innerTail482_9, scan, if_neg (by decide), innerCandidates482_9_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_10
    (if 3 ≤ innerCandidates482_9.length then search edge 3 [3463, 6337] innerCandidates482_9 9800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_10

def innerTail482_8 : List Nat := 6211 :: innerTail482_9
def innerCandidates482_8 : List Nat := [6883]
set_option maxRecDepth 4096 in
theorem innerCandidates482_8_checked : candidates edge 4 3463 6211 innerTail482_9 best.bound = innerCandidates482_8 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_8 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_8 best = best := by
  rw [innerTail482_8, scan, if_neg (by decide), innerCandidates482_8_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_9
    (if 3 ≤ innerCandidates482_8.length then search edge 3 [3463, 6211] innerCandidates482_8 9674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_9

def innerTail482_7 : List Nat := 6079 :: innerTail482_8
def innerCandidates482_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates482_7_checked : candidates edge 4 3463 6079 innerTail482_8 best.bound = innerCandidates482_7 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_7 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_7 best = best := by
  rw [innerTail482_7, scan, if_neg (by decide), innerCandidates482_7_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_8
    (if 3 ≤ innerCandidates482_7.length then search edge 3 [3463, 6079] innerCandidates482_7 9542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_8

def innerTail482_6 : List Nat := 6037 :: innerTail482_7
def innerCandidates482_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates482_6_checked : candidates edge 4 3463 6037 innerTail482_7 best.bound = innerCandidates482_6 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_6 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_6 best = best := by
  rw [innerTail482_6, scan, if_neg (by decide), innerCandidates482_6_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_7
    (if 3 ≤ innerCandidates482_6.length then search edge 3 [3463, 6037] innerCandidates482_6 9500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_7

def innerTail482_5 : List Nat := 5869 :: innerTail482_6
def innerCandidates482_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates482_5_checked : candidates edge 4 3463 5869 innerTail482_6 best.bound = innerCandidates482_5 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_5 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_5 best = best := by
  rw [innerTail482_5, scan, if_neg (by decide), innerCandidates482_5_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_6
    (if 3 ≤ innerCandidates482_5.length then search edge 3 [3463, 5869] innerCandidates482_5 9332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_6

def innerTail482_4 : List Nat := 5479 :: innerTail482_5
def innerCandidates482_4 : List Nat := [5869]
set_option maxRecDepth 4096 in
theorem innerCandidates482_4_checked : candidates edge 4 3463 5479 innerTail482_5 best.bound = innerCandidates482_4 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_4 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_4 best = best := by
  rw [innerTail482_4, scan, if_neg (by decide), innerCandidates482_4_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_5
    (if 3 ≤ innerCandidates482_4.length then search edge 3 [3463, 5479] innerCandidates482_4 8942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_5

def innerTail482_3 : List Nat := 4969 :: innerTail482_4
def innerCandidates482_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates482_3_checked : candidates edge 4 3463 4969 innerTail482_4 best.bound = innerCandidates482_3 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_3 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_3 best = best := by
  rw [innerTail482_3, scan, if_neg (by decide), innerCandidates482_3_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_4
    (if 3 ≤ innerCandidates482_3.length then search edge 3 [3463, 4969] innerCandidates482_3 8432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_4

def innerTail482_2 : List Nat := 4801 :: innerTail482_3
def innerCandidates482_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates482_2_checked : candidates edge 4 3463 4801 innerTail482_3 best.bound = innerCandidates482_2 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_2 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_2 best = best := by
  rw [innerTail482_2, scan, if_neg (by decide), innerCandidates482_2_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_3
    (if 3 ≤ innerCandidates482_2.length then search edge 3 [3463, 4801] innerCandidates482_2 8264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_3

def innerTail482_1 : List Nat := 4447 :: innerTail482_2
def innerCandidates482_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates482_1_checked : candidates edge 4 3463 4447 innerTail482_2 best.bound = innerCandidates482_1 := by
  unfold candidates
  have row : edge 4447 = SearchData.row4447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_1 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_1 best = best := by
  rw [innerTail482_1, scan, if_neg (by decide), innerCandidates482_1_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_2
    (if 3 ≤ innerCandidates482_1.length then search edge 3 [3463, 4447] innerCandidates482_1 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_2

def innerTail482_0 : List Nat := 4051 :: innerTail482_1
def innerCandidates482_0 : List Nat := [6211]
set_option maxRecDepth 4096 in
theorem innerCandidates482_0_checked : candidates edge 4 3463 4051 innerTail482_1 best.bound = innerCandidates482_0 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan482_0 : scan edge (search edge 3) 4 [3463] 3463 innerTail482_0 best = best := by
  rw [innerTail482_0, scan, if_neg (by decide), innerCandidates482_0_checked]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_1
    (if 3 ≤ innerCandidates482_0.length then search edge 3 [3463, 4051] innerCandidates482_0 7514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan482_1

theorem rootBranch482 : search edge 4 [3463] rootCandidates482 3463 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3463] 3463 innerTail482_0 best = best
  exact innerScan482_0

def rootCandidates481 : List Nat := [3659, 4583, 4967, 5333, 6011, 6221, 6737, 7523, 7823, 8117]
set_option maxRecDepth 4096 in
theorem rootCandidates481_checked : candidates edge 5 0 3461 tail482 best.bound = rootCandidates481 := by
  unfold candidates
  have row : edge 3461 = SearchData.row3461.contains := by funext q; rfl
  rw [row]
  decide

def innerTail481_10 : List Nat := []
theorem innerScan481_10 : scan edge (search edge 3) 4 [3461] 3461 innerTail481_10 best = best := rfl

def innerTail481_9 : List Nat := 8117 :: innerTail481_10
def innerCandidates481_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates481_9_checked : candidates edge 4 3461 8117 innerTail481_10 best.bound = innerCandidates481_9 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan481_9 : scan edge (search edge 3) 4 [3461] 3461 innerTail481_9 best = best := by
  rw [innerTail481_9, scan, if_neg (by decide), innerCandidates481_9_checked]
  change scan edge (search edge 3) 4 [3461] 3461 innerTail481_10
    (if 3 ≤ innerCandidates481_9.length then search edge 3 [3461, 8117] innerCandidates481_9 11578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan481_10

def innerTail481_8 : List Nat := 7823 :: innerTail481_9
def innerCandidates481_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates481_8_checked : candidates edge 4 3461 7823 innerTail481_9 best.bound = innerCandidates481_8 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan481_8 : scan edge (search edge 3) 4 [3461] 3461 innerTail481_8 best = best := by
  rw [innerTail481_8, scan, if_neg (by decide), innerCandidates481_8_checked]
  change scan edge (search edge 3) 4 [3461] 3461 innerTail481_9
    (if 3 ≤ innerCandidates481_8.length then search edge 3 [3461, 7823] innerCandidates481_8 11284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan481_9

def innerTail481_7 : List Nat := 7523 :: innerTail481_8
def innerCandidates481_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates481_7_checked : candidates edge 4 3461 7523 innerTail481_8 best.bound = innerCandidates481_7 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan481_7 : scan edge (search edge 3) 4 [3461] 3461 innerTail481_7 best = best := by
  rw [innerTail481_7, scan, if_neg (by decide), innerCandidates481_7_checked]
  change scan edge (search edge 3) 4 [3461] 3461 innerTail481_8
    (if 3 ≤ innerCandidates481_7.length then search edge 3 [3461, 7523] innerCandidates481_7 10984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan481_8

def innerTail481_6 : List Nat := 6737 :: innerTail481_7
def innerCandidates481_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates481_6_checked : candidates edge 4 3461 6737 innerTail481_7 best.bound = innerCandidates481_6 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan481_6 : scan edge (search edge 3) 4 [3461] 3461 innerTail481_6 best = best := by
  rw [innerTail481_6, scan, if_neg (by decide), innerCandidates481_6_checked]
  change scan edge (search edge 3) 4 [3461] 3461 innerTail481_7
    (if 3 ≤ innerCandidates481_6.length then search edge 3 [3461, 6737] innerCandidates481_6 10198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan481_7

def innerTail481_5 : List Nat := 6221 :: innerTail481_6
def innerCandidates481_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates481_5_checked : candidates edge 4 3461 6221 innerTail481_6 best.bound = innerCandidates481_5 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan481_5 : scan edge (search edge 3) 4 [3461] 3461 innerTail481_5 best = best := by
  rw [innerTail481_5, scan, if_neg (by decide), innerCandidates481_5_checked]
  change scan edge (search edge 3) 4 [3461] 3461 innerTail481_6
    (if 3 ≤ innerCandidates481_5.length then search edge 3 [3461, 6221] innerCandidates481_5 9682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan481_6

def innerTail481_4 : List Nat := 6011 :: innerTail481_5
def innerCandidates481_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates481_4_checked : candidates edge 4 3461 6011 innerTail481_5 best.bound = innerCandidates481_4 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan481_4 : scan edge (search edge 3) 4 [3461] 3461 innerTail481_4 best = best := by
  rw [innerTail481_4, scan, if_neg (by decide), innerCandidates481_4_checked]
  change scan edge (search edge 3) 4 [3461] 3461 innerTail481_5
    (if 3 ≤ innerCandidates481_4.length then search edge 3 [3461, 6011] innerCandidates481_4 9472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan481_5

def innerTail481_3 : List Nat := 5333 :: innerTail481_4
def innerCandidates481_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates481_3_checked : candidates edge 4 3461 5333 innerTail481_4 best.bound = innerCandidates481_3 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan481_3 : scan edge (search edge 3) 4 [3461] 3461 innerTail481_3 best = best := by
  rw [innerTail481_3, scan, if_neg (by decide), innerCandidates481_3_checked]
  change scan edge (search edge 3) 4 [3461] 3461 innerTail481_4
    (if 3 ≤ innerCandidates481_3.length then search edge 3 [3461, 5333] innerCandidates481_3 8794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan481_4

def innerTail481_2 : List Nat := 4967 :: innerTail481_3
def innerCandidates481_2 : List Nat := [8117]
set_option maxRecDepth 4096 in
theorem innerCandidates481_2_checked : candidates edge 4 3461 4967 innerTail481_3 best.bound = innerCandidates481_2 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan481_2 : scan edge (search edge 3) 4 [3461] 3461 innerTail481_2 best = best := by
  rw [innerTail481_2, scan, if_neg (by decide), innerCandidates481_2_checked]
  change scan edge (search edge 3) 4 [3461] 3461 innerTail481_3
    (if 3 ≤ innerCandidates481_2.length then search edge 3 [3461, 4967] innerCandidates481_2 8428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan481_3

def innerTail481_1 : List Nat := 4583 :: innerTail481_2
def innerCandidates481_1 : List Nat := [7523]
set_option maxRecDepth 4096 in
theorem innerCandidates481_1_checked : candidates edge 4 3461 4583 innerTail481_2 best.bound = innerCandidates481_1 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan481_1 : scan edge (search edge 3) 4 [3461] 3461 innerTail481_1 best = best := by
  rw [innerTail481_1, scan, if_neg (by decide), innerCandidates481_1_checked]
  change scan edge (search edge 3) 4 [3461] 3461 innerTail481_2
    (if 3 ≤ innerCandidates481_1.length then search edge 3 [3461, 4583] innerCandidates481_1 8044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan481_2

def innerTail481_0 : List Nat := 3659 :: innerTail481_1
def innerCandidates481_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates481_0_checked : candidates edge 4 3461 3659 innerTail481_1 best.bound = innerCandidates481_0 := by
  unfold candidates
  have row : edge 3659 = SearchData.row3659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan481_0 : scan edge (search edge 3) 4 [3461] 3461 innerTail481_0 best = best := by
  rw [innerTail481_0, scan, if_neg (by decide), innerCandidates481_0_checked]
  change scan edge (search edge 3) 4 [3461] 3461 innerTail481_1
    (if 3 ≤ innerCandidates481_0.length then search edge 3 [3461, 3659] innerCandidates481_0 7120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan481_1

theorem rootBranch481 : search edge 4 [3461] rootCandidates481 3461 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3461] 3461 innerTail481_0 best = best
  exact innerScan481_0

def rootCandidates480 : List Nat := [3499, 4297, 4327, 4567, 4759, 4789, 5113, 5197, 5419, 5449, 5869, 6247, 6679, 6823, 7603, 8209, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates480_checked : candidates edge 5 0 3457 tail481 best.bound = rootCandidates480 := by
  unfold candidates
  have row : edge 3457 = SearchData.row3457.contains := by funext q; rfl
  rw [row]
  decide

def innerTail480_17 : List Nat := []
theorem innerScan480_17 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_17 best = best := rfl

def innerTail480_16 : List Nat := 8377 :: innerTail480_17
def innerCandidates480_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates480_16_checked : candidates edge 4 3457 8377 innerTail480_17 best.bound = innerCandidates480_16 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_16 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_16 best = best := by
  rw [innerTail480_16, scan, if_neg (by decide), innerCandidates480_16_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_17
    (if 3 ≤ innerCandidates480_16.length then search edge 3 [3457, 8377] innerCandidates480_16 11834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_17

def innerTail480_15 : List Nat := 8209 :: innerTail480_16
def innerCandidates480_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates480_15_checked : candidates edge 4 3457 8209 innerTail480_16 best.bound = innerCandidates480_15 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_15 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_15 best = best := by
  rw [innerTail480_15, scan, if_neg (by decide), innerCandidates480_15_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_16
    (if 3 ≤ innerCandidates480_15.length then search edge 3 [3457, 8209] innerCandidates480_15 11666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_16

def innerTail480_14 : List Nat := 7603 :: innerTail480_15
def innerCandidates480_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates480_14_checked : candidates edge 4 3457 7603 innerTail480_15 best.bound = innerCandidates480_14 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_14 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_14 best = best := by
  rw [innerTail480_14, scan, if_neg (by decide), innerCandidates480_14_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_15
    (if 3 ≤ innerCandidates480_14.length then search edge 3 [3457, 7603] innerCandidates480_14 11060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_15

def innerTail480_13 : List Nat := 6823 :: innerTail480_14
def innerCandidates480_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates480_13_checked : candidates edge 4 3457 6823 innerTail480_14 best.bound = innerCandidates480_13 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_13 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_13 best = best := by
  rw [innerTail480_13, scan, if_neg (by decide), innerCandidates480_13_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_14
    (if 3 ≤ innerCandidates480_13.length then search edge 3 [3457, 6823] innerCandidates480_13 10280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_14

def innerTail480_12 : List Nat := 6679 :: innerTail480_13
def innerCandidates480_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates480_12_checked : candidates edge 4 3457 6679 innerTail480_13 best.bound = innerCandidates480_12 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_12 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_12 best = best := by
  rw [innerTail480_12, scan, if_neg (by decide), innerCandidates480_12_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_13
    (if 3 ≤ innerCandidates480_12.length then search edge 3 [3457, 6679] innerCandidates480_12 10136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_13

def innerTail480_11 : List Nat := 6247 :: innerTail480_12
def innerCandidates480_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates480_11_checked : candidates edge 4 3457 6247 innerTail480_12 best.bound = innerCandidates480_11 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_11 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_11 best = best := by
  rw [innerTail480_11, scan, if_neg (by decide), innerCandidates480_11_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_12
    (if 3 ≤ innerCandidates480_11.length then search edge 3 [3457, 6247] innerCandidates480_11 9704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_12

def innerTail480_10 : List Nat := 5869 :: innerTail480_11
def innerCandidates480_10 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem innerCandidates480_10_checked : candidates edge 4 3457 5869 innerTail480_11 best.bound = innerCandidates480_10 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_10 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_10 best = best := by
  rw [innerTail480_10, scan, if_neg (by decide), innerCandidates480_10_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_11
    (if 3 ≤ innerCandidates480_10.length then search edge 3 [3457, 5869] innerCandidates480_10 9326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_11

def innerTail480_9 : List Nat := 5449 :: innerTail480_10
def innerCandidates480_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates480_9_checked : candidates edge 4 3457 5449 innerTail480_10 best.bound = innerCandidates480_9 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_9 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_9 best = best := by
  rw [innerTail480_9, scan, if_neg (by decide), innerCandidates480_9_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_10
    (if 3 ≤ innerCandidates480_9.length then search edge 3 [3457, 5449] innerCandidates480_9 8906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_10

def innerTail480_8 : List Nat := 5419 :: innerTail480_9
def innerCandidates480_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates480_8_checked : candidates edge 4 3457 5419 innerTail480_9 best.bound = innerCandidates480_8 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_8 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_8 best = best := by
  rw [innerTail480_8, scan, if_neg (by decide), innerCandidates480_8_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_9
    (if 3 ≤ innerCandidates480_8.length then search edge 3 [3457, 5419] innerCandidates480_8 8876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_9

def innerTail480_7 : List Nat := 5197 :: innerTail480_8
def innerCandidates480_7 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem innerCandidates480_7_checked : candidates edge 4 3457 5197 innerTail480_8 best.bound = innerCandidates480_7 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_7 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_7 best = best := by
  rw [innerTail480_7, scan, if_neg (by decide), innerCandidates480_7_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_8
    (if 3 ≤ innerCandidates480_7.length then search edge 3 [3457, 5197] innerCandidates480_7 8654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_8

def innerTail480_6 : List Nat := 5113 :: innerTail480_7
def innerCandidates480_6 : List Nat := [5449, 6823]
set_option maxRecDepth 4096 in
theorem innerCandidates480_6_checked : candidates edge 4 3457 5113 innerTail480_7 best.bound = innerCandidates480_6 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_6 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_6 best = best := by
  rw [innerTail480_6, scan, if_neg (by decide), innerCandidates480_6_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_7
    (if 3 ≤ innerCandidates480_6.length then search edge 3 [3457, 5113] innerCandidates480_6 8570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_7

def innerTail480_5 : List Nat := 4789 :: innerTail480_6
def innerCandidates480_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates480_5_checked : candidates edge 4 3457 4789 innerTail480_6 best.bound = innerCandidates480_5 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_5 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_5 best = best := by
  rw [innerTail480_5, scan, if_neg (by decide), innerCandidates480_5_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_6
    (if 3 ≤ innerCandidates480_5.length then search edge 3 [3457, 4789] innerCandidates480_5 8246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_6

def innerTail480_4 : List Nat := 4759 :: innerTail480_5
def innerCandidates480_4 : List Nat := [6823]
set_option maxRecDepth 4096 in
theorem innerCandidates480_4_checked : candidates edge 4 3457 4759 innerTail480_5 best.bound = innerCandidates480_4 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_4 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_4 best = best := by
  rw [innerTail480_4, scan, if_neg (by decide), innerCandidates480_4_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_5
    (if 3 ≤ innerCandidates480_4.length then search edge 3 [3457, 4759] innerCandidates480_4 8216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_5

def innerTail480_3 : List Nat := 4567 :: innerTail480_4
def innerCandidates480_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates480_3_checked : candidates edge 4 3457 4567 innerTail480_4 best.bound = innerCandidates480_3 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_3 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_3 best = best := by
  rw [innerTail480_3, scan, if_neg (by decide), innerCandidates480_3_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_4
    (if 3 ≤ innerCandidates480_3.length then search edge 3 [3457, 4567] innerCandidates480_3 8024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_4

def innerTail480_2 : List Nat := 4327 :: innerTail480_3
def innerCandidates480_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates480_2_checked : candidates edge 4 3457 4327 innerTail480_3 best.bound = innerCandidates480_2 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_2 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_2 best = best := by
  rw [innerTail480_2, scan, if_neg (by decide), innerCandidates480_2_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_3
    (if 3 ≤ innerCandidates480_2.length then search edge 3 [3457, 4327] innerCandidates480_2 7784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_3

def innerTail480_1 : List Nat := 4297 :: innerTail480_2
def innerCandidates480_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates480_1_checked : candidates edge 4 3457 4297 innerTail480_2 best.bound = innerCandidates480_1 := by
  unfold candidates
  have row : edge 4297 = SearchData.row4297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_1 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_1 best = best := by
  rw [innerTail480_1, scan, if_neg (by decide), innerCandidates480_1_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_2
    (if 3 ≤ innerCandidates480_1.length then search edge 3 [3457, 4297] innerCandidates480_1 7754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_2

def innerTail480_0 : List Nat := 3499 :: innerTail480_1
def innerCandidates480_0 : List Nat := [5449, 8377]
set_option maxRecDepth 4096 in
theorem innerCandidates480_0_checked : candidates edge 4 3457 3499 innerTail480_1 best.bound = innerCandidates480_0 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan480_0 : scan edge (search edge 3) 4 [3457] 3457 innerTail480_0 best = best := by
  rw [innerTail480_0, scan, if_neg (by decide), innerCandidates480_0_checked]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_1
    (if 3 ≤ innerCandidates480_0.length then search edge 3 [3457, 3499] innerCandidates480_0 6956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan480_1

theorem rootBranch480 : search edge 4 [3457] rootCandidates480 3457 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3457] 3457 innerTail480_0 best = best
  exact innerScan480_0

def rootCandidates479 : List Nat := [4013, 4451, 5501, 5693, 6359, 7433, 7457, 7817, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates479_checked : candidates edge 5 0 3449 tail480 best.bound = rootCandidates479 := by
  unfold candidates
  have row : edge 3449 = SearchData.row3449.contains := by funext q; rfl
  rw [row]
  decide

def innerTail479_9 : List Nat := []
theorem innerScan479_9 : scan edge (search edge 3) 4 [3449] 3449 innerTail479_9 best = best := rfl

def innerTail479_8 : List Nat := 8111 :: innerTail479_9
def innerCandidates479_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates479_8_checked : candidates edge 4 3449 8111 innerTail479_9 best.bound = innerCandidates479_8 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan479_8 : scan edge (search edge 3) 4 [3449] 3449 innerTail479_8 best = best := by
  rw [innerTail479_8, scan, if_neg (by decide), innerCandidates479_8_checked]
  change scan edge (search edge 3) 4 [3449] 3449 innerTail479_9
    (if 3 ≤ innerCandidates479_8.length then search edge 3 [3449, 8111] innerCandidates479_8 11560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan479_9

def innerTail479_7 : List Nat := 7817 :: innerTail479_8
def innerCandidates479_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates479_7_checked : candidates edge 4 3449 7817 innerTail479_8 best.bound = innerCandidates479_7 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan479_7 : scan edge (search edge 3) 4 [3449] 3449 innerTail479_7 best = best := by
  rw [innerTail479_7, scan, if_neg (by decide), innerCandidates479_7_checked]
  change scan edge (search edge 3) 4 [3449] 3449 innerTail479_8
    (if 3 ≤ innerCandidates479_7.length then search edge 3 [3449, 7817] innerCandidates479_7 11266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan479_8

def innerTail479_6 : List Nat := 7457 :: innerTail479_7
def innerCandidates479_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates479_6_checked : candidates edge 4 3449 7457 innerTail479_7 best.bound = innerCandidates479_6 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan479_6 : scan edge (search edge 3) 4 [3449] 3449 innerTail479_6 best = best := by
  rw [innerTail479_6, scan, if_neg (by decide), innerCandidates479_6_checked]
  change scan edge (search edge 3) 4 [3449] 3449 innerTail479_7
    (if 3 ≤ innerCandidates479_6.length then search edge 3 [3449, 7457] innerCandidates479_6 10906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan479_7

def innerTail479_5 : List Nat := 7433 :: innerTail479_6
def innerCandidates479_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates479_5_checked : candidates edge 4 3449 7433 innerTail479_6 best.bound = innerCandidates479_5 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan479_5 : scan edge (search edge 3) 4 [3449] 3449 innerTail479_5 best = best := by
  rw [innerTail479_5, scan, if_neg (by decide), innerCandidates479_5_checked]
  change scan edge (search edge 3) 4 [3449] 3449 innerTail479_6
    (if 3 ≤ innerCandidates479_5.length then search edge 3 [3449, 7433] innerCandidates479_5 10882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan479_6

def innerTail479_4 : List Nat := 6359 :: innerTail479_5
def innerCandidates479_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates479_4_checked : candidates edge 4 3449 6359 innerTail479_5 best.bound = innerCandidates479_4 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan479_4 : scan edge (search edge 3) 4 [3449] 3449 innerTail479_4 best = best := by
  rw [innerTail479_4, scan, if_neg (by decide), innerCandidates479_4_checked]
  change scan edge (search edge 3) 4 [3449] 3449 innerTail479_5
    (if 3 ≤ innerCandidates479_4.length then search edge 3 [3449, 6359] innerCandidates479_4 9808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan479_5

def innerTail479_3 : List Nat := 5693 :: innerTail479_4
def innerCandidates479_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates479_3_checked : candidates edge 4 3449 5693 innerTail479_4 best.bound = innerCandidates479_3 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan479_3 : scan edge (search edge 3) 4 [3449] 3449 innerTail479_3 best = best := by
  rw [innerTail479_3, scan, if_neg (by decide), innerCandidates479_3_checked]
  change scan edge (search edge 3) 4 [3449] 3449 innerTail479_4
    (if 3 ≤ innerCandidates479_3.length then search edge 3 [3449, 5693] innerCandidates479_3 9142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan479_4

def innerTail479_2 : List Nat := 5501 :: innerTail479_3
def innerCandidates479_2 : List Nat := [6359]
set_option maxRecDepth 4096 in
theorem innerCandidates479_2_checked : candidates edge 4 3449 5501 innerTail479_3 best.bound = innerCandidates479_2 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan479_2 : scan edge (search edge 3) 4 [3449] 3449 innerTail479_2 best = best := by
  rw [innerTail479_2, scan, if_neg (by decide), innerCandidates479_2_checked]
  change scan edge (search edge 3) 4 [3449] 3449 innerTail479_3
    (if 3 ≤ innerCandidates479_2.length then search edge 3 [3449, 5501] innerCandidates479_2 8950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan479_3

def innerTail479_1 : List Nat := 4451 :: innerTail479_2
def innerCandidates479_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates479_1_checked : candidates edge 4 3449 4451 innerTail479_2 best.bound = innerCandidates479_1 := by
  unfold candidates
  have row : edge 4451 = SearchData.row4451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan479_1 : scan edge (search edge 3) 4 [3449] 3449 innerTail479_1 best = best := by
  rw [innerTail479_1, scan, if_neg (by decide), innerCandidates479_1_checked]
  change scan edge (search edge 3) 4 [3449] 3449 innerTail479_2
    (if 3 ≤ innerCandidates479_1.length then search edge 3 [3449, 4451] innerCandidates479_1 7900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan479_2

def innerTail479_0 : List Nat := 4013 :: innerTail479_1
def innerCandidates479_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates479_0_checked : candidates edge 4 3449 4013 innerTail479_1 best.bound = innerCandidates479_0 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan479_0 : scan edge (search edge 3) 4 [3449] 3449 innerTail479_0 best = best := by
  rw [innerTail479_0, scan, if_neg (by decide), innerCandidates479_0_checked]
  change scan edge (search edge 3) 4 [3449] 3449 innerTail479_1
    (if 3 ≤ innerCandidates479_0.length then search edge 3 [3449, 4013] innerCandidates479_0 7462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan479_1

theorem rootBranch479 : search edge 4 [3449] rootCandidates479 3449 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3449] 3449 innerTail479_0 best = best
  exact innerScan479_0

end ProjectEulerSolutions.PrimePairUpper.Search
