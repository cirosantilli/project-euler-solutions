import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates497 : List Nat := [4111, 4363, 4801, 4957, 4969, 5023, 6007, 6451, 6883, 7699, 7963]
set_option maxRecDepth 4096 in
theorem rootCandidates497_checked : candidates edge 5 0 3571 tail498 best.bound = rootCandidates497 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext q; rfl
  rw [row]
  decide

def innerTail497_11 : List Nat := []
theorem innerScan497_11 : scan edge (search edge 3) 4 [3571] 3571 innerTail497_11 best = best := rfl

def innerTail497_10 : List Nat := 7963 :: innerTail497_11
def innerCandidates497_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates497_10_checked : candidates edge 4 3571 7963 innerTail497_11 best.bound = innerCandidates497_10 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan497_10 : scan edge (search edge 3) 4 [3571] 3571 innerTail497_10 best = best := by
  rw [innerTail497_10, scan, if_neg (by decide), innerCandidates497_10_checked]
  change scan edge (search edge 3) 4 [3571] 3571 innerTail497_11
    (if 3 ≤ innerCandidates497_10.length then search edge 3 [3571, 7963] innerCandidates497_10 11534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan497_11

def innerTail497_9 : List Nat := 7699 :: innerTail497_10
def innerCandidates497_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates497_9_checked : candidates edge 4 3571 7699 innerTail497_10 best.bound = innerCandidates497_9 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan497_9 : scan edge (search edge 3) 4 [3571] 3571 innerTail497_9 best = best := by
  rw [innerTail497_9, scan, if_neg (by decide), innerCandidates497_9_checked]
  change scan edge (search edge 3) 4 [3571] 3571 innerTail497_10
    (if 3 ≤ innerCandidates497_9.length then search edge 3 [3571, 7699] innerCandidates497_9 11270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan497_10

def innerTail497_8 : List Nat := 6883 :: innerTail497_9
def innerCandidates497_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates497_8_checked : candidates edge 4 3571 6883 innerTail497_9 best.bound = innerCandidates497_8 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan497_8 : scan edge (search edge 3) 4 [3571] 3571 innerTail497_8 best = best := by
  rw [innerTail497_8, scan, if_neg (by decide), innerCandidates497_8_checked]
  change scan edge (search edge 3) 4 [3571] 3571 innerTail497_9
    (if 3 ≤ innerCandidates497_8.length then search edge 3 [3571, 6883] innerCandidates497_8 10454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan497_9

def innerTail497_7 : List Nat := 6451 :: innerTail497_8
def innerCandidates497_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates497_7_checked : candidates edge 4 3571 6451 innerTail497_8 best.bound = innerCandidates497_7 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan497_7 : scan edge (search edge 3) 4 [3571] 3571 innerTail497_7 best = best := by
  rw [innerTail497_7, scan, if_neg (by decide), innerCandidates497_7_checked]
  change scan edge (search edge 3) 4 [3571] 3571 innerTail497_8
    (if 3 ≤ innerCandidates497_7.length then search edge 3 [3571, 6451] innerCandidates497_7 10022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan497_8

def innerTail497_6 : List Nat := 6007 :: innerTail497_7
def innerCandidates497_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates497_6_checked : candidates edge 4 3571 6007 innerTail497_7 best.bound = innerCandidates497_6 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan497_6 : scan edge (search edge 3) 4 [3571] 3571 innerTail497_6 best = best := by
  rw [innerTail497_6, scan, if_neg (by decide), innerCandidates497_6_checked]
  change scan edge (search edge 3) 4 [3571] 3571 innerTail497_7
    (if 3 ≤ innerCandidates497_6.length then search edge 3 [3571, 6007] innerCandidates497_6 9578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan497_7

def innerTail497_5 : List Nat := 5023 :: innerTail497_6
def innerCandidates497_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates497_5_checked : candidates edge 4 3571 5023 innerTail497_6 best.bound = innerCandidates497_5 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan497_5 : scan edge (search edge 3) 4 [3571] 3571 innerTail497_5 best = best := by
  rw [innerTail497_5, scan, if_neg (by decide), innerCandidates497_5_checked]
  change scan edge (search edge 3) 4 [3571] 3571 innerTail497_6
    (if 3 ≤ innerCandidates497_5.length then search edge 3 [3571, 5023] innerCandidates497_5 8594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan497_6

def innerTail497_4 : List Nat := 4969 :: innerTail497_5
def innerCandidates497_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates497_4_checked : candidates edge 4 3571 4969 innerTail497_5 best.bound = innerCandidates497_4 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan497_4 : scan edge (search edge 3) 4 [3571] 3571 innerTail497_4 best = best := by
  rw [innerTail497_4, scan, if_neg (by decide), innerCandidates497_4_checked]
  change scan edge (search edge 3) 4 [3571] 3571 innerTail497_5
    (if 3 ≤ innerCandidates497_4.length then search edge 3 [3571, 4969] innerCandidates497_4 8540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan497_5

def innerTail497_3 : List Nat := 4957 :: innerTail497_4
def innerCandidates497_3 : List Nat := [5023]
set_option maxRecDepth 4096 in
theorem innerCandidates497_3_checked : candidates edge 4 3571 4957 innerTail497_4 best.bound = innerCandidates497_3 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan497_3 : scan edge (search edge 3) 4 [3571] 3571 innerTail497_3 best = best := by
  rw [innerTail497_3, scan, if_neg (by decide), innerCandidates497_3_checked]
  change scan edge (search edge 3) 4 [3571] 3571 innerTail497_4
    (if 3 ≤ innerCandidates497_3.length then search edge 3 [3571, 4957] innerCandidates497_3 8528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan497_4

def innerTail497_2 : List Nat := 4801 :: innerTail497_3
def innerCandidates497_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates497_2_checked : candidates edge 4 3571 4801 innerTail497_3 best.bound = innerCandidates497_2 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan497_2 : scan edge (search edge 3) 4 [3571] 3571 innerTail497_2 best = best := by
  rw [innerTail497_2, scan, if_neg (by decide), innerCandidates497_2_checked]
  change scan edge (search edge 3) 4 [3571] 3571 innerTail497_3
    (if 3 ≤ innerCandidates497_2.length then search edge 3 [3571, 4801] innerCandidates497_2 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan497_3

def innerTail497_1 : List Nat := 4363 :: innerTail497_2
def innerCandidates497_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates497_1_checked : candidates edge 4 3571 4363 innerTail497_2 best.bound = innerCandidates497_1 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan497_1 : scan edge (search edge 3) 4 [3571] 3571 innerTail497_1 best = best := by
  rw [innerTail497_1, scan, if_neg (by decide), innerCandidates497_1_checked]
  change scan edge (search edge 3) 4 [3571] 3571 innerTail497_2
    (if 3 ≤ innerCandidates497_1.length then search edge 3 [3571, 4363] innerCandidates497_1 7934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan497_2

def innerTail497_0 : List Nat := 4111 :: innerTail497_1
def innerCandidates497_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates497_0_checked : candidates edge 4 3571 4111 innerTail497_1 best.bound = innerCandidates497_0 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan497_0 : scan edge (search edge 3) 4 [3571] 3571 innerTail497_0 best = best := by
  rw [innerTail497_0, scan, if_neg (by decide), innerCandidates497_0_checked]
  change scan edge (search edge 3) 4 [3571] 3571 innerTail497_1
    (if 3 ≤ innerCandidates497_0.length then search edge 3 [3571, 4111] innerCandidates497_0 7682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan497_1

theorem rootBranch497 : search edge 4 [3571] rootCandidates497 3571 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3571] 3571 innerTail497_0 best = best
  exact innerScan497_0

def rootCandidates496 : List Nat := [3583, 3709, 3769, 3931, 4021, 5449, 5869, 6037, 6607, 6619, 6961, 7369, 7417, 8053]
set_option maxRecDepth 4096 in
theorem rootCandidates496_checked : candidates edge 5 0 3559 tail497 best.bound = rootCandidates496 := by
  unfold candidates
  have row : edge 3559 = SearchData.row3559.contains := by funext q; rfl
  rw [row]
  decide

def innerTail496_14 : List Nat := []
theorem innerScan496_14 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_14 best = best := rfl

def innerTail496_13 : List Nat := 8053 :: innerTail496_14
def innerCandidates496_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates496_13_checked : candidates edge 4 3559 8053 innerTail496_14 best.bound = innerCandidates496_13 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_13 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_13 best = best := by
  rw [innerTail496_13, scan, if_neg (by decide), innerCandidates496_13_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_14
    (if 3 ≤ innerCandidates496_13.length then search edge 3 [3559, 8053] innerCandidates496_13 11612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_14

def innerTail496_12 : List Nat := 7417 :: innerTail496_13
def innerCandidates496_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates496_12_checked : candidates edge 4 3559 7417 innerTail496_13 best.bound = innerCandidates496_12 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_12 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_12 best = best := by
  rw [innerTail496_12, scan, if_neg (by decide), innerCandidates496_12_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_13
    (if 3 ≤ innerCandidates496_12.length then search edge 3 [3559, 7417] innerCandidates496_12 10976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_13

def innerTail496_11 : List Nat := 7369 :: innerTail496_12
def innerCandidates496_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates496_11_checked : candidates edge 4 3559 7369 innerTail496_12 best.bound = innerCandidates496_11 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_11 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_11 best = best := by
  rw [innerTail496_11, scan, if_neg (by decide), innerCandidates496_11_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_12
    (if 3 ≤ innerCandidates496_11.length then search edge 3 [3559, 7369] innerCandidates496_11 10928 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_12

def innerTail496_10 : List Nat := 6961 :: innerTail496_11
def innerCandidates496_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates496_10_checked : candidates edge 4 3559 6961 innerTail496_11 best.bound = innerCandidates496_10 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_10 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_10 best = best := by
  rw [innerTail496_10, scan, if_neg (by decide), innerCandidates496_10_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_11
    (if 3 ≤ innerCandidates496_10.length then search edge 3 [3559, 6961] innerCandidates496_10 10520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_11

def innerTail496_9 : List Nat := 6619 :: innerTail496_10
def innerCandidates496_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates496_9_checked : candidates edge 4 3559 6619 innerTail496_10 best.bound = innerCandidates496_9 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_9 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_9 best = best := by
  rw [innerTail496_9, scan, if_neg (by decide), innerCandidates496_9_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_10
    (if 3 ≤ innerCandidates496_9.length then search edge 3 [3559, 6619] innerCandidates496_9 10178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_10

def innerTail496_8 : List Nat := 6607 :: innerTail496_9
def innerCandidates496_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates496_8_checked : candidates edge 4 3559 6607 innerTail496_9 best.bound = innerCandidates496_8 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_8 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_8 best = best := by
  rw [innerTail496_8, scan, if_neg (by decide), innerCandidates496_8_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_9
    (if 3 ≤ innerCandidates496_8.length then search edge 3 [3559, 6607] innerCandidates496_8 10166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_9

def innerTail496_7 : List Nat := 6037 :: innerTail496_8
def innerCandidates496_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates496_7_checked : candidates edge 4 3559 6037 innerTail496_8 best.bound = innerCandidates496_7 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_7 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_7 best = best := by
  rw [innerTail496_7, scan, if_neg (by decide), innerCandidates496_7_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_8
    (if 3 ≤ innerCandidates496_7.length then search edge 3 [3559, 6037] innerCandidates496_7 9596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_8

def innerTail496_6 : List Nat := 5869 :: innerTail496_7
def innerCandidates496_6 : List Nat := [6961]
set_option maxRecDepth 4096 in
theorem innerCandidates496_6_checked : candidates edge 4 3559 5869 innerTail496_7 best.bound = innerCandidates496_6 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_6 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_6 best = best := by
  rw [innerTail496_6, scan, if_neg (by decide), innerCandidates496_6_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_7
    (if 3 ≤ innerCandidates496_6.length then search edge 3 [3559, 5869] innerCandidates496_6 9428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_7

def innerTail496_5 : List Nat := 5449 :: innerTail496_6
def innerCandidates496_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates496_5_checked : candidates edge 4 3559 5449 innerTail496_6 best.bound = innerCandidates496_5 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_5 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_5 best = best := by
  rw [innerTail496_5, scan, if_neg (by decide), innerCandidates496_5_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_6
    (if 3 ≤ innerCandidates496_5.length then search edge 3 [3559, 5449] innerCandidates496_5 9008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_6

def innerTail496_4 : List Nat := 4021 :: innerTail496_5
def innerCandidates496_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates496_4_checked : candidates edge 4 3559 4021 innerTail496_5 best.bound = innerCandidates496_4 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_4 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_4 best = best := by
  rw [innerTail496_4, scan, if_neg (by decide), innerCandidates496_4_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_5
    (if 3 ≤ innerCandidates496_4.length then search edge 3 [3559, 4021] innerCandidates496_4 7580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_5

def innerTail496_3 : List Nat := 3931 :: innerTail496_4
def innerCandidates496_3 : List Nat := [5449]
set_option maxRecDepth 4096 in
theorem innerCandidates496_3_checked : candidates edge 4 3559 3931 innerTail496_4 best.bound = innerCandidates496_3 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_3 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_3 best = best := by
  rw [innerTail496_3, scan, if_neg (by decide), innerCandidates496_3_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_4
    (if 3 ≤ innerCandidates496_3.length then search edge 3 [3559, 3931] innerCandidates496_3 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_4

def innerTail496_2 : List Nat := 3769 :: innerTail496_3
def innerCandidates496_2 : List Nat := [6619]
set_option maxRecDepth 4096 in
theorem innerCandidates496_2_checked : candidates edge 4 3559 3769 innerTail496_3 best.bound = innerCandidates496_2 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_2 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_2 best = best := by
  rw [innerTail496_2, scan, if_neg (by decide), innerCandidates496_2_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_3
    (if 3 ≤ innerCandidates496_2.length then search edge 3 [3559, 3769] innerCandidates496_2 7328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_3

def innerTail496_1 : List Nat := 3709 :: innerTail496_2
def innerCandidates496_1 : List Nat := [7369]
set_option maxRecDepth 4096 in
theorem innerCandidates496_1_checked : candidates edge 4 3559 3709 innerTail496_2 best.bound = innerCandidates496_1 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_1 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_1 best = best := by
  rw [innerTail496_1, scan, if_neg (by decide), innerCandidates496_1_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_2
    (if 3 ≤ innerCandidates496_1.length then search edge 3 [3559, 3709] innerCandidates496_1 7268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_2

def innerTail496_0 : List Nat := 3583 :: innerTail496_1
def innerCandidates496_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates496_0_checked : candidates edge 4 3559 3583 innerTail496_1 best.bound = innerCandidates496_0 := by
  unfold candidates
  have row : edge 3583 = SearchData.row3583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan496_0 : scan edge (search edge 3) 4 [3559] 3559 innerTail496_0 best = best := by
  rw [innerTail496_0, scan, if_neg (by decide), innerCandidates496_0_checked]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_1
    (if 3 ≤ innerCandidates496_0.length then search edge 3 [3559, 3583] innerCandidates496_0 7142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan496_1

theorem rootBranch496 : search edge 4 [3559] rootCandidates496 3559 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3559] 3559 innerTail496_0 best = best
  exact innerScan496_0

def rootCandidates495 : List Nat := [3803, 4139, 5021, 5051, 5279, 5417, 5783, 6131, 6329, 6551]
set_option maxRecDepth 4096 in
theorem rootCandidates495_checked : candidates edge 5 0 3557 tail496 best.bound = rootCandidates495 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext q; rfl
  rw [row]
  decide

def innerTail495_10 : List Nat := []
theorem innerScan495_10 : scan edge (search edge 3) 4 [3557] 3557 innerTail495_10 best = best := rfl

def innerTail495_9 : List Nat := 6551 :: innerTail495_10
def innerCandidates495_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates495_9_checked : candidates edge 4 3557 6551 innerTail495_10 best.bound = innerCandidates495_9 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan495_9 : scan edge (search edge 3) 4 [3557] 3557 innerTail495_9 best = best := by
  rw [innerTail495_9, scan, if_neg (by decide), innerCandidates495_9_checked]
  change scan edge (search edge 3) 4 [3557] 3557 innerTail495_10
    (if 3 ≤ innerCandidates495_9.length then search edge 3 [3557, 6551] innerCandidates495_9 10108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan495_10

def innerTail495_8 : List Nat := 6329 :: innerTail495_9
def innerCandidates495_8 : List Nat := [6551]
set_option maxRecDepth 4096 in
theorem innerCandidates495_8_checked : candidates edge 4 3557 6329 innerTail495_9 best.bound = innerCandidates495_8 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan495_8 : scan edge (search edge 3) 4 [3557] 3557 innerTail495_8 best = best := by
  rw [innerTail495_8, scan, if_neg (by decide), innerCandidates495_8_checked]
  change scan edge (search edge 3) 4 [3557] 3557 innerTail495_9
    (if 3 ≤ innerCandidates495_8.length then search edge 3 [3557, 6329] innerCandidates495_8 9886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan495_9

def innerTail495_7 : List Nat := 6131 :: innerTail495_8
def innerCandidates495_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates495_7_checked : candidates edge 4 3557 6131 innerTail495_8 best.bound = innerCandidates495_7 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan495_7 : scan edge (search edge 3) 4 [3557] 3557 innerTail495_7 best = best := by
  rw [innerTail495_7, scan, if_neg (by decide), innerCandidates495_7_checked]
  change scan edge (search edge 3) 4 [3557] 3557 innerTail495_8
    (if 3 ≤ innerCandidates495_7.length then search edge 3 [3557, 6131] innerCandidates495_7 9688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan495_8

def innerTail495_6 : List Nat := 5783 :: innerTail495_7
def innerCandidates495_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates495_6_checked : candidates edge 4 3557 5783 innerTail495_7 best.bound = innerCandidates495_6 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan495_6 : scan edge (search edge 3) 4 [3557] 3557 innerTail495_6 best = best := by
  rw [innerTail495_6, scan, if_neg (by decide), innerCandidates495_6_checked]
  change scan edge (search edge 3) 4 [3557] 3557 innerTail495_7
    (if 3 ≤ innerCandidates495_6.length then search edge 3 [3557, 5783] innerCandidates495_6 9340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan495_7

def innerTail495_5 : List Nat := 5417 :: innerTail495_6
def innerCandidates495_5 : List Nat := [6131]
set_option maxRecDepth 4096 in
theorem innerCandidates495_5_checked : candidates edge 4 3557 5417 innerTail495_6 best.bound = innerCandidates495_5 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan495_5 : scan edge (search edge 3) 4 [3557] 3557 innerTail495_5 best = best := by
  rw [innerTail495_5, scan, if_neg (by decide), innerCandidates495_5_checked]
  change scan edge (search edge 3) 4 [3557] 3557 innerTail495_6
    (if 3 ≤ innerCandidates495_5.length then search edge 3 [3557, 5417] innerCandidates495_5 8974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan495_6

def innerTail495_4 : List Nat := 5279 :: innerTail495_5
def innerCandidates495_4 : List Nat := [5783]
set_option maxRecDepth 4096 in
theorem innerCandidates495_4_checked : candidates edge 4 3557 5279 innerTail495_5 best.bound = innerCandidates495_4 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan495_4 : scan edge (search edge 3) 4 [3557] 3557 innerTail495_4 best = best := by
  rw [innerTail495_4, scan, if_neg (by decide), innerCandidates495_4_checked]
  change scan edge (search edge 3) 4 [3557] 3557 innerTail495_5
    (if 3 ≤ innerCandidates495_4.length then search edge 3 [3557, 5279] innerCandidates495_4 8836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan495_5

def innerTail495_3 : List Nat := 5051 :: innerTail495_4
def innerCandidates495_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates495_3_checked : candidates edge 4 3557 5051 innerTail495_4 best.bound = innerCandidates495_3 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan495_3 : scan edge (search edge 3) 4 [3557] 3557 innerTail495_3 best = best := by
  rw [innerTail495_3, scan, if_neg (by decide), innerCandidates495_3_checked]
  change scan edge (search edge 3) 4 [3557] 3557 innerTail495_4
    (if 3 ≤ innerCandidates495_3.length then search edge 3 [3557, 5051] innerCandidates495_3 8608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan495_4

def innerTail495_2 : List Nat := 5021 :: innerTail495_3
def innerCandidates495_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates495_2_checked : candidates edge 4 3557 5021 innerTail495_3 best.bound = innerCandidates495_2 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan495_2 : scan edge (search edge 3) 4 [3557] 3557 innerTail495_2 best = best := by
  rw [innerTail495_2, scan, if_neg (by decide), innerCandidates495_2_checked]
  change scan edge (search edge 3) 4 [3557] 3557 innerTail495_3
    (if 3 ≤ innerCandidates495_2.length then search edge 3 [3557, 5021] innerCandidates495_2 8578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan495_3

def innerTail495_1 : List Nat := 4139 :: innerTail495_2
def innerCandidates495_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates495_1_checked : candidates edge 4 3557 4139 innerTail495_2 best.bound = innerCandidates495_1 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan495_1 : scan edge (search edge 3) 4 [3557] 3557 innerTail495_1 best = best := by
  rw [innerTail495_1, scan, if_neg (by decide), innerCandidates495_1_checked]
  change scan edge (search edge 3) 4 [3557] 3557 innerTail495_2
    (if 3 ≤ innerCandidates495_1.length then search edge 3 [3557, 4139] innerCandidates495_1 7696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan495_2

def innerTail495_0 : List Nat := 3803 :: innerTail495_1
def innerCandidates495_0 : List Nat := [5051]
set_option maxRecDepth 4096 in
theorem innerCandidates495_0_checked : candidates edge 4 3557 3803 innerTail495_1 best.bound = innerCandidates495_0 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan495_0 : scan edge (search edge 3) 4 [3557] 3557 innerTail495_0 best = best := by
  rw [innerTail495_0, scan, if_neg (by decide), innerCandidates495_0_checked]
  change scan edge (search edge 3) 4 [3557] 3557 innerTail495_1
    (if 3 ≤ innerCandidates495_0.length then search edge 3 [3557, 3803] innerCandidates495_0 7360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan495_1

theorem rootBranch495 : search edge 4 [3557] rootCandidates495 3557 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3557] 3557 innerTail495_0 best = best
  exact innerScan495_0

def rootCandidates494 : List Nat := [3643, 4357, 5449, 5827, 6151, 6481, 6709, 7213, 7243, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates494_checked : candidates edge 5 0 3547 tail495 best.bound = rootCandidates494 := by
  unfold candidates
  have row : edge 3547 = SearchData.row3547.contains := by funext q; rfl
  rw [row]
  decide

def innerTail494_10 : List Nat := []
theorem innerScan494_10 : scan edge (search edge 3) 4 [3547] 3547 innerTail494_10 best = best := rfl

def innerTail494_9 : List Nat := 7993 :: innerTail494_10
def innerCandidates494_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates494_9_checked : candidates edge 4 3547 7993 innerTail494_10 best.bound = innerCandidates494_9 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan494_9 : scan edge (search edge 3) 4 [3547] 3547 innerTail494_9 best = best := by
  rw [innerTail494_9, scan, if_neg (by decide), innerCandidates494_9_checked]
  change scan edge (search edge 3) 4 [3547] 3547 innerTail494_10
    (if 3 ≤ innerCandidates494_9.length then search edge 3 [3547, 7993] innerCandidates494_9 11540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan494_10

def innerTail494_8 : List Nat := 7243 :: innerTail494_9
def innerCandidates494_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates494_8_checked : candidates edge 4 3547 7243 innerTail494_9 best.bound = innerCandidates494_8 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan494_8 : scan edge (search edge 3) 4 [3547] 3547 innerTail494_8 best = best := by
  rw [innerTail494_8, scan, if_neg (by decide), innerCandidates494_8_checked]
  change scan edge (search edge 3) 4 [3547] 3547 innerTail494_9
    (if 3 ≤ innerCandidates494_8.length then search edge 3 [3547, 7243] innerCandidates494_8 10790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan494_9

def innerTail494_7 : List Nat := 7213 :: innerTail494_8
def innerCandidates494_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates494_7_checked : candidates edge 4 3547 7213 innerTail494_8 best.bound = innerCandidates494_7 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan494_7 : scan edge (search edge 3) 4 [3547] 3547 innerTail494_7 best = best := by
  rw [innerTail494_7, scan, if_neg (by decide), innerCandidates494_7_checked]
  change scan edge (search edge 3) 4 [3547] 3547 innerTail494_8
    (if 3 ≤ innerCandidates494_7.length then search edge 3 [3547, 7213] innerCandidates494_7 10760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan494_8

def innerTail494_6 : List Nat := 6709 :: innerTail494_7
def innerCandidates494_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates494_6_checked : candidates edge 4 3547 6709 innerTail494_7 best.bound = innerCandidates494_6 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan494_6 : scan edge (search edge 3) 4 [3547] 3547 innerTail494_6 best = best := by
  rw [innerTail494_6, scan, if_neg (by decide), innerCandidates494_6_checked]
  change scan edge (search edge 3) 4 [3547] 3547 innerTail494_7
    (if 3 ≤ innerCandidates494_6.length then search edge 3 [3547, 6709] innerCandidates494_6 10256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan494_7

def innerTail494_5 : List Nat := 6481 :: innerTail494_6
def innerCandidates494_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates494_5_checked : candidates edge 4 3547 6481 innerTail494_6 best.bound = innerCandidates494_5 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan494_5 : scan edge (search edge 3) 4 [3547] 3547 innerTail494_5 best = best := by
  rw [innerTail494_5, scan, if_neg (by decide), innerCandidates494_5_checked]
  change scan edge (search edge 3) 4 [3547] 3547 innerTail494_6
    (if 3 ≤ innerCandidates494_5.length then search edge 3 [3547, 6481] innerCandidates494_5 10028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan494_6

def innerTail494_4 : List Nat := 6151 :: innerTail494_5
def innerCandidates494_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates494_4_checked : candidates edge 4 3547 6151 innerTail494_5 best.bound = innerCandidates494_4 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan494_4 : scan edge (search edge 3) 4 [3547] 3547 innerTail494_4 best = best := by
  rw [innerTail494_4, scan, if_neg (by decide), innerCandidates494_4_checked]
  change scan edge (search edge 3) 4 [3547] 3547 innerTail494_5
    (if 3 ≤ innerCandidates494_4.length then search edge 3 [3547, 6151] innerCandidates494_4 9698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan494_5

def innerTail494_3 : List Nat := 5827 :: innerTail494_4
def innerCandidates494_3 : List Nat := [6151, 7213]
set_option maxRecDepth 4096 in
theorem innerCandidates494_3_checked : candidates edge 4 3547 5827 innerTail494_4 best.bound = innerCandidates494_3 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan494_3 : scan edge (search edge 3) 4 [3547] 3547 innerTail494_3 best = best := by
  rw [innerTail494_3, scan, if_neg (by decide), innerCandidates494_3_checked]
  change scan edge (search edge 3) 4 [3547] 3547 innerTail494_4
    (if 3 ≤ innerCandidates494_3.length then search edge 3 [3547, 5827] innerCandidates494_3 9374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan494_4

def innerTail494_2 : List Nat := 5449 :: innerTail494_3
def innerCandidates494_2 : List Nat := [7243]
set_option maxRecDepth 4096 in
theorem innerCandidates494_2_checked : candidates edge 4 3547 5449 innerTail494_3 best.bound = innerCandidates494_2 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan494_2 : scan edge (search edge 3) 4 [3547] 3547 innerTail494_2 best = best := by
  rw [innerTail494_2, scan, if_neg (by decide), innerCandidates494_2_checked]
  change scan edge (search edge 3) 4 [3547] 3547 innerTail494_3
    (if 3 ≤ innerCandidates494_2.length then search edge 3 [3547, 5449] innerCandidates494_2 8996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan494_3

def innerTail494_1 : List Nat := 4357 :: innerTail494_2
def innerCandidates494_1 : List Nat := [5449]
set_option maxRecDepth 4096 in
theorem innerCandidates494_1_checked : candidates edge 4 3547 4357 innerTail494_2 best.bound = innerCandidates494_1 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan494_1 : scan edge (search edge 3) 4 [3547] 3547 innerTail494_1 best = best := by
  rw [innerTail494_1, scan, if_neg (by decide), innerCandidates494_1_checked]
  change scan edge (search edge 3) 4 [3547] 3547 innerTail494_2
    (if 3 ≤ innerCandidates494_1.length then search edge 3 [3547, 4357] innerCandidates494_1 7904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan494_2

def innerTail494_0 : List Nat := 3643 :: innerTail494_1
def innerCandidates494_0 : List Nat := [5449]
set_option maxRecDepth 4096 in
theorem innerCandidates494_0_checked : candidates edge 4 3547 3643 innerTail494_1 best.bound = innerCandidates494_0 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan494_0 : scan edge (search edge 3) 4 [3547] 3547 innerTail494_0 best = best := by
  rw [innerTail494_0, scan, if_neg (by decide), innerCandidates494_0_checked]
  change scan edge (search edge 3) 4 [3547] 3547 innerTail494_1
    (if 3 ≤ innerCandidates494_0.length then search edge 3 [3547, 3643] innerCandidates494_0 7190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan494_1

theorem rootBranch494 : search edge 4 [3547] rootCandidates494 3547 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3547] 3547 innerTail494_0 best = best
  exact innerScan494_0

def rootCandidates493 : List Nat := [4567, 7243, 7351, 7489, 7951, 8011, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates493_checked : candidates edge 5 0 3541 tail494 best.bound = rootCandidates493 := by
  unfold candidates
  have row : edge 3541 = SearchData.row3541.contains := by funext q; rfl
  rw [row]
  decide

def innerTail493_7 : List Nat := []
theorem innerScan493_7 : scan edge (search edge 3) 4 [3541] 3541 innerTail493_7 best = best := rfl

def innerTail493_6 : List Nat := 8263 :: innerTail493_7
def innerCandidates493_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates493_6_checked : candidates edge 4 3541 8263 innerTail493_7 best.bound = innerCandidates493_6 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan493_6 : scan edge (search edge 3) 4 [3541] 3541 innerTail493_6 best = best := by
  rw [innerTail493_6, scan, if_neg (by decide), innerCandidates493_6_checked]
  change scan edge (search edge 3) 4 [3541] 3541 innerTail493_7
    (if 3 ≤ innerCandidates493_6.length then search edge 3 [3541, 8263] innerCandidates493_6 11804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan493_7

def innerTail493_5 : List Nat := 8011 :: innerTail493_6
def innerCandidates493_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates493_5_checked : candidates edge 4 3541 8011 innerTail493_6 best.bound = innerCandidates493_5 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan493_5 : scan edge (search edge 3) 4 [3541] 3541 innerTail493_5 best = best := by
  rw [innerTail493_5, scan, if_neg (by decide), innerCandidates493_5_checked]
  change scan edge (search edge 3) 4 [3541] 3541 innerTail493_6
    (if 3 ≤ innerCandidates493_5.length then search edge 3 [3541, 8011] innerCandidates493_5 11552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan493_6

def innerTail493_4 : List Nat := 7951 :: innerTail493_5
def innerCandidates493_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates493_4_checked : candidates edge 4 3541 7951 innerTail493_5 best.bound = innerCandidates493_4 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan493_4 : scan edge (search edge 3) 4 [3541] 3541 innerTail493_4 best = best := by
  rw [innerTail493_4, scan, if_neg (by decide), innerCandidates493_4_checked]
  change scan edge (search edge 3) 4 [3541] 3541 innerTail493_5
    (if 3 ≤ innerCandidates493_4.length then search edge 3 [3541, 7951] innerCandidates493_4 11492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan493_5

def innerTail493_3 : List Nat := 7489 :: innerTail493_4
def innerCandidates493_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates493_3_checked : candidates edge 4 3541 7489 innerTail493_4 best.bound = innerCandidates493_3 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan493_3 : scan edge (search edge 3) 4 [3541] 3541 innerTail493_3 best = best := by
  rw [innerTail493_3, scan, if_neg (by decide), innerCandidates493_3_checked]
  change scan edge (search edge 3) 4 [3541] 3541 innerTail493_4
    (if 3 ≤ innerCandidates493_3.length then search edge 3 [3541, 7489] innerCandidates493_3 11030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan493_4

def innerTail493_2 : List Nat := 7351 :: innerTail493_3
def innerCandidates493_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates493_2_checked : candidates edge 4 3541 7351 innerTail493_3 best.bound = innerCandidates493_2 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan493_2 : scan edge (search edge 3) 4 [3541] 3541 innerTail493_2 best = best := by
  rw [innerTail493_2, scan, if_neg (by decide), innerCandidates493_2_checked]
  change scan edge (search edge 3) 4 [3541] 3541 innerTail493_3
    (if 3 ≤ innerCandidates493_2.length then search edge 3 [3541, 7351] innerCandidates493_2 10892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan493_3

def innerTail493_1 : List Nat := 7243 :: innerTail493_2
def innerCandidates493_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates493_1_checked : candidates edge 4 3541 7243 innerTail493_2 best.bound = innerCandidates493_1 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan493_1 : scan edge (search edge 3) 4 [3541] 3541 innerTail493_1 best = best := by
  rw [innerTail493_1, scan, if_neg (by decide), innerCandidates493_1_checked]
  change scan edge (search edge 3) 4 [3541] 3541 innerTail493_2
    (if 3 ≤ innerCandidates493_1.length then search edge 3 [3541, 7243] innerCandidates493_1 10784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan493_2

def innerTail493_0 : List Nat := 4567 :: innerTail493_1
def innerCandidates493_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates493_0_checked : candidates edge 4 3541 4567 innerTail493_1 best.bound = innerCandidates493_0 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan493_0 : scan edge (search edge 3) 4 [3541] 3541 innerTail493_0 best = best := by
  rw [innerTail493_0, scan, if_neg (by decide), innerCandidates493_0_checked]
  change scan edge (search edge 3) 4 [3541] 3541 innerTail493_1
    (if 3 ≤ innerCandidates493_0.length then search edge 3 [3541, 4567] innerCandidates493_0 8108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan493_1

theorem rootBranch493 : search edge 4 [3541] rootCandidates493 3541 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3541] 3541 innerTail493_0 best = best
  exact innerScan493_0

def rootCandidates492 : List Nat := [3671, 4013, 4649, 5441, 5471, 5861, 6521, 7001, 7829]
set_option maxRecDepth 4096 in
theorem rootCandidates492_checked : candidates edge 5 0 3539 tail493 best.bound = rootCandidates492 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext q; rfl
  rw [row]
  decide

def innerTail492_9 : List Nat := []
theorem innerScan492_9 : scan edge (search edge 3) 4 [3539] 3539 innerTail492_9 best = best := rfl

def innerTail492_8 : List Nat := 7829 :: innerTail492_9
def innerCandidates492_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates492_8_checked : candidates edge 4 3539 7829 innerTail492_9 best.bound = innerCandidates492_8 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan492_8 : scan edge (search edge 3) 4 [3539] 3539 innerTail492_8 best = best := by
  rw [innerTail492_8, scan, if_neg (by decide), innerCandidates492_8_checked]
  change scan edge (search edge 3) 4 [3539] 3539 innerTail492_9
    (if 3 ≤ innerCandidates492_8.length then search edge 3 [3539, 7829] innerCandidates492_8 11368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan492_9

def innerTail492_7 : List Nat := 7001 :: innerTail492_8
def innerCandidates492_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates492_7_checked : candidates edge 4 3539 7001 innerTail492_8 best.bound = innerCandidates492_7 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan492_7 : scan edge (search edge 3) 4 [3539] 3539 innerTail492_7 best = best := by
  rw [innerTail492_7, scan, if_neg (by decide), innerCandidates492_7_checked]
  change scan edge (search edge 3) 4 [3539] 3539 innerTail492_8
    (if 3 ≤ innerCandidates492_7.length then search edge 3 [3539, 7001] innerCandidates492_7 10540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan492_8

def innerTail492_6 : List Nat := 6521 :: innerTail492_7
def innerCandidates492_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates492_6_checked : candidates edge 4 3539 6521 innerTail492_7 best.bound = innerCandidates492_6 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan492_6 : scan edge (search edge 3) 4 [3539] 3539 innerTail492_6 best = best := by
  rw [innerTail492_6, scan, if_neg (by decide), innerCandidates492_6_checked]
  change scan edge (search edge 3) 4 [3539] 3539 innerTail492_7
    (if 3 ≤ innerCandidates492_6.length then search edge 3 [3539, 6521] innerCandidates492_6 10060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan492_7

def innerTail492_5 : List Nat := 5861 :: innerTail492_6
def innerCandidates492_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates492_5_checked : candidates edge 4 3539 5861 innerTail492_6 best.bound = innerCandidates492_5 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan492_5 : scan edge (search edge 3) 4 [3539] 3539 innerTail492_5 best = best := by
  rw [innerTail492_5, scan, if_neg (by decide), innerCandidates492_5_checked]
  change scan edge (search edge 3) 4 [3539] 3539 innerTail492_6
    (if 3 ≤ innerCandidates492_5.length then search edge 3 [3539, 5861] innerCandidates492_5 9400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan492_6

def innerTail492_4 : List Nat := 5471 :: innerTail492_5
def innerCandidates492_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates492_4_checked : candidates edge 4 3539 5471 innerTail492_5 best.bound = innerCandidates492_4 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan492_4 : scan edge (search edge 3) 4 [3539] 3539 innerTail492_4 best = best := by
  rw [innerTail492_4, scan, if_neg (by decide), innerCandidates492_4_checked]
  change scan edge (search edge 3) 4 [3539] 3539 innerTail492_5
    (if 3 ≤ innerCandidates492_4.length then search edge 3 [3539, 5471] innerCandidates492_4 9010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan492_5

def innerTail492_3 : List Nat := 5441 :: innerTail492_4
def innerCandidates492_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates492_3_checked : candidates edge 4 3539 5441 innerTail492_4 best.bound = innerCandidates492_3 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan492_3 : scan edge (search edge 3) 4 [3539] 3539 innerTail492_3 best = best := by
  rw [innerTail492_3, scan, if_neg (by decide), innerCandidates492_3_checked]
  change scan edge (search edge 3) 4 [3539] 3539 innerTail492_4
    (if 3 ≤ innerCandidates492_3.length then search edge 3 [3539, 5441] innerCandidates492_3 8980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan492_4

def innerTail492_2 : List Nat := 4649 :: innerTail492_3
def innerCandidates492_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates492_2_checked : candidates edge 4 3539 4649 innerTail492_3 best.bound = innerCandidates492_2 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan492_2 : scan edge (search edge 3) 4 [3539] 3539 innerTail492_2 best = best := by
  rw [innerTail492_2, scan, if_neg (by decide), innerCandidates492_2_checked]
  change scan edge (search edge 3) 4 [3539] 3539 innerTail492_3
    (if 3 ≤ innerCandidates492_2.length then search edge 3 [3539, 4649] innerCandidates492_2 8188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan492_3

def innerTail492_1 : List Nat := 4013 :: innerTail492_2
def innerCandidates492_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates492_1_checked : candidates edge 4 3539 4013 innerTail492_2 best.bound = innerCandidates492_1 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan492_1 : scan edge (search edge 3) 4 [3539] 3539 innerTail492_1 best = best := by
  rw [innerTail492_1, scan, if_neg (by decide), innerCandidates492_1_checked]
  change scan edge (search edge 3) 4 [3539] 3539 innerTail492_2
    (if 3 ≤ innerCandidates492_1.length then search edge 3 [3539, 4013] innerCandidates492_1 7552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan492_2

def innerTail492_0 : List Nat := 3671 :: innerTail492_1
def innerCandidates492_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates492_0_checked : candidates edge 4 3539 3671 innerTail492_1 best.bound = innerCandidates492_0 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan492_0 : scan edge (search edge 3) 4 [3539] 3539 innerTail492_0 best = best := by
  rw [innerTail492_0, scan, if_neg (by decide), innerCandidates492_0_checked]
  change scan edge (search edge 3) 4 [3539] 3539 innerTail492_1
    (if 3 ≤ innerCandidates492_0.length then search edge 3 [3539, 3671] innerCandidates492_0 7210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan492_1

theorem rootBranch492 : search edge 4 [3539] rootCandidates492 3539 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3539] 3539 innerTail492_0 best = best
  exact innerScan492_0

end ProjectEulerSolutions.PrimePairUpper.Search
