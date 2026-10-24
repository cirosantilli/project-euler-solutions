import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates408 : List Nat := [3037, 3229, 4021, 4591, 4759, 5113, 5683, 5851, 5869, 6361, 7297, 7723]
set_option maxRecDepth 4096 in
theorem rootCandidates408_checked : candidates edge 5 0 2833 tail409 best.bound = rootCandidates408 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext q; rfl
  rw [row]
  decide

def innerTail408_12 : List Nat := []
theorem innerScan408_12 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_12 best = best := rfl

def innerTail408_11 : List Nat := 7723 :: innerTail408_12
def innerCandidates408_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates408_11_checked : candidates edge 4 2833 7723 innerTail408_12 best.bound = innerCandidates408_11 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan408_11 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_11 best = best := by
  rw [innerTail408_11, scan, if_neg (by decide), innerCandidates408_11_checked]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_12
    (if 3 ≤ innerCandidates408_11.length then search edge 3 [2833, 7723] innerCandidates408_11 10556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan408_12

def innerTail408_10 : List Nat := 7297 :: innerTail408_11
def innerCandidates408_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates408_10_checked : candidates edge 4 2833 7297 innerTail408_11 best.bound = innerCandidates408_10 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan408_10 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_10 best = best := by
  rw [innerTail408_10, scan, if_neg (by decide), innerCandidates408_10_checked]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_11
    (if 3 ≤ innerCandidates408_10.length then search edge 3 [2833, 7297] innerCandidates408_10 10130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan408_11

def innerTail408_9 : List Nat := 6361 :: innerTail408_10
def innerCandidates408_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates408_9_checked : candidates edge 4 2833 6361 innerTail408_10 best.bound = innerCandidates408_9 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan408_9 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_9 best = best := by
  rw [innerTail408_9, scan, if_neg (by decide), innerCandidates408_9_checked]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_10
    (if 3 ≤ innerCandidates408_9.length then search edge 3 [2833, 6361] innerCandidates408_9 9194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan408_10

def innerTail408_8 : List Nat := 5869 :: innerTail408_9
def innerCandidates408_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates408_8_checked : candidates edge 4 2833 5869 innerTail408_9 best.bound = innerCandidates408_8 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan408_8 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_8 best = best := by
  rw [innerTail408_8, scan, if_neg (by decide), innerCandidates408_8_checked]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_9
    (if 3 ≤ innerCandidates408_8.length then search edge 3 [2833, 5869] innerCandidates408_8 8702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan408_9

def innerTail408_7 : List Nat := 5851 :: innerTail408_8
def innerCandidates408_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates408_7_checked : candidates edge 4 2833 5851 innerTail408_8 best.bound = innerCandidates408_7 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan408_7 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_7 best = best := by
  rw [innerTail408_7, scan, if_neg (by decide), innerCandidates408_7_checked]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_8
    (if 3 ≤ innerCandidates408_7.length then search edge 3 [2833, 5851] innerCandidates408_7 8684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan408_8

def innerTail408_6 : List Nat := 5683 :: innerTail408_7
def innerCandidates408_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates408_6_checked : candidates edge 4 2833 5683 innerTail408_7 best.bound = innerCandidates408_6 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan408_6 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_6 best = best := by
  rw [innerTail408_6, scan, if_neg (by decide), innerCandidates408_6_checked]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_7
    (if 3 ≤ innerCandidates408_6.length then search edge 3 [2833, 5683] innerCandidates408_6 8516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan408_7

def innerTail408_5 : List Nat := 5113 :: innerTail408_6
def innerCandidates408_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates408_5_checked : candidates edge 4 2833 5113 innerTail408_6 best.bound = innerCandidates408_5 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan408_5 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_5 best = best := by
  rw [innerTail408_5, scan, if_neg (by decide), innerCandidates408_5_checked]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_6
    (if 3 ≤ innerCandidates408_5.length then search edge 3 [2833, 5113] innerCandidates408_5 7946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan408_6

def innerTail408_4 : List Nat := 4759 :: innerTail408_5
def innerCandidates408_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates408_4_checked : candidates edge 4 2833 4759 innerTail408_5 best.bound = innerCandidates408_4 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan408_4 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_4 best = best := by
  rw [innerTail408_4, scan, if_neg (by decide), innerCandidates408_4_checked]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_5
    (if 3 ≤ innerCandidates408_4.length then search edge 3 [2833, 4759] innerCandidates408_4 7592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan408_5

def innerTail408_3 : List Nat := 4591 :: innerTail408_4
def innerCandidates408_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates408_3_checked : candidates edge 4 2833 4591 innerTail408_4 best.bound = innerCandidates408_3 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan408_3 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_3 best = best := by
  rw [innerTail408_3, scan, if_neg (by decide), innerCandidates408_3_checked]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_4
    (if 3 ≤ innerCandidates408_3.length then search edge 3 [2833, 4591] innerCandidates408_3 7424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan408_4

def innerTail408_2 : List Nat := 4021 :: innerTail408_3
def innerCandidates408_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates408_2_checked : candidates edge 4 2833 4021 innerTail408_3 best.bound = innerCandidates408_2 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan408_2 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_2 best = best := by
  rw [innerTail408_2, scan, if_neg (by decide), innerCandidates408_2_checked]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_3
    (if 3 ≤ innerCandidates408_2.length then search edge 3 [2833, 4021] innerCandidates408_2 6854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan408_3

def innerTail408_1 : List Nat := 3229 :: innerTail408_2
def innerCandidates408_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates408_1_checked : candidates edge 4 2833 3229 innerTail408_2 best.bound = innerCandidates408_1 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan408_1 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_1 best = best := by
  rw [innerTail408_1, scan, if_neg (by decide), innerCandidates408_1_checked]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_2
    (if 3 ≤ innerCandidates408_1.length then search edge 3 [2833, 3229] innerCandidates408_1 6062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan408_2

def innerTail408_0 : List Nat := 3037 :: innerTail408_1
def innerCandidates408_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates408_0_checked : candidates edge 4 2833 3037 innerTail408_1 best.bound = innerCandidates408_0 := by
  unfold candidates
  have row : edge 3037 = SearchData.row3037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan408_0 : scan edge (search edge 3) 4 [2833] 2833 innerTail408_0 best = best := by
  rw [innerTail408_0, scan, if_neg (by decide), innerCandidates408_0_checked]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_1
    (if 3 ≤ innerCandidates408_0.length then search edge 3 [2833, 3037] innerCandidates408_0 5870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan408_1

theorem rootBranch408 : search edge 4 [2833] rootCandidates408 2833 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2833] 2833 innerTail408_0 best = best
  exact innerScan408_0

def rootCandidates407 : List Nat := [2897, 3617, 4079, 4787, 5039, 5351, 5381, 5477, 5879, 5927, 6653, 7187, 7523, 7607, 8081]
set_option maxRecDepth 4096 in
theorem rootCandidates407_checked : candidates edge 5 0 2819 tail408 best.bound = rootCandidates407 := by
  unfold candidates
  have row : edge 2819 = SearchData.row2819.contains := by funext q; rfl
  rw [row]
  decide

def innerTail407_15 : List Nat := []
theorem innerScan407_15 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_15 best = best := rfl

def innerTail407_14 : List Nat := 8081 :: innerTail407_15
def innerCandidates407_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates407_14_checked : candidates edge 4 2819 8081 innerTail407_15 best.bound = innerCandidates407_14 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_14 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_14 best = best := by
  rw [innerTail407_14, scan, if_neg (by decide), innerCandidates407_14_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_15
    (if 3 ≤ innerCandidates407_14.length then search edge 3 [2819, 8081] innerCandidates407_14 10900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_15

def innerTail407_13 : List Nat := 7607 :: innerTail407_14
def innerCandidates407_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates407_13_checked : candidates edge 4 2819 7607 innerTail407_14 best.bound = innerCandidates407_13 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_13 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_13 best = best := by
  rw [innerTail407_13, scan, if_neg (by decide), innerCandidates407_13_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_14
    (if 3 ≤ innerCandidates407_13.length then search edge 3 [2819, 7607] innerCandidates407_13 10426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_14

def innerTail407_12 : List Nat := 7523 :: innerTail407_13
def innerCandidates407_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates407_12_checked : candidates edge 4 2819 7523 innerTail407_13 best.bound = innerCandidates407_12 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_12 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_12 best = best := by
  rw [innerTail407_12, scan, if_neg (by decide), innerCandidates407_12_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_13
    (if 3 ≤ innerCandidates407_12.length then search edge 3 [2819, 7523] innerCandidates407_12 10342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_13

def innerTail407_11 : List Nat := 7187 :: innerTail407_12
def innerCandidates407_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates407_11_checked : candidates edge 4 2819 7187 innerTail407_12 best.bound = innerCandidates407_11 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_11 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_11 best = best := by
  rw [innerTail407_11, scan, if_neg (by decide), innerCandidates407_11_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_12
    (if 3 ≤ innerCandidates407_11.length then search edge 3 [2819, 7187] innerCandidates407_11 10006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_12

def innerTail407_10 : List Nat := 6653 :: innerTail407_11
def innerCandidates407_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates407_10_checked : candidates edge 4 2819 6653 innerTail407_11 best.bound = innerCandidates407_10 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_10 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_10 best = best := by
  rw [innerTail407_10, scan, if_neg (by decide), innerCandidates407_10_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_11
    (if 3 ≤ innerCandidates407_10.length then search edge 3 [2819, 6653] innerCandidates407_10 9472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_11

def innerTail407_9 : List Nat := 5927 :: innerTail407_10
def innerCandidates407_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates407_9_checked : candidates edge 4 2819 5927 innerTail407_10 best.bound = innerCandidates407_9 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_9 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_9 best = best := by
  rw [innerTail407_9, scan, if_neg (by decide), innerCandidates407_9_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_10
    (if 3 ≤ innerCandidates407_9.length then search edge 3 [2819, 5927] innerCandidates407_9 8746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_10

def innerTail407_8 : List Nat := 5879 :: innerTail407_9
def innerCandidates407_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates407_8_checked : candidates edge 4 2819 5879 innerTail407_9 best.bound = innerCandidates407_8 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_8 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_8 best = best := by
  rw [innerTail407_8, scan, if_neg (by decide), innerCandidates407_8_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_9
    (if 3 ≤ innerCandidates407_8.length then search edge 3 [2819, 5879] innerCandidates407_8 8698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_9

def innerTail407_7 : List Nat := 5477 :: innerTail407_8
def innerCandidates407_7 : List Nat := [5927, 8081]
set_option maxRecDepth 4096 in
theorem innerCandidates407_7_checked : candidates edge 4 2819 5477 innerTail407_8 best.bound = innerCandidates407_7 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_7 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_7 best = best := by
  rw [innerTail407_7, scan, if_neg (by decide), innerCandidates407_7_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_8
    (if 3 ≤ innerCandidates407_7.length then search edge 3 [2819, 5477] innerCandidates407_7 8296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_8

def innerTail407_6 : List Nat := 5381 :: innerTail407_7
def innerCandidates407_6 : List Nat := [8081]
set_option maxRecDepth 4096 in
theorem innerCandidates407_6_checked : candidates edge 4 2819 5381 innerTail407_7 best.bound = innerCandidates407_6 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_6 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_6 best = best := by
  rw [innerTail407_6, scan, if_neg (by decide), innerCandidates407_6_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_7
    (if 3 ≤ innerCandidates407_6.length then search edge 3 [2819, 5381] innerCandidates407_6 8200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_7

def innerTail407_5 : List Nat := 5351 :: innerTail407_6
def innerCandidates407_5 : List Nat := [5879, 5927]
set_option maxRecDepth 4096 in
theorem innerCandidates407_5_checked : candidates edge 4 2819 5351 innerTail407_6 best.bound = innerCandidates407_5 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_5 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_5 best = best := by
  rw [innerTail407_5, scan, if_neg (by decide), innerCandidates407_5_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_6
    (if 3 ≤ innerCandidates407_5.length then search edge 3 [2819, 5351] innerCandidates407_5 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_6

def innerTail407_4 : List Nat := 5039 :: innerTail407_5
def innerCandidates407_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates407_4_checked : candidates edge 4 2819 5039 innerTail407_5 best.bound = innerCandidates407_4 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_4 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_4 best = best := by
  rw [innerTail407_4, scan, if_neg (by decide), innerCandidates407_4_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_5
    (if 3 ≤ innerCandidates407_4.length then search edge 3 [2819, 5039] innerCandidates407_4 7858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_5

def innerTail407_3 : List Nat := 4787 :: innerTail407_4
def innerCandidates407_3 : List Nat := [7187]
set_option maxRecDepth 4096 in
theorem innerCandidates407_3_checked : candidates edge 4 2819 4787 innerTail407_4 best.bound = innerCandidates407_3 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_3 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_3 best = best := by
  rw [innerTail407_3, scan, if_neg (by decide), innerCandidates407_3_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_4
    (if 3 ≤ innerCandidates407_3.length then search edge 3 [2819, 4787] innerCandidates407_3 7606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_4

def innerTail407_2 : List Nat := 4079 :: innerTail407_3
def innerCandidates407_2 : List Nat := [5879, 5927]
set_option maxRecDepth 4096 in
theorem innerCandidates407_2_checked : candidates edge 4 2819 4079 innerTail407_3 best.bound = innerCandidates407_2 := by
  unfold candidates
  have row : edge 4079 = SearchData.row4079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_2 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_2 best = best := by
  rw [innerTail407_2, scan, if_neg (by decide), innerCandidates407_2_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_3
    (if 3 ≤ innerCandidates407_2.length then search edge 3 [2819, 4079] innerCandidates407_2 6898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_3

def innerTail407_1 : List Nat := 3617 :: innerTail407_2
def innerCandidates407_1 : List Nat := [7607]
set_option maxRecDepth 4096 in
theorem innerCandidates407_1_checked : candidates edge 4 2819 3617 innerTail407_2 best.bound = innerCandidates407_1 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_1 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_1 best = best := by
  rw [innerTail407_1, scan, if_neg (by decide), innerCandidates407_1_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_2
    (if 3 ≤ innerCandidates407_1.length then search edge 3 [2819, 3617] innerCandidates407_1 6436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_2

def innerTail407_0 : List Nat := 2897 :: innerTail407_1
def innerCandidates407_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates407_0_checked : candidates edge 4 2819 2897 innerTail407_1 best.bound = innerCandidates407_0 := by
  unfold candidates
  have row : edge 2897 = SearchData.row2897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan407_0 : scan edge (search edge 3) 4 [2819] 2819 innerTail407_0 best = best := by
  rw [innerTail407_0, scan, if_neg (by decide), innerCandidates407_0_checked]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_1
    (if 3 ≤ innerCandidates407_0.length then search edge 3 [2819, 2897] innerCandidates407_0 5716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan407_1

theorem rootBranch407 : search edge 4 [2819] rootCandidates407 2819 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2819] 2819 innerTail407_0 best = best
  exact innerScan407_0

def rootCandidates406 : List Nat := [3079, 3631, 3769, 4051, 4423, 4507, 4549, 4813, 4969, 5107, 5701, 5851, 6199, 6361, 6733, 7549, 7717, 7951, 7993, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates406_checked : candidates edge 5 0 2803 tail407 best.bound = rootCandidates406 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext q; rfl
  rw [row]
  decide

def innerTail406_20 : List Nat := []
theorem innerScan406_20 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_20 best = best := rfl

def innerTail406_19 : List Nat := 8377 :: innerTail406_20
def innerCandidates406_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates406_19_checked : candidates edge 4 2803 8377 innerTail406_20 best.bound = innerCandidates406_19 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_19 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_19 best = best := by
  rw [innerTail406_19, scan, if_neg (by decide), innerCandidates406_19_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_20
    (if 3 ≤ innerCandidates406_19.length then search edge 3 [2803, 8377] innerCandidates406_19 11180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_20

def innerTail406_18 : List Nat := 7993 :: innerTail406_19
def innerCandidates406_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates406_18_checked : candidates edge 4 2803 7993 innerTail406_19 best.bound = innerCandidates406_18 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_18 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_18 best = best := by
  rw [innerTail406_18, scan, if_neg (by decide), innerCandidates406_18_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_19
    (if 3 ≤ innerCandidates406_18.length then search edge 3 [2803, 7993] innerCandidates406_18 10796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_19

def innerTail406_17 : List Nat := 7951 :: innerTail406_18
def innerCandidates406_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates406_17_checked : candidates edge 4 2803 7951 innerTail406_18 best.bound = innerCandidates406_17 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_17 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_17 best = best := by
  rw [innerTail406_17, scan, if_neg (by decide), innerCandidates406_17_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_18
    (if 3 ≤ innerCandidates406_17.length then search edge 3 [2803, 7951] innerCandidates406_17 10754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_18

def innerTail406_16 : List Nat := 7717 :: innerTail406_17
def innerCandidates406_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates406_16_checked : candidates edge 4 2803 7717 innerTail406_17 best.bound = innerCandidates406_16 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_16 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_16 best = best := by
  rw [innerTail406_16, scan, if_neg (by decide), innerCandidates406_16_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_17
    (if 3 ≤ innerCandidates406_16.length then search edge 3 [2803, 7717] innerCandidates406_16 10520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_17

def innerTail406_15 : List Nat := 7549 :: innerTail406_16
def innerCandidates406_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates406_15_checked : candidates edge 4 2803 7549 innerTail406_16 best.bound = innerCandidates406_15 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_15 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_15 best = best := by
  rw [innerTail406_15, scan, if_neg (by decide), innerCandidates406_15_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_16
    (if 3 ≤ innerCandidates406_15.length then search edge 3 [2803, 7549] innerCandidates406_15 10352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_16

def innerTail406_14 : List Nat := 6733 :: innerTail406_15
def innerCandidates406_14 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates406_14_checked : candidates edge 4 2803 6733 innerTail406_15 best.bound = innerCandidates406_14 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_14 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_14 best = best := by
  rw [innerTail406_14, scan, if_neg (by decide), innerCandidates406_14_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_15
    (if 3 ≤ innerCandidates406_14.length then search edge 3 [2803, 6733] innerCandidates406_14 9536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_15

def innerTail406_13 : List Nat := 6361 :: innerTail406_14
def innerCandidates406_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates406_13_checked : candidates edge 4 2803 6361 innerTail406_14 best.bound = innerCandidates406_13 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_13 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_13 best = best := by
  rw [innerTail406_13, scan, if_neg (by decide), innerCandidates406_13_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_14
    (if 3 ≤ innerCandidates406_13.length then search edge 3 [2803, 6361] innerCandidates406_13 9164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_14

def innerTail406_12 : List Nat := 6199 :: innerTail406_13
def innerCandidates406_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates406_12_checked : candidates edge 4 2803 6199 innerTail406_13 best.bound = innerCandidates406_12 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_12 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_12 best = best := by
  rw [innerTail406_12, scan, if_neg (by decide), innerCandidates406_12_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_13
    (if 3 ≤ innerCandidates406_12.length then search edge 3 [2803, 6199] innerCandidates406_12 9002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_13

def innerTail406_11 : List Nat := 5851 :: innerTail406_12
def innerCandidates406_11 : List Nat := [8377]
set_option maxRecDepth 4096 in
theorem innerCandidates406_11_checked : candidates edge 4 2803 5851 innerTail406_12 best.bound = innerCandidates406_11 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_11 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_11 best = best := by
  rw [innerTail406_11, scan, if_neg (by decide), innerCandidates406_11_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_12
    (if 3 ≤ innerCandidates406_11.length then search edge 3 [2803, 5851] innerCandidates406_11 8654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_12

def innerTail406_10 : List Nat := 5701 :: innerTail406_11
def innerCandidates406_10 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates406_10_checked : candidates edge 4 2803 5701 innerTail406_11 best.bound = innerCandidates406_10 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_10 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_10 best = best := by
  rw [innerTail406_10, scan, if_neg (by decide), innerCandidates406_10_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_11
    (if 3 ≤ innerCandidates406_10.length then search edge 3 [2803, 5701] innerCandidates406_10 8504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_11

def innerTail406_9 : List Nat := 5107 :: innerTail406_10
def innerCandidates406_9 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates406_9_checked : candidates edge 4 2803 5107 innerTail406_10 best.bound = innerCandidates406_9 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_9 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_9 best = best := by
  rw [innerTail406_9, scan, if_neg (by decide), innerCandidates406_9_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_10
    (if 3 ≤ innerCandidates406_9.length then search edge 3 [2803, 5107] innerCandidates406_9 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_10

def innerTail406_8 : List Nat := 4969 :: innerTail406_9
def innerCandidates406_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates406_8_checked : candidates edge 4 2803 4969 innerTail406_9 best.bound = innerCandidates406_8 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_8 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_8 best = best := by
  rw [innerTail406_8, scan, if_neg (by decide), innerCandidates406_8_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_9
    (if 3 ≤ innerCandidates406_8.length then search edge 3 [2803, 4969] innerCandidates406_8 7772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_9

def innerTail406_7 : List Nat := 4813 :: innerTail406_8
def innerCandidates406_7 : List Nat := [5701]
set_option maxRecDepth 4096 in
theorem innerCandidates406_7_checked : candidates edge 4 2803 4813 innerTail406_8 best.bound = innerCandidates406_7 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_7 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_7 best = best := by
  rw [innerTail406_7, scan, if_neg (by decide), innerCandidates406_7_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_8
    (if 3 ≤ innerCandidates406_7.length then search edge 3 [2803, 4813] innerCandidates406_7 7616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_8

def innerTail406_6 : List Nat := 4549 :: innerTail406_7
def innerCandidates406_6 : List Nat := [5851, 6361, 7717]
set_option maxRecDepth 4096 in
theorem innerCandidates406_6_checked : candidates edge 4 2803 4549 innerTail406_7 best.bound = innerCandidates406_6 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild406_6 : search edge 3 [2803, 4549] innerCandidates406_6 7352 best = best := by decide

theorem innerScan406_6 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_6 best = best := by
  rw [innerTail406_6, scan, if_neg (by decide), innerCandidates406_6_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_7
    (if 3 ≤ innerCandidates406_6.length then search edge 3 [2803, 4549] innerCandidates406_6 7352 best else best) = best
  rw [if_pos (by decide), innerChild406_6]
  exact innerScan406_7

def innerTail406_5 : List Nat := 4507 :: innerTail406_6
def innerCandidates406_5 : List Nat := [7993, 8377]
set_option maxRecDepth 4096 in
theorem innerCandidates406_5_checked : candidates edge 4 2803 4507 innerTail406_6 best.bound = innerCandidates406_5 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_5 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_5 best = best := by
  rw [innerTail406_5, scan, if_neg (by decide), innerCandidates406_5_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_6
    (if 3 ≤ innerCandidates406_5.length then search edge 3 [2803, 4507] innerCandidates406_5 7310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_6

def innerTail406_4 : List Nat := 4423 :: innerTail406_5
def innerCandidates406_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates406_4_checked : candidates edge 4 2803 4423 innerTail406_5 best.bound = innerCandidates406_4 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_4 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_4 best = best := by
  rw [innerTail406_4, scan, if_neg (by decide), innerCandidates406_4_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_5
    (if 3 ≤ innerCandidates406_4.length then search edge 3 [2803, 4423] innerCandidates406_4 7226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_5

def innerTail406_3 : List Nat := 4051 :: innerTail406_4
def innerCandidates406_3 : List Nat := [5701, 7549, 7951]
set_option maxRecDepth 4096 in
theorem innerCandidates406_3_checked : candidates edge 4 2803 4051 innerTail406_4 best.bound = innerCandidates406_3 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild406_3 : search edge 3 [2803, 4051] innerCandidates406_3 6854 best = best := by decide

theorem innerScan406_3 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_3 best = best := by
  rw [innerTail406_3, scan, if_neg (by decide), innerCandidates406_3_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_4
    (if 3 ≤ innerCandidates406_3.length then search edge 3 [2803, 4051] innerCandidates406_3 6854 best else best) = best
  rw [if_pos (by decide), innerChild406_3]
  exact innerScan406_4

def innerTail406_2 : List Nat := 3769 :: innerTail406_3
def innerCandidates406_2 : List Nat := [4051]
set_option maxRecDepth 4096 in
theorem innerCandidates406_2_checked : candidates edge 4 2803 3769 innerTail406_3 best.bound = innerCandidates406_2 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_2 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_2 best = best := by
  rw [innerTail406_2, scan, if_neg (by decide), innerCandidates406_2_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_3
    (if 3 ≤ innerCandidates406_2.length then search edge 3 [2803, 3769] innerCandidates406_2 6572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_3

def innerTail406_1 : List Nat := 3631 :: innerTail406_2
def innerCandidates406_1 : List Nat := [8377]
set_option maxRecDepth 4096 in
theorem innerCandidates406_1_checked : candidates edge 4 2803 3631 innerTail406_2 best.bound = innerCandidates406_1 := by
  unfold candidates
  have row : edge 3631 = SearchData.row3631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_1 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_1 best = best := by
  rw [innerTail406_1, scan, if_neg (by decide), innerCandidates406_1_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_2
    (if 3 ≤ innerCandidates406_1.length then search edge 3 [2803, 3631] innerCandidates406_1 6434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_2

def innerTail406_0 : List Nat := 3079 :: innerTail406_1
def innerCandidates406_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates406_0_checked : candidates edge 4 2803 3079 innerTail406_1 best.bound = innerCandidates406_0 := by
  unfold candidates
  have row : edge 3079 = SearchData.row3079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan406_0 : scan edge (search edge 3) 4 [2803] 2803 innerTail406_0 best = best := by
  rw [innerTail406_0, scan, if_neg (by decide), innerCandidates406_0_checked]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_1
    (if 3 ≤ innerCandidates406_0.length then search edge 3 [2803, 3079] innerCandidates406_0 5882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan406_1

theorem rootBranch406 : search edge 4 [2803] rootCandidates406 2803 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2803] 2803 innerTail406_0 best = best
  exact innerScan406_0

end ProjectEulerSolutions.PrimePairUpper.Search
