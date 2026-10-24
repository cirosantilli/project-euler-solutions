import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates412 : List Nat := [3049, 3163, 3229, 3469, 3499, 3709, 3907, 4519, 4549, 4561, 4783, 4957, 5641, 5881, 6091, 7573, 7699, 7933]
set_option maxRecDepth 4096 in
theorem rootCandidates412_checked : candidates edge 5 0 2857 tail413 best.bound = rootCandidates412 := by
  unfold candidates
  have row : edge 2857 = SearchData.row2857.contains := by funext q; rfl
  rw [row]
  decide

def innerTail412_18 : List Nat := []
theorem innerScan412_18 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_18 best = best := rfl

def innerTail412_17 : List Nat := 7933 :: innerTail412_18
def innerCandidates412_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_17_checked : candidates edge 4 2857 7933 innerTail412_18 best.bound = innerCandidates412_17 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_17 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_17 best = best := by
  rw [innerTail412_17, scan, if_neg (by decide), innerCandidates412_17_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_18
    (if 3 ≤ innerCandidates412_17.length then search edge 3 [2857, 7933] innerCandidates412_17 10790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_18

def innerTail412_16 : List Nat := 7699 :: innerTail412_17
def innerCandidates412_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_16_checked : candidates edge 4 2857 7699 innerTail412_17 best.bound = innerCandidates412_16 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_16 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_16 best = best := by
  rw [innerTail412_16, scan, if_neg (by decide), innerCandidates412_16_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_17
    (if 3 ≤ innerCandidates412_16.length then search edge 3 [2857, 7699] innerCandidates412_16 10556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_17

def innerTail412_15 : List Nat := 7573 :: innerTail412_16
def innerCandidates412_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_15_checked : candidates edge 4 2857 7573 innerTail412_16 best.bound = innerCandidates412_15 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_15 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_15 best = best := by
  rw [innerTail412_15, scan, if_neg (by decide), innerCandidates412_15_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_16
    (if 3 ≤ innerCandidates412_15.length then search edge 3 [2857, 7573] innerCandidates412_15 10430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_16

def innerTail412_14 : List Nat := 6091 :: innerTail412_15
def innerCandidates412_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_14_checked : candidates edge 4 2857 6091 innerTail412_15 best.bound = innerCandidates412_14 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_14 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_14 best = best := by
  rw [innerTail412_14, scan, if_neg (by decide), innerCandidates412_14_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_15
    (if 3 ≤ innerCandidates412_14.length then search edge 3 [2857, 6091] innerCandidates412_14 8948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_15

def innerTail412_13 : List Nat := 5881 :: innerTail412_14
def innerCandidates412_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_13_checked : candidates edge 4 2857 5881 innerTail412_14 best.bound = innerCandidates412_13 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_13 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_13 best = best := by
  rw [innerTail412_13, scan, if_neg (by decide), innerCandidates412_13_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_14
    (if 3 ≤ innerCandidates412_13.length then search edge 3 [2857, 5881] innerCandidates412_13 8738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_14

def innerTail412_12 : List Nat := 5641 :: innerTail412_13
def innerCandidates412_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_12_checked : candidates edge 4 2857 5641 innerTail412_13 best.bound = innerCandidates412_12 := by
  unfold candidates
  have row : edge 5641 = SearchData.row5641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_12 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_12 best = best := by
  rw [innerTail412_12, scan, if_neg (by decide), innerCandidates412_12_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_13
    (if 3 ≤ innerCandidates412_12.length then search edge 3 [2857, 5641] innerCandidates412_12 8498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_13

def innerTail412_11 : List Nat := 4957 :: innerTail412_12
def innerCandidates412_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_11_checked : candidates edge 4 2857 4957 innerTail412_12 best.bound = innerCandidates412_11 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_11 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_11 best = best := by
  rw [innerTail412_11, scan, if_neg (by decide), innerCandidates412_11_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_12
    (if 3 ≤ innerCandidates412_11.length then search edge 3 [2857, 4957] innerCandidates412_11 7814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_12

def innerTail412_10 : List Nat := 4783 :: innerTail412_11
def innerCandidates412_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_10_checked : candidates edge 4 2857 4783 innerTail412_11 best.bound = innerCandidates412_10 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_10 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_10 best = best := by
  rw [innerTail412_10, scan, if_neg (by decide), innerCandidates412_10_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_11
    (if 3 ≤ innerCandidates412_10.length then search edge 3 [2857, 4783] innerCandidates412_10 7640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_11

def innerTail412_9 : List Nat := 4561 :: innerTail412_10
def innerCandidates412_9 : List Nat := [4783]
set_option maxRecDepth 4096 in
theorem innerCandidates412_9_checked : candidates edge 4 2857 4561 innerTail412_10 best.bound = innerCandidates412_9 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_9 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_9 best = best := by
  rw [innerTail412_9, scan, if_neg (by decide), innerCandidates412_9_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_10
    (if 3 ≤ innerCandidates412_9.length then search edge 3 [2857, 4561] innerCandidates412_9 7418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_10

def innerTail412_8 : List Nat := 4549 :: innerTail412_9
def innerCandidates412_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_8_checked : candidates edge 4 2857 4549 innerTail412_9 best.bound = innerCandidates412_8 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_8 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_8 best = best := by
  rw [innerTail412_8, scan, if_neg (by decide), innerCandidates412_8_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_9
    (if 3 ≤ innerCandidates412_8.length then search edge 3 [2857, 4549] innerCandidates412_8 7406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_9

def innerTail412_7 : List Nat := 4519 :: innerTail412_8
def innerCandidates412_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_7_checked : candidates edge 4 2857 4519 innerTail412_8 best.bound = innerCandidates412_7 := by
  unfold candidates
  have row : edge 4519 = SearchData.row4519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_7 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_7 best = best := by
  rw [innerTail412_7, scan, if_neg (by decide), innerCandidates412_7_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_8
    (if 3 ≤ innerCandidates412_7.length then search edge 3 [2857, 4519] innerCandidates412_7 7376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_8

def innerTail412_6 : List Nat := 3907 :: innerTail412_7
def innerCandidates412_6 : List Nat := [7573]
set_option maxRecDepth 4096 in
theorem innerCandidates412_6_checked : candidates edge 4 2857 3907 innerTail412_7 best.bound = innerCandidates412_6 := by
  unfold candidates
  have row : edge 3907 = SearchData.row3907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_6 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_6 best = best := by
  rw [innerTail412_6, scan, if_neg (by decide), innerCandidates412_6_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_7
    (if 3 ≤ innerCandidates412_6.length then search edge 3 [2857, 3907] innerCandidates412_6 6764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_7

def innerTail412_5 : List Nat := 3709 :: innerTail412_6
def innerCandidates412_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_5_checked : candidates edge 4 2857 3709 innerTail412_6 best.bound = innerCandidates412_5 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_5 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_5 best = best := by
  rw [innerTail412_5, scan, if_neg (by decide), innerCandidates412_5_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_6
    (if 3 ≤ innerCandidates412_5.length then search edge 3 [2857, 3709] innerCandidates412_5 6566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_6

def innerTail412_4 : List Nat := 3499 :: innerTail412_5
def innerCandidates412_4 : List Nat := [5881]
set_option maxRecDepth 4096 in
theorem innerCandidates412_4_checked : candidates edge 4 2857 3499 innerTail412_5 best.bound = innerCandidates412_4 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_4 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_4 best = best := by
  rw [innerTail412_4, scan, if_neg (by decide), innerCandidates412_4_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_5
    (if 3 ≤ innerCandidates412_4.length then search edge 3 [2857, 3499] innerCandidates412_4 6356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_5

def innerTail412_3 : List Nat := 3469 :: innerTail412_4
def innerCandidates412_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_3_checked : candidates edge 4 2857 3469 innerTail412_4 best.bound = innerCandidates412_3 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_3 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_3 best = best := by
  rw [innerTail412_3, scan, if_neg (by decide), innerCandidates412_3_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_4
    (if 3 ≤ innerCandidates412_3.length then search edge 3 [2857, 3469] innerCandidates412_3 6326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_4

def innerTail412_2 : List Nat := 3229 :: innerTail412_3
def innerCandidates412_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_2_checked : candidates edge 4 2857 3229 innerTail412_3 best.bound = innerCandidates412_2 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_2 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_2 best = best := by
  rw [innerTail412_2, scan, if_neg (by decide), innerCandidates412_2_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_3
    (if 3 ≤ innerCandidates412_2.length then search edge 3 [2857, 3229] innerCandidates412_2 6086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_3

def innerTail412_1 : List Nat := 3163 :: innerTail412_2
def innerCandidates412_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_1_checked : candidates edge 4 2857 3163 innerTail412_2 best.bound = innerCandidates412_1 := by
  unfold candidates
  have row : edge 3163 = SearchData.row3163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_1 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_1 best = best := by
  rw [innerTail412_1, scan, if_neg (by decide), innerCandidates412_1_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_2
    (if 3 ≤ innerCandidates412_1.length then search edge 3 [2857, 3163] innerCandidates412_1 6020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_2

def innerTail412_0 : List Nat := 3049 :: innerTail412_1
def innerCandidates412_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates412_0_checked : candidates edge 4 2857 3049 innerTail412_1 best.bound = innerCandidates412_0 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan412_0 : scan edge (search edge 3) 4 [2857] 2857 innerTail412_0 best = best := by
  rw [innerTail412_0, scan, if_neg (by decide), innerCandidates412_0_checked]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_1
    (if 3 ≤ innerCandidates412_0.length then search edge 3 [2857, 3049] innerCandidates412_0 5906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan412_1

theorem rootBranch412 : search edge 4 [2857] rootCandidates412 2857 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2857] 2857 innerTail412_0 best = best
  exact innerScan412_0

def rootCandidates411 : List Nat := [3187, 3313, 3733, 4951, 5107, 5119, 5737, 6091, 6277, 6949, 7219]
set_option maxRecDepth 4096 in
theorem rootCandidates411_checked : candidates edge 5 0 2851 tail412 best.bound = rootCandidates411 := by
  unfold candidates
  have row : edge 2851 = SearchData.row2851.contains := by funext q; rfl
  rw [row]
  decide

def innerTail411_11 : List Nat := []
theorem innerScan411_11 : scan edge (search edge 3) 4 [2851] 2851 innerTail411_11 best = best := rfl

def innerTail411_10 : List Nat := 7219 :: innerTail411_11
def innerCandidates411_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates411_10_checked : candidates edge 4 2851 7219 innerTail411_11 best.bound = innerCandidates411_10 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan411_10 : scan edge (search edge 3) 4 [2851] 2851 innerTail411_10 best = best := by
  rw [innerTail411_10, scan, if_neg (by decide), innerCandidates411_10_checked]
  change scan edge (search edge 3) 4 [2851] 2851 innerTail411_11
    (if 3 ≤ innerCandidates411_10.length then search edge 3 [2851, 7219] innerCandidates411_10 10070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan411_11

def innerTail411_9 : List Nat := 6949 :: innerTail411_10
def innerCandidates411_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates411_9_checked : candidates edge 4 2851 6949 innerTail411_10 best.bound = innerCandidates411_9 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan411_9 : scan edge (search edge 3) 4 [2851] 2851 innerTail411_9 best = best := by
  rw [innerTail411_9, scan, if_neg (by decide), innerCandidates411_9_checked]
  change scan edge (search edge 3) 4 [2851] 2851 innerTail411_10
    (if 3 ≤ innerCandidates411_9.length then search edge 3 [2851, 6949] innerCandidates411_9 9800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan411_10

def innerTail411_8 : List Nat := 6277 :: innerTail411_9
def innerCandidates411_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates411_8_checked : candidates edge 4 2851 6277 innerTail411_9 best.bound = innerCandidates411_8 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan411_8 : scan edge (search edge 3) 4 [2851] 2851 innerTail411_8 best = best := by
  rw [innerTail411_8, scan, if_neg (by decide), innerCandidates411_8_checked]
  change scan edge (search edge 3) 4 [2851] 2851 innerTail411_9
    (if 3 ≤ innerCandidates411_8.length then search edge 3 [2851, 6277] innerCandidates411_8 9128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan411_9

def innerTail411_7 : List Nat := 6091 :: innerTail411_8
def innerCandidates411_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates411_7_checked : candidates edge 4 2851 6091 innerTail411_8 best.bound = innerCandidates411_7 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan411_7 : scan edge (search edge 3) 4 [2851] 2851 innerTail411_7 best = best := by
  rw [innerTail411_7, scan, if_neg (by decide), innerCandidates411_7_checked]
  change scan edge (search edge 3) 4 [2851] 2851 innerTail411_8
    (if 3 ≤ innerCandidates411_7.length then search edge 3 [2851, 6091] innerCandidates411_7 8942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan411_8

def innerTail411_6 : List Nat := 5737 :: innerTail411_7
def innerCandidates411_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates411_6_checked : candidates edge 4 2851 5737 innerTail411_7 best.bound = innerCandidates411_6 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan411_6 : scan edge (search edge 3) 4 [2851] 2851 innerTail411_6 best = best := by
  rw [innerTail411_6, scan, if_neg (by decide), innerCandidates411_6_checked]
  change scan edge (search edge 3) 4 [2851] 2851 innerTail411_7
    (if 3 ≤ innerCandidates411_6.length then search edge 3 [2851, 5737] innerCandidates411_6 8588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan411_7

def innerTail411_5 : List Nat := 5119 :: innerTail411_6
def innerCandidates411_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates411_5_checked : candidates edge 4 2851 5119 innerTail411_6 best.bound = innerCandidates411_5 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan411_5 : scan edge (search edge 3) 4 [2851] 2851 innerTail411_5 best = best := by
  rw [innerTail411_5, scan, if_neg (by decide), innerCandidates411_5_checked]
  change scan edge (search edge 3) 4 [2851] 2851 innerTail411_6
    (if 3 ≤ innerCandidates411_5.length then search edge 3 [2851, 5119] innerCandidates411_5 7970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan411_6

def innerTail411_4 : List Nat := 5107 :: innerTail411_5
def innerCandidates411_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates411_4_checked : candidates edge 4 2851 5107 innerTail411_5 best.bound = innerCandidates411_4 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan411_4 : scan edge (search edge 3) 4 [2851] 2851 innerTail411_4 best = best := by
  rw [innerTail411_4, scan, if_neg (by decide), innerCandidates411_4_checked]
  change scan edge (search edge 3) 4 [2851] 2851 innerTail411_5
    (if 3 ≤ innerCandidates411_4.length then search edge 3 [2851, 5107] innerCandidates411_4 7958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan411_5

def innerTail411_3 : List Nat := 4951 :: innerTail411_4
def innerCandidates411_3 : List Nat := [5107]
set_option maxRecDepth 4096 in
theorem innerCandidates411_3_checked : candidates edge 4 2851 4951 innerTail411_4 best.bound = innerCandidates411_3 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan411_3 : scan edge (search edge 3) 4 [2851] 2851 innerTail411_3 best = best := by
  rw [innerTail411_3, scan, if_neg (by decide), innerCandidates411_3_checked]
  change scan edge (search edge 3) 4 [2851] 2851 innerTail411_4
    (if 3 ≤ innerCandidates411_3.length then search edge 3 [2851, 4951] innerCandidates411_3 7802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan411_4

def innerTail411_2 : List Nat := 3733 :: innerTail411_3
def innerCandidates411_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates411_2_checked : candidates edge 4 2851 3733 innerTail411_3 best.bound = innerCandidates411_2 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan411_2 : scan edge (search edge 3) 4 [2851] 2851 innerTail411_2 best = best := by
  rw [innerTail411_2, scan, if_neg (by decide), innerCandidates411_2_checked]
  change scan edge (search edge 3) 4 [2851] 2851 innerTail411_3
    (if 3 ≤ innerCandidates411_2.length then search edge 3 [2851, 3733] innerCandidates411_2 6584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan411_3

def innerTail411_1 : List Nat := 3313 :: innerTail411_2
def innerCandidates411_1 : List Nat := [3733, 4951]
set_option maxRecDepth 4096 in
theorem innerCandidates411_1_checked : candidates edge 4 2851 3313 innerTail411_2 best.bound = innerCandidates411_1 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan411_1 : scan edge (search edge 3) 4 [2851] 2851 innerTail411_1 best = best := by
  rw [innerTail411_1, scan, if_neg (by decide), innerCandidates411_1_checked]
  change scan edge (search edge 3) 4 [2851] 2851 innerTail411_2
    (if 3 ≤ innerCandidates411_1.length then search edge 3 [2851, 3313] innerCandidates411_1 6164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan411_2

def innerTail411_0 : List Nat := 3187 :: innerTail411_1
def innerCandidates411_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates411_0_checked : candidates edge 4 2851 3187 innerTail411_1 best.bound = innerCandidates411_0 := by
  unfold candidates
  have row : edge 3187 = SearchData.row3187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan411_0 : scan edge (search edge 3) 4 [2851] 2851 innerTail411_0 best = best := by
  rw [innerTail411_0, scan, if_neg (by decide), innerCandidates411_0_checked]
  change scan edge (search edge 3) 4 [2851] 2851 innerTail411_1
    (if 3 ≤ innerCandidates411_0.length then search edge 3 [2851, 3187] innerCandidates411_0 6038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan411_1

theorem rootBranch411 : search edge 4 [2851] rootCandidates411 2851 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2851] 2851 innerTail411_0 best = best
  exact innerScan411_0

def rootCandidates410 : List Nat := [3089, 3203, 3389, 3947, 3989, 4253, 4817, 5657, 5783, 5807, 6047, 6899, 7841, 8171, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates410_checked : candidates edge 5 0 2843 tail411 best.bound = rootCandidates410 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext q; rfl
  rw [row]
  decide

def innerTail410_15 : List Nat := []
theorem innerScan410_15 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_15 best = best := rfl

def innerTail410_14 : List Nat := 8273 :: innerTail410_15
def innerCandidates410_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates410_14_checked : candidates edge 4 2843 8273 innerTail410_15 best.bound = innerCandidates410_14 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_14 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_14 best = best := by
  rw [innerTail410_14, scan, if_neg (by decide), innerCandidates410_14_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_15
    (if 3 ≤ innerCandidates410_14.length then search edge 3 [2843, 8273] innerCandidates410_14 11116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_15

def innerTail410_13 : List Nat := 8171 :: innerTail410_14
def innerCandidates410_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates410_13_checked : candidates edge 4 2843 8171 innerTail410_14 best.bound = innerCandidates410_13 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_13 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_13 best = best := by
  rw [innerTail410_13, scan, if_neg (by decide), innerCandidates410_13_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_14
    (if 3 ≤ innerCandidates410_13.length then search edge 3 [2843, 8171] innerCandidates410_13 11014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_14

def innerTail410_12 : List Nat := 7841 :: innerTail410_13
def innerCandidates410_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates410_12_checked : candidates edge 4 2843 7841 innerTail410_13 best.bound = innerCandidates410_12 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_12 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_12 best = best := by
  rw [innerTail410_12, scan, if_neg (by decide), innerCandidates410_12_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_13
    (if 3 ≤ innerCandidates410_12.length then search edge 3 [2843, 7841] innerCandidates410_12 10684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_13

def innerTail410_11 : List Nat := 6899 :: innerTail410_12
def innerCandidates410_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates410_11_checked : candidates edge 4 2843 6899 innerTail410_12 best.bound = innerCandidates410_11 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_11 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_11 best = best := by
  rw [innerTail410_11, scan, if_neg (by decide), innerCandidates410_11_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_12
    (if 3 ≤ innerCandidates410_11.length then search edge 3 [2843, 6899] innerCandidates410_11 9742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_12

def innerTail410_10 : List Nat := 6047 :: innerTail410_11
def innerCandidates410_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates410_10_checked : candidates edge 4 2843 6047 innerTail410_11 best.bound = innerCandidates410_10 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_10 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_10 best = best := by
  rw [innerTail410_10, scan, if_neg (by decide), innerCandidates410_10_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_11
    (if 3 ≤ innerCandidates410_10.length then search edge 3 [2843, 6047] innerCandidates410_10 8890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_11

def innerTail410_9 : List Nat := 5807 :: innerTail410_10
def innerCandidates410_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates410_9_checked : candidates edge 4 2843 5807 innerTail410_10 best.bound = innerCandidates410_9 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_9 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_9 best = best := by
  rw [innerTail410_9, scan, if_neg (by decide), innerCandidates410_9_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_10
    (if 3 ≤ innerCandidates410_9.length then search edge 3 [2843, 5807] innerCandidates410_9 8650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_10

def innerTail410_8 : List Nat := 5783 :: innerTail410_9
def innerCandidates410_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates410_8_checked : candidates edge 4 2843 5783 innerTail410_9 best.bound = innerCandidates410_8 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_8 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_8 best = best := by
  rw [innerTail410_8, scan, if_neg (by decide), innerCandidates410_8_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_9
    (if 3 ≤ innerCandidates410_8.length then search edge 3 [2843, 5783] innerCandidates410_8 8626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_9

def innerTail410_7 : List Nat := 5657 :: innerTail410_8
def innerCandidates410_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates410_7_checked : candidates edge 4 2843 5657 innerTail410_8 best.bound = innerCandidates410_7 := by
  unfold candidates
  have row : edge 5657 = SearchData.row5657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_7 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_7 best = best := by
  rw [innerTail410_7, scan, if_neg (by decide), innerCandidates410_7_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_8
    (if 3 ≤ innerCandidates410_7.length then search edge 3 [2843, 5657] innerCandidates410_7 8500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_8

def innerTail410_6 : List Nat := 4817 :: innerTail410_7
def innerCandidates410_6 : List Nat := [5783]
set_option maxRecDepth 4096 in
theorem innerCandidates410_6_checked : candidates edge 4 2843 4817 innerTail410_7 best.bound = innerCandidates410_6 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_6 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_6 best = best := by
  rw [innerTail410_6, scan, if_neg (by decide), innerCandidates410_6_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_7
    (if 3 ≤ innerCandidates410_6.length then search edge 3 [2843, 4817] innerCandidates410_6 7660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_7

def innerTail410_5 : List Nat := 4253 :: innerTail410_6
def innerCandidates410_5 : List Nat := [5657]
set_option maxRecDepth 4096 in
theorem innerCandidates410_5_checked : candidates edge 4 2843 4253 innerTail410_6 best.bound = innerCandidates410_5 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_5 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_5 best = best := by
  rw [innerTail410_5, scan, if_neg (by decide), innerCandidates410_5_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_6
    (if 3 ≤ innerCandidates410_5.length then search edge 3 [2843, 4253] innerCandidates410_5 7096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_6

def innerTail410_4 : List Nat := 3989 :: innerTail410_5
def innerCandidates410_4 : List Nat := [5783]
set_option maxRecDepth 4096 in
theorem innerCandidates410_4_checked : candidates edge 4 2843 3989 innerTail410_5 best.bound = innerCandidates410_4 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_4 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_4 best = best := by
  rw [innerTail410_4, scan, if_neg (by decide), innerCandidates410_4_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_5
    (if 3 ≤ innerCandidates410_4.length then search edge 3 [2843, 3989] innerCandidates410_4 6832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_5

def innerTail410_3 : List Nat := 3947 :: innerTail410_4
def innerCandidates410_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates410_3_checked : candidates edge 4 2843 3947 innerTail410_4 best.bound = innerCandidates410_3 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_3 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_3 best = best := by
  rw [innerTail410_3, scan, if_neg (by decide), innerCandidates410_3_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_4
    (if 3 ≤ innerCandidates410_3.length then search edge 3 [2843, 3947] innerCandidates410_3 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_4

def innerTail410_2 : List Nat := 3389 :: innerTail410_3
def innerCandidates410_2 : List Nat := [7841]
set_option maxRecDepth 4096 in
theorem innerCandidates410_2_checked : candidates edge 4 2843 3389 innerTail410_3 best.bound = innerCandidates410_2 := by
  unfold candidates
  have row : edge 3389 = SearchData.row3389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_2 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_2 best = best := by
  rw [innerTail410_2, scan, if_neg (by decide), innerCandidates410_2_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_3
    (if 3 ≤ innerCandidates410_2.length then search edge 3 [2843, 3389] innerCandidates410_2 6232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_3

def innerTail410_1 : List Nat := 3203 :: innerTail410_2
def innerCandidates410_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates410_1_checked : candidates edge 4 2843 3203 innerTail410_2 best.bound = innerCandidates410_1 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_1 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_1 best = best := by
  rw [innerTail410_1, scan, if_neg (by decide), innerCandidates410_1_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_2
    (if 3 ≤ innerCandidates410_1.length then search edge 3 [2843, 3203] innerCandidates410_1 6046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_2

def innerTail410_0 : List Nat := 3089 :: innerTail410_1
def innerCandidates410_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates410_0_checked : candidates edge 4 2843 3089 innerTail410_1 best.bound = innerCandidates410_0 := by
  unfold candidates
  have row : edge 3089 = SearchData.row3089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan410_0 : scan edge (search edge 3) 4 [2843] 2843 innerTail410_0 best = best := by
  rw [innerTail410_0, scan, if_neg (by decide), innerCandidates410_0_checked]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_1
    (if 3 ≤ innerCandidates410_0.length then search edge 3 [2843, 3089] innerCandidates410_0 5932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan410_1

theorem rootBranch410 : search edge 4 [2843] rootCandidates410 2843 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2843] 2843 innerTail410_0 best = best
  exact innerScan410_0

def rootCandidates409 : List Nat := [3089, 3299, 3407, 3617, 3767, 4817, 5003, 5153, 5297, 6449, 7001, 7691, 7757, 7793]
set_option maxRecDepth 4096 in
theorem rootCandidates409_checked : candidates edge 5 0 2837 tail410 best.bound = rootCandidates409 := by
  unfold candidates
  have row : edge 2837 = SearchData.row2837.contains := by funext q; rfl
  rw [row]
  decide

def innerTail409_14 : List Nat := []
theorem innerScan409_14 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_14 best = best := rfl

def innerTail409_13 : List Nat := 7793 :: innerTail409_14
def innerCandidates409_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates409_13_checked : candidates edge 4 2837 7793 innerTail409_14 best.bound = innerCandidates409_13 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_13 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_13 best = best := by
  rw [innerTail409_13, scan, if_neg (by decide), innerCandidates409_13_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_14
    (if 3 ≤ innerCandidates409_13.length then search edge 3 [2837, 7793] innerCandidates409_13 10630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_14

def innerTail409_12 : List Nat := 7757 :: innerTail409_13
def innerCandidates409_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates409_12_checked : candidates edge 4 2837 7757 innerTail409_13 best.bound = innerCandidates409_12 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_12 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_12 best = best := by
  rw [innerTail409_12, scan, if_neg (by decide), innerCandidates409_12_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_13
    (if 3 ≤ innerCandidates409_12.length then search edge 3 [2837, 7757] innerCandidates409_12 10594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_13

def innerTail409_11 : List Nat := 7691 :: innerTail409_12
def innerCandidates409_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates409_11_checked : candidates edge 4 2837 7691 innerTail409_12 best.bound = innerCandidates409_11 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_11 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_11 best = best := by
  rw [innerTail409_11, scan, if_neg (by decide), innerCandidates409_11_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_12
    (if 3 ≤ innerCandidates409_11.length then search edge 3 [2837, 7691] innerCandidates409_11 10528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_12

def innerTail409_10 : List Nat := 7001 :: innerTail409_11
def innerCandidates409_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates409_10_checked : candidates edge 4 2837 7001 innerTail409_11 best.bound = innerCandidates409_10 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_10 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_10 best = best := by
  rw [innerTail409_10, scan, if_neg (by decide), innerCandidates409_10_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_11
    (if 3 ≤ innerCandidates409_10.length then search edge 3 [2837, 7001] innerCandidates409_10 9838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_11

def innerTail409_9 : List Nat := 6449 :: innerTail409_10
def innerCandidates409_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates409_9_checked : candidates edge 4 2837 6449 innerTail409_10 best.bound = innerCandidates409_9 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_9 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_9 best = best := by
  rw [innerTail409_9, scan, if_neg (by decide), innerCandidates409_9_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_10
    (if 3 ≤ innerCandidates409_9.length then search edge 3 [2837, 6449] innerCandidates409_9 9286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_10

def innerTail409_8 : List Nat := 5297 :: innerTail409_9
def innerCandidates409_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates409_8_checked : candidates edge 4 2837 5297 innerTail409_9 best.bound = innerCandidates409_8 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_8 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_8 best = best := by
  rw [innerTail409_8, scan, if_neg (by decide), innerCandidates409_8_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_9
    (if 3 ≤ innerCandidates409_8.length then search edge 3 [2837, 5297] innerCandidates409_8 8134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_9

def innerTail409_7 : List Nat := 5153 :: innerTail409_8
def innerCandidates409_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates409_7_checked : candidates edge 4 2837 5153 innerTail409_8 best.bound = innerCandidates409_7 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_7 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_7 best = best := by
  rw [innerTail409_7, scan, if_neg (by decide), innerCandidates409_7_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_8
    (if 3 ≤ innerCandidates409_7.length then search edge 3 [2837, 5153] innerCandidates409_7 7990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_8

def innerTail409_6 : List Nat := 5003 :: innerTail409_7
def innerCandidates409_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates409_6_checked : candidates edge 4 2837 5003 innerTail409_7 best.bound = innerCandidates409_6 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_6 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_6 best = best := by
  rw [innerTail409_6, scan, if_neg (by decide), innerCandidates409_6_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_7
    (if 3 ≤ innerCandidates409_6.length then search edge 3 [2837, 5003] innerCandidates409_6 7840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_7

def innerTail409_5 : List Nat := 4817 :: innerTail409_6
def innerCandidates409_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates409_5_checked : candidates edge 4 2837 4817 innerTail409_6 best.bound = innerCandidates409_5 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_5 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_5 best = best := by
  rw [innerTail409_5, scan, if_neg (by decide), innerCandidates409_5_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_6
    (if 3 ≤ innerCandidates409_5.length then search edge 3 [2837, 4817] innerCandidates409_5 7654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_6

def innerTail409_4 : List Nat := 3767 :: innerTail409_5
def innerCandidates409_4 : List Nat := [7001, 7793]
set_option maxRecDepth 4096 in
theorem innerCandidates409_4_checked : candidates edge 4 2837 3767 innerTail409_5 best.bound = innerCandidates409_4 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_4 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_4 best = best := by
  rw [innerTail409_4, scan, if_neg (by decide), innerCandidates409_4_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_5
    (if 3 ≤ innerCandidates409_4.length then search edge 3 [2837, 3767] innerCandidates409_4 6604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_5

def innerTail409_3 : List Nat := 3617 :: innerTail409_4
def innerCandidates409_3 : List Nat := [5003]
set_option maxRecDepth 4096 in
theorem innerCandidates409_3_checked : candidates edge 4 2837 3617 innerTail409_4 best.bound = innerCandidates409_3 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_3 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_3 best = best := by
  rw [innerTail409_3, scan, if_neg (by decide), innerCandidates409_3_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_4
    (if 3 ≤ innerCandidates409_3.length then search edge 3 [2837, 3617] innerCandidates409_3 6454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_4

def innerTail409_2 : List Nat := 3407 :: innerTail409_3
def innerCandidates409_2 : List Nat := [7691]
set_option maxRecDepth 4096 in
theorem innerCandidates409_2_checked : candidates edge 4 2837 3407 innerTail409_3 best.bound = innerCandidates409_2 := by
  unfold candidates
  have row : edge 3407 = SearchData.row3407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_2 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_2 best = best := by
  rw [innerTail409_2, scan, if_neg (by decide), innerCandidates409_2_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_3
    (if 3 ≤ innerCandidates409_2.length then search edge 3 [2837, 3407] innerCandidates409_2 6244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_3

def innerTail409_1 : List Nat := 3299 :: innerTail409_2
def innerCandidates409_1 : List Nat := [3767, 4817]
set_option maxRecDepth 4096 in
theorem innerCandidates409_1_checked : candidates edge 4 2837 3299 innerTail409_2 best.bound = innerCandidates409_1 := by
  unfold candidates
  have row : edge 3299 = SearchData.row3299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_1 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_1 best = best := by
  rw [innerTail409_1, scan, if_neg (by decide), innerCandidates409_1_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_2
    (if 3 ≤ innerCandidates409_1.length then search edge 3 [2837, 3299] innerCandidates409_1 6136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_2

def innerTail409_0 : List Nat := 3089 :: innerTail409_1
def innerCandidates409_0 : List Nat := [3299]
set_option maxRecDepth 4096 in
theorem innerCandidates409_0_checked : candidates edge 4 2837 3089 innerTail409_1 best.bound = innerCandidates409_0 := by
  unfold candidates
  have row : edge 3089 = SearchData.row3089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan409_0 : scan edge (search edge 3) 4 [2837] 2837 innerTail409_0 best = best := by
  rw [innerTail409_0, scan, if_neg (by decide), innerCandidates409_0_checked]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_1
    (if 3 ≤ innerCandidates409_0.length then search edge 3 [2837, 3089] innerCandidates409_0 5926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan409_1

theorem rootBranch409 : search edge 4 [2837] rootCandidates409 2837 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2837] 2837 innerTail409_0 best = best
  exact innerScan409_0

end ProjectEulerSolutions.PrimePairUpper.Search
