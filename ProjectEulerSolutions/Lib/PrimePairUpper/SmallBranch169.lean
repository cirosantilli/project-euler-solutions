import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates174 : List Nat := [2017, 2221, 2347, 2389, 2617, 2713, 3067, 3271, 3331, 3583, 3709, 3739, 4003, 4057, 4507, 4729, 4813, 4909, 5443, 5839, 6301, 6607, 7867, 8311, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates174_checked : candidates edge 5 0 1051 tail175 best.bound = rootCandidates174 := by
  unfold candidates
  have row : edge 1051 = PrimePairCache.smallRow1051.contains := by funext q; rfl
  rw [row]
  decide

def innerTail174_25 : List Nat := []
theorem innerScan174_25 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_25 best = best := rfl

def innerTail174_24 : List Nat := 8353 :: innerTail174_25
def innerCandidates174_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_24_checked : candidates edge 4 1051 8353 innerTail174_25 best.bound = innerCandidates174_24 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_24 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_24 best = best := by
  rw [innerTail174_24, scan, if_neg (by decide), innerCandidates174_24_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_25
    (if 3 ≤ innerCandidates174_24.length then search edge 3 [1051, 8353] innerCandidates174_24 9404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_25

def innerTail174_23 : List Nat := 8311 :: innerTail174_24
def innerCandidates174_23 : List Nat := [8353]
set_option maxRecDepth 4096 in
theorem innerCandidates174_23_checked : candidates edge 4 1051 8311 innerTail174_24 best.bound = innerCandidates174_23 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_23 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_23 best = best := by
  rw [innerTail174_23, scan, if_neg (by decide), innerCandidates174_23_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_24
    (if 3 ≤ innerCandidates174_23.length then search edge 3 [1051, 8311] innerCandidates174_23 9362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_24

def innerTail174_22 : List Nat := 7867 :: innerTail174_23
def innerCandidates174_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_22_checked : candidates edge 4 1051 7867 innerTail174_23 best.bound = innerCandidates174_22 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_22 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_22 best = best := by
  rw [innerTail174_22, scan, if_neg (by decide), innerCandidates174_22_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_23
    (if 3 ≤ innerCandidates174_22.length then search edge 3 [1051, 7867] innerCandidates174_22 8918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_23

def innerTail174_21 : List Nat := 6607 :: innerTail174_22
def innerCandidates174_21 : List Nat := [7867]
set_option maxRecDepth 4096 in
theorem innerCandidates174_21_checked : candidates edge 4 1051 6607 innerTail174_22 best.bound = innerCandidates174_21 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_21 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_21 best = best := by
  rw [innerTail174_21, scan, if_neg (by decide), innerCandidates174_21_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_22
    (if 3 ≤ innerCandidates174_21.length then search edge 3 [1051, 6607] innerCandidates174_21 7658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_22

def innerTail174_20 : List Nat := 6301 :: innerTail174_21
def innerCandidates174_20 : List Nat := [6607]
set_option maxRecDepth 4096 in
theorem innerCandidates174_20_checked : candidates edge 4 1051 6301 innerTail174_21 best.bound = innerCandidates174_20 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_20 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_20 best = best := by
  rw [innerTail174_20, scan, if_neg (by decide), innerCandidates174_20_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_21
    (if 3 ≤ innerCandidates174_20.length then search edge 3 [1051, 6301] innerCandidates174_20 7352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_21

def innerTail174_19 : List Nat := 5839 :: innerTail174_20
def innerCandidates174_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_19_checked : candidates edge 4 1051 5839 innerTail174_20 best.bound = innerCandidates174_19 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_19 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_19 best = best := by
  rw [innerTail174_19, scan, if_neg (by decide), innerCandidates174_19_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_20
    (if 3 ≤ innerCandidates174_19.length then search edge 3 [1051, 5839] innerCandidates174_19 6890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_20

def innerTail174_18 : List Nat := 5443 :: innerTail174_19
def innerCandidates174_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_18_checked : candidates edge 4 1051 5443 innerTail174_19 best.bound = innerCandidates174_18 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_18 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_18 best = best := by
  rw [innerTail174_18, scan, if_neg (by decide), innerCandidates174_18_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_19
    (if 3 ≤ innerCandidates174_18.length then search edge 3 [1051, 5443] innerCandidates174_18 6494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_19

def innerTail174_17 : List Nat := 4909 :: innerTail174_18
def innerCandidates174_17 : List Nat := [5443]
set_option maxRecDepth 4096 in
theorem innerCandidates174_17_checked : candidates edge 4 1051 4909 innerTail174_18 best.bound = innerCandidates174_17 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_17 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_17 best = best := by
  rw [innerTail174_17, scan, if_neg (by decide), innerCandidates174_17_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_18
    (if 3 ≤ innerCandidates174_17.length then search edge 3 [1051, 4909] innerCandidates174_17 5960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_18

def innerTail174_16 : List Nat := 4813 :: innerTail174_17
def innerCandidates174_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_16_checked : candidates edge 4 1051 4813 innerTail174_17 best.bound = innerCandidates174_16 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_16 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_16 best = best := by
  rw [innerTail174_16, scan, if_neg (by decide), innerCandidates174_16_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_17
    (if 3 ≤ innerCandidates174_16.length then search edge 3 [1051, 4813] innerCandidates174_16 5864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_17

def innerTail174_15 : List Nat := 4729 :: innerTail174_16
def innerCandidates174_15 : List Nat := [4909]
set_option maxRecDepth 4096 in
theorem innerCandidates174_15_checked : candidates edge 4 1051 4729 innerTail174_16 best.bound = innerCandidates174_15 := by
  unfold candidates
  have row : edge 4729 = SearchData.row4729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_15 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_15 best = best := by
  rw [innerTail174_15, scan, if_neg (by decide), innerCandidates174_15_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_16
    (if 3 ≤ innerCandidates174_15.length then search edge 3 [1051, 4729] innerCandidates174_15 5780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_16

def innerTail174_14 : List Nat := 4507 :: innerTail174_15
def innerCandidates174_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_14_checked : candidates edge 4 1051 4507 innerTail174_15 best.bound = innerCandidates174_14 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_14 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_14 best = best := by
  rw [innerTail174_14, scan, if_neg (by decide), innerCandidates174_14_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_15
    (if 3 ≤ innerCandidates174_14.length then search edge 3 [1051, 4507] innerCandidates174_14 5558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_15

def innerTail174_13 : List Nat := 4057 :: innerTail174_14
def innerCandidates174_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_13_checked : candidates edge 4 1051 4057 innerTail174_14 best.bound = innerCandidates174_13 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_13 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_13 best = best := by
  rw [innerTail174_13, scan, if_neg (by decide), innerCandidates174_13_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_14
    (if 3 ≤ innerCandidates174_13.length then search edge 3 [1051, 4057] innerCandidates174_13 5108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_14

def innerTail174_12 : List Nat := 4003 :: innerTail174_13
def innerCandidates174_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_12_checked : candidates edge 4 1051 4003 innerTail174_13 best.bound = innerCandidates174_12 := by
  unfold candidates
  have row : edge 4003 = SearchData.row4003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_12 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_12 best = best := by
  rw [innerTail174_12, scan, if_neg (by decide), innerCandidates174_12_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_13
    (if 3 ≤ innerCandidates174_12.length then search edge 3 [1051, 4003] innerCandidates174_12 5054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_13

def innerTail174_11 : List Nat := 3739 :: innerTail174_12
def innerCandidates174_11 : List Nat := [8353]
set_option maxRecDepth 4096 in
theorem innerCandidates174_11_checked : candidates edge 4 1051 3739 innerTail174_12 best.bound = innerCandidates174_11 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_11 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_11 best = best := by
  rw [innerTail174_11, scan, if_neg (by decide), innerCandidates174_11_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_12
    (if 3 ≤ innerCandidates174_11.length then search edge 3 [1051, 3739] innerCandidates174_11 4790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_12

def innerTail174_10 : List Nat := 3709 :: innerTail174_11
def innerCandidates174_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_10_checked : candidates edge 4 1051 3709 innerTail174_11 best.bound = innerCandidates174_10 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_10 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_10 best = best := by
  rw [innerTail174_10, scan, if_neg (by decide), innerCandidates174_10_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_11
    (if 3 ≤ innerCandidates174_10.length then search edge 3 [1051, 3709] innerCandidates174_10 4760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_11

def innerTail174_9 : List Nat := 3583 :: innerTail174_10
def innerCandidates174_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_9_checked : candidates edge 4 1051 3583 innerTail174_10 best.bound = innerCandidates174_9 := by
  unfold candidates
  have row : edge 3583 = SearchData.row3583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_9 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_9 best = best := by
  rw [innerTail174_9, scan, if_neg (by decide), innerCandidates174_9_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_10
    (if 3 ≤ innerCandidates174_9.length then search edge 3 [1051, 3583] innerCandidates174_9 4634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_10

def innerTail174_8 : List Nat := 3331 :: innerTail174_9
def innerCandidates174_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_8_checked : candidates edge 4 1051 3331 innerTail174_9 best.bound = innerCandidates174_8 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_8 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_8 best = best := by
  rw [innerTail174_8, scan, if_neg (by decide), innerCandidates174_8_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_9
    (if 3 ≤ innerCandidates174_8.length then search edge 3 [1051, 3331] innerCandidates174_8 4382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_9

def innerTail174_7 : List Nat := 3271 :: innerTail174_8
def innerCandidates174_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_7_checked : candidates edge 4 1051 3271 innerTail174_8 best.bound = innerCandidates174_7 := by
  unfold candidates
  have row : edge 3271 = SearchData.row3271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_7 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_7 best = best := by
  rw [innerTail174_7, scan, if_neg (by decide), innerCandidates174_7_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_8
    (if 3 ≤ innerCandidates174_7.length then search edge 3 [1051, 3271] innerCandidates174_7 4322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_8

def innerTail174_6 : List Nat := 3067 :: innerTail174_7
def innerCandidates174_6 : List Nat := [4909]
set_option maxRecDepth 4096 in
theorem innerCandidates174_6_checked : candidates edge 4 1051 3067 innerTail174_7 best.bound = innerCandidates174_6 := by
  unfold candidates
  have row : edge 3067 = SearchData.row3067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_6 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_6 best = best := by
  rw [innerTail174_6, scan, if_neg (by decide), innerCandidates174_6_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_7
    (if 3 ≤ innerCandidates174_6.length then search edge 3 [1051, 3067] innerCandidates174_6 4118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_7

def innerTail174_5 : List Nat := 2713 :: innerTail174_6
def innerCandidates174_5 : List Nat := [3739, 5443]
set_option maxRecDepth 4096 in
theorem innerCandidates174_5_checked : candidates edge 4 1051 2713 innerTail174_6 best.bound = innerCandidates174_5 := by
  unfold candidates
  have row : edge 2713 = SearchData.row2713.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_5 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_5 best = best := by
  rw [innerTail174_5, scan, if_neg (by decide), innerCandidates174_5_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_6
    (if 3 ≤ innerCandidates174_5.length then search edge 3 [1051, 2713] innerCandidates174_5 3764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_6

def innerTail174_4 : List Nat := 2617 :: innerTail174_5
def innerCandidates174_4 : List Nat := [5839]
set_option maxRecDepth 4096 in
theorem innerCandidates174_4_checked : candidates edge 4 1051 2617 innerTail174_5 best.bound = innerCandidates174_4 := by
  unfold candidates
  have row : edge 2617 = PrimePairCache.smallRow2617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_4 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_4 best = best := by
  rw [innerTail174_4, scan, if_neg (by decide), innerCandidates174_4_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_5
    (if 3 ≤ innerCandidates174_4.length then search edge 3 [1051, 2617] innerCandidates174_4 3668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_5

def innerTail174_3 : List Nat := 2389 :: innerTail174_4
def innerCandidates174_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates174_3_checked : candidates edge 4 1051 2389 innerTail174_4 best.bound = innerCandidates174_3 := by
  unfold candidates
  have row : edge 2389 = PrimePairCache.smallRow2389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_3 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_3 best = best := by
  rw [innerTail174_3, scan, if_neg (by decide), innerCandidates174_3_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_4
    (if 3 ≤ innerCandidates174_3.length then search edge 3 [1051, 2389] innerCandidates174_3 3440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_4

def innerTail174_2 : List Nat := 2347 :: innerTail174_3
def innerCandidates174_2 : List Nat := [3067]
set_option maxRecDepth 4096 in
theorem innerCandidates174_2_checked : candidates edge 4 1051 2347 innerTail174_3 best.bound = innerCandidates174_2 := by
  unfold candidates
  have row : edge 2347 = PrimePairCache.smallRow2347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan174_2 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_2 best = best := by
  rw [innerTail174_2, scan, if_neg (by decide), innerCandidates174_2_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_3
    (if 3 ≤ innerCandidates174_2.length then search edge 3 [1051, 2347] innerCandidates174_2 3398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan174_3

def innerTail174_1 : List Nat := 2221 :: innerTail174_2
def innerCandidates174_1 : List Nat := [2389, 4909, 5839]
set_option maxRecDepth 4096 in
theorem innerCandidates174_1_checked : candidates edge 4 1051 2221 innerTail174_2 best.bound = innerCandidates174_1 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild174_1 : search edge 3 [1051, 2221] innerCandidates174_1 3272 best = best := by decide

theorem innerScan174_1 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_1 best = best := by
  rw [innerTail174_1, scan, if_neg (by decide), innerCandidates174_1_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_2
    (if 3 ≤ innerCandidates174_1.length then search edge 3 [1051, 2221] innerCandidates174_1 3272 best else best) = best
  rw [if_pos (by decide), innerChild174_1]
  exact innerScan174_2

def innerTail174_0 : List Nat := 2017 :: innerTail174_1
def innerCandidates174_0 : List Nat := [2221, 2617, 8311]
set_option maxRecDepth 4096 in
theorem innerCandidates174_0_checked : candidates edge 4 1051 2017 innerTail174_1 best.bound = innerCandidates174_0 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild174_0 : search edge 3 [1051, 2017] innerCandidates174_0 3068 best = best := by decide

theorem innerScan174_0 : scan edge (search edge 3) 4 [1051] 1051 innerTail174_0 best = best := by
  rw [innerTail174_0, scan, if_neg (by decide), innerCandidates174_0_checked]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_1
    (if 3 ≤ innerCandidates174_0.length then search edge 3 [1051, 2017] innerCandidates174_0 3068 best else best) = best
  rw [if_pos (by decide), innerChild174_0]
  exact innerScan174_1

theorem rootBranch174 : search edge 4 [1051] rootCandidates174 1051 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1051] 1051 innerTail174_0 best = best
  exact innerScan174_0

def rootCandidates173 : List Nat := [1109, 1181, 1847, 2087, 2297, 2633, 3449, 3617, 3701, 3851, 4493, 4787, 5297, 5393, 5897, 6203, 6971, 7013, 7229, 7523, 7559, 7577, 7607, 7841, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates173_checked : candidates edge 5 0 1049 tail174 best.bound = rootCandidates173 := by
  unfold candidates
  have row : edge 1049 = PrimePairCache.smallRow1049.contains := by funext q; rfl
  rw [row]
  decide

def innerTail173_25 : List Nat := []
theorem innerScan173_25 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_25 best = best := rfl

def innerTail173_24 : List Nat := 8219 :: innerTail173_25
def innerCandidates173_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_24_checked : candidates edge 4 1049 8219 innerTail173_25 best.bound = innerCandidates173_24 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_24 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_24 best = best := by
  rw [innerTail173_24, scan, if_neg (by decide), innerCandidates173_24_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_25
    (if 3 ≤ innerCandidates173_24.length then search edge 3 [1049, 8219] innerCandidates173_24 9268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_25

def innerTail173_23 : List Nat := 7841 :: innerTail173_24
def innerCandidates173_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_23_checked : candidates edge 4 1049 7841 innerTail173_24 best.bound = innerCandidates173_23 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_23 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_23 best = best := by
  rw [innerTail173_23, scan, if_neg (by decide), innerCandidates173_23_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_24
    (if 3 ≤ innerCandidates173_23.length then search edge 3 [1049, 7841] innerCandidates173_23 8890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_24

def innerTail173_22 : List Nat := 7607 :: innerTail173_23
def innerCandidates173_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_22_checked : candidates edge 4 1049 7607 innerTail173_23 best.bound = innerCandidates173_22 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_22 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_22 best = best := by
  rw [innerTail173_22, scan, if_neg (by decide), innerCandidates173_22_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_23
    (if 3 ≤ innerCandidates173_22.length then search edge 3 [1049, 7607] innerCandidates173_22 8656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_23

def innerTail173_21 : List Nat := 7577 :: innerTail173_22
def innerCandidates173_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_21_checked : candidates edge 4 1049 7577 innerTail173_22 best.bound = innerCandidates173_21 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_21 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_21 best = best := by
  rw [innerTail173_21, scan, if_neg (by decide), innerCandidates173_21_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_22
    (if 3 ≤ innerCandidates173_21.length then search edge 3 [1049, 7577] innerCandidates173_21 8626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_22

def innerTail173_20 : List Nat := 7559 :: innerTail173_21
def innerCandidates173_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_20_checked : candidates edge 4 1049 7559 innerTail173_21 best.bound = innerCandidates173_20 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_20 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_20 best = best := by
  rw [innerTail173_20, scan, if_neg (by decide), innerCandidates173_20_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_21
    (if 3 ≤ innerCandidates173_20.length then search edge 3 [1049, 7559] innerCandidates173_20 8608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_21

def innerTail173_19 : List Nat := 7523 :: innerTail173_20
def innerCandidates173_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_19_checked : candidates edge 4 1049 7523 innerTail173_20 best.bound = innerCandidates173_19 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_19 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_19 best = best := by
  rw [innerTail173_19, scan, if_neg (by decide), innerCandidates173_19_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_20
    (if 3 ≤ innerCandidates173_19.length then search edge 3 [1049, 7523] innerCandidates173_19 8572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_20

def innerTail173_18 : List Nat := 7229 :: innerTail173_19
def innerCandidates173_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_18_checked : candidates edge 4 1049 7229 innerTail173_19 best.bound = innerCandidates173_18 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_18 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_18 best = best := by
  rw [innerTail173_18, scan, if_neg (by decide), innerCandidates173_18_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_19
    (if 3 ≤ innerCandidates173_18.length then search edge 3 [1049, 7229] innerCandidates173_18 8278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_19

def innerTail173_17 : List Nat := 7013 :: innerTail173_18
def innerCandidates173_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_17_checked : candidates edge 4 1049 7013 innerTail173_18 best.bound = innerCandidates173_17 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_17 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_17 best = best := by
  rw [innerTail173_17, scan, if_neg (by decide), innerCandidates173_17_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_18
    (if 3 ≤ innerCandidates173_17.length then search edge 3 [1049, 7013] innerCandidates173_17 8062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_18

def innerTail173_16 : List Nat := 6971 :: innerTail173_17
def innerCandidates173_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_16_checked : candidates edge 4 1049 6971 innerTail173_17 best.bound = innerCandidates173_16 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_16 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_16 best = best := by
  rw [innerTail173_16, scan, if_neg (by decide), innerCandidates173_16_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_17
    (if 3 ≤ innerCandidates173_16.length then search edge 3 [1049, 6971] innerCandidates173_16 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_17

def innerTail173_15 : List Nat := 6203 :: innerTail173_16
def innerCandidates173_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_15_checked : candidates edge 4 1049 6203 innerTail173_16 best.bound = innerCandidates173_15 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_15 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_15 best = best := by
  rw [innerTail173_15, scan, if_neg (by decide), innerCandidates173_15_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_16
    (if 3 ≤ innerCandidates173_15.length then search edge 3 [1049, 6203] innerCandidates173_15 7252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_16

def innerTail173_14 : List Nat := 5897 :: innerTail173_15
def innerCandidates173_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_14_checked : candidates edge 4 1049 5897 innerTail173_15 best.bound = innerCandidates173_14 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_14 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_14 best = best := by
  rw [innerTail173_14, scan, if_neg (by decide), innerCandidates173_14_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_15
    (if 3 ≤ innerCandidates173_14.length then search edge 3 [1049, 5897] innerCandidates173_14 6946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_15

def innerTail173_13 : List Nat := 5393 :: innerTail173_14
def innerCandidates173_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_13_checked : candidates edge 4 1049 5393 innerTail173_14 best.bound = innerCandidates173_13 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_13 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_13 best = best := by
  rw [innerTail173_13, scan, if_neg (by decide), innerCandidates173_13_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_14
    (if 3 ≤ innerCandidates173_13.length then search edge 3 [1049, 5393] innerCandidates173_13 6442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_14

def innerTail173_12 : List Nat := 5297 :: innerTail173_13
def innerCandidates173_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_12_checked : candidates edge 4 1049 5297 innerTail173_13 best.bound = innerCandidates173_12 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_12 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_12 best = best := by
  rw [innerTail173_12, scan, if_neg (by decide), innerCandidates173_12_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_13
    (if 3 ≤ innerCandidates173_12.length then search edge 3 [1049, 5297] innerCandidates173_12 6346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_13

def innerTail173_11 : List Nat := 4787 :: innerTail173_12
def innerCandidates173_11 : List Nat := [7013, 7229]
set_option maxRecDepth 4096 in
theorem innerCandidates173_11_checked : candidates edge 4 1049 4787 innerTail173_12 best.bound = innerCandidates173_11 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_11 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_11 best = best := by
  rw [innerTail173_11, scan, if_neg (by decide), innerCandidates173_11_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_12
    (if 3 ≤ innerCandidates173_11.length then search edge 3 [1049, 4787] innerCandidates173_11 5836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_12

def innerTail173_10 : List Nat := 4493 :: innerTail173_11
def innerCandidates173_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_10_checked : candidates edge 4 1049 4493 innerTail173_11 best.bound = innerCandidates173_10 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_10 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_10 best = best := by
  rw [innerTail173_10, scan, if_neg (by decide), innerCandidates173_10_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_11
    (if 3 ≤ innerCandidates173_10.length then search edge 3 [1049, 4493] innerCandidates173_10 5542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_11

def innerTail173_9 : List Nat := 3851 :: innerTail173_10
def innerCandidates173_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_9_checked : candidates edge 4 1049 3851 innerTail173_10 best.bound = innerCandidates173_9 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_9 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_9 best = best := by
  rw [innerTail173_9, scan, if_neg (by decide), innerCandidates173_9_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_10
    (if 3 ≤ innerCandidates173_9.length then search edge 3 [1049, 3851] innerCandidates173_9 4900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_10

def innerTail173_8 : List Nat := 3701 :: innerTail173_9
def innerCandidates173_8 : List Nat := [4493]
set_option maxRecDepth 4096 in
theorem innerCandidates173_8_checked : candidates edge 4 1049 3701 innerTail173_9 best.bound = innerCandidates173_8 := by
  unfold candidates
  have row : edge 3701 = SearchData.row3701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_8 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_8 best = best := by
  rw [innerTail173_8, scan, if_neg (by decide), innerCandidates173_8_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_9
    (if 3 ≤ innerCandidates173_8.length then search edge 3 [1049, 3701] innerCandidates173_8 4750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_9

def innerTail173_7 : List Nat := 3617 :: innerTail173_8
def innerCandidates173_7 : List Nat := [7607]
set_option maxRecDepth 4096 in
theorem innerCandidates173_7_checked : candidates edge 4 1049 3617 innerTail173_8 best.bound = innerCandidates173_7 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_7 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_7 best = best := by
  rw [innerTail173_7, scan, if_neg (by decide), innerCandidates173_7_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_8
    (if 3 ≤ innerCandidates173_7.length then search edge 3 [1049, 3617] innerCandidates173_7 4666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_8

def innerTail173_6 : List Nat := 3449 :: innerTail173_7
def innerCandidates173_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_6_checked : candidates edge 4 1049 3449 innerTail173_7 best.bound = innerCandidates173_6 := by
  unfold candidates
  have row : edge 3449 = SearchData.row3449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_6 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_6 best = best := by
  rw [innerTail173_6, scan, if_neg (by decide), innerCandidates173_6_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_7
    (if 3 ≤ innerCandidates173_6.length then search edge 3 [1049, 3449] innerCandidates173_6 4498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_7

def innerTail173_5 : List Nat := 2633 :: innerTail173_6
def innerCandidates173_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_5_checked : candidates edge 4 1049 2633 innerTail173_6 best.bound = innerCandidates173_5 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_5 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_5 best = best := by
  rw [innerTail173_5, scan, if_neg (by decide), innerCandidates173_5_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_6
    (if 3 ≤ innerCandidates173_5.length then search edge 3 [1049, 2633] innerCandidates173_5 3682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_6

def innerTail173_4 : List Nat := 2297 :: innerTail173_5
def innerCandidates173_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates173_4_checked : candidates edge 4 1049 2297 innerTail173_5 best.bound = innerCandidates173_4 := by
  unfold candidates
  have row : edge 2297 = PrimePairCache.smallRow2297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_4 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_4 best = best := by
  rw [innerTail173_4, scan, if_neg (by decide), innerCandidates173_4_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_5
    (if 3 ≤ innerCandidates173_4.length then search edge 3 [1049, 2297] innerCandidates173_4 3346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_5

def innerTail173_3 : List Nat := 2087 :: innerTail173_4
def innerCandidates173_3 : List Nat := [7559]
set_option maxRecDepth 4096 in
theorem innerCandidates173_3_checked : candidates edge 4 1049 2087 innerTail173_4 best.bound = innerCandidates173_3 := by
  unfold candidates
  have row : edge 2087 = PrimePairCache.smallRow2087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_3 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_3 best = best := by
  rw [innerTail173_3, scan, if_neg (by decide), innerCandidates173_3_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_4
    (if 3 ≤ innerCandidates173_3.length then search edge 3 [1049, 2087] innerCandidates173_3 3136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_4

def innerTail173_2 : List Nat := 1847 :: innerTail173_3
def innerCandidates173_2 : List Nat := [7523]
set_option maxRecDepth 4096 in
theorem innerCandidates173_2_checked : candidates edge 4 1049 1847 innerTail173_3 best.bound = innerCandidates173_2 := by
  unfold candidates
  have row : edge 1847 = PrimePairCache.smallRow1847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_2 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_2 best = best := by
  rw [innerTail173_2, scan, if_neg (by decide), innerCandidates173_2_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_3
    (if 3 ≤ innerCandidates173_2.length then search edge 3 [1049, 1847] innerCandidates173_2 2896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_3

def innerTail173_1 : List Nat := 1181 :: innerTail173_2
def innerCandidates173_1 : List Nat := [1847, 7577]
set_option maxRecDepth 4096 in
theorem innerCandidates173_1_checked : candidates edge 4 1049 1181 innerTail173_2 best.bound = innerCandidates173_1 := by
  unfold candidates
  have row : edge 1181 = PrimePairCache.smallRow1181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_1 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_1 best = best := by
  rw [innerTail173_1, scan, if_neg (by decide), innerCandidates173_1_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_2
    (if 3 ≤ innerCandidates173_1.length then search edge 3 [1049, 1181] innerCandidates173_1 2230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_2

def innerTail173_0 : List Nat := 1109 :: innerTail173_1
def innerCandidates173_0 : List Nat := [1181, 4493]
set_option maxRecDepth 4096 in
theorem innerCandidates173_0_checked : candidates edge 4 1049 1109 innerTail173_1 best.bound = innerCandidates173_0 := by
  unfold candidates
  have row : edge 1109 = PrimePairCache.smallRow1109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan173_0 : scan edge (search edge 3) 4 [1049] 1049 innerTail173_0 best = best := by
  rw [innerTail173_0, scan, if_neg (by decide), innerCandidates173_0_checked]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_1
    (if 3 ≤ innerCandidates173_0.length then search edge 3 [1049, 1109] innerCandidates173_0 2158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan173_1

theorem rootBranch173 : search edge 4 [1049] rootCandidates173 1049 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1049] 1049 innerTail173_0 best = best
  exact innerScan173_0

end ProjectEulerSolutions.PrimePairUpper.Search
