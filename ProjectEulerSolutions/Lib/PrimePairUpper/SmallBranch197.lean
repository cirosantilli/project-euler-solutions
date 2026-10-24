import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates108 : List Nat := [619, 967, 991, 1297, 1453, 1693, 1999, 2467, 2887, 3037, 3733, 3889, 4363, 4447, 4483, 4561, 4591, 4603, 5077, 5107, 6121, 6337, 6673, 6709, 6883, 7039, 7573, 7639, 8287, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates108_checked : candidates edge 5 0 607 tail109 best.bound = rootCandidates108 := by
  unfold candidates
  have row : edge 607 = PrimePairCache.smallRow607.contains := by funext q; rfl
  rw [row]
  decide

def innerTail108_30 : List Nat := []
theorem innerScan108_30 : scan edge (search edge 3) 4 [607] 607 innerTail108_30 best = best := rfl

def innerTail108_29 : List Nat := 8353 :: innerTail108_30
def innerCandidates108_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_29_checked : candidates edge 4 607 8353 innerTail108_30 best.bound = innerCandidates108_29 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_29 : scan edge (search edge 3) 4 [607] 607 innerTail108_29 best = best := by
  rw [innerTail108_29, scan, if_neg (by decide), innerCandidates108_29_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_30
    (if 3 ≤ innerCandidates108_29.length then search edge 3 [607, 8353] innerCandidates108_29 8960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_30

def innerTail108_28 : List Nat := 8287 :: innerTail108_29
def innerCandidates108_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_28_checked : candidates edge 4 607 8287 innerTail108_29 best.bound = innerCandidates108_28 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_28 : scan edge (search edge 3) 4 [607] 607 innerTail108_28 best = best := by
  rw [innerTail108_28, scan, if_neg (by decide), innerCandidates108_28_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_29
    (if 3 ≤ innerCandidates108_28.length then search edge 3 [607, 8287] innerCandidates108_28 8894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_29

def innerTail108_27 : List Nat := 7639 :: innerTail108_28
def innerCandidates108_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_27_checked : candidates edge 4 607 7639 innerTail108_28 best.bound = innerCandidates108_27 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_27 : scan edge (search edge 3) 4 [607] 607 innerTail108_27 best = best := by
  rw [innerTail108_27, scan, if_neg (by decide), innerCandidates108_27_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_28
    (if 3 ≤ innerCandidates108_27.length then search edge 3 [607, 7639] innerCandidates108_27 8246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_28

def innerTail108_26 : List Nat := 7573 :: innerTail108_27
def innerCandidates108_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_26_checked : candidates edge 4 607 7573 innerTail108_27 best.bound = innerCandidates108_26 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_26 : scan edge (search edge 3) 4 [607] 607 innerTail108_26 best = best := by
  rw [innerTail108_26, scan, if_neg (by decide), innerCandidates108_26_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_27
    (if 3 ≤ innerCandidates108_26.length then search edge 3 [607, 7573] innerCandidates108_26 8180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_27

def innerTail108_25 : List Nat := 7039 :: innerTail108_26
def innerCandidates108_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_25_checked : candidates edge 4 607 7039 innerTail108_26 best.bound = innerCandidates108_25 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_25 : scan edge (search edge 3) 4 [607] 607 innerTail108_25 best = best := by
  rw [innerTail108_25, scan, if_neg (by decide), innerCandidates108_25_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_26
    (if 3 ≤ innerCandidates108_25.length then search edge 3 [607, 7039] innerCandidates108_25 7646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_26

def innerTail108_24 : List Nat := 6883 :: innerTail108_25
def innerCandidates108_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_24_checked : candidates edge 4 607 6883 innerTail108_25 best.bound = innerCandidates108_24 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_24 : scan edge (search edge 3) 4 [607] 607 innerTail108_24 best = best := by
  rw [innerTail108_24, scan, if_neg (by decide), innerCandidates108_24_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_25
    (if 3 ≤ innerCandidates108_24.length then search edge 3 [607, 6883] innerCandidates108_24 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_25

def innerTail108_23 : List Nat := 6709 :: innerTail108_24
def innerCandidates108_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_23_checked : candidates edge 4 607 6709 innerTail108_24 best.bound = innerCandidates108_23 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_23 : scan edge (search edge 3) 4 [607] 607 innerTail108_23 best = best := by
  rw [innerTail108_23, scan, if_neg (by decide), innerCandidates108_23_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_24
    (if 3 ≤ innerCandidates108_23.length then search edge 3 [607, 6709] innerCandidates108_23 7316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_24

def innerTail108_22 : List Nat := 6673 :: innerTail108_23
def innerCandidates108_22 : List Nat := [7639]
set_option maxRecDepth 4096 in
theorem innerCandidates108_22_checked : candidates edge 4 607 6673 innerTail108_23 best.bound = innerCandidates108_22 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_22 : scan edge (search edge 3) 4 [607] 607 innerTail108_22 best = best := by
  rw [innerTail108_22, scan, if_neg (by decide), innerCandidates108_22_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_23
    (if 3 ≤ innerCandidates108_22.length then search edge 3 [607, 6673] innerCandidates108_22 7280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_23

def innerTail108_21 : List Nat := 6337 :: innerTail108_22
def innerCandidates108_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_21_checked : candidates edge 4 607 6337 innerTail108_22 best.bound = innerCandidates108_21 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_21 : scan edge (search edge 3) 4 [607] 607 innerTail108_21 best = best := by
  rw [innerTail108_21, scan, if_neg (by decide), innerCandidates108_21_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_22
    (if 3 ≤ innerCandidates108_21.length then search edge 3 [607, 6337] innerCandidates108_21 6944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_22

def innerTail108_20 : List Nat := 6121 :: innerTail108_21
def innerCandidates108_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_20_checked : candidates edge 4 607 6121 innerTail108_21 best.bound = innerCandidates108_20 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_20 : scan edge (search edge 3) 4 [607] 607 innerTail108_20 best = best := by
  rw [innerTail108_20, scan, if_neg (by decide), innerCandidates108_20_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_21
    (if 3 ≤ innerCandidates108_20.length then search edge 3 [607, 6121] innerCandidates108_20 6728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_21

def innerTail108_19 : List Nat := 5107 :: innerTail108_20
def innerCandidates108_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_19_checked : candidates edge 4 607 5107 innerTail108_20 best.bound = innerCandidates108_19 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_19 : scan edge (search edge 3) 4 [607] 607 innerTail108_19 best = best := by
  rw [innerTail108_19, scan, if_neg (by decide), innerCandidates108_19_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_20
    (if 3 ≤ innerCandidates108_19.length then search edge 3 [607, 5107] innerCandidates108_19 5714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_20

def innerTail108_18 : List Nat := 5077 :: innerTail108_19
def innerCandidates108_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_18_checked : candidates edge 4 607 5077 innerTail108_19 best.bound = innerCandidates108_18 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_18 : scan edge (search edge 3) 4 [607] 607 innerTail108_18 best = best := by
  rw [innerTail108_18, scan, if_neg (by decide), innerCandidates108_18_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_19
    (if 3 ≤ innerCandidates108_18.length then search edge 3 [607, 5077] innerCandidates108_18 5684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_19

def innerTail108_17 : List Nat := 4603 :: innerTail108_18
def innerCandidates108_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_17_checked : candidates edge 4 607 4603 innerTail108_18 best.bound = innerCandidates108_17 := by
  unfold candidates
  have row : edge 4603 = SearchData.row4603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_17 : scan edge (search edge 3) 4 [607] 607 innerTail108_17 best = best := by
  rw [innerTail108_17, scan, if_neg (by decide), innerCandidates108_17_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_18
    (if 3 ≤ innerCandidates108_17.length then search edge 3 [607, 4603] innerCandidates108_17 5210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_18

def innerTail108_16 : List Nat := 4591 :: innerTail108_17
def innerCandidates108_16 : List Nat := [6121, 8287]
set_option maxRecDepth 4096 in
theorem innerCandidates108_16_checked : candidates edge 4 607 4591 innerTail108_17 best.bound = innerCandidates108_16 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_16 : scan edge (search edge 3) 4 [607] 607 innerTail108_16 best = best := by
  rw [innerTail108_16, scan, if_neg (by decide), innerCandidates108_16_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_17
    (if 3 ≤ innerCandidates108_16.length then search edge 3 [607, 4591] innerCandidates108_16 5198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_17

def innerTail108_15 : List Nat := 4561 :: innerTail108_16
def innerCandidates108_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_15_checked : candidates edge 4 607 4561 innerTail108_16 best.bound = innerCandidates108_15 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_15 : scan edge (search edge 3) 4 [607] 607 innerTail108_15 best = best := by
  rw [innerTail108_15, scan, if_neg (by decide), innerCandidates108_15_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_16
    (if 3 ≤ innerCandidates108_15.length then search edge 3 [607, 4561] innerCandidates108_15 5168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_16

def innerTail108_14 : List Nat := 4483 :: innerTail108_15
def innerCandidates108_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_14_checked : candidates edge 4 607 4483 innerTail108_15 best.bound = innerCandidates108_14 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_14 : scan edge (search edge 3) 4 [607] 607 innerTail108_14 best = best := by
  rw [innerTail108_14, scan, if_neg (by decide), innerCandidates108_14_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_15
    (if 3 ≤ innerCandidates108_14.length then search edge 3 [607, 4483] innerCandidates108_14 5090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_15

def innerTail108_13 : List Nat := 4447 :: innerTail108_14
def innerCandidates108_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_13_checked : candidates edge 4 607 4447 innerTail108_14 best.bound = innerCandidates108_13 := by
  unfold candidates
  have row : edge 4447 = SearchData.row4447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_13 : scan edge (search edge 3) 4 [607] 607 innerTail108_13 best = best := by
  rw [innerTail108_13, scan, if_neg (by decide), innerCandidates108_13_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_14
    (if 3 ≤ innerCandidates108_13.length then search edge 3 [607, 4447] innerCandidates108_13 5054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_14

def innerTail108_12 : List Nat := 4363 :: innerTail108_13
def innerCandidates108_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_12_checked : candidates edge 4 607 4363 innerTail108_13 best.bound = innerCandidates108_12 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_12 : scan edge (search edge 3) 4 [607] 607 innerTail108_12 best = best := by
  rw [innerTail108_12, scan, if_neg (by decide), innerCandidates108_12_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_13
    (if 3 ≤ innerCandidates108_12.length then search edge 3 [607, 4363] innerCandidates108_12 4970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_13

def innerTail108_11 : List Nat := 3889 :: innerTail108_12
def innerCandidates108_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_11_checked : candidates edge 4 607 3889 innerTail108_12 best.bound = innerCandidates108_11 := by
  unfold candidates
  have row : edge 3889 = SearchData.row3889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_11 : scan edge (search edge 3) 4 [607] 607 innerTail108_11 best = best := by
  rw [innerTail108_11, scan, if_neg (by decide), innerCandidates108_11_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_12
    (if 3 ≤ innerCandidates108_11.length then search edge 3 [607, 3889] innerCandidates108_11 4496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_12

def innerTail108_10 : List Nat := 3733 :: innerTail108_11
def innerCandidates108_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_10_checked : candidates edge 4 607 3733 innerTail108_11 best.bound = innerCandidates108_10 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_10 : scan edge (search edge 3) 4 [607] 607 innerTail108_10 best = best := by
  rw [innerTail108_10, scan, if_neg (by decide), innerCandidates108_10_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_11
    (if 3 ≤ innerCandidates108_10.length then search edge 3 [607, 3733] innerCandidates108_10 4340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_11

def innerTail108_9 : List Nat := 3037 :: innerTail108_10
def innerCandidates108_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_9_checked : candidates edge 4 607 3037 innerTail108_10 best.bound = innerCandidates108_9 := by
  unfold candidates
  have row : edge 3037 = SearchData.row3037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_9 : scan edge (search edge 3) 4 [607] 607 innerTail108_9 best = best := by
  rw [innerTail108_9, scan, if_neg (by decide), innerCandidates108_9_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_10
    (if 3 ≤ innerCandidates108_9.length then search edge 3 [607, 3037] innerCandidates108_9 3644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_10

def innerTail108_8 : List Nat := 2887 :: innerTail108_9
def innerCandidates108_8 : List Nat := [6709, 7573]
set_option maxRecDepth 4096 in
theorem innerCandidates108_8_checked : candidates edge 4 607 2887 innerTail108_9 best.bound = innerCandidates108_8 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_8 : scan edge (search edge 3) 4 [607] 607 innerTail108_8 best = best := by
  rw [innerTail108_8, scan, if_neg (by decide), innerCandidates108_8_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_9
    (if 3 ≤ innerCandidates108_8.length then search edge 3 [607, 2887] innerCandidates108_8 3494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_9

def innerTail108_7 : List Nat := 2467 :: innerTail108_8
def innerCandidates108_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_7_checked : candidates edge 4 607 2467 innerTail108_8 best.bound = innerCandidates108_7 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_7 : scan edge (search edge 3) 4 [607] 607 innerTail108_7 best = best := by
  rw [innerTail108_7, scan, if_neg (by decide), innerCandidates108_7_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_8
    (if 3 ≤ innerCandidates108_7.length then search edge 3 [607, 2467] innerCandidates108_7 3074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_8

def innerTail108_6 : List Nat := 1999 :: innerTail108_7
def innerCandidates108_6 : List Nat := [3733, 6121]
set_option maxRecDepth 4096 in
theorem innerCandidates108_6_checked : candidates edge 4 607 1999 innerTail108_7 best.bound = innerCandidates108_6 := by
  unfold candidates
  have row : edge 1999 = PrimePairCache.smallRow1999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_6 : scan edge (search edge 3) 4 [607] 607 innerTail108_6 best = best := by
  rw [innerTail108_6, scan, if_neg (by decide), innerCandidates108_6_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_7
    (if 3 ≤ innerCandidates108_6.length then search edge 3 [607, 1999] innerCandidates108_6 2606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_7

def innerTail108_5 : List Nat := 1693 :: innerTail108_6
def innerCandidates108_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_5_checked : candidates edge 4 607 1693 innerTail108_6 best.bound = innerCandidates108_5 := by
  unfold candidates
  have row : edge 1693 = PrimePairCache.smallRow1693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_5 : scan edge (search edge 3) 4 [607] 607 innerTail108_5 best = best := by
  rw [innerTail108_5, scan, if_neg (by decide), innerCandidates108_5_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_6
    (if 3 ≤ innerCandidates108_5.length then search edge 3 [607, 1693] innerCandidates108_5 2300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_6

def innerTail108_4 : List Nat := 1453 :: innerTail108_5
def innerCandidates108_4 : List Nat := [4483]
set_option maxRecDepth 4096 in
theorem innerCandidates108_4_checked : candidates edge 4 607 1453 innerTail108_5 best.bound = innerCandidates108_4 := by
  unfold candidates
  have row : edge 1453 = PrimePairCache.smallRow1453.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_4 : scan edge (search edge 3) 4 [607] 607 innerTail108_4 best = best := by
  rw [innerTail108_4, scan, if_neg (by decide), innerCandidates108_4_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_5
    (if 3 ≤ innerCandidates108_4.length then search edge 3 [607, 1453] innerCandidates108_4 2060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_5

def innerTail108_3 : List Nat := 1297 :: innerTail108_4
def innerCandidates108_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_3_checked : candidates edge 4 607 1297 innerTail108_4 best.bound = innerCandidates108_3 := by
  unfold candidates
  have row : edge 1297 = PrimePairCache.smallRow1297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_3 : scan edge (search edge 3) 4 [607] 607 innerTail108_3 best = best := by
  rw [innerTail108_3, scan, if_neg (by decide), innerCandidates108_3_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_4
    (if 3 ≤ innerCandidates108_3.length then search edge 3 [607, 1297] innerCandidates108_3 1904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_4

def innerTail108_2 : List Nat := 991 :: innerTail108_3
def innerCandidates108_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_2_checked : candidates edge 4 607 991 innerTail108_3 best.bound = innerCandidates108_2 := by
  unfold candidates
  have row : edge 991 = PrimePairCache.smallRow991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_2 : scan edge (search edge 3) 4 [607] 607 innerTail108_2 best = best := by
  rw [innerTail108_2, scan, if_neg (by decide), innerCandidates108_2_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_3
    (if 3 ≤ innerCandidates108_2.length then search edge 3 [607, 991] innerCandidates108_2 1598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_3

def innerTail108_1 : List Nat := 967 :: innerTail108_2
def innerCandidates108_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_1_checked : candidates edge 4 607 967 innerTail108_2 best.bound = innerCandidates108_1 := by
  unfold candidates
  have row : edge 967 = PrimePairCache.smallRow967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_1 : scan edge (search edge 3) 4 [607] 607 innerTail108_1 best = best := by
  rw [innerTail108_1, scan, if_neg (by decide), innerCandidates108_1_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_2
    (if 3 ≤ innerCandidates108_1.length then search edge 3 [607, 967] innerCandidates108_1 1574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_2

def innerTail108_0 : List Nat := 619 :: innerTail108_1
def innerCandidates108_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates108_0_checked : candidates edge 4 607 619 innerTail108_1 best.bound = innerCandidates108_0 := by
  unfold candidates
  have row : edge 619 = PrimePairCache.smallRow619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan108_0 : scan edge (search edge 3) 4 [607] 607 innerTail108_0 best = best := by
  rw [innerTail108_0, scan, if_neg (by decide), innerCandidates108_0_checked]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_1
    (if 3 ≤ innerCandidates108_0.length then search edge 3 [607, 619] innerCandidates108_0 1226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan108_1

theorem rootBranch108 : search edge 4 [607] rootCandidates108 607 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [607] 607 innerTail108_0 best = best
  exact innerScan108_0

def rootCandidates107 : List Nat := [823, 1171, 1321, 1597, 1627, 1867, 1879, 2011, 2689, 2719, 3187, 4159, 4663, 4909, 5197, 7213, 7411, 7603, 7639, 8317, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates107_checked : candidates edge 5 0 601 tail108 best.bound = rootCandidates107 := by
  unfold candidates
  have row : edge 601 = PrimePairCache.smallRow601.contains := by funext q; rfl
  rw [row]
  decide

def innerTail107_21 : List Nat := []
theorem innerScan107_21 : scan edge (search edge 3) 4 [601] 601 innerTail107_21 best = best := rfl

def innerTail107_20 : List Nat := 8377 :: innerTail107_21
def innerCandidates107_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_20_checked : candidates edge 4 601 8377 innerTail107_21 best.bound = innerCandidates107_20 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_20 : scan edge (search edge 3) 4 [601] 601 innerTail107_20 best = best := by
  rw [innerTail107_20, scan, if_neg (by decide), innerCandidates107_20_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_21
    (if 3 ≤ innerCandidates107_20.length then search edge 3 [601, 8377] innerCandidates107_20 8978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_21

def innerTail107_19 : List Nat := 8317 :: innerTail107_20
def innerCandidates107_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_19_checked : candidates edge 4 601 8317 innerTail107_20 best.bound = innerCandidates107_19 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_19 : scan edge (search edge 3) 4 [601] 601 innerTail107_19 best = best := by
  rw [innerTail107_19, scan, if_neg (by decide), innerCandidates107_19_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_20
    (if 3 ≤ innerCandidates107_19.length then search edge 3 [601, 8317] innerCandidates107_19 8918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_20

def innerTail107_18 : List Nat := 7639 :: innerTail107_19
def innerCandidates107_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_18_checked : candidates edge 4 601 7639 innerTail107_19 best.bound = innerCandidates107_18 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_18 : scan edge (search edge 3) 4 [601] 601 innerTail107_18 best = best := by
  rw [innerTail107_18, scan, if_neg (by decide), innerCandidates107_18_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_19
    (if 3 ≤ innerCandidates107_18.length then search edge 3 [601, 7639] innerCandidates107_18 8240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_19

def innerTail107_17 : List Nat := 7603 :: innerTail107_18
def innerCandidates107_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_17_checked : candidates edge 4 601 7603 innerTail107_18 best.bound = innerCandidates107_17 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_17 : scan edge (search edge 3) 4 [601] 601 innerTail107_17 best = best := by
  rw [innerTail107_17, scan, if_neg (by decide), innerCandidates107_17_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_18
    (if 3 ≤ innerCandidates107_17.length then search edge 3 [601, 7603] innerCandidates107_17 8204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_18

def innerTail107_16 : List Nat := 7411 :: innerTail107_17
def innerCandidates107_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_16_checked : candidates edge 4 601 7411 innerTail107_17 best.bound = innerCandidates107_16 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_16 : scan edge (search edge 3) 4 [601] 601 innerTail107_16 best = best := by
  rw [innerTail107_16, scan, if_neg (by decide), innerCandidates107_16_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_17
    (if 3 ≤ innerCandidates107_16.length then search edge 3 [601, 7411] innerCandidates107_16 8012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_17

def innerTail107_15 : List Nat := 7213 :: innerTail107_16
def innerCandidates107_15 : List Nat := [7411]
set_option maxRecDepth 4096 in
theorem innerCandidates107_15_checked : candidates edge 4 601 7213 innerTail107_16 best.bound = innerCandidates107_15 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_15 : scan edge (search edge 3) 4 [601] 601 innerTail107_15 best = best := by
  rw [innerTail107_15, scan, if_neg (by decide), innerCandidates107_15_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_16
    (if 3 ≤ innerCandidates107_15.length then search edge 3 [601, 7213] innerCandidates107_15 7814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_16

def innerTail107_14 : List Nat := 5197 :: innerTail107_15
def innerCandidates107_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_14_checked : candidates edge 4 601 5197 innerTail107_15 best.bound = innerCandidates107_14 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_14 : scan edge (search edge 3) 4 [601] 601 innerTail107_14 best = best := by
  rw [innerTail107_14, scan, if_neg (by decide), innerCandidates107_14_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_15
    (if 3 ≤ innerCandidates107_14.length then search edge 3 [601, 5197] innerCandidates107_14 5798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_15

def innerTail107_13 : List Nat := 4909 :: innerTail107_14
def innerCandidates107_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_13_checked : candidates edge 4 601 4909 innerTail107_14 best.bound = innerCandidates107_13 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_13 : scan edge (search edge 3) 4 [601] 601 innerTail107_13 best = best := by
  rw [innerTail107_13, scan, if_neg (by decide), innerCandidates107_13_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_14
    (if 3 ≤ innerCandidates107_13.length then search edge 3 [601, 4909] innerCandidates107_13 5510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_14

def innerTail107_12 : List Nat := 4663 :: innerTail107_13
def innerCandidates107_12 : List Nat := [7603]
set_option maxRecDepth 4096 in
theorem innerCandidates107_12_checked : candidates edge 4 601 4663 innerTail107_13 best.bound = innerCandidates107_12 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_12 : scan edge (search edge 3) 4 [601] 601 innerTail107_12 best = best := by
  rw [innerTail107_12, scan, if_neg (by decide), innerCandidates107_12_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_13
    (if 3 ≤ innerCandidates107_12.length then search edge 3 [601, 4663] innerCandidates107_12 5264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_13

def innerTail107_11 : List Nat := 4159 :: innerTail107_12
def innerCandidates107_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_11_checked : candidates edge 4 601 4159 innerTail107_12 best.bound = innerCandidates107_11 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_11 : scan edge (search edge 3) 4 [601] 601 innerTail107_11 best = best := by
  rw [innerTail107_11, scan, if_neg (by decide), innerCandidates107_11_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_12
    (if 3 ≤ innerCandidates107_11.length then search edge 3 [601, 4159] innerCandidates107_11 4760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_12

def innerTail107_10 : List Nat := 3187 :: innerTail107_11
def innerCandidates107_10 : List Nat := [7411]
set_option maxRecDepth 4096 in
theorem innerCandidates107_10_checked : candidates edge 4 601 3187 innerTail107_11 best.bound = innerCandidates107_10 := by
  unfold candidates
  have row : edge 3187 = SearchData.row3187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_10 : scan edge (search edge 3) 4 [601] 601 innerTail107_10 best = best := by
  rw [innerTail107_10, scan, if_neg (by decide), innerCandidates107_10_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_11
    (if 3 ≤ innerCandidates107_10.length then search edge 3 [601, 3187] innerCandidates107_10 3788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_11

def innerTail107_9 : List Nat := 2719 :: innerTail107_10
def innerCandidates107_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_9_checked : candidates edge 4 601 2719 innerTail107_10 best.bound = innerCandidates107_9 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_9 : scan edge (search edge 3) 4 [601] 601 innerTail107_9 best = best := by
  rw [innerTail107_9, scan, if_neg (by decide), innerCandidates107_9_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_10
    (if 3 ≤ innerCandidates107_9.length then search edge 3 [601, 2719] innerCandidates107_9 3320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_10

def innerTail107_8 : List Nat := 2689 :: innerTail107_9
def innerCandidates107_8 : List Nat := [8317]
set_option maxRecDepth 4096 in
theorem innerCandidates107_8_checked : candidates edge 4 601 2689 innerTail107_9 best.bound = innerCandidates107_8 := by
  unfold candidates
  have row : edge 2689 = SearchData.row2689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_8 : scan edge (search edge 3) 4 [601] 601 innerTail107_8 best = best := by
  rw [innerTail107_8, scan, if_neg (by decide), innerCandidates107_8_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_9
    (if 3 ≤ innerCandidates107_8.length then search edge 3 [601, 2689] innerCandidates107_8 3290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_9

def innerTail107_7 : List Nat := 2011 :: innerTail107_8
def innerCandidates107_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_7_checked : candidates edge 4 601 2011 innerTail107_8 best.bound = innerCandidates107_7 := by
  unfold candidates
  have row : edge 2011 = PrimePairCache.smallRow2011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_7 : scan edge (search edge 3) 4 [601] 601 innerTail107_7 best = best := by
  rw [innerTail107_7, scan, if_neg (by decide), innerCandidates107_7_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_8
    (if 3 ≤ innerCandidates107_7.length then search edge 3 [601, 2011] innerCandidates107_7 2612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_8

def innerTail107_6 : List Nat := 1879 :: innerTail107_7
def innerCandidates107_6 : List Nat := [7411]
set_option maxRecDepth 4096 in
theorem innerCandidates107_6_checked : candidates edge 4 601 1879 innerTail107_7 best.bound = innerCandidates107_6 := by
  unfold candidates
  have row : edge 1879 = PrimePairCache.smallRow1879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_6 : scan edge (search edge 3) 4 [601] 601 innerTail107_6 best = best := by
  rw [innerTail107_6, scan, if_neg (by decide), innerCandidates107_6_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_7
    (if 3 ≤ innerCandidates107_6.length then search edge 3 [601, 1879] innerCandidates107_6 2480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_7

def innerTail107_5 : List Nat := 1867 :: innerTail107_6
def innerCandidates107_5 : List Nat := [7639]
set_option maxRecDepth 4096 in
theorem innerCandidates107_5_checked : candidates edge 4 601 1867 innerTail107_6 best.bound = innerCandidates107_5 := by
  unfold candidates
  have row : edge 1867 = PrimePairCache.smallRow1867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_5 : scan edge (search edge 3) 4 [601] 601 innerTail107_5 best = best := by
  rw [innerTail107_5, scan, if_neg (by decide), innerCandidates107_5_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_6
    (if 3 ≤ innerCandidates107_5.length then search edge 3 [601, 1867] innerCandidates107_5 2468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_6

def innerTail107_4 : List Nat := 1627 :: innerTail107_5
def innerCandidates107_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_4_checked : candidates edge 4 601 1627 innerTail107_5 best.bound = innerCandidates107_4 := by
  unfold candidates
  have row : edge 1627 = PrimePairCache.smallRow1627.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_4 : scan edge (search edge 3) 4 [601] 601 innerTail107_4 best = best := by
  rw [innerTail107_4, scan, if_neg (by decide), innerCandidates107_4_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_5
    (if 3 ≤ innerCandidates107_4.length then search edge 3 [601, 1627] innerCandidates107_4 2228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_5

def innerTail107_3 : List Nat := 1597 :: innerTail107_4
def innerCandidates107_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_3_checked : candidates edge 4 601 1597 innerTail107_4 best.bound = innerCandidates107_3 := by
  unfold candidates
  have row : edge 1597 = PrimePairCache.smallRow1597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_3 : scan edge (search edge 3) 4 [601] 601 innerTail107_3 best = best := by
  rw [innerTail107_3, scan, if_neg (by decide), innerCandidates107_3_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_4
    (if 3 ≤ innerCandidates107_3.length then search edge 3 [601, 1597] innerCandidates107_3 2198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_4

def innerTail107_2 : List Nat := 1321 :: innerTail107_3
def innerCandidates107_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates107_2_checked : candidates edge 4 601 1321 innerTail107_3 best.bound = innerCandidates107_2 := by
  unfold candidates
  have row : edge 1321 = PrimePairCache.smallRow1321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_2 : scan edge (search edge 3) 4 [601] 601 innerTail107_2 best = best := by
  rw [innerTail107_2, scan, if_neg (by decide), innerCandidates107_2_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_3
    (if 3 ≤ innerCandidates107_2.length then search edge 3 [601, 1321] innerCandidates107_2 1922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_3

def innerTail107_1 : List Nat := 1171 :: innerTail107_2
def innerCandidates107_1 : List Nat := [2011, 4663]
set_option maxRecDepth 4096 in
theorem innerCandidates107_1_checked : candidates edge 4 601 1171 innerTail107_2 best.bound = innerCandidates107_1 := by
  unfold candidates
  have row : edge 1171 = PrimePairCache.smallRow1171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_1 : scan edge (search edge 3) 4 [601] 601 innerTail107_1 best = best := by
  rw [innerTail107_1, scan, if_neg (by decide), innerCandidates107_1_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_2
    (if 3 ≤ innerCandidates107_1.length then search edge 3 [601, 1171] innerCandidates107_1 1772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_2

def innerTail107_0 : List Nat := 823 :: innerTail107_1
def innerCandidates107_0 : List Nat := [4909]
set_option maxRecDepth 4096 in
theorem innerCandidates107_0_checked : candidates edge 4 601 823 innerTail107_1 best.bound = innerCandidates107_0 := by
  unfold candidates
  have row : edge 823 = PrimePairCache.smallRow823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan107_0 : scan edge (search edge 3) 4 [601] 601 innerTail107_0 best = best := by
  rw [innerTail107_0, scan, if_neg (by decide), innerCandidates107_0_checked]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_1
    (if 3 ≤ innerCandidates107_0.length then search edge 3 [601, 823] innerCandidates107_0 1424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan107_1

theorem rootBranch107 : search edge 4 [601] rootCandidates107 601 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [601] 601 innerTail107_0 best = best
  exact innerScan107_0

end ProjectEulerSolutions.PrimePairUpper.Search
