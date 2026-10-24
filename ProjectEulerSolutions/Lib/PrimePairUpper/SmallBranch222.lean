import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates58 : List Nat := [397, 463, 487, 541, 601, 1051, 1237, 1321, 1429, 1489, 1657, 2113, 2287, 2503, 3001, 3529, 4813, 5653, 5737, 5851, 5857, 6397, 6607, 6733, 6763, 7039, 7057, 8191, 8287, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates58_checked : candidates edge 5 0 283 tail59 best.bound = rootCandidates58 := by
  unfold candidates
  have row : edge 283 = PrimePairCache.smallRow283.contains := by funext q; rfl
  rw [row]
  decide

def innerTail58_30 : List Nat := []
theorem innerScan58_30 : scan edge (search edge 3) 4 [283] 283 innerTail58_30 best = best := rfl

def innerTail58_29 : List Nat := 8293 :: innerTail58_30
def innerCandidates58_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates58_29_checked : candidates edge 4 283 8293 innerTail58_30 best.bound = innerCandidates58_29 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_29 : scan edge (search edge 3) 4 [283] 283 innerTail58_29 best = best := by
  rw [innerTail58_29, scan, if_neg (by decide), innerCandidates58_29_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_30
    (if 3 ≤ innerCandidates58_29.length then search edge 3 [283, 8293] innerCandidates58_29 8576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_30

def innerTail58_28 : List Nat := 8287 :: innerTail58_29
def innerCandidates58_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates58_28_checked : candidates edge 4 283 8287 innerTail58_29 best.bound = innerCandidates58_28 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_28 : scan edge (search edge 3) 4 [283] 283 innerTail58_28 best = best := by
  rw [innerTail58_28, scan, if_neg (by decide), innerCandidates58_28_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_29
    (if 3 ≤ innerCandidates58_28.length then search edge 3 [283, 8287] innerCandidates58_28 8570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_29

def innerTail58_27 : List Nat := 8191 :: innerTail58_28
def innerCandidates58_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates58_27_checked : candidates edge 4 283 8191 innerTail58_28 best.bound = innerCandidates58_27 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_27 : scan edge (search edge 3) 4 [283] 283 innerTail58_27 best = best := by
  rw [innerTail58_27, scan, if_neg (by decide), innerCandidates58_27_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_28
    (if 3 ≤ innerCandidates58_27.length then search edge 3 [283, 8191] innerCandidates58_27 8474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_28

def innerTail58_26 : List Nat := 7057 :: innerTail58_27
def innerCandidates58_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates58_26_checked : candidates edge 4 283 7057 innerTail58_27 best.bound = innerCandidates58_26 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_26 : scan edge (search edge 3) 4 [283] 283 innerTail58_26 best = best := by
  rw [innerTail58_26, scan, if_neg (by decide), innerCandidates58_26_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_27
    (if 3 ≤ innerCandidates58_26.length then search edge 3 [283, 7057] innerCandidates58_26 7340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_27

def innerTail58_25 : List Nat := 7039 :: innerTail58_26
def innerCandidates58_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates58_25_checked : candidates edge 4 283 7039 innerTail58_26 best.bound = innerCandidates58_25 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_25 : scan edge (search edge 3) 4 [283] 283 innerTail58_25 best = best := by
  rw [innerTail58_25, scan, if_neg (by decide), innerCandidates58_25_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_26
    (if 3 ≤ innerCandidates58_25.length then search edge 3 [283, 7039] innerCandidates58_25 7322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_26

def innerTail58_24 : List Nat := 6763 :: innerTail58_25
def innerCandidates58_24 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates58_24_checked : candidates edge 4 283 6763 innerTail58_25 best.bound = innerCandidates58_24 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_24 : scan edge (search edge 3) 4 [283] 283 innerTail58_24 best = best := by
  rw [innerTail58_24, scan, if_neg (by decide), innerCandidates58_24_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_25
    (if 3 ≤ innerCandidates58_24.length then search edge 3 [283, 6763] innerCandidates58_24 7046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_25

def innerTail58_23 : List Nat := 6733 :: innerTail58_24
def innerCandidates58_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates58_23_checked : candidates edge 4 283 6733 innerTail58_24 best.bound = innerCandidates58_23 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_23 : scan edge (search edge 3) 4 [283] 283 innerTail58_23 best = best := by
  rw [innerTail58_23, scan, if_neg (by decide), innerCandidates58_23_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_24
    (if 3 ≤ innerCandidates58_23.length then search edge 3 [283, 6733] innerCandidates58_23 7016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_24

def innerTail58_22 : List Nat := 6607 :: innerTail58_23
def innerCandidates58_22 : List Nat := [6763]
set_option maxRecDepth 4096 in
theorem innerCandidates58_22_checked : candidates edge 4 283 6607 innerTail58_23 best.bound = innerCandidates58_22 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_22 : scan edge (search edge 3) 4 [283] 283 innerTail58_22 best = best := by
  rw [innerTail58_22, scan, if_neg (by decide), innerCandidates58_22_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_23
    (if 3 ≤ innerCandidates58_22.length then search edge 3 [283, 6607] innerCandidates58_22 6890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_23

def innerTail58_21 : List Nat := 6397 :: innerTail58_22
def innerCandidates58_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates58_21_checked : candidates edge 4 283 6397 innerTail58_22 best.bound = innerCandidates58_21 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_21 : scan edge (search edge 3) 4 [283] 283 innerTail58_21 best = best := by
  rw [innerTail58_21, scan, if_neg (by decide), innerCandidates58_21_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_22
    (if 3 ≤ innerCandidates58_21.length then search edge 3 [283, 6397] innerCandidates58_21 6680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_22

def innerTail58_20 : List Nat := 5857 :: innerTail58_21
def innerCandidates58_20 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates58_20_checked : candidates edge 4 283 5857 innerTail58_21 best.bound = innerCandidates58_20 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_20 : scan edge (search edge 3) 4 [283] 283 innerTail58_20 best = best := by
  rw [innerTail58_20, scan, if_neg (by decide), innerCandidates58_20_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_21
    (if 3 ≤ innerCandidates58_20.length then search edge 3 [283, 5857] innerCandidates58_20 6140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_21

def innerTail58_19 : List Nat := 5851 :: innerTail58_20
def innerCandidates58_19 : List Nat := [6607]
set_option maxRecDepth 4096 in
theorem innerCandidates58_19_checked : candidates edge 4 283 5851 innerTail58_20 best.bound = innerCandidates58_19 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_19 : scan edge (search edge 3) 4 [283] 283 innerTail58_19 best = best := by
  rw [innerTail58_19, scan, if_neg (by decide), innerCandidates58_19_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_20
    (if 3 ≤ innerCandidates58_19.length then search edge 3 [283, 5851] innerCandidates58_19 6134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_20

def innerTail58_18 : List Nat := 5737 :: innerTail58_19
def innerCandidates58_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates58_18_checked : candidates edge 4 283 5737 innerTail58_19 best.bound = innerCandidates58_18 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_18 : scan edge (search edge 3) 4 [283] 283 innerTail58_18 best = best := by
  rw [innerTail58_18, scan, if_neg (by decide), innerCandidates58_18_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_19
    (if 3 ≤ innerCandidates58_18.length then search edge 3 [283, 5737] innerCandidates58_18 6020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_19

def innerTail58_17 : List Nat := 5653 :: innerTail58_18
def innerCandidates58_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates58_17_checked : candidates edge 4 283 5653 innerTail58_18 best.bound = innerCandidates58_17 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_17 : scan edge (search edge 3) 4 [283] 283 innerTail58_17 best = best := by
  rw [innerTail58_17, scan, if_neg (by decide), innerCandidates58_17_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_18
    (if 3 ≤ innerCandidates58_17.length then search edge 3 [283, 5653] innerCandidates58_17 5936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_18

def innerTail58_16 : List Nat := 4813 :: innerTail58_17
def innerCandidates58_16 : List Nat := [7057]
set_option maxRecDepth 4096 in
theorem innerCandidates58_16_checked : candidates edge 4 283 4813 innerTail58_17 best.bound = innerCandidates58_16 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_16 : scan edge (search edge 3) 4 [283] 283 innerTail58_16 best = best := by
  rw [innerTail58_16, scan, if_neg (by decide), innerCandidates58_16_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_17
    (if 3 ≤ innerCandidates58_16.length then search edge 3 [283, 4813] innerCandidates58_16 5096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_17

def innerTail58_15 : List Nat := 3529 :: innerTail58_16
def innerCandidates58_15 : List Nat := [5851, 6397, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates58_15_checked : candidates edge 4 283 3529 innerTail58_16 best.bound = innerCandidates58_15 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild58_15 : search edge 3 [283, 3529] innerCandidates58_15 3812 best = best := by decide

theorem innerScan58_15 : scan edge (search edge 3) 4 [283] 283 innerTail58_15 best = best := by
  rw [innerTail58_15, scan, if_neg (by decide), innerCandidates58_15_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_16
    (if 3 ≤ innerCandidates58_15.length then search edge 3 [283, 3529] innerCandidates58_15 3812 best else best) = best
  rw [if_pos (by decide), innerChild58_15]
  exact innerScan58_16

def innerTail58_14 : List Nat := 3001 :: innerTail58_15
def innerCandidates58_14 : List Nat := [6607, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates58_14_checked : candidates edge 4 283 3001 innerTail58_15 best.bound = innerCandidates58_14 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_14 : scan edge (search edge 3) 4 [283] 283 innerTail58_14 best = best := by
  rw [innerTail58_14, scan, if_neg (by decide), innerCandidates58_14_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_15
    (if 3 ≤ innerCandidates58_14.length then search edge 3 [283, 3001] innerCandidates58_14 3284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_15

def innerTail58_13 : List Nat := 2503 :: innerTail58_14
def innerCandidates58_13 : List Nat := [6763]
set_option maxRecDepth 4096 in
theorem innerCandidates58_13_checked : candidates edge 4 283 2503 innerTail58_14 best.bound = innerCandidates58_13 := by
  unfold candidates
  have row : edge 2503 = PrimePairCache.smallRow2503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_13 : scan edge (search edge 3) 4 [283] 283 innerTail58_13 best = best := by
  rw [innerTail58_13, scan, if_neg (by decide), innerCandidates58_13_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_14
    (if 3 ≤ innerCandidates58_13.length then search edge 3 [283, 2503] innerCandidates58_13 2786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_14

def innerTail58_12 : List Nat := 2287 :: innerTail58_13
def innerCandidates58_12 : List Nat := [5857]
set_option maxRecDepth 4096 in
theorem innerCandidates58_12_checked : candidates edge 4 283 2287 innerTail58_13 best.bound = innerCandidates58_12 := by
  unfold candidates
  have row : edge 2287 = PrimePairCache.smallRow2287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_12 : scan edge (search edge 3) 4 [283] 283 innerTail58_12 best = best := by
  rw [innerTail58_12, scan, if_neg (by decide), innerCandidates58_12_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_13
    (if 3 ≤ innerCandidates58_12.length then search edge 3 [283, 2287] innerCandidates58_12 2570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_13

def innerTail58_11 : List Nat := 2113 :: innerTail58_12
def innerCandidates58_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates58_11_checked : candidates edge 4 283 2113 innerTail58_12 best.bound = innerCandidates58_11 := by
  unfold candidates
  have row : edge 2113 = PrimePairCache.smallRow2113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_11 : scan edge (search edge 3) 4 [283] 283 innerTail58_11 best = best := by
  rw [innerTail58_11, scan, if_neg (by decide), innerCandidates58_11_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_12
    (if 3 ≤ innerCandidates58_11.length then search edge 3 [283, 2113] innerCandidates58_11 2396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_12

def innerTail58_10 : List Nat := 1657 :: innerTail58_11
def innerCandidates58_10 : List Nat := [6397, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates58_10_checked : candidates edge 4 283 1657 innerTail58_11 best.bound = innerCandidates58_10 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_10 : scan edge (search edge 3) 4 [283] 283 innerTail58_10 best = best := by
  rw [innerTail58_10, scan, if_neg (by decide), innerCandidates58_10_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_11
    (if 3 ≤ innerCandidates58_10.length then search edge 3 [283, 1657] innerCandidates58_10 1940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_11

def innerTail58_9 : List Nat := 1489 :: innerTail58_10
def innerCandidates58_9 : List Nat := [5857]
set_option maxRecDepth 4096 in
theorem innerCandidates58_9_checked : candidates edge 4 283 1489 innerTail58_10 best.bound = innerCandidates58_9 := by
  unfold candidates
  have row : edge 1489 = PrimePairCache.smallRow1489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_9 : scan edge (search edge 3) 4 [283] 283 innerTail58_9 best = best := by
  rw [innerTail58_9, scan, if_neg (by decide), innerCandidates58_9_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_10
    (if 3 ≤ innerCandidates58_9.length then search edge 3 [283, 1489] innerCandidates58_9 1772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_10

def innerTail58_8 : List Nat := 1429 :: innerTail58_9
def innerCandidates58_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates58_8_checked : candidates edge 4 283 1429 innerTail58_9 best.bound = innerCandidates58_8 := by
  unfold candidates
  have row : edge 1429 = PrimePairCache.smallRow1429.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_8 : scan edge (search edge 3) 4 [283] 283 innerTail58_8 best = best := by
  rw [innerTail58_8, scan, if_neg (by decide), innerCandidates58_8_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_9
    (if 3 ≤ innerCandidates58_8.length then search edge 3 [283, 1429] innerCandidates58_8 1712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_9

def innerTail58_7 : List Nat := 1321 :: innerTail58_8
def innerCandidates58_7 : List Nat := [3001, 6607]
set_option maxRecDepth 4096 in
theorem innerCandidates58_7_checked : candidates edge 4 283 1321 innerTail58_8 best.bound = innerCandidates58_7 := by
  unfold candidates
  have row : edge 1321 = PrimePairCache.smallRow1321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_7 : scan edge (search edge 3) 4 [283] 283 innerTail58_7 best = best := by
  rw [innerTail58_7, scan, if_neg (by decide), innerCandidates58_7_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_8
    (if 3 ≤ innerCandidates58_7.length then search edge 3 [283, 1321] innerCandidates58_7 1604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_8

def innerTail58_6 : List Nat := 1237 :: innerTail58_7
def innerCandidates58_6 : List Nat := [1657]
set_option maxRecDepth 4096 in
theorem innerCandidates58_6_checked : candidates edge 4 283 1237 innerTail58_7 best.bound = innerCandidates58_6 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_6 : scan edge (search edge 3) 4 [283] 283 innerTail58_6 best = best := by
  rw [innerTail58_6, scan, if_neg (by decide), innerCandidates58_6_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_7
    (if 3 ≤ innerCandidates58_6.length then search edge 3 [283, 1237] innerCandidates58_6 1520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_7

def innerTail58_5 : List Nat := 1051 :: innerTail58_6
def innerCandidates58_5 : List Nat := [4813, 6607]
set_option maxRecDepth 4096 in
theorem innerCandidates58_5_checked : candidates edge 4 283 1051 innerTail58_6 best.bound = innerCandidates58_5 := by
  unfold candidates
  have row : edge 1051 = PrimePairCache.smallRow1051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_5 : scan edge (search edge 3) 4 [283] 283 innerTail58_5 best = best := by
  rw [innerTail58_5, scan, if_neg (by decide), innerCandidates58_5_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_6
    (if 3 ≤ innerCandidates58_5.length then search edge 3 [283, 1051] innerCandidates58_5 1334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_6

def innerTail58_4 : List Nat := 601 :: innerTail58_5
def innerCandidates58_4 : List Nat := [1321]
set_option maxRecDepth 4096 in
theorem innerCandidates58_4_checked : candidates edge 4 283 601 innerTail58_5 best.bound = innerCandidates58_4 := by
  unfold candidates
  have row : edge 601 = PrimePairCache.smallRow601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_4 : scan edge (search edge 3) 4 [283] 283 innerTail58_4 best = best := by
  rw [innerTail58_4, scan, if_neg (by decide), innerCandidates58_4_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_5
    (if 3 ≤ innerCandidates58_4.length then search edge 3 [283, 601] innerCandidates58_4 884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_5

def innerTail58_3 : List Nat := 541 :: innerTail58_4
def innerCandidates58_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates58_3_checked : candidates edge 4 283 541 innerTail58_4 best.bound = innerCandidates58_3 := by
  unfold candidates
  have row : edge 541 = PrimePairCache.smallRow541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_3 : scan edge (search edge 3) 4 [283] 283 innerTail58_3 best = best := by
  rw [innerTail58_3, scan, if_neg (by decide), innerCandidates58_3_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_4
    (if 3 ≤ innerCandidates58_3.length then search edge 3 [283, 541] innerCandidates58_3 824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_4

def innerTail58_2 : List Nat := 487 :: innerTail58_3
def innerCandidates58_2 : List Nat := [601, 6607]
set_option maxRecDepth 4096 in
theorem innerCandidates58_2_checked : candidates edge 4 283 487 innerTail58_3 best.bound = innerCandidates58_2 := by
  unfold candidates
  have row : edge 487 = PrimePairCache.smallRow487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_2 : scan edge (search edge 3) 4 [283] 283 innerTail58_2 best = best := by
  rw [innerTail58_2, scan, if_neg (by decide), innerCandidates58_2_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_3
    (if 3 ≤ innerCandidates58_2.length then search edge 3 [283, 487] innerCandidates58_2 770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_3

def innerTail58_1 : List Nat := 463 :: innerTail58_2
def innerCandidates58_1 : List Nat := [1489, 3001]
set_option maxRecDepth 4096 in
theorem innerCandidates58_1_checked : candidates edge 4 283 463 innerTail58_2 best.bound = innerCandidates58_1 := by
  unfold candidates
  have row : edge 463 = PrimePairCache.smallRow463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_1 : scan edge (search edge 3) 4 [283] 283 innerTail58_1 best = best := by
  rw [innerTail58_1, scan, if_neg (by decide), innerCandidates58_1_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_2
    (if 3 ≤ innerCandidates58_1.length then search edge 3 [283, 463] innerCandidates58_1 746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_2

def innerTail58_0 : List Nat := 397 :: innerTail58_1
def innerCandidates58_0 : List Nat := [6397]
set_option maxRecDepth 4096 in
theorem innerCandidates58_0_checked : candidates edge 4 283 397 innerTail58_1 best.bound = innerCandidates58_0 := by
  unfold candidates
  have row : edge 397 = PrimePairCache.smallRow397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan58_0 : scan edge (search edge 3) 4 [283] 283 innerTail58_0 best = best := by
  rw [innerTail58_0, scan, if_neg (by decide), innerCandidates58_0_checked]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_1
    (if 3 ≤ innerCandidates58_0.length then search edge 3 [283, 397] innerCandidates58_0 680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan58_1

theorem rootBranch58 : search edge 4 [283] rootCandidates58 283 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [283] 283 innerTail58_0 best = best
  exact innerScan58_0

def rootCandidates57 : List Nat := [419, 509, 557, 653, 719, 797, 857, 971, 1181, 1427, 1559, 1619, 1973, 2207, 2267, 2687, 2819, 3191, 3329, 3677, 3917, 4787, 5009, 5399, 5843, 6029, 6263, 6899, 7229, 7481, 7487, 8111, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates57_checked : candidates edge 5 0 281 tail58 best.bound = rootCandidates57 := by
  unfold candidates
  have row : edge 281 = PrimePairCache.smallRow281.contains := by funext q; rfl
  rw [row]
  decide

def innerTail57_33 : List Nat := []
theorem innerScan57_33 : scan edge (search edge 3) 4 [281] 281 innerTail57_33 best = best := rfl

def innerTail57_32 : List Nat := 8369 :: innerTail57_33
def innerCandidates57_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_32_checked : candidates edge 4 281 8369 innerTail57_33 best.bound = innerCandidates57_32 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_32 : scan edge (search edge 3) 4 [281] 281 innerTail57_32 best = best := by
  rw [innerTail57_32, scan, if_neg (by decide), innerCandidates57_32_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_33
    (if 3 ≤ innerCandidates57_32.length then search edge 3 [281, 8369] innerCandidates57_32 8650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_33

def innerTail57_31 : List Nat := 8111 :: innerTail57_32
def innerCandidates57_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_31_checked : candidates edge 4 281 8111 innerTail57_32 best.bound = innerCandidates57_31 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_31 : scan edge (search edge 3) 4 [281] 281 innerTail57_31 best = best := by
  rw [innerTail57_31, scan, if_neg (by decide), innerCandidates57_31_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_32
    (if 3 ≤ innerCandidates57_31.length then search edge 3 [281, 8111] innerCandidates57_31 8392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_32

def innerTail57_30 : List Nat := 7487 :: innerTail57_31
def innerCandidates57_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_30_checked : candidates edge 4 281 7487 innerTail57_31 best.bound = innerCandidates57_30 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_30 : scan edge (search edge 3) 4 [281] 281 innerTail57_30 best = best := by
  rw [innerTail57_30, scan, if_neg (by decide), innerCandidates57_30_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_31
    (if 3 ≤ innerCandidates57_30.length then search edge 3 [281, 7487] innerCandidates57_30 7768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_31

def innerTail57_29 : List Nat := 7481 :: innerTail57_30
def innerCandidates57_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_29_checked : candidates edge 4 281 7481 innerTail57_30 best.bound = innerCandidates57_29 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_29 : scan edge (search edge 3) 4 [281] 281 innerTail57_29 best = best := by
  rw [innerTail57_29, scan, if_neg (by decide), innerCandidates57_29_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_30
    (if 3 ≤ innerCandidates57_29.length then search edge 3 [281, 7481] innerCandidates57_29 7762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_30

def innerTail57_28 : List Nat := 7229 :: innerTail57_29
def innerCandidates57_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_28_checked : candidates edge 4 281 7229 innerTail57_29 best.bound = innerCandidates57_28 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_28 : scan edge (search edge 3) 4 [281] 281 innerTail57_28 best = best := by
  rw [innerTail57_28, scan, if_neg (by decide), innerCandidates57_28_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_29
    (if 3 ≤ innerCandidates57_28.length then search edge 3 [281, 7229] innerCandidates57_28 7510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_29

def innerTail57_27 : List Nat := 6899 :: innerTail57_28
def innerCandidates57_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_27_checked : candidates edge 4 281 6899 innerTail57_28 best.bound = innerCandidates57_27 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_27 : scan edge (search edge 3) 4 [281] 281 innerTail57_27 best = best := by
  rw [innerTail57_27, scan, if_neg (by decide), innerCandidates57_27_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_28
    (if 3 ≤ innerCandidates57_27.length then search edge 3 [281, 6899] innerCandidates57_27 7180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_28

def innerTail57_26 : List Nat := 6263 :: innerTail57_27
def innerCandidates57_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_26_checked : candidates edge 4 281 6263 innerTail57_27 best.bound = innerCandidates57_26 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_26 : scan edge (search edge 3) 4 [281] 281 innerTail57_26 best = best := by
  rw [innerTail57_26, scan, if_neg (by decide), innerCandidates57_26_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_27
    (if 3 ≤ innerCandidates57_26.length then search edge 3 [281, 6263] innerCandidates57_26 6544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_27

def innerTail57_25 : List Nat := 6029 :: innerTail57_26
def innerCandidates57_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_25_checked : candidates edge 4 281 6029 innerTail57_26 best.bound = innerCandidates57_25 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_25 : scan edge (search edge 3) 4 [281] 281 innerTail57_25 best = best := by
  rw [innerTail57_25, scan, if_neg (by decide), innerCandidates57_25_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_26
    (if 3 ≤ innerCandidates57_25.length then search edge 3 [281, 6029] innerCandidates57_25 6310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_26

def innerTail57_24 : List Nat := 5843 :: innerTail57_25
def innerCandidates57_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_24_checked : candidates edge 4 281 5843 innerTail57_25 best.bound = innerCandidates57_24 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_24 : scan edge (search edge 3) 4 [281] 281 innerTail57_24 best = best := by
  rw [innerTail57_24, scan, if_neg (by decide), innerCandidates57_24_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_25
    (if 3 ≤ innerCandidates57_24.length then search edge 3 [281, 5843] innerCandidates57_24 6124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_25

def innerTail57_23 : List Nat := 5399 :: innerTail57_24
def innerCandidates57_23 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates57_23_checked : candidates edge 4 281 5399 innerTail57_24 best.bound = innerCandidates57_23 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_23 : scan edge (search edge 3) 4 [281] 281 innerTail57_23 best = best := by
  rw [innerTail57_23, scan, if_neg (by decide), innerCandidates57_23_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_24
    (if 3 ≤ innerCandidates57_23.length then search edge 3 [281, 5399] innerCandidates57_23 5680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_24

def innerTail57_22 : List Nat := 5009 :: innerTail57_23
def innerCandidates57_22 : List Nat := [6029]
set_option maxRecDepth 4096 in
theorem innerCandidates57_22_checked : candidates edge 4 281 5009 innerTail57_23 best.bound = innerCandidates57_22 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_22 : scan edge (search edge 3) 4 [281] 281 innerTail57_22 best = best := by
  rw [innerTail57_22, scan, if_neg (by decide), innerCandidates57_22_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_23
    (if 3 ≤ innerCandidates57_22.length then search edge 3 [281, 5009] innerCandidates57_22 5290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_23

def innerTail57_21 : List Nat := 4787 :: innerTail57_22
def innerCandidates57_21 : List Nat := [7229]
set_option maxRecDepth 4096 in
theorem innerCandidates57_21_checked : candidates edge 4 281 4787 innerTail57_22 best.bound = innerCandidates57_21 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_21 : scan edge (search edge 3) 4 [281] 281 innerTail57_21 best = best := by
  rw [innerTail57_21, scan, if_neg (by decide), innerCandidates57_21_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_22
    (if 3 ≤ innerCandidates57_21.length then search edge 3 [281, 4787] innerCandidates57_21 5068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_22

def innerTail57_20 : List Nat := 3917 :: innerTail57_21
def innerCandidates57_20 : List Nat := [5399, 6899]
set_option maxRecDepth 4096 in
theorem innerCandidates57_20_checked : candidates edge 4 281 3917 innerTail57_21 best.bound = innerCandidates57_20 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_20 : scan edge (search edge 3) 4 [281] 281 innerTail57_20 best = best := by
  rw [innerTail57_20, scan, if_neg (by decide), innerCandidates57_20_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_21
    (if 3 ≤ innerCandidates57_20.length then search edge 3 [281, 3917] innerCandidates57_20 4198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_21

def innerTail57_19 : List Nat := 3677 :: innerTail57_20
def innerCandidates57_19 : List Nat := [7229]
set_option maxRecDepth 4096 in
theorem innerCandidates57_19_checked : candidates edge 4 281 3677 innerTail57_20 best.bound = innerCandidates57_19 := by
  unfold candidates
  have row : edge 3677 = SearchData.row3677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_19 : scan edge (search edge 3) 4 [281] 281 innerTail57_19 best = best := by
  rw [innerTail57_19, scan, if_neg (by decide), innerCandidates57_19_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_20
    (if 3 ≤ innerCandidates57_19.length then search edge 3 [281, 3677] innerCandidates57_19 3958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_20

def innerTail57_18 : List Nat := 3329 :: innerTail57_19
def innerCandidates57_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_18_checked : candidates edge 4 281 3329 innerTail57_19 best.bound = innerCandidates57_18 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_18 : scan edge (search edge 3) 4 [281] 281 innerTail57_18 best = best := by
  rw [innerTail57_18, scan, if_neg (by decide), innerCandidates57_18_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_19
    (if 3 ≤ innerCandidates57_18.length then search edge 3 [281, 3329] innerCandidates57_18 3610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_19

def innerTail57_17 : List Nat := 3191 :: innerTail57_18
def innerCandidates57_17 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates57_17_checked : candidates edge 4 281 3191 innerTail57_18 best.bound = innerCandidates57_17 := by
  unfold candidates
  have row : edge 3191 = SearchData.row3191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_17 : scan edge (search edge 3) 4 [281] 281 innerTail57_17 best = best := by
  rw [innerTail57_17, scan, if_neg (by decide), innerCandidates57_17_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_18
    (if 3 ≤ innerCandidates57_17.length then search edge 3 [281, 3191] innerCandidates57_17 3472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_18

def innerTail57_16 : List Nat := 2819 :: innerTail57_17
def innerCandidates57_16 : List Nat := [4787]
set_option maxRecDepth 4096 in
theorem innerCandidates57_16_checked : candidates edge 4 281 2819 innerTail57_17 best.bound = innerCandidates57_16 := by
  unfold candidates
  have row : edge 2819 = SearchData.row2819.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_16 : scan edge (search edge 3) 4 [281] 281 innerTail57_16 best = best := by
  rw [innerTail57_16, scan, if_neg (by decide), innerCandidates57_16_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_17
    (if 3 ≤ innerCandidates57_16.length then search edge 3 [281, 2819] innerCandidates57_16 3100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_17

def innerTail57_15 : List Nat := 2687 :: innerTail57_16
def innerCandidates57_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_15_checked : candidates edge 4 281 2687 innerTail57_16 best.bound = innerCandidates57_15 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_15 : scan edge (search edge 3) 4 [281] 281 innerTail57_15 best = best := by
  rw [innerTail57_15, scan, if_neg (by decide), innerCandidates57_15_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_16
    (if 3 ≤ innerCandidates57_15.length then search edge 3 [281, 2687] innerCandidates57_15 2968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_16

def innerTail57_14 : List Nat := 2267 :: innerTail57_15
def innerCandidates57_14 : List Nat := [5843]
set_option maxRecDepth 4096 in
theorem innerCandidates57_14_checked : candidates edge 4 281 2267 innerTail57_15 best.bound = innerCandidates57_14 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_14 : scan edge (search edge 3) 4 [281] 281 innerTail57_14 best = best := by
  rw [innerTail57_14, scan, if_neg (by decide), innerCandidates57_14_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_15
    (if 3 ≤ innerCandidates57_14.length then search edge 3 [281, 2267] innerCandidates57_14 2548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_15

def innerTail57_13 : List Nat := 2207 :: innerTail57_14
def innerCandidates57_13 : List Nat := [2819]
set_option maxRecDepth 4096 in
theorem innerCandidates57_13_checked : candidates edge 4 281 2207 innerTail57_14 best.bound = innerCandidates57_13 := by
  unfold candidates
  have row : edge 2207 = PrimePairCache.smallRow2207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_13 : scan edge (search edge 3) 4 [281] 281 innerTail57_13 best = best := by
  rw [innerTail57_13, scan, if_neg (by decide), innerCandidates57_13_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_14
    (if 3 ≤ innerCandidates57_13.length then search edge 3 [281, 2207] innerCandidates57_13 2488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_14

def innerTail57_12 : List Nat := 1973 :: innerTail57_13
def innerCandidates57_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_12_checked : candidates edge 4 281 1973 innerTail57_13 best.bound = innerCandidates57_12 := by
  unfold candidates
  have row : edge 1973 = PrimePairCache.smallRow1973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_12 : scan edge (search edge 3) 4 [281] 281 innerTail57_12 best = best := by
  rw [innerTail57_12, scan, if_neg (by decide), innerCandidates57_12_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_13
    (if 3 ≤ innerCandidates57_12.length then search edge 3 [281, 1973] innerCandidates57_12 2254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_13

def innerTail57_11 : List Nat := 1619 :: innerTail57_12
def innerCandidates57_11 : List Nat := [3677]
set_option maxRecDepth 4096 in
theorem innerCandidates57_11_checked : candidates edge 4 281 1619 innerTail57_12 best.bound = innerCandidates57_11 := by
  unfold candidates
  have row : edge 1619 = PrimePairCache.smallRow1619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_11 : scan edge (search edge 3) 4 [281] 281 innerTail57_11 best = best := by
  rw [innerTail57_11, scan, if_neg (by decide), innerCandidates57_11_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_12
    (if 3 ≤ innerCandidates57_11.length then search edge 3 [281, 1619] innerCandidates57_11 1900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_12

def innerTail57_10 : List Nat := 1559 :: innerTail57_11
def innerCandidates57_10 : List Nat := [3191]
set_option maxRecDepth 4096 in
theorem innerCandidates57_10_checked : candidates edge 4 281 1559 innerTail57_11 best.bound = innerCandidates57_10 := by
  unfold candidates
  have row : edge 1559 = PrimePairCache.smallRow1559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_10 : scan edge (search edge 3) 4 [281] 281 innerTail57_10 best = best := by
  rw [innerTail57_10, scan, if_neg (by decide), innerCandidates57_10_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_11
    (if 3 ≤ innerCandidates57_10.length then search edge 3 [281, 1559] innerCandidates57_10 1840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_11

def innerTail57_9 : List Nat := 1427 :: innerTail57_10
def innerCandidates57_9 : List Nat := [5843, 8111]
set_option maxRecDepth 4096 in
theorem innerCandidates57_9_checked : candidates edge 4 281 1427 innerTail57_10 best.bound = innerCandidates57_9 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_9 : scan edge (search edge 3) 4 [281] 281 innerTail57_9 best = best := by
  rw [innerTail57_9, scan, if_neg (by decide), innerCandidates57_9_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_10
    (if 3 ≤ innerCandidates57_9.length then search edge 3 [281, 1427] innerCandidates57_9 1708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_10

def innerTail57_8 : List Nat := 1181 :: innerTail57_9
def innerCandidates57_8 : List Nat := [1427, 2687]
set_option maxRecDepth 4096 in
theorem innerCandidates57_8_checked : candidates edge 4 281 1181 innerTail57_9 best.bound = innerCandidates57_8 := by
  unfold candidates
  have row : edge 1181 = PrimePairCache.smallRow1181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_8 : scan edge (search edge 3) 4 [281] 281 innerTail57_8 best = best := by
  rw [innerTail57_8, scan, if_neg (by decide), innerCandidates57_8_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_9
    (if 3 ≤ innerCandidates57_8.length then search edge 3 [281, 1181] innerCandidates57_8 1462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_9

def innerTail57_7 : List Nat := 971 :: innerTail57_8
def innerCandidates57_7 : List Nat := [5009]
set_option maxRecDepth 4096 in
theorem innerCandidates57_7_checked : candidates edge 4 281 971 innerTail57_8 best.bound = innerCandidates57_7 := by
  unfold candidates
  have row : edge 971 = PrimePairCache.smallRow971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_7 : scan edge (search edge 3) 4 [281] 281 innerTail57_7 best = best := by
  rw [innerTail57_7, scan, if_neg (by decide), innerCandidates57_7_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_8
    (if 3 ≤ innerCandidates57_7.length then search edge 3 [281, 971] innerCandidates57_7 1252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_8

def innerTail57_6 : List Nat := 857 :: innerTail57_7
def innerCandidates57_6 : List Nat := [1973]
set_option maxRecDepth 4096 in
theorem innerCandidates57_6_checked : candidates edge 4 281 857 innerTail57_7 best.bound = innerCandidates57_6 := by
  unfold candidates
  have row : edge 857 = PrimePairCache.smallRow857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_6 : scan edge (search edge 3) 4 [281] 281 innerTail57_6 best = best := by
  rw [innerTail57_6, scan, if_neg (by decide), innerCandidates57_6_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_7
    (if 3 ≤ innerCandidates57_6.length then search edge 3 [281, 857] innerCandidates57_6 1138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_7

def innerTail57_5 : List Nat := 797 :: innerTail57_6
def innerCandidates57_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_5_checked : candidates edge 4 281 797 innerTail57_6 best.bound = innerCandidates57_5 := by
  unfold candidates
  have row : edge 797 = PrimePairCache.smallRow797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_5 : scan edge (search edge 3) 4 [281] 281 innerTail57_5 best = best := by
  rw [innerTail57_5, scan, if_neg (by decide), innerCandidates57_5_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_6
    (if 3 ≤ innerCandidates57_5.length then search edge 3 [281, 797] innerCandidates57_5 1078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_6

def innerTail57_4 : List Nat := 719 :: innerTail57_5
def innerCandidates57_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_4_checked : candidates edge 4 281 719 innerTail57_5 best.bound = innerCandidates57_4 := by
  unfold candidates
  have row : edge 719 = PrimePairCache.smallRow719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_4 : scan edge (search edge 3) 4 [281] 281 innerTail57_4 best = best := by
  rw [innerTail57_4, scan, if_neg (by decide), innerCandidates57_4_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_5
    (if 3 ≤ innerCandidates57_4.length then search edge 3 [281, 719] innerCandidates57_4 1000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_5

def innerTail57_3 : List Nat := 653 :: innerTail57_4
def innerCandidates57_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates57_3_checked : candidates edge 4 281 653 innerTail57_4 best.bound = innerCandidates57_3 := by
  unfold candidates
  have row : edge 653 = PrimePairCache.smallRow653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_3 : scan edge (search edge 3) 4 [281] 281 innerTail57_3 best = best := by
  rw [innerTail57_3, scan, if_neg (by decide), innerCandidates57_3_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_4
    (if 3 ≤ innerCandidates57_3.length then search edge 3 [281, 653] innerCandidates57_3 934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_4

def innerTail57_2 : List Nat := 557 :: innerTail57_3
def innerCandidates57_2 : List Nat := [2687]
set_option maxRecDepth 4096 in
theorem innerCandidates57_2_checked : candidates edge 4 281 557 innerTail57_3 best.bound = innerCandidates57_2 := by
  unfold candidates
  have row : edge 557 = PrimePairCache.smallRow557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_2 : scan edge (search edge 3) 4 [281] 281 innerTail57_2 best = best := by
  rw [innerTail57_2, scan, if_neg (by decide), innerCandidates57_2_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_3
    (if 3 ≤ innerCandidates57_2.length then search edge 3 [281, 557] innerCandidates57_2 838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_3

def innerTail57_1 : List Nat := 509 :: innerTail57_2
def innerCandidates57_1 : List Nat := [797, 1973, 8369]
set_option maxRecDepth 4096 in
theorem innerCandidates57_1_checked : candidates edge 4 281 509 innerTail57_2 best.bound = innerCandidates57_1 := by
  unfold candidates
  have row : edge 509 = PrimePairCache.smallRow509.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild57_1 : search edge 3 [281, 509] innerCandidates57_1 790 best = best := by decide

theorem innerScan57_1 : scan edge (search edge 3) 4 [281] 281 innerTail57_1 best = best := by
  rw [innerTail57_1, scan, if_neg (by decide), innerCandidates57_1_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_2
    (if 3 ≤ innerCandidates57_1.length then search edge 3 [281, 509] innerCandidates57_1 790 best else best) = best
  rw [if_pos (by decide), innerChild57_1]
  exact innerScan57_2

def innerTail57_0 : List Nat := 419 :: innerTail57_1
def innerCandidates57_0 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates57_0_checked : candidates edge 4 281 419 innerTail57_1 best.bound = innerCandidates57_0 := by
  unfold candidates
  have row : edge 419 = PrimePairCache.smallRow419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan57_0 : scan edge (search edge 3) 4 [281] 281 innerTail57_0 best = best := by
  rw [innerTail57_0, scan, if_neg (by decide), innerCandidates57_0_checked]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_1
    (if 3 ≤ innerCandidates57_0.length then search edge 3 [281, 419] innerCandidates57_0 700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan57_1

theorem rootBranch57 : search edge 4 [281] rootCandidates57 281 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [281] 281 innerTail57_0 best = best
  exact innerScan57_0

end ProjectEulerSolutions.PrimePairUpper.Search
