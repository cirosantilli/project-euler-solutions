import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates188 : List Nat := [1231, 1543, 1699, 1759, 1801, 2383, 2473, 2803, 3163, 3253, 3673, 3853, 3919, 3943, 4093, 4423, 4597, 4951, 5179, 5437, 6037, 6067, 6361, 6673, 6691, 7243, 7507, 7573, 7681, 7741, 7993, 8017]
set_option maxRecDepth 4096 in
theorem rootCandidates188_checked : candidates edge 5 0 1153 tail189 best.bound = rootCandidates188 := by
  unfold candidates
  have row : edge 1153 = PrimePairCache.smallRow1153.contains := by funext q; rfl
  rw [row]
  decide

def innerTail188_32 : List Nat := []
theorem innerScan188_32 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_32 best = best := rfl

def innerTail188_31 : List Nat := 8017 :: innerTail188_32
def innerCandidates188_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_31_checked : candidates edge 4 1153 8017 innerTail188_32 best.bound = innerCandidates188_31 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_31 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_31 best = best := by
  rw [innerTail188_31, scan, if_neg (by decide), innerCandidates188_31_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_32
    (if 3 ≤ innerCandidates188_31.length then search edge 3 [1153, 8017] innerCandidates188_31 9170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_32

def innerTail188_30 : List Nat := 7993 :: innerTail188_31
def innerCandidates188_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_30_checked : candidates edge 4 1153 7993 innerTail188_31 best.bound = innerCandidates188_30 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_30 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_30 best = best := by
  rw [innerTail188_30, scan, if_neg (by decide), innerCandidates188_30_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_31
    (if 3 ≤ innerCandidates188_30.length then search edge 3 [1153, 7993] innerCandidates188_30 9146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_31

def innerTail188_29 : List Nat := 7741 :: innerTail188_30
def innerCandidates188_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_29_checked : candidates edge 4 1153 7741 innerTail188_30 best.bound = innerCandidates188_29 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_29 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_29 best = best := by
  rw [innerTail188_29, scan, if_neg (by decide), innerCandidates188_29_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_30
    (if 3 ≤ innerCandidates188_29.length then search edge 3 [1153, 7741] innerCandidates188_29 8894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_30

def innerTail188_28 : List Nat := 7681 :: innerTail188_29
def innerCandidates188_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_28_checked : candidates edge 4 1153 7681 innerTail188_29 best.bound = innerCandidates188_28 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_28 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_28 best = best := by
  rw [innerTail188_28, scan, if_neg (by decide), innerCandidates188_28_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_29
    (if 3 ≤ innerCandidates188_28.length then search edge 3 [1153, 7681] innerCandidates188_28 8834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_29

def innerTail188_27 : List Nat := 7573 :: innerTail188_28
def innerCandidates188_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_27_checked : candidates edge 4 1153 7573 innerTail188_28 best.bound = innerCandidates188_27 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_27 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_27 best = best := by
  rw [innerTail188_27, scan, if_neg (by decide), innerCandidates188_27_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_28
    (if 3 ≤ innerCandidates188_27.length then search edge 3 [1153, 7573] innerCandidates188_27 8726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_28

def innerTail188_26 : List Nat := 7507 :: innerTail188_27
def innerCandidates188_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_26_checked : candidates edge 4 1153 7507 innerTail188_27 best.bound = innerCandidates188_26 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_26 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_26 best = best := by
  rw [innerTail188_26, scan, if_neg (by decide), innerCandidates188_26_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_27
    (if 3 ≤ innerCandidates188_26.length then search edge 3 [1153, 7507] innerCandidates188_26 8660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_27

def innerTail188_25 : List Nat := 7243 :: innerTail188_26
def innerCandidates188_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_25_checked : candidates edge 4 1153 7243 innerTail188_26 best.bound = innerCandidates188_25 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_25 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_25 best = best := by
  rw [innerTail188_25, scan, if_neg (by decide), innerCandidates188_25_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_26
    (if 3 ≤ innerCandidates188_25.length then search edge 3 [1153, 7243] innerCandidates188_25 8396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_26

def innerTail188_24 : List Nat := 6691 :: innerTail188_25
def innerCandidates188_24 : List Nat := [7243, 7507]
set_option maxRecDepth 4096 in
theorem innerCandidates188_24_checked : candidates edge 4 1153 6691 innerTail188_25 best.bound = innerCandidates188_24 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_24 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_24 best = best := by
  rw [innerTail188_24, scan, if_neg (by decide), innerCandidates188_24_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_25
    (if 3 ≤ innerCandidates188_24.length then search edge 3 [1153, 6691] innerCandidates188_24 7844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_25

def innerTail188_23 : List Nat := 6673 :: innerTail188_24
def innerCandidates188_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_23_checked : candidates edge 4 1153 6673 innerTail188_24 best.bound = innerCandidates188_23 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_23 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_23 best = best := by
  rw [innerTail188_23, scan, if_neg (by decide), innerCandidates188_23_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_24
    (if 3 ≤ innerCandidates188_23.length then search edge 3 [1153, 6673] innerCandidates188_23 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_24

def innerTail188_22 : List Nat := 6361 :: innerTail188_23
def innerCandidates188_22 : List Nat := [6673, 7507]
set_option maxRecDepth 4096 in
theorem innerCandidates188_22_checked : candidates edge 4 1153 6361 innerTail188_23 best.bound = innerCandidates188_22 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_22 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_22 best = best := by
  rw [innerTail188_22, scan, if_neg (by decide), innerCandidates188_22_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_23
    (if 3 ≤ innerCandidates188_22.length then search edge 3 [1153, 6361] innerCandidates188_22 7514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_23

def innerTail188_21 : List Nat := 6067 :: innerTail188_22
def innerCandidates188_21 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates188_21_checked : candidates edge 4 1153 6067 innerTail188_22 best.bound = innerCandidates188_21 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_21 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_21 best = best := by
  rw [innerTail188_21, scan, if_neg (by decide), innerCandidates188_21_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_22
    (if 3 ≤ innerCandidates188_21.length then search edge 3 [1153, 6067] innerCandidates188_21 7220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_22

def innerTail188_20 : List Nat := 6037 :: innerTail188_21
def innerCandidates188_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_20_checked : candidates edge 4 1153 6037 innerTail188_21 best.bound = innerCandidates188_20 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_20 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_20 best = best := by
  rw [innerTail188_20, scan, if_neg (by decide), innerCandidates188_20_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_21
    (if 3 ≤ innerCandidates188_20.length then search edge 3 [1153, 6037] innerCandidates188_20 7190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_21

def innerTail188_19 : List Nat := 5437 :: innerTail188_20
def innerCandidates188_19 : List Nat := [8017]
set_option maxRecDepth 4096 in
theorem innerCandidates188_19_checked : candidates edge 4 1153 5437 innerTail188_20 best.bound = innerCandidates188_19 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_19 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_19 best = best := by
  rw [innerTail188_19, scan, if_neg (by decide), innerCandidates188_19_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_20
    (if 3 ≤ innerCandidates188_19.length then search edge 3 [1153, 5437] innerCandidates188_19 6590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_20

def innerTail188_18 : List Nat := 5179 :: innerTail188_19
def innerCandidates188_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_18_checked : candidates edge 4 1153 5179 innerTail188_19 best.bound = innerCandidates188_18 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_18 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_18 best = best := by
  rw [innerTail188_18, scan, if_neg (by decide), innerCandidates188_18_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_19
    (if 3 ≤ innerCandidates188_18.length then search edge 3 [1153, 5179] innerCandidates188_18 6332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_19

def innerTail188_17 : List Nat := 4951 :: innerTail188_18
def innerCandidates188_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_17_checked : candidates edge 4 1153 4951 innerTail188_18 best.bound = innerCandidates188_17 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_17 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_17 best = best := by
  rw [innerTail188_17, scan, if_neg (by decide), innerCandidates188_17_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_18
    (if 3 ≤ innerCandidates188_17.length then search edge 3 [1153, 4951] innerCandidates188_17 6104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_18

def innerTail188_16 : List Nat := 4597 :: innerTail188_17
def innerCandidates188_16 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates188_16_checked : candidates edge 4 1153 4597 innerTail188_17 best.bound = innerCandidates188_16 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_16 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_16 best = best := by
  rw [innerTail188_16, scan, if_neg (by decide), innerCandidates188_16_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_17
    (if 3 ≤ innerCandidates188_16.length then search edge 3 [1153, 4597] innerCandidates188_16 5750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_17

def innerTail188_15 : List Nat := 4423 :: innerTail188_16
def innerCandidates188_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_15_checked : candidates edge 4 1153 4423 innerTail188_16 best.bound = innerCandidates188_15 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_15 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_15 best = best := by
  rw [innerTail188_15, scan, if_neg (by decide), innerCandidates188_15_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_16
    (if 3 ≤ innerCandidates188_15.length then search edge 3 [1153, 4423] innerCandidates188_15 5576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_16

def innerTail188_14 : List Nat := 4093 :: innerTail188_15
def innerCandidates188_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_14_checked : candidates edge 4 1153 4093 innerTail188_15 best.bound = innerCandidates188_14 := by
  unfold candidates
  have row : edge 4093 = SearchData.row4093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_14 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_14 best = best := by
  rw [innerTail188_14, scan, if_neg (by decide), innerCandidates188_14_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_15
    (if 3 ≤ innerCandidates188_14.length then search edge 3 [1153, 4093] innerCandidates188_14 5246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_15

def innerTail188_13 : List Nat := 3943 :: innerTail188_14
def innerCandidates188_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates188_13_checked : candidates edge 4 1153 3943 innerTail188_14 best.bound = innerCandidates188_13 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_13 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_13 best = best := by
  rw [innerTail188_13, scan, if_neg (by decide), innerCandidates188_13_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_14
    (if 3 ≤ innerCandidates188_13.length then search edge 3 [1153, 3943] innerCandidates188_13 5096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_14

def innerTail188_12 : List Nat := 3919 :: innerTail188_13
def innerCandidates188_12 : List Nat := [5179]
set_option maxRecDepth 4096 in
theorem innerCandidates188_12_checked : candidates edge 4 1153 3919 innerTail188_13 best.bound = innerCandidates188_12 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_12 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_12 best = best := by
  rw [innerTail188_12, scan, if_neg (by decide), innerCandidates188_12_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_13
    (if 3 ≤ innerCandidates188_12.length then search edge 3 [1153, 3919] innerCandidates188_12 5072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_13

def innerTail188_11 : List Nat := 3853 :: innerTail188_12
def innerCandidates188_11 : List Nat := [6361, 7243]
set_option maxRecDepth 4096 in
theorem innerCandidates188_11_checked : candidates edge 4 1153 3853 innerTail188_12 best.bound = innerCandidates188_11 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_11 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_11 best = best := by
  rw [innerTail188_11, scan, if_neg (by decide), innerCandidates188_11_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_12
    (if 3 ≤ innerCandidates188_11.length then search edge 3 [1153, 3853] innerCandidates188_11 5006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_12

def innerTail188_10 : List Nat := 3673 :: innerTail188_11
def innerCandidates188_10 : List Nat := [6691]
set_option maxRecDepth 4096 in
theorem innerCandidates188_10_checked : candidates edge 4 1153 3673 innerTail188_11 best.bound = innerCandidates188_10 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_10 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_10 best = best := by
  rw [innerTail188_10, scan, if_neg (by decide), innerCandidates188_10_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_11
    (if 3 ≤ innerCandidates188_10.length then search edge 3 [1153, 3673] innerCandidates188_10 4826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_11

def innerTail188_9 : List Nat := 3253 :: innerTail188_10
def innerCandidates188_9 : List Nat := [6067]
set_option maxRecDepth 4096 in
theorem innerCandidates188_9_checked : candidates edge 4 1153 3253 innerTail188_10 best.bound = innerCandidates188_9 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_9 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_9 best = best := by
  rw [innerTail188_9, scan, if_neg (by decide), innerCandidates188_9_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_10
    (if 3 ≤ innerCandidates188_9.length then search edge 3 [1153, 3253] innerCandidates188_9 4406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_10

def innerTail188_8 : List Nat := 3163 :: innerTail188_9
def innerCandidates188_8 : List Nat := [6037]
set_option maxRecDepth 4096 in
theorem innerCandidates188_8_checked : candidates edge 4 1153 3163 innerTail188_9 best.bound = innerCandidates188_8 := by
  unfold candidates
  have row : edge 3163 = SearchData.row3163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_8 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_8 best = best := by
  rw [innerTail188_8, scan, if_neg (by decide), innerCandidates188_8_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_9
    (if 3 ≤ innerCandidates188_8.length then search edge 3 [1153, 3163] innerCandidates188_8 4316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_9

def innerTail188_7 : List Nat := 2803 :: innerTail188_8
def innerCandidates188_7 : List Nat := [4423, 6361, 7993]
set_option maxRecDepth 4096 in
theorem innerCandidates188_7_checked : candidates edge 4 1153 2803 innerTail188_8 best.bound = innerCandidates188_7 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild188_7 : search edge 3 [1153, 2803] innerCandidates188_7 3956 best = best := by decide

theorem innerScan188_7 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_7 best = best := by
  rw [innerTail188_7, scan, if_neg (by decide), innerCandidates188_7_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_8
    (if 3 ≤ innerCandidates188_7.length then search edge 3 [1153, 2803] innerCandidates188_7 3956 best else best) = best
  rw [if_pos (by decide), innerChild188_7]
  exact innerScan188_8

def innerTail188_6 : List Nat := 2473 :: innerTail188_7
def innerCandidates188_6 : List Nat := [5437]
set_option maxRecDepth 4096 in
theorem innerCandidates188_6_checked : candidates edge 4 1153 2473 innerTail188_7 best.bound = innerCandidates188_6 := by
  unfold candidates
  have row : edge 2473 = PrimePairCache.smallRow2473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_6 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_6 best = best := by
  rw [innerTail188_6, scan, if_neg (by decide), innerCandidates188_6_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_7
    (if 3 ≤ innerCandidates188_6.length then search edge 3 [1153, 2473] innerCandidates188_6 3626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_7

def innerTail188_5 : List Nat := 2383 :: innerTail188_6
def innerCandidates188_5 : List Nat := [7573]
set_option maxRecDepth 4096 in
theorem innerCandidates188_5_checked : candidates edge 4 1153 2383 innerTail188_6 best.bound = innerCandidates188_5 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_5 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_5 best = best := by
  rw [innerTail188_5, scan, if_neg (by decide), innerCandidates188_5_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_6
    (if 3 ≤ innerCandidates188_5.length then search edge 3 [1153, 2383] innerCandidates188_5 3536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_6

def innerTail188_4 : List Nat := 1801 :: innerTail188_5
def innerCandidates188_4 : List Nat := [3253]
set_option maxRecDepth 4096 in
theorem innerCandidates188_4_checked : candidates edge 4 1153 1801 innerTail188_5 best.bound = innerCandidates188_4 := by
  unfold candidates
  have row : edge 1801 = PrimePairCache.smallRow1801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_4 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_4 best = best := by
  rw [innerTail188_4, scan, if_neg (by decide), innerCandidates188_4_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_5
    (if 3 ≤ innerCandidates188_4.length then search edge 3 [1153, 1801] innerCandidates188_4 2954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_5

def innerTail188_3 : List Nat := 1759 :: innerTail188_4
def innerCandidates188_3 : List Nat := [6361]
set_option maxRecDepth 4096 in
theorem innerCandidates188_3_checked : candidates edge 4 1153 1759 innerTail188_4 best.bound = innerCandidates188_3 := by
  unfold candidates
  have row : edge 1759 = PrimePairCache.smallRow1759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_3 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_3 best = best := by
  rw [innerTail188_3, scan, if_neg (by decide), innerCandidates188_3_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_4
    (if 3 ≤ innerCandidates188_3.length then search edge 3 [1153, 1759] innerCandidates188_3 2912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_4

def innerTail188_2 : List Nat := 1699 :: innerTail188_3
def innerCandidates188_2 : List Nat := [1759, 3853, 6691, 7507]
set_option maxRecDepth 4096 in
theorem innerCandidates188_2_checked : candidates edge 4 1153 1699 innerTail188_3 best.bound = innerCandidates188_2 := by
  unfold candidates
  have row : edge 1699 = PrimePairCache.smallRow1699.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild188_2 : search edge 3 [1153, 1699] innerCandidates188_2 2852 best = best := by decide

theorem innerScan188_2 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_2 best = best := by
  rw [innerTail188_2, scan, if_neg (by decide), innerCandidates188_2_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_3
    (if 3 ≤ innerCandidates188_2.length then search edge 3 [1153, 1699] innerCandidates188_2 2852 best else best) = best
  rw [if_pos (by decide), innerChild188_2]
  exact innerScan188_3

def innerTail188_1 : List Nat := 1543 :: innerTail188_2
def innerCandidates188_1 : List Nat := [4093, 4423]
set_option maxRecDepth 4096 in
theorem innerCandidates188_1_checked : candidates edge 4 1153 1543 innerTail188_2 best.bound = innerCandidates188_1 := by
  unfold candidates
  have row : edge 1543 = PrimePairCache.smallRow1543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_1 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_1 best = best := by
  rw [innerTail188_1, scan, if_neg (by decide), innerCandidates188_1_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_2
    (if 3 ≤ innerCandidates188_1.length then search edge 3 [1153, 1543] innerCandidates188_1 2696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_2

def innerTail188_0 : List Nat := 1231 :: innerTail188_1
def innerCandidates188_0 : List Nat := [4597, 6037]
set_option maxRecDepth 4096 in
theorem innerCandidates188_0_checked : candidates edge 4 1153 1231 innerTail188_1 best.bound = innerCandidates188_0 := by
  unfold candidates
  have row : edge 1231 = PrimePairCache.smallRow1231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan188_0 : scan edge (search edge 3) 4 [1153] 1153 innerTail188_0 best = best := by
  rw [innerTail188_0, scan, if_neg (by decide), innerCandidates188_0_checked]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_1
    (if 3 ≤ innerCandidates188_0.length then search edge 3 [1153, 1231] innerCandidates188_0 2384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan188_1

theorem rootBranch188 : search edge 4 [1153] rootCandidates188 1153 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1153] 1153 innerTail188_0 best = best
  exact innerScan188_0

def rootCandidates187 : List Nat := [1217, 1277, 1361, 1433, 1481, 1559, 1607, 1823, 2399, 2789, 3371, 3527, 4001, 4127, 4337, 4871, 5333, 5417, 5897, 6101, 6473, 6521, 6863, 7019, 7577]
set_option maxRecDepth 4096 in
theorem rootCandidates187_checked : candidates edge 5 0 1151 tail188 best.bound = rootCandidates187 := by
  unfold candidates
  have row : edge 1151 = PrimePairCache.smallRow1151.contains := by funext q; rfl
  rw [row]
  decide

def innerTail187_25 : List Nat := []
theorem innerScan187_25 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_25 best = best := rfl

def innerTail187_24 : List Nat := 7577 :: innerTail187_25
def innerCandidates187_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_24_checked : candidates edge 4 1151 7577 innerTail187_25 best.bound = innerCandidates187_24 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_24 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_24 best = best := by
  rw [innerTail187_24, scan, if_neg (by decide), innerCandidates187_24_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_25
    (if 3 ≤ innerCandidates187_24.length then search edge 3 [1151, 7577] innerCandidates187_24 8728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_25

def innerTail187_23 : List Nat := 7019 :: innerTail187_24
def innerCandidates187_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_23_checked : candidates edge 4 1151 7019 innerTail187_24 best.bound = innerCandidates187_23 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_23 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_23 best = best := by
  rw [innerTail187_23, scan, if_neg (by decide), innerCandidates187_23_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_24
    (if 3 ≤ innerCandidates187_23.length then search edge 3 [1151, 7019] innerCandidates187_23 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_24

def innerTail187_22 : List Nat := 6863 :: innerTail187_23
def innerCandidates187_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_22_checked : candidates edge 4 1151 6863 innerTail187_23 best.bound = innerCandidates187_22 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_22 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_22 best = best := by
  rw [innerTail187_22, scan, if_neg (by decide), innerCandidates187_22_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_23
    (if 3 ≤ innerCandidates187_22.length then search edge 3 [1151, 6863] innerCandidates187_22 8014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_23

def innerTail187_21 : List Nat := 6521 :: innerTail187_22
def innerCandidates187_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_21_checked : candidates edge 4 1151 6521 innerTail187_22 best.bound = innerCandidates187_21 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_21 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_21 best = best := by
  rw [innerTail187_21, scan, if_neg (by decide), innerCandidates187_21_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_22
    (if 3 ≤ innerCandidates187_21.length then search edge 3 [1151, 6521] innerCandidates187_21 7672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_22

def innerTail187_20 : List Nat := 6473 :: innerTail187_21
def innerCandidates187_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_20_checked : candidates edge 4 1151 6473 innerTail187_21 best.bound = innerCandidates187_20 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_20 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_20 best = best := by
  rw [innerTail187_20, scan, if_neg (by decide), innerCandidates187_20_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_21
    (if 3 ≤ innerCandidates187_20.length then search edge 3 [1151, 6473] innerCandidates187_20 7624 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_21

def innerTail187_19 : List Nat := 6101 :: innerTail187_20
def innerCandidates187_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_19_checked : candidates edge 4 1151 6101 innerTail187_20 best.bound = innerCandidates187_19 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_19 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_19 best = best := by
  rw [innerTail187_19, scan, if_neg (by decide), innerCandidates187_19_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_20
    (if 3 ≤ innerCandidates187_19.length then search edge 3 [1151, 6101] innerCandidates187_19 7252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_20

def innerTail187_18 : List Nat := 5897 :: innerTail187_19
def innerCandidates187_18 : List Nat := [7019]
set_option maxRecDepth 4096 in
theorem innerCandidates187_18_checked : candidates edge 4 1151 5897 innerTail187_19 best.bound = innerCandidates187_18 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_18 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_18 best = best := by
  rw [innerTail187_18, scan, if_neg (by decide), innerCandidates187_18_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_19
    (if 3 ≤ innerCandidates187_18.length then search edge 3 [1151, 5897] innerCandidates187_18 7048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_19

def innerTail187_17 : List Nat := 5417 :: innerTail187_18
def innerCandidates187_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_17_checked : candidates edge 4 1151 5417 innerTail187_18 best.bound = innerCandidates187_17 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_17 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_17 best = best := by
  rw [innerTail187_17, scan, if_neg (by decide), innerCandidates187_17_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_18
    (if 3 ≤ innerCandidates187_17.length then search edge 3 [1151, 5417] innerCandidates187_17 6568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_18

def innerTail187_16 : List Nat := 5333 :: innerTail187_17
def innerCandidates187_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_16_checked : candidates edge 4 1151 5333 innerTail187_17 best.bound = innerCandidates187_16 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_16 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_16 best = best := by
  rw [innerTail187_16, scan, if_neg (by decide), innerCandidates187_16_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_17
    (if 3 ≤ innerCandidates187_16.length then search edge 3 [1151, 5333] innerCandidates187_16 6484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_17

def innerTail187_15 : List Nat := 4871 :: innerTail187_16
def innerCandidates187_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_15_checked : candidates edge 4 1151 4871 innerTail187_16 best.bound = innerCandidates187_15 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_15 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_15 best = best := by
  rw [innerTail187_15, scan, if_neg (by decide), innerCandidates187_15_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_16
    (if 3 ≤ innerCandidates187_15.length then search edge 3 [1151, 4871] innerCandidates187_15 6022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_16

def innerTail187_14 : List Nat := 4337 :: innerTail187_15
def innerCandidates187_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_14_checked : candidates edge 4 1151 4337 innerTail187_15 best.bound = innerCandidates187_14 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_14 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_14 best = best := by
  rw [innerTail187_14, scan, if_neg (by decide), innerCandidates187_14_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_15
    (if 3 ≤ innerCandidates187_14.length then search edge 3 [1151, 4337] innerCandidates187_14 5488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_15

def innerTail187_13 : List Nat := 4127 :: innerTail187_14
def innerCandidates187_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_13_checked : candidates edge 4 1151 4127 innerTail187_14 best.bound = innerCandidates187_13 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_13 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_13 best = best := by
  rw [innerTail187_13, scan, if_neg (by decide), innerCandidates187_13_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_14
    (if 3 ≤ innerCandidates187_13.length then search edge 3 [1151, 4127] innerCandidates187_13 5278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_14

def innerTail187_12 : List Nat := 4001 :: innerTail187_13
def innerCandidates187_12 : List Nat := [7577]
set_option maxRecDepth 4096 in
theorem innerCandidates187_12_checked : candidates edge 4 1151 4001 innerTail187_13 best.bound = innerCandidates187_12 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_12 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_12 best = best := by
  rw [innerTail187_12, scan, if_neg (by decide), innerCandidates187_12_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_13
    (if 3 ≤ innerCandidates187_12.length then search edge 3 [1151, 4001] innerCandidates187_12 5152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_13

def innerTail187_11 : List Nat := 3527 :: innerTail187_12
def innerCandidates187_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_11_checked : candidates edge 4 1151 3527 innerTail187_12 best.bound = innerCandidates187_11 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_11 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_11 best = best := by
  rw [innerTail187_11, scan, if_neg (by decide), innerCandidates187_11_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_12
    (if 3 ≤ innerCandidates187_11.length then search edge 3 [1151, 3527] innerCandidates187_11 4678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_12

def innerTail187_10 : List Nat := 3371 :: innerTail187_11
def innerCandidates187_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_10_checked : candidates edge 4 1151 3371 innerTail187_11 best.bound = innerCandidates187_10 := by
  unfold candidates
  have row : edge 3371 = SearchData.row3371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_10 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_10 best = best := by
  rw [innerTail187_10, scan, if_neg (by decide), innerCandidates187_10_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_11
    (if 3 ≤ innerCandidates187_10.length then search edge 3 [1151, 3371] innerCandidates187_10 4522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_11

def innerTail187_9 : List Nat := 2789 :: innerTail187_10
def innerCandidates187_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_9_checked : candidates edge 4 1151 2789 innerTail187_10 best.bound = innerCandidates187_9 := by
  unfold candidates
  have row : edge 2789 = SearchData.row2789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_9 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_9 best = best := by
  rw [innerTail187_9, scan, if_neg (by decide), innerCandidates187_9_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_10
    (if 3 ≤ innerCandidates187_9.length then search edge 3 [1151, 2789] innerCandidates187_9 3940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_10

def innerTail187_8 : List Nat := 2399 :: innerTail187_9
def innerCandidates187_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_8_checked : candidates edge 4 1151 2399 innerTail187_9 best.bound = innerCandidates187_8 := by
  unfold candidates
  have row : edge 2399 = PrimePairCache.smallRow2399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_8 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_8 best = best := by
  rw [innerTail187_8, scan, if_neg (by decide), innerCandidates187_8_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_9
    (if 3 ≤ innerCandidates187_8.length then search edge 3 [1151, 2399] innerCandidates187_8 3550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_9

def innerTail187_7 : List Nat := 1823 :: innerTail187_8
def innerCandidates187_7 : List Nat := [4001]
set_option maxRecDepth 4096 in
theorem innerCandidates187_7_checked : candidates edge 4 1151 1823 innerTail187_8 best.bound = innerCandidates187_7 := by
  unfold candidates
  have row : edge 1823 = PrimePairCache.smallRow1823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_7 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_7 best = best := by
  rw [innerTail187_7, scan, if_neg (by decide), innerCandidates187_7_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_8
    (if 3 ≤ innerCandidates187_7.length then search edge 3 [1151, 1823] innerCandidates187_7 2974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_8

def innerTail187_6 : List Nat := 1607 :: innerTail187_7
def innerCandidates187_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_6_checked : candidates edge 4 1151 1607 innerTail187_7 best.bound = innerCandidates187_6 := by
  unfold candidates
  have row : edge 1607 = PrimePairCache.smallRow1607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_6 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_6 best = best := by
  rw [innerTail187_6, scan, if_neg (by decide), innerCandidates187_6_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_7
    (if 3 ≤ innerCandidates187_6.length then search edge 3 [1151, 1607] innerCandidates187_6 2758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_7

def innerTail187_5 : List Nat := 1559 :: innerTail187_6
def innerCandidates187_5 : List Nat := [5897]
set_option maxRecDepth 4096 in
theorem innerCandidates187_5_checked : candidates edge 4 1151 1559 innerTail187_6 best.bound = innerCandidates187_5 := by
  unfold candidates
  have row : edge 1559 = PrimePairCache.smallRow1559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_5 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_5 best = best := by
  rw [innerTail187_5, scan, if_neg (by decide), innerCandidates187_5_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_6
    (if 3 ≤ innerCandidates187_5.length then search edge 3 [1151, 1559] innerCandidates187_5 2710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_6

def innerTail187_4 : List Nat := 1481 :: innerTail187_5
def innerCandidates187_4 : List Nat := [4127]
set_option maxRecDepth 4096 in
theorem innerCandidates187_4_checked : candidates edge 4 1151 1481 innerTail187_5 best.bound = innerCandidates187_4 := by
  unfold candidates
  have row : edge 1481 = PrimePairCache.smallRow1481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_4 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_4 best = best := by
  rw [innerTail187_4, scan, if_neg (by decide), innerCandidates187_4_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_5
    (if 3 ≤ innerCandidates187_4.length then search edge 3 [1151, 1481] innerCandidates187_4 2632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_5

def innerTail187_3 : List Nat := 1433 :: innerTail187_4
def innerCandidates187_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_3_checked : candidates edge 4 1151 1433 innerTail187_4 best.bound = innerCandidates187_3 := by
  unfold candidates
  have row : edge 1433 = PrimePairCache.smallRow1433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_3 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_3 best = best := by
  rw [innerTail187_3, scan, if_neg (by decide), innerCandidates187_3_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_4
    (if 3 ≤ innerCandidates187_3.length then search edge 3 [1151, 1433] innerCandidates187_3 2584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_4

def innerTail187_2 : List Nat := 1361 :: innerTail187_3
def innerCandidates187_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates187_2_checked : candidates edge 4 1151 1361 innerTail187_3 best.bound = innerCandidates187_2 := by
  unfold candidates
  have row : edge 1361 = PrimePairCache.smallRow1361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_2 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_2 best = best := by
  rw [innerTail187_2, scan, if_neg (by decide), innerCandidates187_2_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_3
    (if 3 ≤ innerCandidates187_2.length then search edge 3 [1151, 1361] innerCandidates187_2 2512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_3

def innerTail187_1 : List Nat := 1277 :: innerTail187_2
def innerCandidates187_1 : List Nat := [7577]
set_option maxRecDepth 4096 in
theorem innerCandidates187_1_checked : candidates edge 4 1151 1277 innerTail187_2 best.bound = innerCandidates187_1 := by
  unfold candidates
  have row : edge 1277 = PrimePairCache.smallRow1277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan187_1 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_1 best = best := by
  rw [innerTail187_1, scan, if_neg (by decide), innerCandidates187_1_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_2
    (if 3 ≤ innerCandidates187_1.length then search edge 3 [1151, 1277] innerCandidates187_1 2428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan187_2

def innerTail187_0 : List Nat := 1217 :: innerTail187_1
def innerCandidates187_0 : List Nat := [1361, 1433, 7019]
set_option maxRecDepth 4096 in
theorem innerCandidates187_0_checked : candidates edge 4 1151 1217 innerTail187_1 best.bound = innerCandidates187_0 := by
  unfold candidates
  have row : edge 1217 = PrimePairCache.smallRow1217.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild187_0 : search edge 3 [1151, 1217] innerCandidates187_0 2368 best = best := by decide

theorem innerScan187_0 : scan edge (search edge 3) 4 [1151] 1151 innerTail187_0 best = best := by
  rw [innerTail187_0, scan, if_neg (by decide), innerCandidates187_0_checked]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_1
    (if 3 ≤ innerCandidates187_0.length then search edge 3 [1151, 1217] innerCandidates187_0 2368 best else best) = best
  rw [if_pos (by decide), innerChild187_0]
  exact innerScan187_1

theorem rootBranch187 : search edge 4 [1151] rootCandidates187 1151 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1151] 1151 innerTail187_0 best = best
  exact innerScan187_0

end ProjectEulerSolutions.PrimePairUpper.Search
