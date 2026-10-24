import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates28 : List Nat := [157, 163, 241, 271, 331, 373, 601, 607, 733, 1033, 1069, 1129, 1987, 2281, 2749, 3001, 3343, 3541, 3673, 4363, 4423, 4759, 5179, 5563, 5683, 5749, 5779, 6271, 6361, 6481, 6571, 6733, 6871, 6949, 7369, 7723, 7753, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates28_checked : candidates edge 5 0 127 tail29 best.bound = rootCandidates28 := by
  unfold candidates
  have row : edge 127 = PrimePairCache.smallRow127.contains := by funext q; rfl
  rw [row]
  decide

def innerTail28_38 : List Nat := []
theorem innerScan28_38 : scan edge (search edge 3) 4 [127] 127 innerTail28_38 best = best := rfl

def innerTail28_37 : List Nat := 7993 :: innerTail28_38
def innerCandidates28_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_37_checked : candidates edge 4 127 7993 innerTail28_38 best.bound = innerCandidates28_37 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_37 : scan edge (search edge 3) 4 [127] 127 innerTail28_37 best = best := by
  rw [innerTail28_37, scan, if_neg (by decide), innerCandidates28_37_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_38
    (if 3 ≤ innerCandidates28_37.length then search edge 3 [127, 7993] innerCandidates28_37 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_38

def innerTail28_36 : List Nat := 7753 :: innerTail28_37
def innerCandidates28_36 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_36_checked : candidates edge 4 127 7753 innerTail28_37 best.bound = innerCandidates28_36 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_36 : scan edge (search edge 3) 4 [127] 127 innerTail28_36 best = best := by
  rw [innerTail28_36, scan, if_neg (by decide), innerCandidates28_36_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_37
    (if 3 ≤ innerCandidates28_36.length then search edge 3 [127, 7753] innerCandidates28_36 7880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_37

def innerTail28_35 : List Nat := 7723 :: innerTail28_36
def innerCandidates28_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_35_checked : candidates edge 4 127 7723 innerTail28_36 best.bound = innerCandidates28_35 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_35 : scan edge (search edge 3) 4 [127] 127 innerTail28_35 best = best := by
  rw [innerTail28_35, scan, if_neg (by decide), innerCandidates28_35_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_36
    (if 3 ≤ innerCandidates28_35.length then search edge 3 [127, 7723] innerCandidates28_35 7850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_36

def innerTail28_34 : List Nat := 7369 :: innerTail28_35
def innerCandidates28_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_34_checked : candidates edge 4 127 7369 innerTail28_35 best.bound = innerCandidates28_34 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_34 : scan edge (search edge 3) 4 [127] 127 innerTail28_34 best = best := by
  rw [innerTail28_34, scan, if_neg (by decide), innerCandidates28_34_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_35
    (if 3 ≤ innerCandidates28_34.length then search edge 3 [127, 7369] innerCandidates28_34 7496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_35

def innerTail28_33 : List Nat := 6949 :: innerTail28_34
def innerCandidates28_33 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates28_33_checked : candidates edge 4 127 6949 innerTail28_34 best.bound = innerCandidates28_33 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_33 : scan edge (search edge 3) 4 [127] 127 innerTail28_33 best = best := by
  rw [innerTail28_33, scan, if_neg (by decide), innerCandidates28_33_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_34
    (if 3 ≤ innerCandidates28_33.length then search edge 3 [127, 6949] innerCandidates28_33 7076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_34

def innerTail28_32 : List Nat := 6871 :: innerTail28_33
def innerCandidates28_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_32_checked : candidates edge 4 127 6871 innerTail28_33 best.bound = innerCandidates28_32 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_32 : scan edge (search edge 3) 4 [127] 127 innerTail28_32 best = best := by
  rw [innerTail28_32, scan, if_neg (by decide), innerCandidates28_32_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_33
    (if 3 ≤ innerCandidates28_32.length then search edge 3 [127, 6871] innerCandidates28_32 6998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_33

def innerTail28_31 : List Nat := 6733 :: innerTail28_32
def innerCandidates28_31 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates28_31_checked : candidates edge 4 127 6733 innerTail28_32 best.bound = innerCandidates28_31 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_31 : scan edge (search edge 3) 4 [127] 127 innerTail28_31 best = best := by
  rw [innerTail28_31, scan, if_neg (by decide), innerCandidates28_31_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_32
    (if 3 ≤ innerCandidates28_31.length then search edge 3 [127, 6733] innerCandidates28_31 6860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_32

def innerTail28_30 : List Nat := 6571 :: innerTail28_31
def innerCandidates28_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_30_checked : candidates edge 4 127 6571 innerTail28_31 best.bound = innerCandidates28_30 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_30 : scan edge (search edge 3) 4 [127] 127 innerTail28_30 best = best := by
  rw [innerTail28_30, scan, if_neg (by decide), innerCandidates28_30_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_31
    (if 3 ≤ innerCandidates28_30.length then search edge 3 [127, 6571] innerCandidates28_30 6698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_31

def innerTail28_29 : List Nat := 6481 :: innerTail28_30
def innerCandidates28_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_29_checked : candidates edge 4 127 6481 innerTail28_30 best.bound = innerCandidates28_29 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_29 : scan edge (search edge 3) 4 [127] 127 innerTail28_29 best = best := by
  rw [innerTail28_29, scan, if_neg (by decide), innerCandidates28_29_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_30
    (if 3 ≤ innerCandidates28_29.length then search edge 3 [127, 6481] innerCandidates28_29 6608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_30

def innerTail28_28 : List Nat := 6361 :: innerTail28_29
def innerCandidates28_28 : List Nat := [6481]
set_option maxRecDepth 4096 in
theorem innerCandidates28_28_checked : candidates edge 4 127 6361 innerTail28_29 best.bound = innerCandidates28_28 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_28 : scan edge (search edge 3) 4 [127] 127 innerTail28_28 best = best := by
  rw [innerTail28_28, scan, if_neg (by decide), innerCandidates28_28_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_29
    (if 3 ≤ innerCandidates28_28.length then search edge 3 [127, 6361] innerCandidates28_28 6488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_29

def innerTail28_27 : List Nat := 6271 :: innerTail28_28
def innerCandidates28_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_27_checked : candidates edge 4 127 6271 innerTail28_28 best.bound = innerCandidates28_27 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_27 : scan edge (search edge 3) 4 [127] 127 innerTail28_27 best = best := by
  rw [innerTail28_27, scan, if_neg (by decide), innerCandidates28_27_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_28
    (if 3 ≤ innerCandidates28_27.length then search edge 3 [127, 6271] innerCandidates28_27 6398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_28

def innerTail28_26 : List Nat := 5779 :: innerTail28_27
def innerCandidates28_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_26_checked : candidates edge 4 127 5779 innerTail28_27 best.bound = innerCandidates28_26 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_26 : scan edge (search edge 3) 4 [127] 127 innerTail28_26 best = best := by
  rw [innerTail28_26, scan, if_neg (by decide), innerCandidates28_26_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_27
    (if 3 ≤ innerCandidates28_26.length then search edge 3 [127, 5779] innerCandidates28_26 5906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_27

def innerTail28_25 : List Nat := 5749 :: innerTail28_26
def innerCandidates28_25 : List Nat := [7753]
set_option maxRecDepth 4096 in
theorem innerCandidates28_25_checked : candidates edge 4 127 5749 innerTail28_26 best.bound = innerCandidates28_25 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_25 : scan edge (search edge 3) 4 [127] 127 innerTail28_25 best = best := by
  rw [innerTail28_25, scan, if_neg (by decide), innerCandidates28_25_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_26
    (if 3 ≤ innerCandidates28_25.length then search edge 3 [127, 5749] innerCandidates28_25 5876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_26

def innerTail28_24 : List Nat := 5683 :: innerTail28_25
def innerCandidates28_24 : List Nat := [6271, 7369]
set_option maxRecDepth 4096 in
theorem innerCandidates28_24_checked : candidates edge 4 127 5683 innerTail28_25 best.bound = innerCandidates28_24 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_24 : scan edge (search edge 3) 4 [127] 127 innerTail28_24 best = best := by
  rw [innerTail28_24, scan, if_neg (by decide), innerCandidates28_24_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_25
    (if 3 ≤ innerCandidates28_24.length then search edge 3 [127, 5683] innerCandidates28_24 5810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_25

def innerTail28_23 : List Nat := 5563 :: innerTail28_24
def innerCandidates28_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_23_checked : candidates edge 4 127 5563 innerTail28_24 best.bound = innerCandidates28_23 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_23 : scan edge (search edge 3) 4 [127] 127 innerTail28_23 best = best := by
  rw [innerTail28_23, scan, if_neg (by decide), innerCandidates28_23_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_24
    (if 3 ≤ innerCandidates28_23.length then search edge 3 [127, 5563] innerCandidates28_23 5690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_24

def innerTail28_22 : List Nat := 5179 :: innerTail28_23
def innerCandidates28_22 : List Nat := [5683]
set_option maxRecDepth 4096 in
theorem innerCandidates28_22_checked : candidates edge 4 127 5179 innerTail28_23 best.bound = innerCandidates28_22 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_22 : scan edge (search edge 3) 4 [127] 127 innerTail28_22 best = best := by
  rw [innerTail28_22, scan, if_neg (by decide), innerCandidates28_22_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_23
    (if 3 ≤ innerCandidates28_22.length then search edge 3 [127, 5179] innerCandidates28_22 5306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_23

def innerTail28_21 : List Nat := 4759 :: innerTail28_22
def innerCandidates28_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_21_checked : candidates edge 4 127 4759 innerTail28_22 best.bound = innerCandidates28_21 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_21 : scan edge (search edge 3) 4 [127] 127 innerTail28_21 best = best := by
  rw [innerTail28_21, scan, if_neg (by decide), innerCandidates28_21_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_22
    (if 3 ≤ innerCandidates28_21.length then search edge 3 [127, 4759] innerCandidates28_21 4886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_22

def innerTail28_20 : List Nat := 4423 :: innerTail28_21
def innerCandidates28_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_20_checked : candidates edge 4 127 4423 innerTail28_21 best.bound = innerCandidates28_20 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_20 : scan edge (search edge 3) 4 [127] 127 innerTail28_20 best = best := by
  rw [innerTail28_20, scan, if_neg (by decide), innerCandidates28_20_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_21
    (if 3 ≤ innerCandidates28_20.length then search edge 3 [127, 4423] innerCandidates28_20 4550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_21

def innerTail28_19 : List Nat := 4363 :: innerTail28_20
def innerCandidates28_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_19_checked : candidates edge 4 127 4363 innerTail28_20 best.bound = innerCandidates28_19 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_19 : scan edge (search edge 3) 4 [127] 127 innerTail28_19 best = best := by
  rw [innerTail28_19, scan, if_neg (by decide), innerCandidates28_19_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_20
    (if 3 ≤ innerCandidates28_19.length then search edge 3 [127, 4363] innerCandidates28_19 4490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_20

def innerTail28_18 : List Nat := 3673 :: innerTail28_19
def innerCandidates28_18 : List Nat := [4759]
set_option maxRecDepth 4096 in
theorem innerCandidates28_18_checked : candidates edge 4 127 3673 innerTail28_19 best.bound = innerCandidates28_18 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_18 : scan edge (search edge 3) 4 [127] 127 innerTail28_18 best = best := by
  rw [innerTail28_18, scan, if_neg (by decide), innerCandidates28_18_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_19
    (if 3 ≤ innerCandidates28_18.length then search edge 3 [127, 3673] innerCandidates28_18 3800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_19

def innerTail28_17 : List Nat := 3541 :: innerTail28_18
def innerCandidates28_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_17_checked : candidates edge 4 127 3541 innerTail28_18 best.bound = innerCandidates28_17 := by
  unfold candidates
  have row : edge 3541 = SearchData.row3541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_17 : scan edge (search edge 3) 4 [127] 127 innerTail28_17 best = best := by
  rw [innerTail28_17, scan, if_neg (by decide), innerCandidates28_17_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_18
    (if 3 ≤ innerCandidates28_17.length then search edge 3 [127, 3541] innerCandidates28_17 3668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_18

def innerTail28_16 : List Nat := 3343 :: innerTail28_17
def innerCandidates28_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_16_checked : candidates edge 4 127 3343 innerTail28_17 best.bound = innerCandidates28_16 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_16 : scan edge (search edge 3) 4 [127] 127 innerTail28_16 best = best := by
  rw [innerTail28_16, scan, if_neg (by decide), innerCandidates28_16_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_17
    (if 3 ≤ innerCandidates28_16.length then search edge 3 [127, 3343] innerCandidates28_16 3470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_17

def innerTail28_15 : List Nat := 3001 :: innerTail28_16
def innerCandidates28_15 : List Nat := [5749, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates28_15_checked : candidates edge 4 127 3001 innerTail28_16 best.bound = innerCandidates28_15 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_15 : scan edge (search edge 3) 4 [127] 127 innerTail28_15 best = best := by
  rw [innerTail28_15, scan, if_neg (by decide), innerCandidates28_15_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_16
    (if 3 ≤ innerCandidates28_15.length then search edge 3 [127, 3001] innerCandidates28_15 3128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_16

def innerTail28_14 : List Nat := 2749 :: innerTail28_15
def innerCandidates28_14 : List Nat := [3541]
set_option maxRecDepth 4096 in
theorem innerCandidates28_14_checked : candidates edge 4 127 2749 innerTail28_15 best.bound = innerCandidates28_14 := by
  unfold candidates
  have row : edge 2749 = SearchData.row2749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_14 : scan edge (search edge 3) 4 [127] 127 innerTail28_14 best = best := by
  rw [innerTail28_14, scan, if_neg (by decide), innerCandidates28_14_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_15
    (if 3 ≤ innerCandidates28_14.length then search edge 3 [127, 2749] innerCandidates28_14 2876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_15

def innerTail28_13 : List Nat := 2281 :: innerTail28_14
def innerCandidates28_13 : List Nat := [6361, 6481]
set_option maxRecDepth 4096 in
theorem innerCandidates28_13_checked : candidates edge 4 127 2281 innerTail28_14 best.bound = innerCandidates28_13 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_13 : scan edge (search edge 3) 4 [127] 127 innerTail28_13 best = best := by
  rw [innerTail28_13, scan, if_neg (by decide), innerCandidates28_13_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_14
    (if 3 ≤ innerCandidates28_13.length then search edge 3 [127, 2281] innerCandidates28_13 2408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_14

def innerTail28_12 : List Nat := 1987 :: innerTail28_13
def innerCandidates28_12 : List Nat := [4423, 5563, 6481]
set_option maxRecDepth 4096 in
theorem innerCandidates28_12_checked : candidates edge 4 127 1987 innerTail28_13 best.bound = innerCandidates28_12 := by
  unfold candidates
  have row : edge 1987 = PrimePairCache.smallRow1987.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild28_12 : search edge 3 [127, 1987] innerCandidates28_12 2114 best = best := by decide

theorem innerScan28_12 : scan edge (search edge 3) 4 [127] 127 innerTail28_12 best = best := by
  rw [innerTail28_12, scan, if_neg (by decide), innerCandidates28_12_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_13
    (if 3 ≤ innerCandidates28_12.length then search edge 3 [127, 1987] innerCandidates28_12 2114 best else best) = best
  rw [if_pos (by decide), innerChild28_12]
  exact innerScan28_13

def innerTail28_11 : List Nat := 1129 :: innerTail28_12
def innerCandidates28_11 : List Nat := [3001, 6871]
set_option maxRecDepth 4096 in
theorem innerCandidates28_11_checked : candidates edge 4 127 1129 innerTail28_12 best.bound = innerCandidates28_11 := by
  unfold candidates
  have row : edge 1129 = PrimePairCache.smallRow1129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_11 : scan edge (search edge 3) 4 [127] 127 innerTail28_11 best = best := by
  rw [innerTail28_11, scan, if_neg (by decide), innerCandidates28_11_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_12
    (if 3 ≤ innerCandidates28_11.length then search edge 3 [127, 1129] innerCandidates28_11 1256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_12

def innerTail28_10 : List Nat := 1069 :: innerTail28_11
def innerCandidates28_10 : List Nat := [2281, 5683, 7993]
set_option maxRecDepth 4096 in
theorem innerCandidates28_10_checked : candidates edge 4 127 1069 innerTail28_11 best.bound = innerCandidates28_10 := by
  unfold candidates
  have row : edge 1069 = PrimePairCache.smallRow1069.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild28_10 : search edge 3 [127, 1069] innerCandidates28_10 1196 best = best := by decide

theorem innerScan28_10 : scan edge (search edge 3) 4 [127] 127 innerTail28_10 best = best := by
  rw [innerTail28_10, scan, if_neg (by decide), innerCandidates28_10_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_11
    (if 3 ≤ innerCandidates28_10.length then search edge 3 [127, 1069] innerCandidates28_10 1196 best else best) = best
  rw [if_pos (by decide), innerChild28_10]
  exact innerScan28_11

def innerTail28_9 : List Nat := 1033 :: innerTail28_10
def innerCandidates28_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_9_checked : candidates edge 4 127 1033 innerTail28_10 best.bound = innerCandidates28_9 := by
  unfold candidates
  have row : edge 1033 = PrimePairCache.smallRow1033.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_9 : scan edge (search edge 3) 4 [127] 127 innerTail28_9 best = best := by
  rw [innerTail28_9, scan, if_neg (by decide), innerCandidates28_9_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_10
    (if 3 ≤ innerCandidates28_9.length then search edge 3 [127, 1033] innerCandidates28_9 1160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_10

def innerTail28_8 : List Nat := 733 :: innerTail28_9
def innerCandidates28_8 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates28_8_checked : candidates edge 4 127 733 innerTail28_9 best.bound = innerCandidates28_8 := by
  unfold candidates
  have row : edge 733 = PrimePairCache.smallRow733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_8 : scan edge (search edge 3) 4 [127] 127 innerTail28_8 best = best := by
  rw [innerTail28_8, scan, if_neg (by decide), innerCandidates28_8_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_9
    (if 3 ≤ innerCandidates28_8.length then search edge 3 [127, 733] innerCandidates28_8 860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_9

def innerTail28_7 : List Nat := 607 :: innerTail28_8
def innerCandidates28_7 : List Nat := [4363]
set_option maxRecDepth 4096 in
theorem innerCandidates28_7_checked : candidates edge 4 127 607 innerTail28_8 best.bound = innerCandidates28_7 := by
  unfold candidates
  have row : edge 607 = PrimePairCache.smallRow607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_7 : scan edge (search edge 3) 4 [127] 127 innerTail28_7 best = best := by
  rw [innerTail28_7, scan, if_neg (by decide), innerCandidates28_7_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_8
    (if 3 ≤ innerCandidates28_7.length then search edge 3 [127, 607] innerCandidates28_7 734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_8

def innerTail28_6 : List Nat := 601 :: innerTail28_7
def innerCandidates28_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_6_checked : candidates edge 4 127 601 innerTail28_7 best.bound = innerCandidates28_6 := by
  unfold candidates
  have row : edge 601 = PrimePairCache.smallRow601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_6 : scan edge (search edge 3) 4 [127] 127 innerTail28_6 best = best := by
  rw [innerTail28_6, scan, if_neg (by decide), innerCandidates28_6_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_7
    (if 3 ≤ innerCandidates28_6.length then search edge 3 [127, 601] innerCandidates28_6 728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_7

def innerTail28_5 : List Nat := 373 :: innerTail28_6
def innerCandidates28_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates28_5_checked : candidates edge 4 127 373 innerTail28_6 best.bound = innerCandidates28_5 := by
  unfold candidates
  have row : edge 373 = PrimePairCache.smallRow373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_5 : scan edge (search edge 3) 4 [127] 127 innerTail28_5 best = best := by
  rw [innerTail28_5, scan, if_neg (by decide), innerCandidates28_5_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_6
    (if 3 ≤ innerCandidates28_5.length then search edge 3 [127, 373] innerCandidates28_5 500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_6

def innerTail28_4 : List Nat := 331 :: innerTail28_5
def innerCandidates28_4 : List Nat := [5779, 6361, 6949]
set_option maxRecDepth 4096 in
theorem innerCandidates28_4_checked : candidates edge 4 127 331 innerTail28_5 best.bound = innerCandidates28_4 := by
  unfold candidates
  have row : edge 331 = PrimePairCache.smallRow331.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild28_4 : search edge 3 [127, 331] innerCandidates28_4 458 best = best := by decide

theorem innerScan28_4 : scan edge (search edge 3) 4 [127] 127 innerTail28_4 best = best := by
  rw [innerTail28_4, scan, if_neg (by decide), innerCandidates28_4_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_5
    (if 3 ≤ innerCandidates28_4.length then search edge 3 [127, 331] innerCandidates28_4 458 best else best) = best
  rw [if_pos (by decide), innerChild28_4]
  exact innerScan28_5

def innerTail28_3 : List Nat := 271 :: innerTail28_4
def innerCandidates28_3 : List Nat := [3541, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates28_3_checked : candidates edge 4 127 271 innerTail28_4 best.bound = innerCandidates28_3 := by
  unfold candidates
  have row : edge 271 = PrimePairCache.smallRow271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_3 : scan edge (search edge 3) 4 [127] 127 innerTail28_3 best = best := by
  rw [innerTail28_3, scan, if_neg (by decide), innerCandidates28_3_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_4
    (if 3 ≤ innerCandidates28_3.length then search edge 3 [127, 271] innerCandidates28_3 398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_4

def innerTail28_2 : List Nat := 241 :: innerTail28_3
def innerCandidates28_2 : List Nat := [271, 601, 6361]
set_option maxRecDepth 4096 in
theorem innerCandidates28_2_checked : candidates edge 4 127 241 innerTail28_3 best.bound = innerCandidates28_2 := by
  unfold candidates
  have row : edge 241 = PrimePairCache.smallRow241.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild28_2 : search edge 3 [127, 241] innerCandidates28_2 368 best = best := by decide

theorem innerScan28_2 : scan edge (search edge 3) 4 [127] 127 innerTail28_2 best = best := by
  rw [innerTail28_2, scan, if_neg (by decide), innerCandidates28_2_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_3
    (if 3 ≤ innerCandidates28_2.length then search edge 3 [127, 241] innerCandidates28_2 368 best else best) = best
  rw [if_pos (by decide), innerChild28_2]
  exact innerScan28_3

def innerTail28_1 : List Nat := 163 :: innerTail28_2
def innerCandidates28_1 : List Nat := [5563]
set_option maxRecDepth 4096 in
theorem innerCandidates28_1_checked : candidates edge 4 127 163 innerTail28_2 best.bound = innerCandidates28_1 := by
  unfold candidates
  have row : edge 163 = PrimePairCache.smallRow163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_1 : scan edge (search edge 3) 4 [127] 127 innerTail28_1 best = best := by
  rw [innerTail28_1, scan, if_neg (by decide), innerCandidates28_1_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_2
    (if 3 ≤ innerCandidates28_1.length then search edge 3 [127, 163] innerCandidates28_1 290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_2

def innerTail28_0 : List Nat := 157 :: innerTail28_1
def innerCandidates28_0 : List Nat := [733, 2749]
set_option maxRecDepth 4096 in
theorem innerCandidates28_0_checked : candidates edge 4 127 157 innerTail28_1 best.bound = innerCandidates28_0 := by
  unfold candidates
  have row : edge 157 = PrimePairCache.smallRow157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan28_0 : scan edge (search edge 3) 4 [127] 127 innerTail28_0 best = best := by
  rw [innerTail28_0, scan, if_neg (by decide), innerCandidates28_0_checked]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_1
    (if 3 ≤ innerCandidates28_0.length then search edge 3 [127, 157] innerCandidates28_0 284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan28_1

theorem rootBranch28 : search edge 4 [127] rootCandidates28 127 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [127] 127 innerTail28_0 best = best
  exact innerScan28_0

end ProjectEulerSolutions.PrimePairUpper.Search
