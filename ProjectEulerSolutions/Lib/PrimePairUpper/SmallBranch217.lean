import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates68 : List Nat := [359, 443, 797, 1433, 1481, 1499, 1877, 2213, 2687, 2741, 3203, 3371, 3701, 3779, 4337, 4397, 4493, 4673, 4919, 4967, 5087, 5573, 5849, 5879, 6131, 6389, 6551, 6791, 7517, 7559, 7907, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates68_checked : candidates edge 5 0 353 tail69 best.bound = rootCandidates68 := by
  unfold candidates
  have row : edge 353 = PrimePairCache.smallRow353.contains := by funext q; rfl
  rw [row]
  decide

def innerTail68_32 : List Nat := []
theorem innerScan68_32 : scan edge (search edge 3) 4 [353] 353 innerTail68_32 best = best := rfl

def innerTail68_31 : List Nat := 8219 :: innerTail68_32
def innerCandidates68_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_31_checked : candidates edge 4 353 8219 innerTail68_32 best.bound = innerCandidates68_31 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_31 : scan edge (search edge 3) 4 [353] 353 innerTail68_31 best = best := by
  rw [innerTail68_31, scan, if_neg (by decide), innerCandidates68_31_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_32
    (if 3 ≤ innerCandidates68_31.length then search edge 3 [353, 8219] innerCandidates68_31 8572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_32

def innerTail68_30 : List Nat := 7907 :: innerTail68_31
def innerCandidates68_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_30_checked : candidates edge 4 353 7907 innerTail68_31 best.bound = innerCandidates68_30 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_30 : scan edge (search edge 3) 4 [353] 353 innerTail68_30 best = best := by
  rw [innerTail68_30, scan, if_neg (by decide), innerCandidates68_30_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_31
    (if 3 ≤ innerCandidates68_30.length then search edge 3 [353, 7907] innerCandidates68_30 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_31

def innerTail68_29 : List Nat := 7559 :: innerTail68_30
def innerCandidates68_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_29_checked : candidates edge 4 353 7559 innerTail68_30 best.bound = innerCandidates68_29 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_29 : scan edge (search edge 3) 4 [353] 353 innerTail68_29 best = best := by
  rw [innerTail68_29, scan, if_neg (by decide), innerCandidates68_29_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_30
    (if 3 ≤ innerCandidates68_29.length then search edge 3 [353, 7559] innerCandidates68_29 7912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_30

def innerTail68_28 : List Nat := 7517 :: innerTail68_29
def innerCandidates68_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_28_checked : candidates edge 4 353 7517 innerTail68_29 best.bound = innerCandidates68_28 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_28 : scan edge (search edge 3) 4 [353] 353 innerTail68_28 best = best := by
  rw [innerTail68_28, scan, if_neg (by decide), innerCandidates68_28_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_29
    (if 3 ≤ innerCandidates68_28.length then search edge 3 [353, 7517] innerCandidates68_28 7870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_29

def innerTail68_27 : List Nat := 6791 :: innerTail68_28
def innerCandidates68_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_27_checked : candidates edge 4 353 6791 innerTail68_28 best.bound = innerCandidates68_27 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_27 : scan edge (search edge 3) 4 [353] 353 innerTail68_27 best = best := by
  rw [innerTail68_27, scan, if_neg (by decide), innerCandidates68_27_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_28
    (if 3 ≤ innerCandidates68_27.length then search edge 3 [353, 6791] innerCandidates68_27 7144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_28

def innerTail68_26 : List Nat := 6551 :: innerTail68_27
def innerCandidates68_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_26_checked : candidates edge 4 353 6551 innerTail68_27 best.bound = innerCandidates68_26 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_26 : scan edge (search edge 3) 4 [353] 353 innerTail68_26 best = best := by
  rw [innerTail68_26, scan, if_neg (by decide), innerCandidates68_26_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_27
    (if 3 ≤ innerCandidates68_26.length then search edge 3 [353, 6551] innerCandidates68_26 6904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_27

def innerTail68_25 : List Nat := 6389 :: innerTail68_26
def innerCandidates68_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_25_checked : candidates edge 4 353 6389 innerTail68_26 best.bound = innerCandidates68_25 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_25 : scan edge (search edge 3) 4 [353] 353 innerTail68_25 best = best := by
  rw [innerTail68_25, scan, if_neg (by decide), innerCandidates68_25_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_26
    (if 3 ≤ innerCandidates68_25.length then search edge 3 [353, 6389] innerCandidates68_25 6742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_26

def innerTail68_24 : List Nat := 6131 :: innerTail68_25
def innerCandidates68_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_24_checked : candidates edge 4 353 6131 innerTail68_25 best.bound = innerCandidates68_24 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_24 : scan edge (search edge 3) 4 [353] 353 innerTail68_24 best = best := by
  rw [innerTail68_24, scan, if_neg (by decide), innerCandidates68_24_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_25
    (if 3 ≤ innerCandidates68_24.length then search edge 3 [353, 6131] innerCandidates68_24 6484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_25

def innerTail68_23 : List Nat := 5879 :: innerTail68_24
def innerCandidates68_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_23_checked : candidates edge 4 353 5879 innerTail68_24 best.bound = innerCandidates68_23 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_23 : scan edge (search edge 3) 4 [353] 353 innerTail68_23 best = best := by
  rw [innerTail68_23, scan, if_neg (by decide), innerCandidates68_23_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_24
    (if 3 ≤ innerCandidates68_23.length then search edge 3 [353, 5879] innerCandidates68_23 6232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_24

def innerTail68_22 : List Nat := 5849 :: innerTail68_23
def innerCandidates68_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_22_checked : candidates edge 4 353 5849 innerTail68_23 best.bound = innerCandidates68_22 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_22 : scan edge (search edge 3) 4 [353] 353 innerTail68_22 best = best := by
  rw [innerTail68_22, scan, if_neg (by decide), innerCandidates68_22_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_23
    (if 3 ≤ innerCandidates68_22.length then search edge 3 [353, 5849] innerCandidates68_22 6202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_23

def innerTail68_21 : List Nat := 5573 :: innerTail68_22
def innerCandidates68_21 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates68_21_checked : candidates edge 4 353 5573 innerTail68_22 best.bound = innerCandidates68_21 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_21 : scan edge (search edge 3) 4 [353] 353 innerTail68_21 best = best := by
  rw [innerTail68_21, scan, if_neg (by decide), innerCandidates68_21_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_22
    (if 3 ≤ innerCandidates68_21.length then search edge 3 [353, 5573] innerCandidates68_21 5926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_22

def innerTail68_20 : List Nat := 5087 :: innerTail68_21
def innerCandidates68_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_20_checked : candidates edge 4 353 5087 innerTail68_21 best.bound = innerCandidates68_20 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_20 : scan edge (search edge 3) 4 [353] 353 innerTail68_20 best = best := by
  rw [innerTail68_20, scan, if_neg (by decide), innerCandidates68_20_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_21
    (if 3 ≤ innerCandidates68_20.length then search edge 3 [353, 5087] innerCandidates68_20 5440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_21

def innerTail68_19 : List Nat := 4967 :: innerTail68_20
def innerCandidates68_19 : List Nat := [5573, 5849, 8219]
set_option maxRecDepth 4096 in
theorem innerCandidates68_19_checked : candidates edge 4 353 4967 innerTail68_20 best.bound = innerCandidates68_19 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild68_19 : search edge 3 [353, 4967] innerCandidates68_19 5320 best = best := by decide

theorem innerScan68_19 : scan edge (search edge 3) 4 [353] 353 innerTail68_19 best = best := by
  rw [innerTail68_19, scan, if_neg (by decide), innerCandidates68_19_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_20
    (if 3 ≤ innerCandidates68_19.length then search edge 3 [353, 4967] innerCandidates68_19 5320 best else best) = best
  rw [if_pos (by decide), innerChild68_19]
  exact innerScan68_20

def innerTail68_18 : List Nat := 4919 :: innerTail68_19
def innerCandidates68_18 : List Nat := [7559]
set_option maxRecDepth 4096 in
theorem innerCandidates68_18_checked : candidates edge 4 353 4919 innerTail68_19 best.bound = innerCandidates68_18 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_18 : scan edge (search edge 3) 4 [353] 353 innerTail68_18 best = best := by
  rw [innerTail68_18, scan, if_neg (by decide), innerCandidates68_18_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_19
    (if 3 ≤ innerCandidates68_18.length then search edge 3 [353, 4919] innerCandidates68_18 5272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_19

def innerTail68_17 : List Nat := 4673 :: innerTail68_18
def innerCandidates68_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_17_checked : candidates edge 4 353 4673 innerTail68_18 best.bound = innerCandidates68_17 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_17 : scan edge (search edge 3) 4 [353] 353 innerTail68_17 best = best := by
  rw [innerTail68_17, scan, if_neg (by decide), innerCandidates68_17_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_18
    (if 3 ≤ innerCandidates68_17.length then search edge 3 [353, 4673] innerCandidates68_17 5026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_18

def innerTail68_16 : List Nat := 4493 :: innerTail68_17
def innerCandidates68_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_16_checked : candidates edge 4 353 4493 innerTail68_17 best.bound = innerCandidates68_16 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_16 : scan edge (search edge 3) 4 [353] 353 innerTail68_16 best = best := by
  rw [innerTail68_16, scan, if_neg (by decide), innerCandidates68_16_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_17
    (if 3 ≤ innerCandidates68_16.length then search edge 3 [353, 4493] innerCandidates68_16 4846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_17

def innerTail68_15 : List Nat := 4397 :: innerTail68_16
def innerCandidates68_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_15_checked : candidates edge 4 353 4397 innerTail68_16 best.bound = innerCandidates68_15 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_15 : scan edge (search edge 3) 4 [353] 353 innerTail68_15 best = best := by
  rw [innerTail68_15, scan, if_neg (by decide), innerCandidates68_15_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_16
    (if 3 ≤ innerCandidates68_15.length then search edge 3 [353, 4397] innerCandidates68_15 4750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_16

def innerTail68_14 : List Nat := 4337 :: innerTail68_15
def innerCandidates68_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_14_checked : candidates edge 4 353 4337 innerTail68_15 best.bound = innerCandidates68_14 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_14 : scan edge (search edge 3) 4 [353] 353 innerTail68_14 best = best := by
  rw [innerTail68_14, scan, if_neg (by decide), innerCandidates68_14_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_15
    (if 3 ≤ innerCandidates68_14.length then search edge 3 [353, 4337] innerCandidates68_14 4690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_15

def innerTail68_13 : List Nat := 3779 :: innerTail68_14
def innerCandidates68_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_13_checked : candidates edge 4 353 3779 innerTail68_14 best.bound = innerCandidates68_13 := by
  unfold candidates
  have row : edge 3779 = SearchData.row3779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_13 : scan edge (search edge 3) 4 [353] 353 innerTail68_13 best = best := by
  rw [innerTail68_13, scan, if_neg (by decide), innerCandidates68_13_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_14
    (if 3 ≤ innerCandidates68_13.length then search edge 3 [353, 3779] innerCandidates68_13 4132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_14

def innerTail68_12 : List Nat := 3701 :: innerTail68_13
def innerCandidates68_12 : List Nat := [4493, 4919]
set_option maxRecDepth 4096 in
theorem innerCandidates68_12_checked : candidates edge 4 353 3701 innerTail68_13 best.bound = innerCandidates68_12 := by
  unfold candidates
  have row : edge 3701 = SearchData.row3701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_12 : scan edge (search edge 3) 4 [353] 353 innerTail68_12 best = best := by
  rw [innerTail68_12, scan, if_neg (by decide), innerCandidates68_12_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_13
    (if 3 ≤ innerCandidates68_12.length then search edge 3 [353, 3701] innerCandidates68_12 4054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_13

def innerTail68_11 : List Nat := 3371 :: innerTail68_12
def innerCandidates68_11 : List Nat := [5849]
set_option maxRecDepth 4096 in
theorem innerCandidates68_11_checked : candidates edge 4 353 3371 innerTail68_12 best.bound = innerCandidates68_11 := by
  unfold candidates
  have row : edge 3371 = SearchData.row3371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_11 : scan edge (search edge 3) 4 [353] 353 innerTail68_11 best = best := by
  rw [innerTail68_11, scan, if_neg (by decide), innerCandidates68_11_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_12
    (if 3 ≤ innerCandidates68_11.length then search edge 3 [353, 3371] innerCandidates68_11 3724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_12

def innerTail68_10 : List Nat := 3203 :: innerTail68_11
def innerCandidates68_10 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates68_10_checked : candidates edge 4 353 3203 innerTail68_11 best.bound = innerCandidates68_10 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_10 : scan edge (search edge 3) 4 [353] 353 innerTail68_10 best = best := by
  rw [innerTail68_10, scan, if_neg (by decide), innerCandidates68_10_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_11
    (if 3 ≤ innerCandidates68_10.length then search edge 3 [353, 3203] innerCandidates68_10 3556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_11

def innerTail68_9 : List Nat := 2741 :: innerTail68_10
def innerCandidates68_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_9_checked : candidates edge 4 353 2741 innerTail68_10 best.bound = innerCandidates68_9 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_9 : scan edge (search edge 3) 4 [353] 353 innerTail68_9 best = best := by
  rw [innerTail68_9, scan, if_neg (by decide), innerCandidates68_9_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_10
    (if 3 ≤ innerCandidates68_9.length then search edge 3 [353, 2741] innerCandidates68_9 3094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_10

def innerTail68_8 : List Nat := 2687 :: innerTail68_9
def innerCandidates68_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_8_checked : candidates edge 4 353 2687 innerTail68_9 best.bound = innerCandidates68_8 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_8 : scan edge (search edge 3) 4 [353] 353 innerTail68_8 best = best := by
  rw [innerTail68_8, scan, if_neg (by decide), innerCandidates68_8_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_9
    (if 3 ≤ innerCandidates68_8.length then search edge 3 [353, 2687] innerCandidates68_8 3040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_9

def innerTail68_7 : List Nat := 2213 :: innerTail68_8
def innerCandidates68_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_7_checked : candidates edge 4 353 2213 innerTail68_8 best.bound = innerCandidates68_7 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_7 : scan edge (search edge 3) 4 [353] 353 innerTail68_7 best = best := by
  rw [innerTail68_7, scan, if_neg (by decide), innerCandidates68_7_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_8
    (if 3 ≤ innerCandidates68_7.length then search edge 3 [353, 2213] innerCandidates68_7 2566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_8

def innerTail68_6 : List Nat := 1877 :: innerTail68_7
def innerCandidates68_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_6_checked : candidates edge 4 353 1877 innerTail68_7 best.bound = innerCandidates68_6 := by
  unfold candidates
  have row : edge 1877 = PrimePairCache.smallRow1877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_6 : scan edge (search edge 3) 4 [353] 353 innerTail68_6 best = best := by
  rw [innerTail68_6, scan, if_neg (by decide), innerCandidates68_6_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_7
    (if 3 ≤ innerCandidates68_6.length then search edge 3 [353, 1877] innerCandidates68_6 2230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_7

def innerTail68_5 : List Nat := 1499 :: innerTail68_6
def innerCandidates68_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_5_checked : candidates edge 4 353 1499 innerTail68_6 best.bound = innerCandidates68_5 := by
  unfold candidates
  have row : edge 1499 = PrimePairCache.smallRow1499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_5 : scan edge (search edge 3) 4 [353] 353 innerTail68_5 best = best := by
  rw [innerTail68_5, scan, if_neg (by decide), innerCandidates68_5_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_6
    (if 3 ≤ innerCandidates68_5.length then search edge 3 [353, 1499] innerCandidates68_5 1852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_6

def innerTail68_4 : List Nat := 1481 :: innerTail68_5
def innerCandidates68_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_4_checked : candidates edge 4 353 1481 innerTail68_5 best.bound = innerCandidates68_4 := by
  unfold candidates
  have row : edge 1481 = PrimePairCache.smallRow1481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_4 : scan edge (search edge 3) 4 [353] 353 innerTail68_4 best = best := by
  rw [innerTail68_4, scan, if_neg (by decide), innerCandidates68_4_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_5
    (if 3 ≤ innerCandidates68_4.length then search edge 3 [353, 1481] innerCandidates68_4 1834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_5

def innerTail68_3 : List Nat := 1433 :: innerTail68_4
def innerCandidates68_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates68_3_checked : candidates edge 4 353 1433 innerTail68_4 best.bound = innerCandidates68_3 := by
  unfold candidates
  have row : edge 1433 = PrimePairCache.smallRow1433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_3 : scan edge (search edge 3) 4 [353] 353 innerTail68_3 best = best := by
  rw [innerTail68_3, scan, if_neg (by decide), innerCandidates68_3_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_4
    (if 3 ≤ innerCandidates68_3.length then search edge 3 [353, 1433] innerCandidates68_3 1786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_4

def innerTail68_2 : List Nat := 797 :: innerTail68_3
def innerCandidates68_2 : List Nat := [6389]
set_option maxRecDepth 4096 in
theorem innerCandidates68_2_checked : candidates edge 4 353 797 innerTail68_3 best.bound = innerCandidates68_2 := by
  unfold candidates
  have row : edge 797 = PrimePairCache.smallRow797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan68_2 : scan edge (search edge 3) 4 [353] 353 innerTail68_2 best = best := by
  rw [innerTail68_2, scan, if_neg (by decide), innerCandidates68_2_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_3
    (if 3 ≤ innerCandidates68_2.length then search edge 3 [353, 797] innerCandidates68_2 1150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan68_3

def innerTail68_1 : List Nat := 443 :: innerTail68_2
def innerCandidates68_1 : List Nat := [3203, 4673, 6389]
set_option maxRecDepth 4096 in
theorem innerCandidates68_1_checked : candidates edge 4 353 443 innerTail68_2 best.bound = innerCandidates68_1 := by
  unfold candidates
  have row : edge 443 = PrimePairCache.smallRow443.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild68_1 : search edge 3 [353, 443] innerCandidates68_1 796 best = best := by decide

theorem innerScan68_1 : scan edge (search edge 3) 4 [353] 353 innerTail68_1 best = best := by
  rw [innerTail68_1, scan, if_neg (by decide), innerCandidates68_1_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_2
    (if 3 ≤ innerCandidates68_1.length then search edge 3 [353, 443] innerCandidates68_1 796 best else best) = best
  rw [if_pos (by decide), innerChild68_1]
  exact innerScan68_2

def innerTail68_0 : List Nat := 359 :: innerTail68_1
def innerCandidates68_0 : List Nat := [1499, 3203, 7907]
set_option maxRecDepth 4096 in
theorem innerCandidates68_0_checked : candidates edge 4 353 359 innerTail68_1 best.bound = innerCandidates68_0 := by
  unfold candidates
  have row : edge 359 = PrimePairCache.smallRow359.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild68_0 : search edge 3 [353, 359] innerCandidates68_0 712 best = best := by decide

theorem innerScan68_0 : scan edge (search edge 3) 4 [353] 353 innerTail68_0 best = best := by
  rw [innerTail68_0, scan, if_neg (by decide), innerCandidates68_0_checked]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_1
    (if 3 ≤ innerCandidates68_0.length then search edge 3 [353, 359] innerCandidates68_0 712 best else best) = best
  rw [if_pos (by decide), innerChild68_0]
  exact innerScan68_1

theorem rootBranch68 : search edge 4 [353] rootCandidates68 353 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [353] 353 innerTail68_0 best = best
  exact innerScan68_0

def rootCandidates67 : List Nat := [373, 409, 499, 709, 829, 919, 967, 1087, 1123, 1237, 1291, 1429, 1447, 2161, 2341, 2551, 2707, 2719, 2971, 3433, 4159, 4357, 4657, 5581, 5743, 6007, 6151, 6709, 8293, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates67_checked : candidates edge 5 0 349 tail68 best.bound = rootCandidates67 := by
  unfold candidates
  have row : edge 349 = PrimePairCache.smallRow349.contains := by funext q; rfl
  rw [row]
  decide

def innerTail67_30 : List Nat := []
theorem innerScan67_30 : scan edge (search edge 3) 4 [349] 349 innerTail67_30 best = best := rfl

def innerTail67_29 : List Nat := 8377 :: innerTail67_30
def innerCandidates67_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_29_checked : candidates edge 4 349 8377 innerTail67_30 best.bound = innerCandidates67_29 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_29 : scan edge (search edge 3) 4 [349] 349 innerTail67_29 best = best := by
  rw [innerTail67_29, scan, if_neg (by decide), innerCandidates67_29_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_30
    (if 3 ≤ innerCandidates67_29.length then search edge 3 [349, 8377] innerCandidates67_29 8726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_30

def innerTail67_28 : List Nat := 8293 :: innerTail67_29
def innerCandidates67_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_28_checked : candidates edge 4 349 8293 innerTail67_29 best.bound = innerCandidates67_28 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_28 : scan edge (search edge 3) 4 [349] 349 innerTail67_28 best = best := by
  rw [innerTail67_28, scan, if_neg (by decide), innerCandidates67_28_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_29
    (if 3 ≤ innerCandidates67_28.length then search edge 3 [349, 8293] innerCandidates67_28 8642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_29

def innerTail67_27 : List Nat := 6709 :: innerTail67_28
def innerCandidates67_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_27_checked : candidates edge 4 349 6709 innerTail67_28 best.bound = innerCandidates67_27 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_27 : scan edge (search edge 3) 4 [349] 349 innerTail67_27 best = best := by
  rw [innerTail67_27, scan, if_neg (by decide), innerCandidates67_27_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_28
    (if 3 ≤ innerCandidates67_27.length then search edge 3 [349, 6709] innerCandidates67_27 7058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_28

def innerTail67_26 : List Nat := 6151 :: innerTail67_27
def innerCandidates67_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_26_checked : candidates edge 4 349 6151 innerTail67_27 best.bound = innerCandidates67_26 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_26 : scan edge (search edge 3) 4 [349] 349 innerTail67_26 best = best := by
  rw [innerTail67_26, scan, if_neg (by decide), innerCandidates67_26_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_27
    (if 3 ≤ innerCandidates67_26.length then search edge 3 [349, 6151] innerCandidates67_26 6500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_27

def innerTail67_25 : List Nat := 6007 :: innerTail67_26
def innerCandidates67_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_25_checked : candidates edge 4 349 6007 innerTail67_26 best.bound = innerCandidates67_25 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_25 : scan edge (search edge 3) 4 [349] 349 innerTail67_25 best = best := by
  rw [innerTail67_25, scan, if_neg (by decide), innerCandidates67_25_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_26
    (if 3 ≤ innerCandidates67_25.length then search edge 3 [349, 6007] innerCandidates67_25 6356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_26

def innerTail67_24 : List Nat := 5743 :: innerTail67_25
def innerCandidates67_24 : List Nat := [8377]
set_option maxRecDepth 4096 in
theorem innerCandidates67_24_checked : candidates edge 4 349 5743 innerTail67_25 best.bound = innerCandidates67_24 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_24 : scan edge (search edge 3) 4 [349] 349 innerTail67_24 best = best := by
  rw [innerTail67_24, scan, if_neg (by decide), innerCandidates67_24_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_25
    (if 3 ≤ innerCandidates67_24.length then search edge 3 [349, 5743] innerCandidates67_24 6092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_25

def innerTail67_23 : List Nat := 5581 :: innerTail67_24
def innerCandidates67_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_23_checked : candidates edge 4 349 5581 innerTail67_24 best.bound = innerCandidates67_23 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_23 : scan edge (search edge 3) 4 [349] 349 innerTail67_23 best = best := by
  rw [innerTail67_23, scan, if_neg (by decide), innerCandidates67_23_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_24
    (if 3 ≤ innerCandidates67_23.length then search edge 3 [349, 5581] innerCandidates67_23 5930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_24

def innerTail67_22 : List Nat := 4657 :: innerTail67_23
def innerCandidates67_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_22_checked : candidates edge 4 349 4657 innerTail67_23 best.bound = innerCandidates67_22 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_22 : scan edge (search edge 3) 4 [349] 349 innerTail67_22 best = best := by
  rw [innerTail67_22, scan, if_neg (by decide), innerCandidates67_22_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_23
    (if 3 ≤ innerCandidates67_22.length then search edge 3 [349, 4657] innerCandidates67_22 5006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_23

def innerTail67_21 : List Nat := 4357 :: innerTail67_22
def innerCandidates67_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_21_checked : candidates edge 4 349 4357 innerTail67_22 best.bound = innerCandidates67_21 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_21 : scan edge (search edge 3) 4 [349] 349 innerTail67_21 best = best := by
  rw [innerTail67_21, scan, if_neg (by decide), innerCandidates67_21_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_22
    (if 3 ≤ innerCandidates67_21.length then search edge 3 [349, 4357] innerCandidates67_21 4706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_22

def innerTail67_20 : List Nat := 4159 :: innerTail67_21
def innerCandidates67_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_20_checked : candidates edge 4 349 4159 innerTail67_21 best.bound = innerCandidates67_20 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_20 : scan edge (search edge 3) 4 [349] 349 innerTail67_20 best = best := by
  rw [innerTail67_20, scan, if_neg (by decide), innerCandidates67_20_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_21
    (if 3 ≤ innerCandidates67_20.length then search edge 3 [349, 4159] innerCandidates67_20 4508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_21

def innerTail67_19 : List Nat := 3433 :: innerTail67_20
def innerCandidates67_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_19_checked : candidates edge 4 349 3433 innerTail67_20 best.bound = innerCandidates67_19 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_19 : scan edge (search edge 3) 4 [349] 349 innerTail67_19 best = best := by
  rw [innerTail67_19, scan, if_neg (by decide), innerCandidates67_19_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_20
    (if 3 ≤ innerCandidates67_19.length then search edge 3 [349, 3433] innerCandidates67_19 3782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_20

def innerTail67_18 : List Nat := 2971 :: innerTail67_19
def innerCandidates67_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_18_checked : candidates edge 4 349 2971 innerTail67_19 best.bound = innerCandidates67_18 := by
  unfold candidates
  have row : edge 2971 = SearchData.row2971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_18 : scan edge (search edge 3) 4 [349] 349 innerTail67_18 best = best := by
  rw [innerTail67_18, scan, if_neg (by decide), innerCandidates67_18_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_19
    (if 3 ≤ innerCandidates67_18.length then search edge 3 [349, 2971] innerCandidates67_18 3320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_19

def innerTail67_17 : List Nat := 2719 :: innerTail67_18
def innerCandidates67_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_17_checked : candidates edge 4 349 2719 innerTail67_18 best.bound = innerCandidates67_17 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_17 : scan edge (search edge 3) 4 [349] 349 innerTail67_17 best = best := by
  rw [innerTail67_17, scan, if_neg (by decide), innerCandidates67_17_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_18
    (if 3 ≤ innerCandidates67_17.length then search edge 3 [349, 2719] innerCandidates67_17 3068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_18

def innerTail67_16 : List Nat := 2707 :: innerTail67_17
def innerCandidates67_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_16_checked : candidates edge 4 349 2707 innerTail67_17 best.bound = innerCandidates67_16 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_16 : scan edge (search edge 3) 4 [349] 349 innerTail67_16 best = best := by
  rw [innerTail67_16, scan, if_neg (by decide), innerCandidates67_16_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_17
    (if 3 ≤ innerCandidates67_16.length then search edge 3 [349, 2707] innerCandidates67_16 3056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_17

def innerTail67_15 : List Nat := 2551 :: innerTail67_16
def innerCandidates67_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_15_checked : candidates edge 4 349 2551 innerTail67_16 best.bound = innerCandidates67_15 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_15 : scan edge (search edge 3) 4 [349] 349 innerTail67_15 best = best := by
  rw [innerTail67_15, scan, if_neg (by decide), innerCandidates67_15_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_16
    (if 3 ≤ innerCandidates67_15.length then search edge 3 [349, 2551] innerCandidates67_15 2900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_16

def innerTail67_14 : List Nat := 2341 :: innerTail67_15
def innerCandidates67_14 : List Nat := [8377]
set_option maxRecDepth 4096 in
theorem innerCandidates67_14_checked : candidates edge 4 349 2341 innerTail67_15 best.bound = innerCandidates67_14 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_14 : scan edge (search edge 3) 4 [349] 349 innerTail67_14 best = best := by
  rw [innerTail67_14, scan, if_neg (by decide), innerCandidates67_14_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_15
    (if 3 ≤ innerCandidates67_14.length then search edge 3 [349, 2341] innerCandidates67_14 2690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_15

def innerTail67_13 : List Nat := 2161 :: innerTail67_14
def innerCandidates67_13 : List Nat := [4357]
set_option maxRecDepth 4096 in
theorem innerCandidates67_13_checked : candidates edge 4 349 2161 innerTail67_14 best.bound = innerCandidates67_13 := by
  unfold candidates
  have row : edge 2161 = PrimePairCache.smallRow2161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_13 : scan edge (search edge 3) 4 [349] 349 innerTail67_13 best = best := by
  rw [innerTail67_13, scan, if_neg (by decide), innerCandidates67_13_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_14
    (if 3 ≤ innerCandidates67_13.length then search edge 3 [349, 2161] innerCandidates67_13 2510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_14

def innerTail67_12 : List Nat := 1447 :: innerTail67_13
def innerCandidates67_12 : List Nat := [2719]
set_option maxRecDepth 4096 in
theorem innerCandidates67_12_checked : candidates edge 4 349 1447 innerTail67_13 best.bound = innerCandidates67_12 := by
  unfold candidates
  have row : edge 1447 = PrimePairCache.smallRow1447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_12 : scan edge (search edge 3) 4 [349] 349 innerTail67_12 best = best := by
  rw [innerTail67_12, scan, if_neg (by decide), innerCandidates67_12_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_13
    (if 3 ≤ innerCandidates67_12.length then search edge 3 [349, 1447] innerCandidates67_12 1796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_13

def innerTail67_11 : List Nat := 1429 :: innerTail67_12
def innerCandidates67_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_11_checked : candidates edge 4 349 1429 innerTail67_12 best.bound = innerCandidates67_11 := by
  unfold candidates
  have row : edge 1429 = PrimePairCache.smallRow1429.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_11 : scan edge (search edge 3) 4 [349] 349 innerTail67_11 best = best := by
  rw [innerTail67_11, scan, if_neg (by decide), innerCandidates67_11_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_12
    (if 3 ≤ innerCandidates67_11.length then search edge 3 [349, 1429] innerCandidates67_11 1778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_12

def innerTail67_10 : List Nat := 1291 :: innerTail67_11
def innerCandidates67_10 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates67_10_checked : candidates edge 4 349 1291 innerTail67_11 best.bound = innerCandidates67_10 := by
  unfold candidates
  have row : edge 1291 = PrimePairCache.smallRow1291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_10 : scan edge (search edge 3) 4 [349] 349 innerTail67_10 best = best := by
  rw [innerTail67_10, scan, if_neg (by decide), innerCandidates67_10_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_11
    (if 3 ≤ innerCandidates67_10.length then search edge 3 [349, 1291] innerCandidates67_10 1640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_11

def innerTail67_9 : List Nat := 1237 :: innerTail67_10
def innerCandidates67_9 : List Nat := [2341, 5743]
set_option maxRecDepth 4096 in
theorem innerCandidates67_9_checked : candidates edge 4 349 1237 innerTail67_10 best.bound = innerCandidates67_9 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_9 : scan edge (search edge 3) 4 [349] 349 innerTail67_9 best = best := by
  rw [innerTail67_9, scan, if_neg (by decide), innerCandidates67_9_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_10
    (if 3 ≤ innerCandidates67_9.length then search edge 3 [349, 1237] innerCandidates67_9 1586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_10

def innerTail67_8 : List Nat := 1123 :: innerTail67_9
def innerCandidates67_8 : List Nat := [3433]
set_option maxRecDepth 4096 in
theorem innerCandidates67_8_checked : candidates edge 4 349 1123 innerTail67_9 best.bound = innerCandidates67_8 := by
  unfold candidates
  have row : edge 1123 = PrimePairCache.smallRow1123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_8 : scan edge (search edge 3) 4 [349] 349 innerTail67_8 best = best := by
  rw [innerTail67_8, scan, if_neg (by decide), innerCandidates67_8_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_9
    (if 3 ≤ innerCandidates67_8.length then search edge 3 [349, 1123] innerCandidates67_8 1472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_9

def innerTail67_7 : List Nat := 1087 :: innerTail67_8
def innerCandidates67_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_7_checked : candidates edge 4 349 1087 innerTail67_8 best.bound = innerCandidates67_7 := by
  unfold candidates
  have row : edge 1087 = PrimePairCache.smallRow1087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_7 : scan edge (search edge 3) 4 [349] 349 innerTail67_7 best = best := by
  rw [innerTail67_7, scan, if_neg (by decide), innerCandidates67_7_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_8
    (if 3 ≤ innerCandidates67_7.length then search edge 3 [349, 1087] innerCandidates67_7 1436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_8

def innerTail67_6 : List Nat := 967 :: innerTail67_7
def innerCandidates67_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_6_checked : candidates edge 4 349 967 innerTail67_7 best.bound = innerCandidates67_6 := by
  unfold candidates
  have row : edge 967 = PrimePairCache.smallRow967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_6 : scan edge (search edge 3) 4 [349] 349 innerTail67_6 best = best := by
  rw [innerTail67_6, scan, if_neg (by decide), innerCandidates67_6_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_7
    (if 3 ≤ innerCandidates67_6.length then search edge 3 [349, 967] innerCandidates67_6 1316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_7

def innerTail67_5 : List Nat := 919 :: innerTail67_6
def innerCandidates67_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates67_5_checked : candidates edge 4 349 919 innerTail67_6 best.bound = innerCandidates67_5 := by
  unfold candidates
  have row : edge 919 = PrimePairCache.smallRow919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_5 : scan edge (search edge 3) 4 [349] 349 innerTail67_5 best = best := by
  rw [innerTail67_5, scan, if_neg (by decide), innerCandidates67_5_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_6
    (if 3 ≤ innerCandidates67_5.length then search edge 3 [349, 919] innerCandidates67_5 1268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_6

def innerTail67_4 : List Nat := 829 :: innerTail67_5
def innerCandidates67_4 : List Nat := [1237]
set_option maxRecDepth 4096 in
theorem innerCandidates67_4_checked : candidates edge 4 349 829 innerTail67_5 best.bound = innerCandidates67_4 := by
  unfold candidates
  have row : edge 829 = PrimePairCache.smallRow829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_4 : scan edge (search edge 3) 4 [349] 349 innerTail67_4 best = best := by
  rw [innerTail67_4, scan, if_neg (by decide), innerCandidates67_4_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_5
    (if 3 ≤ innerCandidates67_4.length then search edge 3 [349, 829] innerCandidates67_4 1178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_5

def innerTail67_3 : List Nat := 709 :: innerTail67_4
def innerCandidates67_3 : List Nat := [967]
set_option maxRecDepth 4096 in
theorem innerCandidates67_3_checked : candidates edge 4 349 709 innerTail67_4 best.bound = innerCandidates67_3 := by
  unfold candidates
  have row : edge 709 = PrimePairCache.smallRow709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_3 : scan edge (search edge 3) 4 [349] 349 innerTail67_3 best = best := by
  rw [innerTail67_3, scan, if_neg (by decide), innerCandidates67_3_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_4
    (if 3 ≤ innerCandidates67_3.length then search edge 3 [349, 709] innerCandidates67_3 1058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_4

def innerTail67_2 : List Nat := 499 :: innerTail67_3
def innerCandidates67_2 : List Nat := [1237, 2551, 2707, 4357]
set_option maxRecDepth 4096 in
theorem innerCandidates67_2_checked : candidates edge 4 349 499 innerTail67_3 best.bound = innerCandidates67_2 := by
  unfold candidates
  have row : edge 499 = PrimePairCache.smallRow499.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild67_2 : search edge 3 [349, 499] innerCandidates67_2 848 best = best := by decide

theorem innerScan67_2 : scan edge (search edge 3) 4 [349] 349 innerTail67_2 best = best := by
  rw [innerTail67_2, scan, if_neg (by decide), innerCandidates67_2_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_3
    (if 3 ≤ innerCandidates67_2.length then search edge 3 [349, 499] innerCandidates67_2 848 best else best) = best
  rw [if_pos (by decide), innerChild67_2]
  exact innerScan67_3

def innerTail67_1 : List Nat := 409 :: innerTail67_2
def innerCandidates67_1 : List Nat := [709]
set_option maxRecDepth 4096 in
theorem innerCandidates67_1_checked : candidates edge 4 349 409 innerTail67_2 best.bound = innerCandidates67_1 := by
  unfold candidates
  have row : edge 409 = PrimePairCache.smallRow409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan67_1 : scan edge (search edge 3) 4 [349] 349 innerTail67_1 best = best := by
  rw [innerTail67_1, scan, if_neg (by decide), innerCandidates67_1_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_2
    (if 3 ≤ innerCandidates67_1.length then search edge 3 [349, 409] innerCandidates67_1 758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan67_2

def innerTail67_0 : List Nat := 373 :: innerTail67_1
def innerCandidates67_0 : List Nat := [1237, 1447, 2161]
set_option maxRecDepth 4096 in
theorem innerCandidates67_0_checked : candidates edge 4 349 373 innerTail67_1 best.bound = innerCandidates67_0 := by
  unfold candidates
  have row : edge 373 = PrimePairCache.smallRow373.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild67_0 : search edge 3 [349, 373] innerCandidates67_0 722 best = best := by decide

theorem innerScan67_0 : scan edge (search edge 3) 4 [349] 349 innerTail67_0 best = best := by
  rw [innerTail67_0, scan, if_neg (by decide), innerCandidates67_0_checked]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_1
    (if 3 ≤ innerCandidates67_0.length then search edge 3 [349, 373] innerCandidates67_0 722 best else best) = best
  rw [if_pos (by decide), innerChild67_0]
  exact innerScan67_1

theorem rootBranch67 : search edge 4 [349] rootCandidates67 349 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [349] 349 innerTail67_0 best = best
  exact innerScan67_0

end ProjectEulerSolutions.PrimePairUpper.Search
