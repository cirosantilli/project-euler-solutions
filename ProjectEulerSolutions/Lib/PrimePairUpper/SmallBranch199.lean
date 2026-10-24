import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates104 : List Nat := [617, 677, 1163, 1223, 1571, 2357, 2549, 2903, 3011, 3719, 3767, 3851, 4133, 4229, 4397, 4793, 5021, 5087, 6011, 6323, 6911, 6977, 7127, 7247, 7481, 7523, 7829, 8069, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates104_checked : candidates edge 5 0 587 tail105 best.bound = rootCandidates104 := by
  unfold candidates
  have row : edge 587 = PrimePairCache.smallRow587.contains := by funext q; rfl
  rw [row]
  decide

def innerTail104_29 : List Nat := []
theorem innerScan104_29 : scan edge (search edge 3) 4 [587] 587 innerTail104_29 best = best := rfl

def innerTail104_28 : List Nat := 8123 :: innerTail104_29
def innerCandidates104_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_28_checked : candidates edge 4 587 8123 innerTail104_29 best.bound = innerCandidates104_28 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_28 : scan edge (search edge 3) 4 [587] 587 innerTail104_28 best = best := by
  rw [innerTail104_28, scan, if_neg (by decide), innerCandidates104_28_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_29
    (if 3 ≤ innerCandidates104_28.length then search edge 3 [587, 8123] innerCandidates104_28 8710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_29

def innerTail104_27 : List Nat := 8069 :: innerTail104_28
def innerCandidates104_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_27_checked : candidates edge 4 587 8069 innerTail104_28 best.bound = innerCandidates104_27 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_27 : scan edge (search edge 3) 4 [587] 587 innerTail104_27 best = best := by
  rw [innerTail104_27, scan, if_neg (by decide), innerCandidates104_27_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_28
    (if 3 ≤ innerCandidates104_27.length then search edge 3 [587, 8069] innerCandidates104_27 8656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_28

def innerTail104_26 : List Nat := 7829 :: innerTail104_27
def innerCandidates104_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_26_checked : candidates edge 4 587 7829 innerTail104_27 best.bound = innerCandidates104_26 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_26 : scan edge (search edge 3) 4 [587] 587 innerTail104_26 best = best := by
  rw [innerTail104_26, scan, if_neg (by decide), innerCandidates104_26_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_27
    (if 3 ≤ innerCandidates104_26.length then search edge 3 [587, 7829] innerCandidates104_26 8416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_27

def innerTail104_25 : List Nat := 7523 :: innerTail104_26
def innerCandidates104_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_25_checked : candidates edge 4 587 7523 innerTail104_26 best.bound = innerCandidates104_25 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_25 : scan edge (search edge 3) 4 [587] 587 innerTail104_25 best = best := by
  rw [innerTail104_25, scan, if_neg (by decide), innerCandidates104_25_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_26
    (if 3 ≤ innerCandidates104_25.length then search edge 3 [587, 7523] innerCandidates104_25 8110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_26

def innerTail104_24 : List Nat := 7481 :: innerTail104_25
def innerCandidates104_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_24_checked : candidates edge 4 587 7481 innerTail104_25 best.bound = innerCandidates104_24 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_24 : scan edge (search edge 3) 4 [587] 587 innerTail104_24 best = best := by
  rw [innerTail104_24, scan, if_neg (by decide), innerCandidates104_24_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_25
    (if 3 ≤ innerCandidates104_24.length then search edge 3 [587, 7481] innerCandidates104_24 8068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_25

def innerTail104_23 : List Nat := 7247 :: innerTail104_24
def innerCandidates104_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_23_checked : candidates edge 4 587 7247 innerTail104_24 best.bound = innerCandidates104_23 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_23 : scan edge (search edge 3) 4 [587] 587 innerTail104_23 best = best := by
  rw [innerTail104_23, scan, if_neg (by decide), innerCandidates104_23_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_24
    (if 3 ≤ innerCandidates104_23.length then search edge 3 [587, 7247] innerCandidates104_23 7834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_24

def innerTail104_22 : List Nat := 7127 :: innerTail104_23
def innerCandidates104_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_22_checked : candidates edge 4 587 7127 innerTail104_23 best.bound = innerCandidates104_22 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_22 : scan edge (search edge 3) 4 [587] 587 innerTail104_22 best = best := by
  rw [innerTail104_22, scan, if_neg (by decide), innerCandidates104_22_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_23
    (if 3 ≤ innerCandidates104_22.length then search edge 3 [587, 7127] innerCandidates104_22 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_23

def innerTail104_21 : List Nat := 6977 :: innerTail104_22
def innerCandidates104_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_21_checked : candidates edge 4 587 6977 innerTail104_22 best.bound = innerCandidates104_21 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_21 : scan edge (search edge 3) 4 [587] 587 innerTail104_21 best = best := by
  rw [innerTail104_21, scan, if_neg (by decide), innerCandidates104_21_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_22
    (if 3 ≤ innerCandidates104_21.length then search edge 3 [587, 6977] innerCandidates104_21 7564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_22

def innerTail104_20 : List Nat := 6911 :: innerTail104_21
def innerCandidates104_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_20_checked : candidates edge 4 587 6911 innerTail104_21 best.bound = innerCandidates104_20 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_20 : scan edge (search edge 3) 4 [587] 587 innerTail104_20 best = best := by
  rw [innerTail104_20, scan, if_neg (by decide), innerCandidates104_20_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_21
    (if 3 ≤ innerCandidates104_20.length then search edge 3 [587, 6911] innerCandidates104_20 7498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_21

def innerTail104_19 : List Nat := 6323 :: innerTail104_20
def innerCandidates104_19 : List Nat := [7523]
set_option maxRecDepth 4096 in
theorem innerCandidates104_19_checked : candidates edge 4 587 6323 innerTail104_20 best.bound = innerCandidates104_19 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_19 : scan edge (search edge 3) 4 [587] 587 innerTail104_19 best = best := by
  rw [innerTail104_19, scan, if_neg (by decide), innerCandidates104_19_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_20
    (if 3 ≤ innerCandidates104_19.length then search edge 3 [587, 6323] innerCandidates104_19 6910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_20

def innerTail104_18 : List Nat := 6011 :: innerTail104_19
def innerCandidates104_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_18_checked : candidates edge 4 587 6011 innerTail104_19 best.bound = innerCandidates104_18 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_18 : scan edge (search edge 3) 4 [587] 587 innerTail104_18 best = best := by
  rw [innerTail104_18, scan, if_neg (by decide), innerCandidates104_18_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_19
    (if 3 ≤ innerCandidates104_18.length then search edge 3 [587, 6011] innerCandidates104_18 6598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_19

def innerTail104_17 : List Nat := 5087 :: innerTail104_18
def innerCandidates104_17 : List Nat := [7247]
set_option maxRecDepth 4096 in
theorem innerCandidates104_17_checked : candidates edge 4 587 5087 innerTail104_18 best.bound = innerCandidates104_17 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_17 : scan edge (search edge 3) 4 [587] 587 innerTail104_17 best = best := by
  rw [innerTail104_17, scan, if_neg (by decide), innerCandidates104_17_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_18
    (if 3 ≤ innerCandidates104_17.length then search edge 3 [587, 5087] innerCandidates104_17 5674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_18

def innerTail104_16 : List Nat := 5021 :: innerTail104_17
def innerCandidates104_16 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates104_16_checked : candidates edge 4 587 5021 innerTail104_17 best.bound = innerCandidates104_16 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_16 : scan edge (search edge 3) 4 [587] 587 innerTail104_16 best = best := by
  rw [innerTail104_16, scan, if_neg (by decide), innerCandidates104_16_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_17
    (if 3 ≤ innerCandidates104_16.length then search edge 3 [587, 5021] innerCandidates104_16 5608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_17

def innerTail104_15 : List Nat := 4793 :: innerTail104_16
def innerCandidates104_15 : List Nat := [7481, 8123]
set_option maxRecDepth 4096 in
theorem innerCandidates104_15_checked : candidates edge 4 587 4793 innerTail104_16 best.bound = innerCandidates104_15 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_15 : scan edge (search edge 3) 4 [587] 587 innerTail104_15 best = best := by
  rw [innerTail104_15, scan, if_neg (by decide), innerCandidates104_15_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_16
    (if 3 ≤ innerCandidates104_15.length then search edge 3 [587, 4793] innerCandidates104_15 5380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_16

def innerTail104_14 : List Nat := 4397 :: innerTail104_15
def innerCandidates104_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_14_checked : candidates edge 4 587 4397 innerTail104_15 best.bound = innerCandidates104_14 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_14 : scan edge (search edge 3) 4 [587] 587 innerTail104_14 best = best := by
  rw [innerTail104_14, scan, if_neg (by decide), innerCandidates104_14_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_15
    (if 3 ≤ innerCandidates104_14.length then search edge 3 [587, 4397] innerCandidates104_14 4984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_15

def innerTail104_13 : List Nat := 4229 :: innerTail104_14
def innerCandidates104_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_13_checked : candidates edge 4 587 4229 innerTail104_14 best.bound = innerCandidates104_13 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_13 : scan edge (search edge 3) 4 [587] 587 innerTail104_13 best = best := by
  rw [innerTail104_13, scan, if_neg (by decide), innerCandidates104_13_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_14
    (if 3 ≤ innerCandidates104_13.length then search edge 3 [587, 4229] innerCandidates104_13 4816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_14

def innerTail104_12 : List Nat := 4133 :: innerTail104_13
def innerCandidates104_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_12_checked : candidates edge 4 587 4133 innerTail104_13 best.bound = innerCandidates104_12 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_12 : scan edge (search edge 3) 4 [587] 587 innerTail104_12 best = best := by
  rw [innerTail104_12, scan, if_neg (by decide), innerCandidates104_12_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_13
    (if 3 ≤ innerCandidates104_12.length then search edge 3 [587, 4133] innerCandidates104_12 4720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_13

def innerTail104_11 : List Nat := 3851 :: innerTail104_12
def innerCandidates104_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_11_checked : candidates edge 4 587 3851 innerTail104_12 best.bound = innerCandidates104_11 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_11 : scan edge (search edge 3) 4 [587] 587 innerTail104_11 best = best := by
  rw [innerTail104_11, scan, if_neg (by decide), innerCandidates104_11_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_12
    (if 3 ≤ innerCandidates104_11.length then search edge 3 [587, 3851] innerCandidates104_11 4438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_12

def innerTail104_10 : List Nat := 3767 :: innerTail104_11
def innerCandidates104_10 : List Nat := [6911]
set_option maxRecDepth 4096 in
theorem innerCandidates104_10_checked : candidates edge 4 587 3767 innerTail104_11 best.bound = innerCandidates104_10 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_10 : scan edge (search edge 3) 4 [587] 587 innerTail104_10 best = best := by
  rw [innerTail104_10, scan, if_neg (by decide), innerCandidates104_10_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_11
    (if 3 ≤ innerCandidates104_10.length then search edge 3 [587, 3767] innerCandidates104_10 4354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_11

def innerTail104_9 : List Nat := 3719 :: innerTail104_10
def innerCandidates104_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_9_checked : candidates edge 4 587 3719 innerTail104_10 best.bound = innerCandidates104_9 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_9 : scan edge (search edge 3) 4 [587] 587 innerTail104_9 best = best := by
  rw [innerTail104_9, scan, if_neg (by decide), innerCandidates104_9_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_10
    (if 3 ≤ innerCandidates104_9.length then search edge 3 [587, 3719] innerCandidates104_9 4306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_10

def innerTail104_8 : List Nat := 3011 :: innerTail104_9
def innerCandidates104_8 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates104_8_checked : candidates edge 4 587 3011 innerTail104_9 best.bound = innerCandidates104_8 := by
  unfold candidates
  have row : edge 3011 = SearchData.row3011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_8 : scan edge (search edge 3) 4 [587] 587 innerTail104_8 best = best := by
  rw [innerTail104_8, scan, if_neg (by decide), innerCandidates104_8_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_9
    (if 3 ≤ innerCandidates104_8.length then search edge 3 [587, 3011] innerCandidates104_8 3598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_9

def innerTail104_7 : List Nat := 2903 :: innerTail104_8
def innerCandidates104_7 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates104_7_checked : candidates edge 4 587 2903 innerTail104_8 best.bound = innerCandidates104_7 := by
  unfold candidates
  have row : edge 2903 = SearchData.row2903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_7 : scan edge (search edge 3) 4 [587] 587 innerTail104_7 best = best := by
  rw [innerTail104_7, scan, if_neg (by decide), innerCandidates104_7_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_8
    (if 3 ≤ innerCandidates104_7.length then search edge 3 [587, 2903] innerCandidates104_7 3490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_8

def innerTail104_6 : List Nat := 2549 :: innerTail104_7
def innerCandidates104_6 : List Nat := [6323]
set_option maxRecDepth 4096 in
theorem innerCandidates104_6_checked : candidates edge 4 587 2549 innerTail104_7 best.bound = innerCandidates104_6 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_6 : scan edge (search edge 3) 4 [587] 587 innerTail104_6 best = best := by
  rw [innerTail104_6, scan, if_neg (by decide), innerCandidates104_6_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_7
    (if 3 ≤ innerCandidates104_6.length then search edge 3 [587, 2549] innerCandidates104_6 3136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_7

def innerTail104_5 : List Nat := 2357 :: innerTail104_6
def innerCandidates104_5 : List Nat := [2903, 5087, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates104_5_checked : candidates edge 4 587 2357 innerTail104_6 best.bound = innerCandidates104_5 := by
  unfold candidates
  have row : edge 2357 = PrimePairCache.smallRow2357.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild104_5 : search edge 3 [587, 2357] innerCandidates104_5 2944 best = best := by decide

theorem innerScan104_5 : scan edge (search edge 3) 4 [587] 587 innerTail104_5 best = best := by
  rw [innerTail104_5, scan, if_neg (by decide), innerCandidates104_5_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_6
    (if 3 ≤ innerCandidates104_5.length then search edge 3 [587, 2357] innerCandidates104_5 2944 best else best) = best
  rw [if_pos (by decide), innerChild104_5]
  exact innerScan104_6

def innerTail104_4 : List Nat := 1571 :: innerTail104_5
def innerCandidates104_4 : List Nat := [4133]
set_option maxRecDepth 4096 in
theorem innerCandidates104_4_checked : candidates edge 4 587 1571 innerTail104_5 best.bound = innerCandidates104_4 := by
  unfold candidates
  have row : edge 1571 = PrimePairCache.smallRow1571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_4 : scan edge (search edge 3) 4 [587] 587 innerTail104_4 best = best := by
  rw [innerTail104_4, scan, if_neg (by decide), innerCandidates104_4_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_5
    (if 3 ≤ innerCandidates104_4.length then search edge 3 [587, 1571] innerCandidates104_4 2158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_5

def innerTail104_3 : List Nat := 1223 :: innerTail104_4
def innerCandidates104_3 : List Nat := [4793, 7523]
set_option maxRecDepth 4096 in
theorem innerCandidates104_3_checked : candidates edge 4 587 1223 innerTail104_4 best.bound = innerCandidates104_3 := by
  unfold candidates
  have row : edge 1223 = PrimePairCache.smallRow1223.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_3 : scan edge (search edge 3) 4 [587] 587 innerTail104_3 best = best := by
  rw [innerTail104_3, scan, if_neg (by decide), innerCandidates104_3_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_4
    (if 3 ≤ innerCandidates104_3.length then search edge 3 [587, 1223] innerCandidates104_3 1810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_4

def innerTail104_2 : List Nat := 1163 :: innerTail104_3
def innerCandidates104_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_2_checked : candidates edge 4 587 1163 innerTail104_3 best.bound = innerCandidates104_2 := by
  unfold candidates
  have row : edge 1163 = PrimePairCache.smallRow1163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_2 : scan edge (search edge 3) 4 [587] 587 innerTail104_2 best = best := by
  rw [innerTail104_2, scan, if_neg (by decide), innerCandidates104_2_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_3
    (if 3 ≤ innerCandidates104_2.length then search edge 3 [587, 1163] innerCandidates104_2 1750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_3

def innerTail104_1 : List Nat := 677 :: innerTail104_2
def innerCandidates104_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates104_1_checked : candidates edge 4 587 677 innerTail104_2 best.bound = innerCandidates104_1 := by
  unfold candidates
  have row : edge 677 = PrimePairCache.smallRow677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_1 : scan edge (search edge 3) 4 [587] 587 innerTail104_1 best = best := by
  rw [innerTail104_1, scan, if_neg (by decide), innerCandidates104_1_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_2
    (if 3 ≤ innerCandidates104_1.length then search edge 3 [587, 677] innerCandidates104_1 1264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_2

def innerTail104_0 : List Nat := 617 :: innerTail104_1
def innerCandidates104_0 : List Nat := [6011, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates104_0_checked : candidates edge 4 587 617 innerTail104_1 best.bound = innerCandidates104_0 := by
  unfold candidates
  have row : edge 617 = PrimePairCache.smallRow617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan104_0 : scan edge (search edge 3) 4 [587] 587 innerTail104_0 best = best := by
  rw [innerTail104_0, scan, if_neg (by decide), innerCandidates104_0_checked]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_1
    (if 3 ≤ innerCandidates104_0.length then search edge 3 [587, 617] innerCandidates104_0 1204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan104_1

theorem rootBranch104 : search edge 4 [587] rootCandidates104 587 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [587] 587 innerTail104_0 best = best
  exact innerScan104_0

def rootCandidates103 : List Nat := [613, 757, 937, 1327, 1723, 2539, 2617, 3331, 3709, 3919, 4597, 4759, 5431, 5683, 5821, 5839, 6949, 7459, 7951, 7963, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates103_checked : candidates edge 5 0 577 tail104 best.bound = rootCandidates103 := by
  unfold candidates
  have row : edge 577 = PrimePairCache.smallRow577.contains := by funext q; rfl
  rw [row]
  decide

def innerTail103_21 : List Nat := []
theorem innerScan103_21 : scan edge (search edge 3) 4 [577] 577 innerTail103_21 best = best := rfl

def innerTail103_20 : List Nat := 8329 :: innerTail103_21
def innerCandidates103_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_20_checked : candidates edge 4 577 8329 innerTail103_21 best.bound = innerCandidates103_20 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_20 : scan edge (search edge 3) 4 [577] 577 innerTail103_20 best = best := by
  rw [innerTail103_20, scan, if_neg (by decide), innerCandidates103_20_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_21
    (if 3 ≤ innerCandidates103_20.length then search edge 3 [577, 8329] innerCandidates103_20 8906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_21

def innerTail103_19 : List Nat := 7963 :: innerTail103_20
def innerCandidates103_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_19_checked : candidates edge 4 577 7963 innerTail103_20 best.bound = innerCandidates103_19 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_19 : scan edge (search edge 3) 4 [577] 577 innerTail103_19 best = best := by
  rw [innerTail103_19, scan, if_neg (by decide), innerCandidates103_19_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_20
    (if 3 ≤ innerCandidates103_19.length then search edge 3 [577, 7963] innerCandidates103_19 8540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_20

def innerTail103_18 : List Nat := 7951 :: innerTail103_19
def innerCandidates103_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_18_checked : candidates edge 4 577 7951 innerTail103_19 best.bound = innerCandidates103_18 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_18 : scan edge (search edge 3) 4 [577] 577 innerTail103_18 best = best := by
  rw [innerTail103_18, scan, if_neg (by decide), innerCandidates103_18_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_19
    (if 3 ≤ innerCandidates103_18.length then search edge 3 [577, 7951] innerCandidates103_18 8528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_19

def innerTail103_17 : List Nat := 7459 :: innerTail103_18
def innerCandidates103_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_17_checked : candidates edge 4 577 7459 innerTail103_18 best.bound = innerCandidates103_17 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_17 : scan edge (search edge 3) 4 [577] 577 innerTail103_17 best = best := by
  rw [innerTail103_17, scan, if_neg (by decide), innerCandidates103_17_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_18
    (if 3 ≤ innerCandidates103_17.length then search edge 3 [577, 7459] innerCandidates103_17 8036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_18

def innerTail103_16 : List Nat := 6949 :: innerTail103_17
def innerCandidates103_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_16_checked : candidates edge 4 577 6949 innerTail103_17 best.bound = innerCandidates103_16 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_16 : scan edge (search edge 3) 4 [577] 577 innerTail103_16 best = best := by
  rw [innerTail103_16, scan, if_neg (by decide), innerCandidates103_16_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_17
    (if 3 ≤ innerCandidates103_16.length then search edge 3 [577, 6949] innerCandidates103_16 7526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_17

def innerTail103_15 : List Nat := 5839 :: innerTail103_16
def innerCandidates103_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_15_checked : candidates edge 4 577 5839 innerTail103_16 best.bound = innerCandidates103_15 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_15 : scan edge (search edge 3) 4 [577] 577 innerTail103_15 best = best := by
  rw [innerTail103_15, scan, if_neg (by decide), innerCandidates103_15_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_16
    (if 3 ≤ innerCandidates103_15.length then search edge 3 [577, 5839] innerCandidates103_15 6416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_16

def innerTail103_14 : List Nat := 5821 :: innerTail103_15
def innerCandidates103_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_14_checked : candidates edge 4 577 5821 innerTail103_15 best.bound = innerCandidates103_14 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_14 : scan edge (search edge 3) 4 [577] 577 innerTail103_14 best = best := by
  rw [innerTail103_14, scan, if_neg (by decide), innerCandidates103_14_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_15
    (if 3 ≤ innerCandidates103_14.length then search edge 3 [577, 5821] innerCandidates103_14 6398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_15

def innerTail103_13 : List Nat := 5683 :: innerTail103_14
def innerCandidates103_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_13_checked : candidates edge 4 577 5683 innerTail103_14 best.bound = innerCandidates103_13 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_13 : scan edge (search edge 3) 4 [577] 577 innerTail103_13 best = best := by
  rw [innerTail103_13, scan, if_neg (by decide), innerCandidates103_13_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_14
    (if 3 ≤ innerCandidates103_13.length then search edge 3 [577, 5683] innerCandidates103_13 6260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_14

def innerTail103_12 : List Nat := 5431 :: innerTail103_13
def innerCandidates103_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_12_checked : candidates edge 4 577 5431 innerTail103_13 best.bound = innerCandidates103_12 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_12 : scan edge (search edge 3) 4 [577] 577 innerTail103_12 best = best := by
  rw [innerTail103_12, scan, if_neg (by decide), innerCandidates103_12_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_13
    (if 3 ≤ innerCandidates103_12.length then search edge 3 [577, 5431] innerCandidates103_12 6008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_13

def innerTail103_11 : List Nat := 4759 :: innerTail103_12
def innerCandidates103_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_11_checked : candidates edge 4 577 4759 innerTail103_12 best.bound = innerCandidates103_11 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_11 : scan edge (search edge 3) 4 [577] 577 innerTail103_11 best = best := by
  rw [innerTail103_11, scan, if_neg (by decide), innerCandidates103_11_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_12
    (if 3 ≤ innerCandidates103_11.length then search edge 3 [577, 4759] innerCandidates103_11 5336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_12

def innerTail103_10 : List Nat := 4597 :: innerTail103_11
def innerCandidates103_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_10_checked : candidates edge 4 577 4597 innerTail103_11 best.bound = innerCandidates103_10 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_10 : scan edge (search edge 3) 4 [577] 577 innerTail103_10 best = best := by
  rw [innerTail103_10, scan, if_neg (by decide), innerCandidates103_10_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_11
    (if 3 ≤ innerCandidates103_10.length then search edge 3 [577, 4597] innerCandidates103_10 5174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_11

def innerTail103_9 : List Nat := 3919 :: innerTail103_10
def innerCandidates103_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_9_checked : candidates edge 4 577 3919 innerTail103_10 best.bound = innerCandidates103_9 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_9 : scan edge (search edge 3) 4 [577] 577 innerTail103_9 best = best := by
  rw [innerTail103_9, scan, if_neg (by decide), innerCandidates103_9_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_10
    (if 3 ≤ innerCandidates103_9.length then search edge 3 [577, 3919] innerCandidates103_9 4496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_10

def innerTail103_8 : List Nat := 3709 :: innerTail103_9
def innerCandidates103_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_8_checked : candidates edge 4 577 3709 innerTail103_9 best.bound = innerCandidates103_8 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_8 : scan edge (search edge 3) 4 [577] 577 innerTail103_8 best = best := by
  rw [innerTail103_8, scan, if_neg (by decide), innerCandidates103_8_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_9
    (if 3 ≤ innerCandidates103_8.length then search edge 3 [577, 3709] innerCandidates103_8 4286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_9

def innerTail103_7 : List Nat := 3331 :: innerTail103_8
def innerCandidates103_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_7_checked : candidates edge 4 577 3331 innerTail103_8 best.bound = innerCandidates103_7 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_7 : scan edge (search edge 3) 4 [577] 577 innerTail103_7 best = best := by
  rw [innerTail103_7, scan, if_neg (by decide), innerCandidates103_7_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_8
    (if 3 ≤ innerCandidates103_7.length then search edge 3 [577, 3331] innerCandidates103_7 3908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_8

def innerTail103_6 : List Nat := 2617 :: innerTail103_7
def innerCandidates103_6 : List Nat := [5839]
set_option maxRecDepth 4096 in
theorem innerCandidates103_6_checked : candidates edge 4 577 2617 innerTail103_7 best.bound = innerCandidates103_6 := by
  unfold candidates
  have row : edge 2617 = PrimePairCache.smallRow2617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_6 : scan edge (search edge 3) 4 [577] 577 innerTail103_6 best = best := by
  rw [innerTail103_6, scan, if_neg (by decide), innerCandidates103_6_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_7
    (if 3 ≤ innerCandidates103_6.length then search edge 3 [577, 2617] innerCandidates103_6 3194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_7

def innerTail103_5 : List Nat := 2539 :: innerTail103_6
def innerCandidates103_5 : List Nat := [7951]
set_option maxRecDepth 4096 in
theorem innerCandidates103_5_checked : candidates edge 4 577 2539 innerTail103_6 best.bound = innerCandidates103_5 := by
  unfold candidates
  have row : edge 2539 = PrimePairCache.smallRow2539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_5 : scan edge (search edge 3) 4 [577] 577 innerTail103_5 best = best := by
  rw [innerTail103_5, scan, if_neg (by decide), innerCandidates103_5_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_6
    (if 3 ≤ innerCandidates103_5.length then search edge 3 [577, 2539] innerCandidates103_5 3116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_6

def innerTail103_4 : List Nat := 1723 :: innerTail103_5
def innerCandidates103_4 : List Nat := [5431, 7951, 8329]
set_option maxRecDepth 4096 in
theorem innerCandidates103_4_checked : candidates edge 4 577 1723 innerTail103_5 best.bound = innerCandidates103_4 := by
  unfold candidates
  have row : edge 1723 = PrimePairCache.smallRow1723.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild103_4 : search edge 3 [577, 1723] innerCandidates103_4 2300 best = best := by decide

theorem innerScan103_4 : scan edge (search edge 3) 4 [577] 577 innerTail103_4 best = best := by
  rw [innerTail103_4, scan, if_neg (by decide), innerCandidates103_4_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_5
    (if 3 ≤ innerCandidates103_4.length then search edge 3 [577, 1723] innerCandidates103_4 2300 best else best) = best
  rw [if_pos (by decide), innerChild103_4]
  exact innerScan103_5

def innerTail103_3 : List Nat := 1327 :: innerTail103_4
def innerCandidates103_3 : List Nat := [1723, 7963]
set_option maxRecDepth 4096 in
theorem innerCandidates103_3_checked : candidates edge 4 577 1327 innerTail103_4 best.bound = innerCandidates103_3 := by
  unfold candidates
  have row : edge 1327 = PrimePairCache.smallRow1327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_3 : scan edge (search edge 3) 4 [577] 577 innerTail103_3 best = best := by
  rw [innerTail103_3, scan, if_neg (by decide), innerCandidates103_3_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_4
    (if 3 ≤ innerCandidates103_3.length then search edge 3 [577, 1327] innerCandidates103_3 1904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_4

def innerTail103_2 : List Nat := 937 :: innerTail103_3
def innerCandidates103_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_2_checked : candidates edge 4 577 937 innerTail103_3 best.bound = innerCandidates103_2 := by
  unfold candidates
  have row : edge 937 = PrimePairCache.smallRow937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_2 : scan edge (search edge 3) 4 [577] 577 innerTail103_2 best = best := by
  rw [innerTail103_2, scan, if_neg (by decide), innerCandidates103_2_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_3
    (if 3 ≤ innerCandidates103_2.length then search edge 3 [577, 937] innerCandidates103_2 1514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_3

def innerTail103_1 : List Nat := 757 :: innerTail103_2
def innerCandidates103_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_1_checked : candidates edge 4 577 757 innerTail103_2 best.bound = innerCandidates103_1 := by
  unfold candidates
  have row : edge 757 = PrimePairCache.smallRow757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_1 : scan edge (search edge 3) 4 [577] 577 innerTail103_1 best = best := by
  rw [innerTail103_1, scan, if_neg (by decide), innerCandidates103_1_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_2
    (if 3 ≤ innerCandidates103_1.length then search edge 3 [577, 757] innerCandidates103_1 1334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_2

def innerTail103_0 : List Nat := 613 :: innerTail103_1
def innerCandidates103_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates103_0_checked : candidates edge 4 577 613 innerTail103_1 best.bound = innerCandidates103_0 := by
  unfold candidates
  have row : edge 613 = PrimePairCache.smallRow613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan103_0 : scan edge (search edge 3) 4 [577] 577 innerTail103_0 best = best := by
  rw [innerTail103_0, scan, if_neg (by decide), innerCandidates103_0_checked]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_1
    (if 3 ≤ innerCandidates103_0.length then search edge 3 [577, 613] innerCandidates103_0 1190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan103_1

theorem rootBranch103 : search edge 4 [577] rootCandidates103 577 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [577] 577 innerTail103_0 best = best
  exact innerScan103_0

end ProjectEulerSolutions.PrimePairUpper.Search
