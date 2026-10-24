import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates56 : List Nat := [331, 499, 751, 787, 859, 1381, 1471, 1693, 1933, 2281, 2437, 2593, 3037, 3079, 3307, 3919, 4327, 4483, 5119, 5323, 5431, 6343, 6427, 6841, 7213, 7741, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates56_checked : candidates edge 5 0 277 tail57 best.bound = rootCandidates56 := by
  unfold candidates
  have row : edge 277 = PrimePairCache.smallRow277.contains := by funext q; rfl
  rw [row]
  decide

def innerTail56_27 : List Nat := []
theorem innerScan56_27 : scan edge (search edge 3) 4 [277] 277 innerTail56_27 best = best := rfl

def innerTail56_26 : List Nat := 7993 :: innerTail56_27
def innerCandidates56_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_26_checked : candidates edge 4 277 7993 innerTail56_27 best.bound = innerCandidates56_26 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_26 : scan edge (search edge 3) 4 [277] 277 innerTail56_26 best = best := by
  rw [innerTail56_26, scan, if_neg (by decide), innerCandidates56_26_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_27
    (if 3 ≤ innerCandidates56_26.length then search edge 3 [277, 7993] innerCandidates56_26 8270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_27

def innerTail56_25 : List Nat := 7741 :: innerTail56_26
def innerCandidates56_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_25_checked : candidates edge 4 277 7741 innerTail56_26 best.bound = innerCandidates56_25 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_25 : scan edge (search edge 3) 4 [277] 277 innerTail56_25 best = best := by
  rw [innerTail56_25, scan, if_neg (by decide), innerCandidates56_25_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_26
    (if 3 ≤ innerCandidates56_25.length then search edge 3 [277, 7741] innerCandidates56_25 8018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_26

def innerTail56_24 : List Nat := 7213 :: innerTail56_25
def innerCandidates56_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_24_checked : candidates edge 4 277 7213 innerTail56_25 best.bound = innerCandidates56_24 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_24 : scan edge (search edge 3) 4 [277] 277 innerTail56_24 best = best := by
  rw [innerTail56_24, scan, if_neg (by decide), innerCandidates56_24_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_25
    (if 3 ≤ innerCandidates56_24.length then search edge 3 [277, 7213] innerCandidates56_24 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_25

def innerTail56_23 : List Nat := 6841 :: innerTail56_24
def innerCandidates56_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_23_checked : candidates edge 4 277 6841 innerTail56_24 best.bound = innerCandidates56_23 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_23 : scan edge (search edge 3) 4 [277] 277 innerTail56_23 best = best := by
  rw [innerTail56_23, scan, if_neg (by decide), innerCandidates56_23_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_24
    (if 3 ≤ innerCandidates56_23.length then search edge 3 [277, 6841] innerCandidates56_23 7118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_24

def innerTail56_22 : List Nat := 6427 :: innerTail56_23
def innerCandidates56_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_22_checked : candidates edge 4 277 6427 innerTail56_23 best.bound = innerCandidates56_22 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_22 : scan edge (search edge 3) 4 [277] 277 innerTail56_22 best = best := by
  rw [innerTail56_22, scan, if_neg (by decide), innerCandidates56_22_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_23
    (if 3 ≤ innerCandidates56_22.length then search edge 3 [277, 6427] innerCandidates56_22 6704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_23

def innerTail56_21 : List Nat := 6343 :: innerTail56_22
def innerCandidates56_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_21_checked : candidates edge 4 277 6343 innerTail56_22 best.bound = innerCandidates56_21 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_21 : scan edge (search edge 3) 4 [277] 277 innerTail56_21 best = best := by
  rw [innerTail56_21, scan, if_neg (by decide), innerCandidates56_21_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_22
    (if 3 ≤ innerCandidates56_21.length then search edge 3 [277, 6343] innerCandidates56_21 6620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_22

def innerTail56_20 : List Nat := 5431 :: innerTail56_21
def innerCandidates56_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_20_checked : candidates edge 4 277 5431 innerTail56_21 best.bound = innerCandidates56_20 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_20 : scan edge (search edge 3) 4 [277] 277 innerTail56_20 best = best := by
  rw [innerTail56_20, scan, if_neg (by decide), innerCandidates56_20_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_21
    (if 3 ≤ innerCandidates56_20.length then search edge 3 [277, 5431] innerCandidates56_20 5708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_21

def innerTail56_19 : List Nat := 5323 :: innerTail56_20
def innerCandidates56_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_19_checked : candidates edge 4 277 5323 innerTail56_20 best.bound = innerCandidates56_19 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_19 : scan edge (search edge 3) 4 [277] 277 innerTail56_19 best = best := by
  rw [innerTail56_19, scan, if_neg (by decide), innerCandidates56_19_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_20
    (if 3 ≤ innerCandidates56_19.length then search edge 3 [277, 5323] innerCandidates56_19 5600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_20

def innerTail56_18 : List Nat := 5119 :: innerTail56_19
def innerCandidates56_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_18_checked : candidates edge 4 277 5119 innerTail56_19 best.bound = innerCandidates56_18 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_18 : scan edge (search edge 3) 4 [277] 277 innerTail56_18 best = best := by
  rw [innerTail56_18, scan, if_neg (by decide), innerCandidates56_18_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_19
    (if 3 ≤ innerCandidates56_18.length then search edge 3 [277, 5119] innerCandidates56_18 5396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_19

def innerTail56_17 : List Nat := 4483 :: innerTail56_18
def innerCandidates56_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_17_checked : candidates edge 4 277 4483 innerTail56_18 best.bound = innerCandidates56_17 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_17 : scan edge (search edge 3) 4 [277] 277 innerTail56_17 best = best := by
  rw [innerTail56_17, scan, if_neg (by decide), innerCandidates56_17_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_18
    (if 3 ≤ innerCandidates56_17.length then search edge 3 [277, 4483] innerCandidates56_17 4760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_18

def innerTail56_16 : List Nat := 4327 :: innerTail56_17
def innerCandidates56_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_16_checked : candidates edge 4 277 4327 innerTail56_17 best.bound = innerCandidates56_16 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_16 : scan edge (search edge 3) 4 [277] 277 innerTail56_16 best = best := by
  rw [innerTail56_16, scan, if_neg (by decide), innerCandidates56_16_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_17
    (if 3 ≤ innerCandidates56_16.length then search edge 3 [277, 4327] innerCandidates56_16 4604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_17

def innerTail56_15 : List Nat := 3919 :: innerTail56_16
def innerCandidates56_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_15_checked : candidates edge 4 277 3919 innerTail56_16 best.bound = innerCandidates56_15 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_15 : scan edge (search edge 3) 4 [277] 277 innerTail56_15 best = best := by
  rw [innerTail56_15, scan, if_neg (by decide), innerCandidates56_15_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_16
    (if 3 ≤ innerCandidates56_15.length then search edge 3 [277, 3919] innerCandidates56_15 4196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_16

def innerTail56_14 : List Nat := 3307 :: innerTail56_15
def innerCandidates56_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_14_checked : candidates edge 4 277 3307 innerTail56_15 best.bound = innerCandidates56_14 := by
  unfold candidates
  have row : edge 3307 = SearchData.row3307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_14 : scan edge (search edge 3) 4 [277] 277 innerTail56_14 best = best := by
  rw [innerTail56_14, scan, if_neg (by decide), innerCandidates56_14_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_15
    (if 3 ≤ innerCandidates56_14.length then search edge 3 [277, 3307] innerCandidates56_14 3584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_15

def innerTail56_13 : List Nat := 3079 :: innerTail56_14
def innerCandidates56_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_13_checked : candidates edge 4 277 3079 innerTail56_14 best.bound = innerCandidates56_13 := by
  unfold candidates
  have row : edge 3079 = SearchData.row3079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_13 : scan edge (search edge 3) 4 [277] 277 innerTail56_13 best = best := by
  rw [innerTail56_13, scan, if_neg (by decide), innerCandidates56_13_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_14
    (if 3 ≤ innerCandidates56_13.length then search edge 3 [277, 3079] innerCandidates56_13 3356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_14

def innerTail56_12 : List Nat := 3037 :: innerTail56_13
def innerCandidates56_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_12_checked : candidates edge 4 277 3037 innerTail56_13 best.bound = innerCandidates56_12 := by
  unfold candidates
  have row : edge 3037 = SearchData.row3037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_12 : scan edge (search edge 3) 4 [277] 277 innerTail56_12 best = best := by
  rw [innerTail56_12, scan, if_neg (by decide), innerCandidates56_12_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_13
    (if 3 ≤ innerCandidates56_12.length then search edge 3 [277, 3037] innerCandidates56_12 3314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_13

def innerTail56_11 : List Nat := 2593 :: innerTail56_12
def innerCandidates56_11 : List Nat := [3307]
set_option maxRecDepth 4096 in
theorem innerCandidates56_11_checked : candidates edge 4 277 2593 innerTail56_12 best.bound = innerCandidates56_11 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_11 : scan edge (search edge 3) 4 [277] 277 innerTail56_11 best = best := by
  rw [innerTail56_11, scan, if_neg (by decide), innerCandidates56_11_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_12
    (if 3 ≤ innerCandidates56_11.length then search edge 3 [277, 2593] innerCandidates56_11 2870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_12

def innerTail56_10 : List Nat := 2437 :: innerTail56_11
def innerCandidates56_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_10_checked : candidates edge 4 277 2437 innerTail56_11 best.bound = innerCandidates56_10 := by
  unfold candidates
  have row : edge 2437 = PrimePairCache.smallRow2437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_10 : scan edge (search edge 3) 4 [277] 277 innerTail56_10 best = best := by
  rw [innerTail56_10, scan, if_neg (by decide), innerCandidates56_10_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_11
    (if 3 ≤ innerCandidates56_10.length then search edge 3 [277, 2437] innerCandidates56_10 2714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_11

def innerTail56_9 : List Nat := 2281 :: innerTail56_10
def innerCandidates56_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_9_checked : candidates edge 4 277 2281 innerTail56_10 best.bound = innerCandidates56_9 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_9 : scan edge (search edge 3) 4 [277] 277 innerTail56_9 best = best := by
  rw [innerTail56_9, scan, if_neg (by decide), innerCandidates56_9_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_10
    (if 3 ≤ innerCandidates56_9.length then search edge 3 [277, 2281] innerCandidates56_9 2558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_10

def innerTail56_8 : List Nat := 1933 :: innerTail56_9
def innerCandidates56_8 : List Nat := [6841]
set_option maxRecDepth 4096 in
theorem innerCandidates56_8_checked : candidates edge 4 277 1933 innerTail56_9 best.bound = innerCandidates56_8 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_8 : scan edge (search edge 3) 4 [277] 277 innerTail56_8 best = best := by
  rw [innerTail56_8, scan, if_neg (by decide), innerCandidates56_8_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_9
    (if 3 ≤ innerCandidates56_8.length then search edge 3 [277, 1933] innerCandidates56_8 2210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_9

def innerTail56_7 : List Nat := 1693 :: innerTail56_8
def innerCandidates56_7 : List Nat := [2593, 3919, 4327]
set_option maxRecDepth 4096 in
theorem innerCandidates56_7_checked : candidates edge 4 277 1693 innerTail56_8 best.bound = innerCandidates56_7 := by
  unfold candidates
  have row : edge 1693 = PrimePairCache.smallRow1693.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild56_7 : search edge 3 [277, 1693] innerCandidates56_7 1970 best = best := by decide

theorem innerScan56_7 : scan edge (search edge 3) 4 [277] 277 innerTail56_7 best = best := by
  rw [innerTail56_7, scan, if_neg (by decide), innerCandidates56_7_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_8
    (if 3 ≤ innerCandidates56_7.length then search edge 3 [277, 1693] innerCandidates56_7 1970 best else best) = best
  rw [if_pos (by decide), innerChild56_7]
  exact innerScan56_8

def innerTail56_6 : List Nat := 1471 :: innerTail56_7
def innerCandidates56_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_6_checked : candidates edge 4 277 1471 innerTail56_7 best.bound = innerCandidates56_6 := by
  unfold candidates
  have row : edge 1471 = PrimePairCache.smallRow1471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_6 : scan edge (search edge 3) 4 [277] 277 innerTail56_6 best = best := by
  rw [innerTail56_6, scan, if_neg (by decide), innerCandidates56_6_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_7
    (if 3 ≤ innerCandidates56_6.length then search edge 3 [277, 1471] innerCandidates56_6 1748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_7

def innerTail56_5 : List Nat := 1381 :: innerTail56_6
def innerCandidates56_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates56_5_checked : candidates edge 4 277 1381 innerTail56_6 best.bound = innerCandidates56_5 := by
  unfold candidates
  have row : edge 1381 = PrimePairCache.smallRow1381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_5 : scan edge (search edge 3) 4 [277] 277 innerTail56_5 best = best := by
  rw [innerTail56_5, scan, if_neg (by decide), innerCandidates56_5_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_6
    (if 3 ≤ innerCandidates56_5.length then search edge 3 [277, 1381] innerCandidates56_5 1658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_6

def innerTail56_4 : List Nat := 859 :: innerTail56_5
def innerCandidates56_4 : List Nat := [1381, 3307, 7213]
set_option maxRecDepth 4096 in
theorem innerCandidates56_4_checked : candidates edge 4 277 859 innerTail56_5 best.bound = innerCandidates56_4 := by
  unfold candidates
  have row : edge 859 = PrimePairCache.smallRow859.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild56_4 : search edge 3 [277, 859] innerCandidates56_4 1136 best = best := by decide

theorem innerScan56_4 : scan edge (search edge 3) 4 [277] 277 innerTail56_4 best = best := by
  rw [innerTail56_4, scan, if_neg (by decide), innerCandidates56_4_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_5
    (if 3 ≤ innerCandidates56_4.length then search edge 3 [277, 859] innerCandidates56_4 1136 best else best) = best
  rw [if_pos (by decide), innerChild56_4]
  exact innerScan56_5

def innerTail56_3 : List Nat := 787 :: innerTail56_4
def innerCandidates56_3 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates56_3_checked : candidates edge 4 277 787 innerTail56_4 best.bound = innerCandidates56_3 := by
  unfold candidates
  have row : edge 787 = PrimePairCache.smallRow787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_3 : scan edge (search edge 3) 4 [277] 277 innerTail56_3 best = best := by
  rw [innerTail56_3, scan, if_neg (by decide), innerCandidates56_3_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_4
    (if 3 ≤ innerCandidates56_3.length then search edge 3 [277, 787] innerCandidates56_3 1064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_4

def innerTail56_2 : List Nat := 751 :: innerTail56_3
def innerCandidates56_2 : List Nat := [787, 2437]
set_option maxRecDepth 4096 in
theorem innerCandidates56_2_checked : candidates edge 4 277 751 innerTail56_3 best.bound = innerCandidates56_2 := by
  unfold candidates
  have row : edge 751 = PrimePairCache.smallRow751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_2 : scan edge (search edge 3) 4 [277] 277 innerTail56_2 best = best := by
  rw [innerTail56_2, scan, if_neg (by decide), innerCandidates56_2_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_3
    (if 3 ≤ innerCandidates56_2.length then search edge 3 [277, 751] innerCandidates56_2 1028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_3

def innerTail56_1 : List Nat := 499 :: innerTail56_2
def innerCandidates56_1 : List Nat := [1933]
set_option maxRecDepth 4096 in
theorem innerCandidates56_1_checked : candidates edge 4 277 499 innerTail56_2 best.bound = innerCandidates56_1 := by
  unfold candidates
  have row : edge 499 = PrimePairCache.smallRow499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_1 : scan edge (search edge 3) 4 [277] 277 innerTail56_1 best = best := by
  rw [innerTail56_1, scan, if_neg (by decide), innerCandidates56_1_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_2
    (if 3 ≤ innerCandidates56_1.length then search edge 3 [277, 499] innerCandidates56_1 776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_2

def innerTail56_0 : List Nat := 331 :: innerTail56_1
def innerCandidates56_0 : List Nat := [1933, 6427]
set_option maxRecDepth 4096 in
theorem innerCandidates56_0_checked : candidates edge 4 277 331 innerTail56_1 best.bound = innerCandidates56_0 := by
  unfold candidates
  have row : edge 331 = PrimePairCache.smallRow331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan56_0 : scan edge (search edge 3) 4 [277] 277 innerTail56_0 best = best := by
  rw [innerTail56_0, scan, if_neg (by decide), innerCandidates56_0_checked]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_1
    (if 3 ≤ innerCandidates56_0.length then search edge 3 [277, 331] innerCandidates56_0 608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan56_1

theorem rootBranch56 : search edge 4 [277] rootCandidates56 277 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [277] 277 innerTail56_0 best = best
  exact innerScan56_0

def rootCandidates55 : List Nat := [409, 1087, 1213, 1399, 1549, 1567, 1747, 1873, 2089, 2131, 2179, 2971, 3457, 3541, 4111, 5701, 5737, 6163, 6451, 6733, 7177, 7549, 7639, 8101, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates55_checked : candidates edge 5 0 271 tail56 best.bound = rootCandidates55 := by
  unfold candidates
  have row : edge 271 = PrimePairCache.smallRow271.contains := by funext q; rfl
  rw [row]
  decide

def innerTail55_25 : List Nat := []
theorem innerScan55_25 : scan edge (search edge 3) 4 [271] 271 innerTail55_25 best = best := rfl

def innerTail55_24 : List Nat := 8269 :: innerTail55_25
def innerCandidates55_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_24_checked : candidates edge 4 271 8269 innerTail55_25 best.bound = innerCandidates55_24 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_24 : scan edge (search edge 3) 4 [271] 271 innerTail55_24 best = best := by
  rw [innerTail55_24, scan, if_neg (by decide), innerCandidates55_24_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_25
    (if 3 ≤ innerCandidates55_24.length then search edge 3 [271, 8269] innerCandidates55_24 8540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_25

def innerTail55_23 : List Nat := 8101 :: innerTail55_24
def innerCandidates55_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_23_checked : candidates edge 4 271 8101 innerTail55_24 best.bound = innerCandidates55_23 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_23 : scan edge (search edge 3) 4 [271] 271 innerTail55_23 best = best := by
  rw [innerTail55_23, scan, if_neg (by decide), innerCandidates55_23_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_24
    (if 3 ≤ innerCandidates55_23.length then search edge 3 [271, 8101] innerCandidates55_23 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_24

def innerTail55_22 : List Nat := 7639 :: innerTail55_23
def innerCandidates55_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_22_checked : candidates edge 4 271 7639 innerTail55_23 best.bound = innerCandidates55_22 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_22 : scan edge (search edge 3) 4 [271] 271 innerTail55_22 best = best := by
  rw [innerTail55_22, scan, if_neg (by decide), innerCandidates55_22_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_23
    (if 3 ≤ innerCandidates55_22.length then search edge 3 [271, 7639] innerCandidates55_22 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_23

def innerTail55_21 : List Nat := 7549 :: innerTail55_22
def innerCandidates55_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_21_checked : candidates edge 4 271 7549 innerTail55_22 best.bound = innerCandidates55_21 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_21 : scan edge (search edge 3) 4 [271] 271 innerTail55_21 best = best := by
  rw [innerTail55_21, scan, if_neg (by decide), innerCandidates55_21_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_22
    (if 3 ≤ innerCandidates55_21.length then search edge 3 [271, 7549] innerCandidates55_21 7820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_22

def innerTail55_20 : List Nat := 7177 :: innerTail55_21
def innerCandidates55_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_20_checked : candidates edge 4 271 7177 innerTail55_21 best.bound = innerCandidates55_20 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_20 : scan edge (search edge 3) 4 [271] 271 innerTail55_20 best = best := by
  rw [innerTail55_20, scan, if_neg (by decide), innerCandidates55_20_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_21
    (if 3 ≤ innerCandidates55_20.length then search edge 3 [271, 7177] innerCandidates55_20 7448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_21

def innerTail55_19 : List Nat := 6733 :: innerTail55_20
def innerCandidates55_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_19_checked : candidates edge 4 271 6733 innerTail55_20 best.bound = innerCandidates55_19 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_19 : scan edge (search edge 3) 4 [271] 271 innerTail55_19 best = best := by
  rw [innerTail55_19, scan, if_neg (by decide), innerCandidates55_19_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_20
    (if 3 ≤ innerCandidates55_19.length then search edge 3 [271, 6733] innerCandidates55_19 7004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_20

def innerTail55_18 : List Nat := 6451 :: innerTail55_19
def innerCandidates55_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_18_checked : candidates edge 4 271 6451 innerTail55_19 best.bound = innerCandidates55_18 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_18 : scan edge (search edge 3) 4 [271] 271 innerTail55_18 best = best := by
  rw [innerTail55_18, scan, if_neg (by decide), innerCandidates55_18_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_19
    (if 3 ≤ innerCandidates55_18.length then search edge 3 [271, 6451] innerCandidates55_18 6722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_19

def innerTail55_17 : List Nat := 6163 :: innerTail55_18
def innerCandidates55_17 : List Nat := [6451]
set_option maxRecDepth 4096 in
theorem innerCandidates55_17_checked : candidates edge 4 271 6163 innerTail55_18 best.bound = innerCandidates55_17 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_17 : scan edge (search edge 3) 4 [271] 271 innerTail55_17 best = best := by
  rw [innerTail55_17, scan, if_neg (by decide), innerCandidates55_17_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_18
    (if 3 ≤ innerCandidates55_17.length then search edge 3 [271, 6163] innerCandidates55_17 6434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_18

def innerTail55_16 : List Nat := 5737 :: innerTail55_17
def innerCandidates55_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_16_checked : candidates edge 4 271 5737 innerTail55_17 best.bound = innerCandidates55_16 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_16 : scan edge (search edge 3) 4 [271] 271 innerTail55_16 best = best := by
  rw [innerTail55_16, scan, if_neg (by decide), innerCandidates55_16_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_17
    (if 3 ≤ innerCandidates55_16.length then search edge 3 [271, 5737] innerCandidates55_16 6008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_17

def innerTail55_15 : List Nat := 5701 :: innerTail55_16
def innerCandidates55_15 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates55_15_checked : candidates edge 4 271 5701 innerTail55_16 best.bound = innerCandidates55_15 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_15 : scan edge (search edge 3) 4 [271] 271 innerTail55_15 best = best := by
  rw [innerTail55_15, scan, if_neg (by decide), innerCandidates55_15_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_16
    (if 3 ≤ innerCandidates55_15.length then search edge 3 [271, 5701] innerCandidates55_15 5972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_16

def innerTail55_14 : List Nat := 4111 :: innerTail55_15
def innerCandidates55_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_14_checked : candidates edge 4 271 4111 innerTail55_15 best.bound = innerCandidates55_14 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_14 : scan edge (search edge 3) 4 [271] 271 innerTail55_14 best = best := by
  rw [innerTail55_14, scan, if_neg (by decide), innerCandidates55_14_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_15
    (if 3 ≤ innerCandidates55_14.length then search edge 3 [271, 4111] innerCandidates55_14 4382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_15

def innerTail55_13 : List Nat := 3541 :: innerTail55_14
def innerCandidates55_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_13_checked : candidates edge 4 271 3541 innerTail55_14 best.bound = innerCandidates55_13 := by
  unfold candidates
  have row : edge 3541 = SearchData.row3541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_13 : scan edge (search edge 3) 4 [271] 271 innerTail55_13 best = best := by
  rw [innerTail55_13, scan, if_neg (by decide), innerCandidates55_13_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_14
    (if 3 ≤ innerCandidates55_13.length then search edge 3 [271, 3541] innerCandidates55_13 3812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_14

def innerTail55_12 : List Nat := 3457 :: innerTail55_13
def innerCandidates55_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_12_checked : candidates edge 4 271 3457 innerTail55_13 best.bound = innerCandidates55_12 := by
  unfold candidates
  have row : edge 3457 = SearchData.row3457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_12 : scan edge (search edge 3) 4 [271] 271 innerTail55_12 best = best := by
  rw [innerTail55_12, scan, if_neg (by decide), innerCandidates55_12_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_13
    (if 3 ≤ innerCandidates55_12.length then search edge 3 [271, 3457] innerCandidates55_12 3728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_13

def innerTail55_11 : List Nat := 2971 :: innerTail55_12
def innerCandidates55_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_11_checked : candidates edge 4 271 2971 innerTail55_12 best.bound = innerCandidates55_11 := by
  unfold candidates
  have row : edge 2971 = SearchData.row2971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_11 : scan edge (search edge 3) 4 [271] 271 innerTail55_11 best = best := by
  rw [innerTail55_11, scan, if_neg (by decide), innerCandidates55_11_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_12
    (if 3 ≤ innerCandidates55_11.length then search edge 3 [271, 2971] innerCandidates55_11 3242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_12

def innerTail55_10 : List Nat := 2179 :: innerTail55_11
def innerCandidates55_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_10_checked : candidates edge 4 271 2179 innerTail55_11 best.bound = innerCandidates55_10 := by
  unfold candidates
  have row : edge 2179 = PrimePairCache.smallRow2179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_10 : scan edge (search edge 3) 4 [271] 271 innerTail55_10 best = best := by
  rw [innerTail55_10, scan, if_neg (by decide), innerCandidates55_10_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_11
    (if 3 ≤ innerCandidates55_10.length then search edge 3 [271, 2179] innerCandidates55_10 2450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_11

def innerTail55_9 : List Nat := 2131 :: innerTail55_10
def innerCandidates55_9 : List Nat := [3541]
set_option maxRecDepth 4096 in
theorem innerCandidates55_9_checked : candidates edge 4 271 2131 innerTail55_10 best.bound = innerCandidates55_9 := by
  unfold candidates
  have row : edge 2131 = PrimePairCache.smallRow2131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_9 : scan edge (search edge 3) 4 [271] 271 innerTail55_9 best = best := by
  rw [innerTail55_9, scan, if_neg (by decide), innerCandidates55_9_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_10
    (if 3 ≤ innerCandidates55_9.length then search edge 3 [271, 2131] innerCandidates55_9 2402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_10

def innerTail55_8 : List Nat := 2089 :: innerTail55_9
def innerCandidates55_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_8_checked : candidates edge 4 271 2089 innerTail55_9 best.bound = innerCandidates55_8 := by
  unfold candidates
  have row : edge 2089 = PrimePairCache.smallRow2089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_8 : scan edge (search edge 3) 4 [271] 271 innerTail55_8 best = best := by
  rw [innerTail55_8, scan, if_neg (by decide), innerCandidates55_8_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_9
    (if 3 ≤ innerCandidates55_8.length then search edge 3 [271, 2089] innerCandidates55_8 2360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_9

def innerTail55_7 : List Nat := 1873 :: innerTail55_8
def innerCandidates55_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_7_checked : candidates edge 4 271 1873 innerTail55_8 best.bound = innerCandidates55_7 := by
  unfold candidates
  have row : edge 1873 = PrimePairCache.smallRow1873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_7 : scan edge (search edge 3) 4 [271] 271 innerTail55_7 best = best := by
  rw [innerTail55_7, scan, if_neg (by decide), innerCandidates55_7_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_8
    (if 3 ≤ innerCandidates55_7.length then search edge 3 [271, 1873] innerCandidates55_7 2144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_8

def innerTail55_6 : List Nat := 1747 :: innerTail55_7
def innerCandidates55_6 : List Nat := [1873, 2179, 5737, 7177, 7639]
set_option maxRecDepth 4096 in
theorem innerCandidates55_6_checked : candidates edge 4 271 1747 innerTail55_7 best.bound = innerCandidates55_6 := by
  unfold candidates
  have row : edge 1747 = PrimePairCache.smallRow1747.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild55_6 : search edge 3 [271, 1747] innerCandidates55_6 2018 best = best := by decide

theorem innerScan55_6 : scan edge (search edge 3) 4 [271] 271 innerTail55_6 best = best := by
  rw [innerTail55_6, scan, if_neg (by decide), innerCandidates55_6_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_7
    (if 3 ≤ innerCandidates55_6.length then search edge 3 [271, 1747] innerCandidates55_6 2018 best else best) = best
  rw [if_pos (by decide), innerChild55_6]
  exact innerScan55_7

def innerTail55_5 : List Nat := 1567 :: innerTail55_6
def innerCandidates55_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_5_checked : candidates edge 4 271 1567 innerTail55_6 best.bound = innerCandidates55_5 := by
  unfold candidates
  have row : edge 1567 = PrimePairCache.smallRow1567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_5 : scan edge (search edge 3) 4 [271] 271 innerTail55_5 best = best := by
  rw [innerTail55_5, scan, if_neg (by decide), innerCandidates55_5_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_6
    (if 3 ≤ innerCandidates55_5.length then search edge 3 [271, 1567] innerCandidates55_5 1838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_6

def innerTail55_4 : List Nat := 1549 :: innerTail55_5
def innerCandidates55_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_4_checked : candidates edge 4 271 1549 innerTail55_5 best.bound = innerCandidates55_4 := by
  unfold candidates
  have row : edge 1549 = PrimePairCache.smallRow1549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_4 : scan edge (search edge 3) 4 [271] 271 innerTail55_4 best = best := by
  rw [innerTail55_4, scan, if_neg (by decide), innerCandidates55_4_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_5
    (if 3 ≤ innerCandidates55_4.length then search edge 3 [271, 1549] innerCandidates55_4 1820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_5

def innerTail55_3 : List Nat := 1399 :: innerTail55_4
def innerCandidates55_3 : List Nat := [2131]
set_option maxRecDepth 4096 in
theorem innerCandidates55_3_checked : candidates edge 4 271 1399 innerTail55_4 best.bound = innerCandidates55_3 := by
  unfold candidates
  have row : edge 1399 = PrimePairCache.smallRow1399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_3 : scan edge (search edge 3) 4 [271] 271 innerTail55_3 best = best := by
  rw [innerTail55_3, scan, if_neg (by decide), innerCandidates55_3_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_4
    (if 3 ≤ innerCandidates55_3.length then search edge 3 [271, 1399] innerCandidates55_3 1670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_4

def innerTail55_2 : List Nat := 1213 :: innerTail55_3
def innerCandidates55_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates55_2_checked : candidates edge 4 271 1213 innerTail55_3 best.bound = innerCandidates55_2 := by
  unfold candidates
  have row : edge 1213 = PrimePairCache.smallRow1213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_2 : scan edge (search edge 3) 4 [271] 271 innerTail55_2 best = best := by
  rw [innerTail55_2, scan, if_neg (by decide), innerCandidates55_2_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_3
    (if 3 ≤ innerCandidates55_2.length then search edge 3 [271, 1213] innerCandidates55_2 1484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_3

def innerTail55_1 : List Nat := 1087 :: innerTail55_2
def innerCandidates55_1 : List Nat := [2179, 6451]
set_option maxRecDepth 4096 in
theorem innerCandidates55_1_checked : candidates edge 4 271 1087 innerTail55_2 best.bound = innerCandidates55_1 := by
  unfold candidates
  have row : edge 1087 = PrimePairCache.smallRow1087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_1 : scan edge (search edge 3) 4 [271] 271 innerTail55_1 best = best := by
  rw [innerTail55_1, scan, if_neg (by decide), innerCandidates55_1_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_2
    (if 3 ≤ innerCandidates55_1.length then search edge 3 [271, 1087] innerCandidates55_1 1358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_2

def innerTail55_0 : List Nat := 409 :: innerTail55_1
def innerCandidates55_0 : List Nat := [1873]
set_option maxRecDepth 4096 in
theorem innerCandidates55_0_checked : candidates edge 4 271 409 innerTail55_1 best.bound = innerCandidates55_0 := by
  unfold candidates
  have row : edge 409 = PrimePairCache.smallRow409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan55_0 : scan edge (search edge 3) 4 [271] 271 innerTail55_0 best = best := by
  rw [innerTail55_0, scan, if_neg (by decide), innerCandidates55_0_checked]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_1
    (if 3 ≤ innerCandidates55_0.length then search edge 3 [271, 409] innerCandidates55_0 680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan55_1

theorem rootBranch55 : search edge 4 [271] rootCandidates55 271 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [271] 271 innerTail55_0 best = best
  exact innerScan55_0

end ProjectEulerSolutions.PrimePairUpper.Search
