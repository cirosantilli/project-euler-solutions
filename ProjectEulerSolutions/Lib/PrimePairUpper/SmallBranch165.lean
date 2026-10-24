import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates184 : List Nat := [1201, 1327, 1429, 2269, 2341, 2731, 2797, 2851, 3067, 3229, 3343, 3511, 3637, 3823, 4357, 4363, 4561, 4993, 5077, 5167, 5323, 5869, 6133, 6211, 6637, 6679, 7591]
set_option maxRecDepth 4096 in
theorem rootCandidates184_checked : candidates edge 5 0 1117 tail185 best.bound = rootCandidates184 := by
  unfold candidates
  have row : edge 1117 = PrimePairCache.smallRow1117.contains := by funext q; rfl
  rw [row]
  decide

def innerTail184_27 : List Nat := []
theorem innerScan184_27 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_27 best = best := rfl

def innerTail184_26 : List Nat := 7591 :: innerTail184_27
def innerCandidates184_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_26_checked : candidates edge 4 1117 7591 innerTail184_27 best.bound = innerCandidates184_26 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_26 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_26 best = best := by
  rw [innerTail184_26, scan, if_neg (by decide), innerCandidates184_26_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_27
    (if 3 ≤ innerCandidates184_26.length then search edge 3 [1117, 7591] innerCandidates184_26 8708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_27

def innerTail184_25 : List Nat := 6679 :: innerTail184_26
def innerCandidates184_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_25_checked : candidates edge 4 1117 6679 innerTail184_26 best.bound = innerCandidates184_25 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_25 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_25 best = best := by
  rw [innerTail184_25, scan, if_neg (by decide), innerCandidates184_25_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_26
    (if 3 ≤ innerCandidates184_25.length then search edge 3 [1117, 6679] innerCandidates184_25 7796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_26

def innerTail184_24 : List Nat := 6637 :: innerTail184_25
def innerCandidates184_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_24_checked : candidates edge 4 1117 6637 innerTail184_25 best.bound = innerCandidates184_24 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_24 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_24 best = best := by
  rw [innerTail184_24, scan, if_neg (by decide), innerCandidates184_24_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_25
    (if 3 ≤ innerCandidates184_24.length then search edge 3 [1117, 6637] innerCandidates184_24 7754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_25

def innerTail184_23 : List Nat := 6211 :: innerTail184_24
def innerCandidates184_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_23_checked : candidates edge 4 1117 6211 innerTail184_24 best.bound = innerCandidates184_23 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_23 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_23 best = best := by
  rw [innerTail184_23, scan, if_neg (by decide), innerCandidates184_23_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_24
    (if 3 ≤ innerCandidates184_23.length then search edge 3 [1117, 6211] innerCandidates184_23 7328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_24

def innerTail184_22 : List Nat := 6133 :: innerTail184_23
def innerCandidates184_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_22_checked : candidates edge 4 1117 6133 innerTail184_23 best.bound = innerCandidates184_22 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_22 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_22 best = best := by
  rw [innerTail184_22, scan, if_neg (by decide), innerCandidates184_22_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_23
    (if 3 ≤ innerCandidates184_22.length then search edge 3 [1117, 6133] innerCandidates184_22 7250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_23

def innerTail184_21 : List Nat := 5869 :: innerTail184_22
def innerCandidates184_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_21_checked : candidates edge 4 1117 5869 innerTail184_22 best.bound = innerCandidates184_21 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_21 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_21 best = best := by
  rw [innerTail184_21, scan, if_neg (by decide), innerCandidates184_21_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_22
    (if 3 ≤ innerCandidates184_21.length then search edge 3 [1117, 5869] innerCandidates184_21 6986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_22

def innerTail184_20 : List Nat := 5323 :: innerTail184_21
def innerCandidates184_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_20_checked : candidates edge 4 1117 5323 innerTail184_21 best.bound = innerCandidates184_20 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_20 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_20 best = best := by
  rw [innerTail184_20, scan, if_neg (by decide), innerCandidates184_20_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_21
    (if 3 ≤ innerCandidates184_20.length then search edge 3 [1117, 5323] innerCandidates184_20 6440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_21

def innerTail184_19 : List Nat := 5167 :: innerTail184_20
def innerCandidates184_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_19_checked : candidates edge 4 1117 5167 innerTail184_20 best.bound = innerCandidates184_19 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_19 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_19 best = best := by
  rw [innerTail184_19, scan, if_neg (by decide), innerCandidates184_19_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_20
    (if 3 ≤ innerCandidates184_19.length then search edge 3 [1117, 5167] innerCandidates184_19 6284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_20

def innerTail184_18 : List Nat := 5077 :: innerTail184_19
def innerCandidates184_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_18_checked : candidates edge 4 1117 5077 innerTail184_19 best.bound = innerCandidates184_18 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_18 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_18 best = best := by
  rw [innerTail184_18, scan, if_neg (by decide), innerCandidates184_18_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_19
    (if 3 ≤ innerCandidates184_18.length then search edge 3 [1117, 5077] innerCandidates184_18 6194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_19

def innerTail184_17 : List Nat := 4993 :: innerTail184_18
def innerCandidates184_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_17_checked : candidates edge 4 1117 4993 innerTail184_18 best.bound = innerCandidates184_17 := by
  unfold candidates
  have row : edge 4993 = SearchData.row4993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_17 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_17 best = best := by
  rw [innerTail184_17, scan, if_neg (by decide), innerCandidates184_17_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_18
    (if 3 ≤ innerCandidates184_17.length then search edge 3 [1117, 4993] innerCandidates184_17 6110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_18

def innerTail184_16 : List Nat := 4561 :: innerTail184_17
def innerCandidates184_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_16_checked : candidates edge 4 1117 4561 innerTail184_17 best.bound = innerCandidates184_16 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_16 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_16 best = best := by
  rw [innerTail184_16, scan, if_neg (by decide), innerCandidates184_16_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_17
    (if 3 ≤ innerCandidates184_16.length then search edge 3 [1117, 4561] innerCandidates184_16 5678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_17

def innerTail184_15 : List Nat := 4363 :: innerTail184_16
def innerCandidates184_15 : List Nat := [6133]
set_option maxRecDepth 4096 in
theorem innerCandidates184_15_checked : candidates edge 4 1117 4363 innerTail184_16 best.bound = innerCandidates184_15 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_15 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_15 best = best := by
  rw [innerTail184_15, scan, if_neg (by decide), innerCandidates184_15_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_16
    (if 3 ≤ innerCandidates184_15.length then search edge 3 [1117, 4363] innerCandidates184_15 5480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_16

def innerTail184_14 : List Nat := 4357 :: innerTail184_15
def innerCandidates184_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_14_checked : candidates edge 4 1117 4357 innerTail184_15 best.bound = innerCandidates184_14 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_14 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_14 best = best := by
  rw [innerTail184_14, scan, if_neg (by decide), innerCandidates184_14_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_15
    (if 3 ≤ innerCandidates184_14.length then search edge 3 [1117, 4357] innerCandidates184_14 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_15

def innerTail184_13 : List Nat := 3823 :: innerTail184_14
def innerCandidates184_13 : List Nat := [5167]
set_option maxRecDepth 4096 in
theorem innerCandidates184_13_checked : candidates edge 4 1117 3823 innerTail184_14 best.bound = innerCandidates184_13 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_13 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_13 best = best := by
  rw [innerTail184_13, scan, if_neg (by decide), innerCandidates184_13_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_14
    (if 3 ≤ innerCandidates184_13.length then search edge 3 [1117, 3823] innerCandidates184_13 4940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_14

def innerTail184_12 : List Nat := 3637 :: innerTail184_13
def innerCandidates184_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_12_checked : candidates edge 4 1117 3637 innerTail184_13 best.bound = innerCandidates184_12 := by
  unfold candidates
  have row : edge 3637 = SearchData.row3637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_12 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_12 best = best := by
  rw [innerTail184_12, scan, if_neg (by decide), innerCandidates184_12_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_13
    (if 3 ≤ innerCandidates184_12.length then search edge 3 [1117, 3637] innerCandidates184_12 4754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_13

def innerTail184_11 : List Nat := 3511 :: innerTail184_12
def innerCandidates184_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_11_checked : candidates edge 4 1117 3511 innerTail184_12 best.bound = innerCandidates184_11 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_11 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_11 best = best := by
  rw [innerTail184_11, scan, if_neg (by decide), innerCandidates184_11_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_12
    (if 3 ≤ innerCandidates184_11.length then search edge 3 [1117, 3511] innerCandidates184_11 4628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_12

def innerTail184_10 : List Nat := 3343 :: innerTail184_11
def innerCandidates184_10 : List Nat := [3823, 4561]
set_option maxRecDepth 4096 in
theorem innerCandidates184_10_checked : candidates edge 4 1117 3343 innerTail184_11 best.bound = innerCandidates184_10 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_10 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_10 best = best := by
  rw [innerTail184_10, scan, if_neg (by decide), innerCandidates184_10_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_11
    (if 3 ≤ innerCandidates184_10.length then search edge 3 [1117, 3343] innerCandidates184_10 4460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_11

def innerTail184_9 : List Nat := 3229 :: innerTail184_10
def innerCandidates184_9 : List Nat := [4993]
set_option maxRecDepth 4096 in
theorem innerCandidates184_9_checked : candidates edge 4 1117 3229 innerTail184_10 best.bound = innerCandidates184_9 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_9 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_9 best = best := by
  rw [innerTail184_9, scan, if_neg (by decide), innerCandidates184_9_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_10
    (if 3 ≤ innerCandidates184_9.length then search edge 3 [1117, 3229] innerCandidates184_9 4346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_10

def innerTail184_8 : List Nat := 3067 :: innerTail184_9
def innerCandidates184_8 : List Nat := [5167]
set_option maxRecDepth 4096 in
theorem innerCandidates184_8_checked : candidates edge 4 1117 3067 innerTail184_9 best.bound = innerCandidates184_8 := by
  unfold candidates
  have row : edge 3067 = SearchData.row3067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_8 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_8 best = best := by
  rw [innerTail184_8, scan, if_neg (by decide), innerCandidates184_8_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_9
    (if 3 ≤ innerCandidates184_8.length then search edge 3 [1117, 3067] innerCandidates184_8 4184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_9

def innerTail184_7 : List Nat := 2851 :: innerTail184_8
def innerCandidates184_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_7_checked : candidates edge 4 1117 2851 innerTail184_8 best.bound = innerCandidates184_7 := by
  unfold candidates
  have row : edge 2851 = SearchData.row2851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_7 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_7 best = best := by
  rw [innerTail184_7, scan, if_neg (by decide), innerCandidates184_7_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_8
    (if 3 ≤ innerCandidates184_7.length then search edge 3 [1117, 2851] innerCandidates184_7 3968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_8

def innerTail184_6 : List Nat := 2797 :: innerTail184_7
def innerCandidates184_6 : List Nat := [4561]
set_option maxRecDepth 4096 in
theorem innerCandidates184_6_checked : candidates edge 4 1117 2797 innerTail184_7 best.bound = innerCandidates184_6 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_6 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_6 best = best := by
  rw [innerTail184_6, scan, if_neg (by decide), innerCandidates184_6_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_7
    (if 3 ≤ innerCandidates184_6.length then search edge 3 [1117, 2797] innerCandidates184_6 3914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_7

def innerTail184_5 : List Nat := 2731 :: innerTail184_6
def innerCandidates184_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_5_checked : candidates edge 4 1117 2731 innerTail184_6 best.bound = innerCandidates184_5 := by
  unfold candidates
  have row : edge 2731 = SearchData.row2731.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_5 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_5 best = best := by
  rw [innerTail184_5, scan, if_neg (by decide), innerCandidates184_5_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_6
    (if 3 ≤ innerCandidates184_5.length then search edge 3 [1117, 2731] innerCandidates184_5 3848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_6

def innerTail184_4 : List Nat := 2341 :: innerTail184_5
def innerCandidates184_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates184_4_checked : candidates edge 4 1117 2341 innerTail184_5 best.bound = innerCandidates184_4 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_4 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_4 best = best := by
  rw [innerTail184_4, scan, if_neg (by decide), innerCandidates184_4_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_5
    (if 3 ≤ innerCandidates184_4.length then search edge 3 [1117, 2341] innerCandidates184_4 3458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_5

def innerTail184_3 : List Nat := 2269 :: innerTail184_4
def innerCandidates184_3 : List Nat := [3067]
set_option maxRecDepth 4096 in
theorem innerCandidates184_3_checked : candidates edge 4 1117 2269 innerTail184_4 best.bound = innerCandidates184_3 := by
  unfold candidates
  have row : edge 2269 = PrimePairCache.smallRow2269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_3 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_3 best = best := by
  rw [innerTail184_3, scan, if_neg (by decide), innerCandidates184_3_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_4
    (if 3 ≤ innerCandidates184_3.length then search edge 3 [1117, 2269] innerCandidates184_3 3386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_4

def innerTail184_2 : List Nat := 1429 :: innerTail184_3
def innerCandidates184_2 : List Nat := [6211]
set_option maxRecDepth 4096 in
theorem innerCandidates184_2_checked : candidates edge 4 1117 1429 innerTail184_3 best.bound = innerCandidates184_2 := by
  unfold candidates
  have row : edge 1429 = PrimePairCache.smallRow1429.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_2 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_2 best = best := by
  rw [innerTail184_2, scan, if_neg (by decide), innerCandidates184_2_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_3
    (if 3 ≤ innerCandidates184_2.length then search edge 3 [1117, 1429] innerCandidates184_2 2546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_3

def innerTail184_1 : List Nat := 1327 :: innerTail184_2
def innerCandidates184_1 : List Nat := [1429, 4357]
set_option maxRecDepth 4096 in
theorem innerCandidates184_1_checked : candidates edge 4 1117 1327 innerTail184_2 best.bound = innerCandidates184_1 := by
  unfold candidates
  have row : edge 1327 = PrimePairCache.smallRow1327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan184_1 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_1 best = best := by
  rw [innerTail184_1, scan, if_neg (by decide), innerCandidates184_1_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_2
    (if 3 ≤ innerCandidates184_1.length then search edge 3 [1117, 1327] innerCandidates184_1 2444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan184_2

def innerTail184_0 : List Nat := 1201 :: innerTail184_1
def innerCandidates184_0 : List Nat := [3229, 3637, 5167]
set_option maxRecDepth 4096 in
theorem innerCandidates184_0_checked : candidates edge 4 1117 1201 innerTail184_1 best.bound = innerCandidates184_0 := by
  unfold candidates
  have row : edge 1201 = PrimePairCache.smallRow1201.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild184_0 : search edge 3 [1117, 1201] innerCandidates184_0 2318 best = best := by decide

theorem innerScan184_0 : scan edge (search edge 3) 4 [1117] 1117 innerTail184_0 best = best := by
  rw [innerTail184_0, scan, if_neg (by decide), innerCandidates184_0_checked]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_1
    (if 3 ≤ innerCandidates184_0.length then search edge 3 [1117, 1201] innerCandidates184_0 2318 best else best) = best
  rw [if_pos (by decide), innerChild184_0]
  exact innerScan184_1

theorem rootBranch184 : search edge 4 [1117] rootCandidates184 1117 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1117] 1117 innerTail184_0 best = best
  exact innerScan184_0

def rootCandidates183 : List Nat := [1181, 1229, 1889, 2027, 2069, 2441, 2999, 3209, 3491, 3917, 3923, 4289, 4493, 4871, 5501, 6521, 6719, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates183_checked : candidates edge 5 0 1109 tail184 best.bound = rootCandidates183 := by
  unfold candidates
  have row : edge 1109 = PrimePairCache.smallRow1109.contains := by funext q; rfl
  rw [row]
  decide

def innerTail183_18 : List Nat := []
theorem innerScan183_18 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_18 best = best := rfl

def innerTail183_17 : List Nat := 8369 :: innerTail183_18
def innerCandidates183_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates183_17_checked : candidates edge 4 1109 8369 innerTail183_18 best.bound = innerCandidates183_17 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_17 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_17 best = best := by
  rw [innerTail183_17, scan, if_neg (by decide), innerCandidates183_17_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_18
    (if 3 ≤ innerCandidates183_17.length then search edge 3 [1109, 8369] innerCandidates183_17 9478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_18

def innerTail183_16 : List Nat := 6719 :: innerTail183_17
def innerCandidates183_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates183_16_checked : candidates edge 4 1109 6719 innerTail183_17 best.bound = innerCandidates183_16 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_16 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_16 best = best := by
  rw [innerTail183_16, scan, if_neg (by decide), innerCandidates183_16_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_17
    (if 3 ≤ innerCandidates183_16.length then search edge 3 [1109, 6719] innerCandidates183_16 7828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_17

def innerTail183_15 : List Nat := 6521 :: innerTail183_16
def innerCandidates183_15 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates183_15_checked : candidates edge 4 1109 6521 innerTail183_16 best.bound = innerCandidates183_15 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_15 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_15 best = best := by
  rw [innerTail183_15, scan, if_neg (by decide), innerCandidates183_15_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_16
    (if 3 ≤ innerCandidates183_15.length then search edge 3 [1109, 6521] innerCandidates183_15 7630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_16

def innerTail183_14 : List Nat := 5501 :: innerTail183_15
def innerCandidates183_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates183_14_checked : candidates edge 4 1109 5501 innerTail183_15 best.bound = innerCandidates183_14 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_14 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_14 best = best := by
  rw [innerTail183_14, scan, if_neg (by decide), innerCandidates183_14_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_15
    (if 3 ≤ innerCandidates183_14.length then search edge 3 [1109, 5501] innerCandidates183_14 6610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_15

def innerTail183_13 : List Nat := 4871 :: innerTail183_14
def innerCandidates183_13 : List Nat := [6719]
set_option maxRecDepth 4096 in
theorem innerCandidates183_13_checked : candidates edge 4 1109 4871 innerTail183_14 best.bound = innerCandidates183_13 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_13 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_13 best = best := by
  rw [innerTail183_13, scan, if_neg (by decide), innerCandidates183_13_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_14
    (if 3 ≤ innerCandidates183_13.length then search edge 3 [1109, 4871] innerCandidates183_13 5980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_14

def innerTail183_12 : List Nat := 4493 :: innerTail183_13
def innerCandidates183_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates183_12_checked : candidates edge 4 1109 4493 innerTail183_13 best.bound = innerCandidates183_12 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_12 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_12 best = best := by
  rw [innerTail183_12, scan, if_neg (by decide), innerCandidates183_12_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_13
    (if 3 ≤ innerCandidates183_12.length then search edge 3 [1109, 4493] innerCandidates183_12 5602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_13

def innerTail183_11 : List Nat := 4289 :: innerTail183_12
def innerCandidates183_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates183_11_checked : candidates edge 4 1109 4289 innerTail183_12 best.bound = innerCandidates183_11 := by
  unfold candidates
  have row : edge 4289 = SearchData.row4289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_11 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_11 best = best := by
  rw [innerTail183_11, scan, if_neg (by decide), innerCandidates183_11_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_12
    (if 3 ≤ innerCandidates183_11.length then search edge 3 [1109, 4289] innerCandidates183_11 5398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_12

def innerTail183_10 : List Nat := 3923 :: innerTail183_11
def innerCandidates183_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates183_10_checked : candidates edge 4 1109 3923 innerTail183_11 best.bound = innerCandidates183_10 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_10 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_10 best = best := by
  rw [innerTail183_10, scan, if_neg (by decide), innerCandidates183_10_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_11
    (if 3 ≤ innerCandidates183_10.length then search edge 3 [1109, 3923] innerCandidates183_10 5032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_11

def innerTail183_9 : List Nat := 3917 :: innerTail183_10
def innerCandidates183_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates183_9_checked : candidates edge 4 1109 3917 innerTail183_10 best.bound = innerCandidates183_9 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_9 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_9 best = best := by
  rw [innerTail183_9, scan, if_neg (by decide), innerCandidates183_9_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_10
    (if 3 ≤ innerCandidates183_9.length then search edge 3 [1109, 3917] innerCandidates183_9 5026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_10

def innerTail183_8 : List Nat := 3491 :: innerTail183_9
def innerCandidates183_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates183_8_checked : candidates edge 4 1109 3491 innerTail183_9 best.bound = innerCandidates183_8 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_8 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_8 best = best := by
  rw [innerTail183_8, scan, if_neg (by decide), innerCandidates183_8_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_9
    (if 3 ≤ innerCandidates183_8.length then search edge 3 [1109, 3491] innerCandidates183_8 4600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_9

def innerTail183_7 : List Nat := 3209 :: innerTail183_8
def innerCandidates183_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates183_7_checked : candidates edge 4 1109 3209 innerTail183_8 best.bound = innerCandidates183_7 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_7 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_7 best = best := by
  rw [innerTail183_7, scan, if_neg (by decide), innerCandidates183_7_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_8
    (if 3 ≤ innerCandidates183_7.length then search edge 3 [1109, 3209] innerCandidates183_7 4318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_8

def innerTail183_6 : List Nat := 2999 :: innerTail183_7
def innerCandidates183_6 : List Nat := [4871]
set_option maxRecDepth 4096 in
theorem innerCandidates183_6_checked : candidates edge 4 1109 2999 innerTail183_7 best.bound = innerCandidates183_6 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_6 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_6 best = best := by
  rw [innerTail183_6, scan, if_neg (by decide), innerCandidates183_6_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_7
    (if 3 ≤ innerCandidates183_6.length then search edge 3 [1109, 2999] innerCandidates183_6 4108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_7

def innerTail183_5 : List Nat := 2441 :: innerTail183_6
def innerCandidates183_5 : List Nat := [3209]
set_option maxRecDepth 4096 in
theorem innerCandidates183_5_checked : candidates edge 4 1109 2441 innerTail183_6 best.bound = innerCandidates183_5 := by
  unfold candidates
  have row : edge 2441 = PrimePairCache.smallRow2441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_5 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_5 best = best := by
  rw [innerTail183_5, scan, if_neg (by decide), innerCandidates183_5_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_6
    (if 3 ≤ innerCandidates183_5.length then search edge 3 [1109, 2441] innerCandidates183_5 3550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_6

def innerTail183_4 : List Nat := 2069 :: innerTail183_5
def innerCandidates183_4 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates183_4_checked : candidates edge 4 1109 2069 innerTail183_5 best.bound = innerCandidates183_4 := by
  unfold candidates
  have row : edge 2069 = PrimePairCache.smallRow2069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_4 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_4 best = best := by
  rw [innerTail183_4, scan, if_neg (by decide), innerCandidates183_4_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_5
    (if 3 ≤ innerCandidates183_4.length then search edge 3 [1109, 2069] innerCandidates183_4 3178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_5

def innerTail183_3 : List Nat := 2027 :: innerTail183_4
def innerCandidates183_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates183_3_checked : candidates edge 4 1109 2027 innerTail183_4 best.bound = innerCandidates183_3 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_3 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_3 best = best := by
  rw [innerTail183_3, scan, if_neg (by decide), innerCandidates183_3_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_4
    (if 3 ≤ innerCandidates183_3.length then search edge 3 [1109, 2027] innerCandidates183_3 3136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_4

def innerTail183_2 : List Nat := 1889 :: innerTail183_3
def innerCandidates183_2 : List Nat := [2999, 3491]
set_option maxRecDepth 4096 in
theorem innerCandidates183_2_checked : candidates edge 4 1109 1889 innerTail183_3 best.bound = innerCandidates183_2 := by
  unfold candidates
  have row : edge 1889 = PrimePairCache.smallRow1889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_2 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_2 best = best := by
  rw [innerTail183_2, scan, if_neg (by decide), innerCandidates183_2_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_3
    (if 3 ≤ innerCandidates183_2.length then search edge 3 [1109, 1889] innerCandidates183_2 2998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_3

def innerTail183_1 : List Nat := 1229 :: innerTail183_2
def innerCandidates183_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates183_1_checked : candidates edge 4 1109 1229 innerTail183_2 best.bound = innerCandidates183_1 := by
  unfold candidates
  have row : edge 1229 = PrimePairCache.smallRow1229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan183_1 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_1 best = best := by
  rw [innerTail183_1, scan, if_neg (by decide), innerCandidates183_1_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_2
    (if 3 ≤ innerCandidates183_1.length then search edge 3 [1109, 1229] innerCandidates183_1 2338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan183_2

def innerTail183_0 : List Nat := 1181 :: innerTail183_1
def innerCandidates183_0 : List Nat := [3209, 5501, 6719]
set_option maxRecDepth 4096 in
theorem innerCandidates183_0_checked : candidates edge 4 1109 1181 innerTail183_1 best.bound = innerCandidates183_0 := by
  unfold candidates
  have row : edge 1181 = PrimePairCache.smallRow1181.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild183_0 : search edge 3 [1109, 1181] innerCandidates183_0 2290 best = best := by decide

theorem innerScan183_0 : scan edge (search edge 3) 4 [1109] 1109 innerTail183_0 best = best := by
  rw [innerTail183_0, scan, if_neg (by decide), innerCandidates183_0_checked]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_1
    (if 3 ≤ innerCandidates183_0.length then search edge 3 [1109, 1181] innerCandidates183_0 2290 best else best) = best
  rw [if_pos (by decide), innerChild183_0]
  exact innerScan183_1

theorem rootBranch183 : search edge 4 [1109] rootCandidates183 1109 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1109] 1109 innerTail183_0 best = best
  exact innerScan183_0

end ProjectEulerSolutions.PrimePairUpper.Search
