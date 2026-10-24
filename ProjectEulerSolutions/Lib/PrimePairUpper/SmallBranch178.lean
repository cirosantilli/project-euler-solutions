import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates152 : List Nat := [1213, 1429, 1459, 1471, 1801, 2383, 2473, 2731, 3709, 3919, 4357, 4423, 4567, 4861, 4987, 4993, 5503, 5851, 6229, 6337, 6733, 6841, 6961, 7027, 7591, 7723, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates152_checked : candidates edge 5 0 907 tail153 best.bound = rootCandidates152 := by
  unfold candidates
  have row : edge 907 = PrimePairCache.smallRow907.contains := by funext q; rfl
  rw [row]
  decide

def innerTail152_27 : List Nat := []
theorem innerScan152_27 : scan edge (search edge 3) 4 [907] 907 innerTail152_27 best = best := rfl

def innerTail152_26 : List Nat := 7993 :: innerTail152_27
def innerCandidates152_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_26_checked : candidates edge 4 907 7993 innerTail152_27 best.bound = innerCandidates152_26 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_26 : scan edge (search edge 3) 4 [907] 907 innerTail152_26 best = best := by
  rw [innerTail152_26, scan, if_neg (by decide), innerCandidates152_26_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_27
    (if 3 ≤ innerCandidates152_26.length then search edge 3 [907, 7993] innerCandidates152_26 8900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_27

def innerTail152_25 : List Nat := 7723 :: innerTail152_26
def innerCandidates152_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_25_checked : candidates edge 4 907 7723 innerTail152_26 best.bound = innerCandidates152_25 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_25 : scan edge (search edge 3) 4 [907] 907 innerTail152_25 best = best := by
  rw [innerTail152_25, scan, if_neg (by decide), innerCandidates152_25_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_26
    (if 3 ≤ innerCandidates152_25.length then search edge 3 [907, 7723] innerCandidates152_25 8630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_26

def innerTail152_24 : List Nat := 7591 :: innerTail152_25
def innerCandidates152_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_24_checked : candidates edge 4 907 7591 innerTail152_25 best.bound = innerCandidates152_24 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_24 : scan edge (search edge 3) 4 [907] 907 innerTail152_24 best = best := by
  rw [innerTail152_24, scan, if_neg (by decide), innerCandidates152_24_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_25
    (if 3 ≤ innerCandidates152_24.length then search edge 3 [907, 7591] innerCandidates152_24 8498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_25

def innerTail152_23 : List Nat := 7027 :: innerTail152_24
def innerCandidates152_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_23_checked : candidates edge 4 907 7027 innerTail152_24 best.bound = innerCandidates152_23 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_23 : scan edge (search edge 3) 4 [907] 907 innerTail152_23 best = best := by
  rw [innerTail152_23, scan, if_neg (by decide), innerCandidates152_23_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_24
    (if 3 ≤ innerCandidates152_23.length then search edge 3 [907, 7027] innerCandidates152_23 7934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_24

def innerTail152_22 : List Nat := 6961 :: innerTail152_23
def innerCandidates152_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_22_checked : candidates edge 4 907 6961 innerTail152_23 best.bound = innerCandidates152_22 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_22 : scan edge (search edge 3) 4 [907] 907 innerTail152_22 best = best := by
  rw [innerTail152_22, scan, if_neg (by decide), innerCandidates152_22_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_23
    (if 3 ≤ innerCandidates152_22.length then search edge 3 [907, 6961] innerCandidates152_22 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_23

def innerTail152_21 : List Nat := 6841 :: innerTail152_22
def innerCandidates152_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_21_checked : candidates edge 4 907 6841 innerTail152_22 best.bound = innerCandidates152_21 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_21 : scan edge (search edge 3) 4 [907] 907 innerTail152_21 best = best := by
  rw [innerTail152_21, scan, if_neg (by decide), innerCandidates152_21_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_22
    (if 3 ≤ innerCandidates152_21.length then search edge 3 [907, 6841] innerCandidates152_21 7748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_22

def innerTail152_20 : List Nat := 6733 :: innerTail152_21
def innerCandidates152_20 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates152_20_checked : candidates edge 4 907 6733 innerTail152_21 best.bound = innerCandidates152_20 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_20 : scan edge (search edge 3) 4 [907] 907 innerTail152_20 best = best := by
  rw [innerTail152_20, scan, if_neg (by decide), innerCandidates152_20_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_21
    (if 3 ≤ innerCandidates152_20.length then search edge 3 [907, 6733] innerCandidates152_20 7640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_21

def innerTail152_19 : List Nat := 6337 :: innerTail152_20
def innerCandidates152_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_19_checked : candidates edge 4 907 6337 innerTail152_20 best.bound = innerCandidates152_19 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_19 : scan edge (search edge 3) 4 [907] 907 innerTail152_19 best = best := by
  rw [innerTail152_19, scan, if_neg (by decide), innerCandidates152_19_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_20
    (if 3 ≤ innerCandidates152_19.length then search edge 3 [907, 6337] innerCandidates152_19 7244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_20

def innerTail152_18 : List Nat := 6229 :: innerTail152_19
def innerCandidates152_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_18_checked : candidates edge 4 907 6229 innerTail152_19 best.bound = innerCandidates152_18 := by
  unfold candidates
  have row : edge 6229 = SearchData.row6229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_18 : scan edge (search edge 3) 4 [907] 907 innerTail152_18 best = best := by
  rw [innerTail152_18, scan, if_neg (by decide), innerCandidates152_18_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_19
    (if 3 ≤ innerCandidates152_18.length then search edge 3 [907, 6229] innerCandidates152_18 7136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_19

def innerTail152_17 : List Nat := 5851 :: innerTail152_18
def innerCandidates152_17 : List Nat := [6961]
set_option maxRecDepth 4096 in
theorem innerCandidates152_17_checked : candidates edge 4 907 5851 innerTail152_18 best.bound = innerCandidates152_17 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_17 : scan edge (search edge 3) 4 [907] 907 innerTail152_17 best = best := by
  rw [innerTail152_17, scan, if_neg (by decide), innerCandidates152_17_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_18
    (if 3 ≤ innerCandidates152_17.length then search edge 3 [907, 5851] innerCandidates152_17 6758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_18

def innerTail152_16 : List Nat := 5503 :: innerTail152_17
def innerCandidates152_16 : List Nat := [7027]
set_option maxRecDepth 4096 in
theorem innerCandidates152_16_checked : candidates edge 4 907 5503 innerTail152_17 best.bound = innerCandidates152_16 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_16 : scan edge (search edge 3) 4 [907] 907 innerTail152_16 best = best := by
  rw [innerTail152_16, scan, if_neg (by decide), innerCandidates152_16_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_17
    (if 3 ≤ innerCandidates152_16.length then search edge 3 [907, 5503] innerCandidates152_16 6410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_17

def innerTail152_15 : List Nat := 4993 :: innerTail152_16
def innerCandidates152_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_15_checked : candidates edge 4 907 4993 innerTail152_16 best.bound = innerCandidates152_15 := by
  unfold candidates
  have row : edge 4993 = SearchData.row4993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_15 : scan edge (search edge 3) 4 [907] 907 innerTail152_15 best = best := by
  rw [innerTail152_15, scan, if_neg (by decide), innerCandidates152_15_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_16
    (if 3 ≤ innerCandidates152_15.length then search edge 3 [907, 4993] innerCandidates152_15 5900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_16

def innerTail152_14 : List Nat := 4987 :: innerTail152_15
def innerCandidates152_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_14_checked : candidates edge 4 907 4987 innerTail152_15 best.bound = innerCandidates152_14 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_14 : scan edge (search edge 3) 4 [907] 907 innerTail152_14 best = best := by
  rw [innerTail152_14, scan, if_neg (by decide), innerCandidates152_14_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_15
    (if 3 ≤ innerCandidates152_14.length then search edge 3 [907, 4987] innerCandidates152_14 5894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_15

def innerTail152_13 : List Nat := 4861 :: innerTail152_14
def innerCandidates152_13 : List Nat := [6229]
set_option maxRecDepth 4096 in
theorem innerCandidates152_13_checked : candidates edge 4 907 4861 innerTail152_14 best.bound = innerCandidates152_13 := by
  unfold candidates
  have row : edge 4861 = SearchData.row4861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_13 : scan edge (search edge 3) 4 [907] 907 innerTail152_13 best = best := by
  rw [innerTail152_13, scan, if_neg (by decide), innerCandidates152_13_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_14
    (if 3 ≤ innerCandidates152_13.length then search edge 3 [907, 4861] innerCandidates152_13 5768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_14

def innerTail152_12 : List Nat := 4567 :: innerTail152_13
def innerCandidates152_12 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates152_12_checked : candidates edge 4 907 4567 innerTail152_13 best.bound = innerCandidates152_12 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_12 : scan edge (search edge 3) 4 [907] 907 innerTail152_12 best = best := by
  rw [innerTail152_12, scan, if_neg (by decide), innerCandidates152_12_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_13
    (if 3 ≤ innerCandidates152_12.length then search edge 3 [907, 4567] innerCandidates152_12 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_13

def innerTail152_11 : List Nat := 4423 :: innerTail152_12
def innerCandidates152_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_11_checked : candidates edge 4 907 4423 innerTail152_12 best.bound = innerCandidates152_11 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_11 : scan edge (search edge 3) 4 [907] 907 innerTail152_11 best = best := by
  rw [innerTail152_11, scan, if_neg (by decide), innerCandidates152_11_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_12
    (if 3 ≤ innerCandidates152_11.length then search edge 3 [907, 4423] innerCandidates152_11 5330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_12

def innerTail152_10 : List Nat := 4357 :: innerTail152_11
def innerCandidates152_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_10_checked : candidates edge 4 907 4357 innerTail152_11 best.bound = innerCandidates152_10 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_10 : scan edge (search edge 3) 4 [907] 907 innerTail152_10 best = best := by
  rw [innerTail152_10, scan, if_neg (by decide), innerCandidates152_10_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_11
    (if 3 ≤ innerCandidates152_10.length then search edge 3 [907, 4357] innerCandidates152_10 5264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_11

def innerTail152_9 : List Nat := 3919 :: innerTail152_10
def innerCandidates152_9 : List Nat := [4567, 5503, 6961]
set_option maxRecDepth 4096 in
theorem innerCandidates152_9_checked : candidates edge 4 907 3919 innerTail152_10 best.bound = innerCandidates152_9 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild152_9 : search edge 3 [907, 3919] innerCandidates152_9 4826 best = best := by decide

theorem innerScan152_9 : scan edge (search edge 3) 4 [907] 907 innerTail152_9 best = best := by
  rw [innerTail152_9, scan, if_neg (by decide), innerCandidates152_9_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_10
    (if 3 ≤ innerCandidates152_9.length then search edge 3 [907, 3919] innerCandidates152_9 4826 best else best) = best
  rw [if_pos (by decide), innerChild152_9]
  exact innerScan152_10

def innerTail152_8 : List Nat := 3709 :: innerTail152_9
def innerCandidates152_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_8_checked : candidates edge 4 907 3709 innerTail152_9 best.bound = innerCandidates152_8 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_8 : scan edge (search edge 3) 4 [907] 907 innerTail152_8 best = best := by
  rw [innerTail152_8, scan, if_neg (by decide), innerCandidates152_8_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_9
    (if 3 ≤ innerCandidates152_8.length then search edge 3 [907, 3709] innerCandidates152_8 4616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_9

def innerTail152_7 : List Nat := 2731 :: innerTail152_8
def innerCandidates152_7 : List Nat := [4423, 7993]
set_option maxRecDepth 4096 in
theorem innerCandidates152_7_checked : candidates edge 4 907 2731 innerTail152_8 best.bound = innerCandidates152_7 := by
  unfold candidates
  have row : edge 2731 = SearchData.row2731.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_7 : scan edge (search edge 3) 4 [907] 907 innerTail152_7 best = best := by
  rw [innerTail152_7, scan, if_neg (by decide), innerCandidates152_7_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_8
    (if 3 ≤ innerCandidates152_7.length then search edge 3 [907, 2731] innerCandidates152_7 3638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_8

def innerTail152_6 : List Nat := 2473 :: innerTail152_7
def innerCandidates152_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_6_checked : candidates edge 4 907 2473 innerTail152_7 best.bound = innerCandidates152_6 := by
  unfold candidates
  have row : edge 2473 = PrimePairCache.smallRow2473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_6 : scan edge (search edge 3) 4 [907] 907 innerTail152_6 best = best := by
  rw [innerTail152_6, scan, if_neg (by decide), innerCandidates152_6_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_7
    (if 3 ≤ innerCandidates152_6.length then search edge 3 [907, 2473] innerCandidates152_6 3380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_7

def innerTail152_5 : List Nat := 2383 :: innerTail152_6
def innerCandidates152_5 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates152_5_checked : candidates edge 4 907 2383 innerTail152_6 best.bound = innerCandidates152_5 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_5 : scan edge (search edge 3) 4 [907] 907 innerTail152_5 best = best := by
  rw [innerTail152_5, scan, if_neg (by decide), innerCandidates152_5_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_6
    (if 3 ≤ innerCandidates152_5.length then search edge 3 [907, 2383] innerCandidates152_5 3290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_6

def innerTail152_4 : List Nat := 1801 :: innerTail152_5
def innerCandidates152_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_4_checked : candidates edge 4 907 1801 innerTail152_5 best.bound = innerCandidates152_4 := by
  unfold candidates
  have row : edge 1801 = PrimePairCache.smallRow1801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_4 : scan edge (search edge 3) 4 [907] 907 innerTail152_4 best = best := by
  rw [innerTail152_4, scan, if_neg (by decide), innerCandidates152_4_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_5
    (if 3 ≤ innerCandidates152_4.length then search edge 3 [907, 1801] innerCandidates152_4 2708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_5

def innerTail152_3 : List Nat := 1471 :: innerTail152_4
def innerCandidates152_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_3_checked : candidates edge 4 907 1471 innerTail152_4 best.bound = innerCandidates152_3 := by
  unfold candidates
  have row : edge 1471 = PrimePairCache.smallRow1471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_3 : scan edge (search edge 3) 4 [907] 907 innerTail152_3 best = best := by
  rw [innerTail152_3, scan, if_neg (by decide), innerCandidates152_3_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_4
    (if 3 ≤ innerCandidates152_3.length then search edge 3 [907, 1471] innerCandidates152_3 2378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_4

def innerTail152_2 : List Nat := 1459 :: innerTail152_3
def innerCandidates152_2 : List Nat := [7027]
set_option maxRecDepth 4096 in
theorem innerCandidates152_2_checked : candidates edge 4 907 1459 innerTail152_3 best.bound = innerCandidates152_2 := by
  unfold candidates
  have row : edge 1459 = PrimePairCache.smallRow1459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_2 : scan edge (search edge 3) 4 [907] 907 innerTail152_2 best = best := by
  rw [innerTail152_2, scan, if_neg (by decide), innerCandidates152_2_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_3
    (if 3 ≤ innerCandidates152_2.length then search edge 3 [907, 1459] innerCandidates152_2 2366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_3

def innerTail152_1 : List Nat := 1429 :: innerTail152_2
def innerCandidates152_1 : List Nat := [3919]
set_option maxRecDepth 4096 in
theorem innerCandidates152_1_checked : candidates edge 4 907 1429 innerTail152_2 best.bound = innerCandidates152_1 := by
  unfold candidates
  have row : edge 1429 = PrimePairCache.smallRow1429.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_1 : scan edge (search edge 3) 4 [907] 907 innerTail152_1 best = best := by
  rw [innerTail152_1, scan, if_neg (by decide), innerCandidates152_1_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_2
    (if 3 ≤ innerCandidates152_1.length then search edge 3 [907, 1429] innerCandidates152_1 2336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_2

def innerTail152_0 : List Nat := 1213 :: innerTail152_1
def innerCandidates152_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates152_0_checked : candidates edge 4 907 1213 innerTail152_1 best.bound = innerCandidates152_0 := by
  unfold candidates
  have row : edge 1213 = PrimePairCache.smallRow1213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan152_0 : scan edge (search edge 3) 4 [907] 907 innerTail152_0 best = best := by
  rw [innerTail152_0, scan, if_neg (by decide), innerCandidates152_0_checked]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_1
    (if 3 ≤ innerCandidates152_0.length then search edge 3 [907, 1213] innerCandidates152_0 2120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan152_1

theorem rootBranch152 : search edge 4 [907] rootCandidates152 907 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [907] 907 innerTail152_0 best = best
  exact innerScan152_0

def rootCandidates151 : List Nat := [1091, 1427, 1553, 1697, 1847, 1931, 2099, 2153, 2741, 3089, 3251, 3803, 4049, 4157, 5303, 5417, 5651, 6449, 6653, 6863, 6911, 7013, 7109, 7499, 7727, 8087]
set_option maxRecDepth 4096 in
theorem rootCandidates151_checked : candidates edge 5 0 887 tail152 best.bound = rootCandidates151 := by
  unfold candidates
  have row : edge 887 = PrimePairCache.smallRow887.contains := by funext q; rfl
  rw [row]
  decide

def innerTail151_26 : List Nat := []
theorem innerScan151_26 : scan edge (search edge 3) 4 [887] 887 innerTail151_26 best = best := rfl

def innerTail151_25 : List Nat := 8087 :: innerTail151_26
def innerCandidates151_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_25_checked : candidates edge 4 887 8087 innerTail151_26 best.bound = innerCandidates151_25 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_25 : scan edge (search edge 3) 4 [887] 887 innerTail151_25 best = best := by
  rw [innerTail151_25, scan, if_neg (by decide), innerCandidates151_25_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_26
    (if 3 ≤ innerCandidates151_25.length then search edge 3 [887, 8087] innerCandidates151_25 8974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_26

def innerTail151_24 : List Nat := 7727 :: innerTail151_25
def innerCandidates151_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_24_checked : candidates edge 4 887 7727 innerTail151_25 best.bound = innerCandidates151_24 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_24 : scan edge (search edge 3) 4 [887] 887 innerTail151_24 best = best := by
  rw [innerTail151_24, scan, if_neg (by decide), innerCandidates151_24_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_25
    (if 3 ≤ innerCandidates151_24.length then search edge 3 [887, 7727] innerCandidates151_24 8614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_25

def innerTail151_23 : List Nat := 7499 :: innerTail151_24
def innerCandidates151_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_23_checked : candidates edge 4 887 7499 innerTail151_24 best.bound = innerCandidates151_23 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_23 : scan edge (search edge 3) 4 [887] 887 innerTail151_23 best = best := by
  rw [innerTail151_23, scan, if_neg (by decide), innerCandidates151_23_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_24
    (if 3 ≤ innerCandidates151_23.length then search edge 3 [887, 7499] innerCandidates151_23 8386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_24

def innerTail151_22 : List Nat := 7109 :: innerTail151_23
def innerCandidates151_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_22_checked : candidates edge 4 887 7109 innerTail151_23 best.bound = innerCandidates151_22 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_22 : scan edge (search edge 3) 4 [887] 887 innerTail151_22 best = best := by
  rw [innerTail151_22, scan, if_neg (by decide), innerCandidates151_22_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_23
    (if 3 ≤ innerCandidates151_22.length then search edge 3 [887, 7109] innerCandidates151_22 7996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_23

def innerTail151_21 : List Nat := 7013 :: innerTail151_22
def innerCandidates151_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_21_checked : candidates edge 4 887 7013 innerTail151_22 best.bound = innerCandidates151_21 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_21 : scan edge (search edge 3) 4 [887] 887 innerTail151_21 best = best := by
  rw [innerTail151_21, scan, if_neg (by decide), innerCandidates151_21_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_22
    (if 3 ≤ innerCandidates151_21.length then search edge 3 [887, 7013] innerCandidates151_21 7900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_22

def innerTail151_20 : List Nat := 6911 :: innerTail151_21
def innerCandidates151_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_20_checked : candidates edge 4 887 6911 innerTail151_21 best.bound = innerCandidates151_20 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_20 : scan edge (search edge 3) 4 [887] 887 innerTail151_20 best = best := by
  rw [innerTail151_20, scan, if_neg (by decide), innerCandidates151_20_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_21
    (if 3 ≤ innerCandidates151_20.length then search edge 3 [887, 6911] innerCandidates151_20 7798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_21

def innerTail151_19 : List Nat := 6863 :: innerTail151_20
def innerCandidates151_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_19_checked : candidates edge 4 887 6863 innerTail151_20 best.bound = innerCandidates151_19 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_19 : scan edge (search edge 3) 4 [887] 887 innerTail151_19 best = best := by
  rw [innerTail151_19, scan, if_neg (by decide), innerCandidates151_19_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_20
    (if 3 ≤ innerCandidates151_19.length then search edge 3 [887, 6863] innerCandidates151_19 7750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_20

def innerTail151_18 : List Nat := 6653 :: innerTail151_19
def innerCandidates151_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_18_checked : candidates edge 4 887 6653 innerTail151_19 best.bound = innerCandidates151_18 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_18 : scan edge (search edge 3) 4 [887] 887 innerTail151_18 best = best := by
  rw [innerTail151_18, scan, if_neg (by decide), innerCandidates151_18_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_19
    (if 3 ≤ innerCandidates151_18.length then search edge 3 [887, 6653] innerCandidates151_18 7540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_19

def innerTail151_17 : List Nat := 6449 :: innerTail151_18
def innerCandidates151_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_17_checked : candidates edge 4 887 6449 innerTail151_18 best.bound = innerCandidates151_17 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_17 : scan edge (search edge 3) 4 [887] 887 innerTail151_17 best = best := by
  rw [innerTail151_17, scan, if_neg (by decide), innerCandidates151_17_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_18
    (if 3 ≤ innerCandidates151_17.length then search edge 3 [887, 6449] innerCandidates151_17 7336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_18

def innerTail151_16 : List Nat := 5651 :: innerTail151_17
def innerCandidates151_16 : List Nat := [7499]
set_option maxRecDepth 4096 in
theorem innerCandidates151_16_checked : candidates edge 4 887 5651 innerTail151_17 best.bound = innerCandidates151_16 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_16 : scan edge (search edge 3) 4 [887] 887 innerTail151_16 best = best := by
  rw [innerTail151_16, scan, if_neg (by decide), innerCandidates151_16_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_17
    (if 3 ≤ innerCandidates151_16.length then search edge 3 [887, 5651] innerCandidates151_16 6538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_17

def innerTail151_15 : List Nat := 5417 :: innerTail151_16
def innerCandidates151_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_15_checked : candidates edge 4 887 5417 innerTail151_16 best.bound = innerCandidates151_15 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_15 : scan edge (search edge 3) 4 [887] 887 innerTail151_15 best = best := by
  rw [innerTail151_15, scan, if_neg (by decide), innerCandidates151_15_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_16
    (if 3 ≤ innerCandidates151_15.length then search edge 3 [887, 5417] innerCandidates151_15 6304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_16

def innerTail151_14 : List Nat := 5303 :: innerTail151_15
def innerCandidates151_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_14_checked : candidates edge 4 887 5303 innerTail151_15 best.bound = innerCandidates151_14 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_14 : scan edge (search edge 3) 4 [887] 887 innerTail151_14 best = best := by
  rw [innerTail151_14, scan, if_neg (by decide), innerCandidates151_14_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_15
    (if 3 ≤ innerCandidates151_14.length then search edge 3 [887, 5303] innerCandidates151_14 6190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_15

def innerTail151_13 : List Nat := 4157 :: innerTail151_14
def innerCandidates151_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_13_checked : candidates edge 4 887 4157 innerTail151_14 best.bound = innerCandidates151_13 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_13 : scan edge (search edge 3) 4 [887] 887 innerTail151_13 best = best := by
  rw [innerTail151_13, scan, if_neg (by decide), innerCandidates151_13_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_14
    (if 3 ≤ innerCandidates151_13.length then search edge 3 [887, 4157] innerCandidates151_13 5044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_14

def innerTail151_12 : List Nat := 4049 :: innerTail151_13
def innerCandidates151_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_12_checked : candidates edge 4 887 4049 innerTail151_13 best.bound = innerCandidates151_12 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_12 : scan edge (search edge 3) 4 [887] 887 innerTail151_12 best = best := by
  rw [innerTail151_12, scan, if_neg (by decide), innerCandidates151_12_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_13
    (if 3 ≤ innerCandidates151_12.length then search edge 3 [887, 4049] innerCandidates151_12 4936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_13

def innerTail151_11 : List Nat := 3803 :: innerTail151_12
def innerCandidates151_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_11_checked : candidates edge 4 887 3803 innerTail151_12 best.bound = innerCandidates151_11 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_11 : scan edge (search edge 3) 4 [887] 887 innerTail151_11 best = best := by
  rw [innerTail151_11, scan, if_neg (by decide), innerCandidates151_11_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_12
    (if 3 ≤ innerCandidates151_11.length then search edge 3 [887, 3803] innerCandidates151_11 4690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_12

def innerTail151_10 : List Nat := 3251 :: innerTail151_11
def innerCandidates151_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_10_checked : candidates edge 4 887 3251 innerTail151_11 best.bound = innerCandidates151_10 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_10 : scan edge (search edge 3) 4 [887] 887 innerTail151_10 best = best := by
  rw [innerTail151_10, scan, if_neg (by decide), innerCandidates151_10_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_11
    (if 3 ≤ innerCandidates151_10.length then search edge 3 [887, 3251] innerCandidates151_10 4138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_11

def innerTail151_9 : List Nat := 3089 :: innerTail151_10
def innerCandidates151_9 : List Nat := [5303]
set_option maxRecDepth 4096 in
theorem innerCandidates151_9_checked : candidates edge 4 887 3089 innerTail151_10 best.bound = innerCandidates151_9 := by
  unfold candidates
  have row : edge 3089 = SearchData.row3089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_9 : scan edge (search edge 3) 4 [887] 887 innerTail151_9 best = best := by
  rw [innerTail151_9, scan, if_neg (by decide), innerCandidates151_9_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_10
    (if 3 ≤ innerCandidates151_9.length then search edge 3 [887, 3089] innerCandidates151_9 3976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_10

def innerTail151_8 : List Nat := 2741 :: innerTail151_9
def innerCandidates151_8 : List Nat := [6911]
set_option maxRecDepth 4096 in
theorem innerCandidates151_8_checked : candidates edge 4 887 2741 innerTail151_9 best.bound = innerCandidates151_8 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_8 : scan edge (search edge 3) 4 [887] 887 innerTail151_8 best = best := by
  rw [innerTail151_8, scan, if_neg (by decide), innerCandidates151_8_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_9
    (if 3 ≤ innerCandidates151_8.length then search edge 3 [887, 2741] innerCandidates151_8 3628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_9

def innerTail151_7 : List Nat := 2153 :: innerTail151_8
def innerCandidates151_7 : List Nat := [6863]
set_option maxRecDepth 4096 in
theorem innerCandidates151_7_checked : candidates edge 4 887 2153 innerTail151_8 best.bound = innerCandidates151_7 := by
  unfold candidates
  have row : edge 2153 = PrimePairCache.smallRow2153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_7 : scan edge (search edge 3) 4 [887] 887 innerTail151_7 best = best := by
  rw [innerTail151_7, scan, if_neg (by decide), innerCandidates151_7_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_8
    (if 3 ≤ innerCandidates151_7.length then search edge 3 [887, 2153] innerCandidates151_7 3040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_8

def innerTail151_6 : List Nat := 2099 :: innerTail151_7
def innerCandidates151_6 : List Nat := [6863]
set_option maxRecDepth 4096 in
theorem innerCandidates151_6_checked : candidates edge 4 887 2099 innerTail151_7 best.bound = innerCandidates151_6 := by
  unfold candidates
  have row : edge 2099 = PrimePairCache.smallRow2099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_6 : scan edge (search edge 3) 4 [887] 887 innerTail151_6 best = best := by
  rw [innerTail151_6, scan, if_neg (by decide), innerCandidates151_6_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_7
    (if 3 ≤ innerCandidates151_6.length then search edge 3 [887, 2099] innerCandidates151_6 2986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_7

def innerTail151_5 : List Nat := 1931 :: innerTail151_6
def innerCandidates151_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_5_checked : candidates edge 4 887 1931 innerTail151_6 best.bound = innerCandidates151_5 := by
  unfold candidates
  have row : edge 1931 = PrimePairCache.smallRow1931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_5 : scan edge (search edge 3) 4 [887] 887 innerTail151_5 best = best := by
  rw [innerTail151_5, scan, if_neg (by decide), innerCandidates151_5_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_6
    (if 3 ≤ innerCandidates151_5.length then search edge 3 [887, 1931] innerCandidates151_5 2818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_6

def innerTail151_4 : List Nat := 1847 :: innerTail151_5
def innerCandidates151_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates151_4_checked : candidates edge 4 887 1847 innerTail151_5 best.bound = innerCandidates151_4 := by
  unfold candidates
  have row : edge 1847 = PrimePairCache.smallRow1847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_4 : scan edge (search edge 3) 4 [887] 887 innerTail151_4 best = best := by
  rw [innerTail151_4, scan, if_neg (by decide), innerCandidates151_4_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_5
    (if 3 ≤ innerCandidates151_4.length then search edge 3 [887, 1847] innerCandidates151_4 2734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_5

def innerTail151_3 : List Nat := 1697 :: innerTail151_4
def innerCandidates151_3 : List Nat := [2741]
set_option maxRecDepth 4096 in
theorem innerCandidates151_3_checked : candidates edge 4 887 1697 innerTail151_4 best.bound = innerCandidates151_3 := by
  unfold candidates
  have row : edge 1697 = PrimePairCache.smallRow1697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_3 : scan edge (search edge 3) 4 [887] 887 innerTail151_3 best = best := by
  rw [innerTail151_3, scan, if_neg (by decide), innerCandidates151_3_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_4
    (if 3 ≤ innerCandidates151_3.length then search edge 3 [887, 1697] innerCandidates151_3 2584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_4

def innerTail151_2 : List Nat := 1553 :: innerTail151_3
def innerCandidates151_2 : List Nat := [1697, 4049, 7013]
set_option maxRecDepth 4096 in
theorem innerCandidates151_2_checked : candidates edge 4 887 1553 innerTail151_3 best.bound = innerCandidates151_2 := by
  unfold candidates
  have row : edge 1553 = PrimePairCache.smallRow1553.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild151_2 : search edge 3 [887, 1553] innerCandidates151_2 2440 best = best := by decide

theorem innerScan151_2 : scan edge (search edge 3) 4 [887] 887 innerTail151_2 best = best := by
  rw [innerTail151_2, scan, if_neg (by decide), innerCandidates151_2_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_3
    (if 3 ≤ innerCandidates151_2.length then search edge 3 [887, 1553] innerCandidates151_2 2440 best else best) = best
  rw [if_pos (by decide), innerChild151_2]
  exact innerScan151_3

def innerTail151_1 : List Nat := 1427 :: innerTail151_2
def innerCandidates151_1 : List Nat := [1931, 4157]
set_option maxRecDepth 4096 in
theorem innerCandidates151_1_checked : candidates edge 4 887 1427 innerTail151_2 best.bound = innerCandidates151_1 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_1 : scan edge (search edge 3) 4 [887] 887 innerTail151_1 best = best := by
  rw [innerTail151_1, scan, if_neg (by decide), innerCandidates151_1_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_2
    (if 3 ≤ innerCandidates151_1.length then search edge 3 [887, 1427] innerCandidates151_1 2314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_2

def innerTail151_0 : List Nat := 1091 :: innerTail151_1
def innerCandidates151_0 : List Nat := [1697, 1847]
set_option maxRecDepth 4096 in
theorem innerCandidates151_0_checked : candidates edge 4 887 1091 innerTail151_1 best.bound = innerCandidates151_0 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan151_0 : scan edge (search edge 3) 4 [887] 887 innerTail151_0 best = best := by
  rw [innerTail151_0, scan, if_neg (by decide), innerCandidates151_0_checked]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_1
    (if 3 ≤ innerCandidates151_0.length then search edge 3 [887, 1091] innerCandidates151_0 1978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan151_1

theorem rootBranch151 : search edge 4 [887] rootCandidates151 887 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [887] 887 innerTail151_0 best = best
  exact innerScan151_0

end ProjectEulerSolutions.PrimePairUpper.Search
