import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates23 : List Nat := [107, 149, 197, 359, 383, 467, 641, 719, 1733, 1889, 1979, 2549, 2633, 2657, 2699, 2903, 3329, 3527, 3833, 3851, 4229, 4337, 4817, 5039, 5081, 5471, 6053, 6221, 6689, 6971, 7193]
set_option maxRecDepth 4096 in
theorem rootCandidates23_checked : candidates edge 5 0 101 tail24 best.bound = rootCandidates23 := by
  unfold candidates
  have row : edge 101 = PrimePairCache.smallRow101.contains := by funext q; rfl
  rw [row]
  decide

def innerTail23_31 : List Nat := []
theorem innerScan23_31 : scan edge (search edge 3) 4 [101] 101 innerTail23_31 best = best := rfl

def innerTail23_30 : List Nat := 7193 :: innerTail23_31
def innerCandidates23_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_30_checked : candidates edge 4 101 7193 innerTail23_31 best.bound = innerCandidates23_30 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_30 : scan edge (search edge 3) 4 [101] 101 innerTail23_30 best = best := by
  rw [innerTail23_30, scan, if_neg (by decide), innerCandidates23_30_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_31
    (if 3 ≤ innerCandidates23_30.length then search edge 3 [101, 7193] innerCandidates23_30 7294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_31

def innerTail23_29 : List Nat := 6971 :: innerTail23_30
def innerCandidates23_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_29_checked : candidates edge 4 101 6971 innerTail23_30 best.bound = innerCandidates23_29 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_29 : scan edge (search edge 3) 4 [101] 101 innerTail23_29 best = best := by
  rw [innerTail23_29, scan, if_neg (by decide), innerCandidates23_29_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_30
    (if 3 ≤ innerCandidates23_29.length then search edge 3 [101, 6971] innerCandidates23_29 7072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_30

def innerTail23_28 : List Nat := 6689 :: innerTail23_29
def innerCandidates23_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_28_checked : candidates edge 4 101 6689 innerTail23_29 best.bound = innerCandidates23_28 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_28 : scan edge (search edge 3) 4 [101] 101 innerTail23_28 best = best := by
  rw [innerTail23_28, scan, if_neg (by decide), innerCandidates23_28_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_29
    (if 3 ≤ innerCandidates23_28.length then search edge 3 [101, 6689] innerCandidates23_28 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_29

def innerTail23_27 : List Nat := 6221 :: innerTail23_28
def innerCandidates23_27 : List Nat := [6971]
set_option maxRecDepth 4096 in
theorem innerCandidates23_27_checked : candidates edge 4 101 6221 innerTail23_28 best.bound = innerCandidates23_27 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_27 : scan edge (search edge 3) 4 [101] 101 innerTail23_27 best = best := by
  rw [innerTail23_27, scan, if_neg (by decide), innerCandidates23_27_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_28
    (if 3 ≤ innerCandidates23_27.length then search edge 3 [101, 6221] innerCandidates23_27 6322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_28

def innerTail23_26 : List Nat := 6053 :: innerTail23_27
def innerCandidates23_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_26_checked : candidates edge 4 101 6053 innerTail23_27 best.bound = innerCandidates23_26 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_26 : scan edge (search edge 3) 4 [101] 101 innerTail23_26 best = best := by
  rw [innerTail23_26, scan, if_neg (by decide), innerCandidates23_26_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_27
    (if 3 ≤ innerCandidates23_26.length then search edge 3 [101, 6053] innerCandidates23_26 6154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_27

def innerTail23_25 : List Nat := 5471 :: innerTail23_26
def innerCandidates23_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_25_checked : candidates edge 4 101 5471 innerTail23_26 best.bound = innerCandidates23_25 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_25 : scan edge (search edge 3) 4 [101] 101 innerTail23_25 best = best := by
  rw [innerTail23_25, scan, if_neg (by decide), innerCandidates23_25_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_26
    (if 3 ≤ innerCandidates23_25.length then search edge 3 [101, 5471] innerCandidates23_25 5572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_26

def innerTail23_24 : List Nat := 5081 :: innerTail23_25
def innerCandidates23_24 : List Nat := [6971]
set_option maxRecDepth 4096 in
theorem innerCandidates23_24_checked : candidates edge 4 101 5081 innerTail23_25 best.bound = innerCandidates23_24 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_24 : scan edge (search edge 3) 4 [101] 101 innerTail23_24 best = best := by
  rw [innerTail23_24, scan, if_neg (by decide), innerCandidates23_24_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_25
    (if 3 ≤ innerCandidates23_24.length then search edge 3 [101, 5081] innerCandidates23_24 5182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_25

def innerTail23_23 : List Nat := 5039 :: innerTail23_24
def innerCandidates23_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_23_checked : candidates edge 4 101 5039 innerTail23_24 best.bound = innerCandidates23_23 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_23 : scan edge (search edge 3) 4 [101] 101 innerTail23_23 best = best := by
  rw [innerTail23_23, scan, if_neg (by decide), innerCandidates23_23_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_24
    (if 3 ≤ innerCandidates23_23.length then search edge 3 [101, 5039] innerCandidates23_23 5140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_24

def innerTail23_22 : List Nat := 4817 :: innerTail23_23
def innerCandidates23_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_22_checked : candidates edge 4 101 4817 innerTail23_23 best.bound = innerCandidates23_22 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_22 : scan edge (search edge 3) 4 [101] 101 innerTail23_22 best = best := by
  rw [innerTail23_22, scan, if_neg (by decide), innerCandidates23_22_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_23
    (if 3 ≤ innerCandidates23_22.length then search edge 3 [101, 4817] innerCandidates23_22 4918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_23

def innerTail23_21 : List Nat := 4337 :: innerTail23_22
def innerCandidates23_21 : List Nat := [5471]
set_option maxRecDepth 4096 in
theorem innerCandidates23_21_checked : candidates edge 4 101 4337 innerTail23_22 best.bound = innerCandidates23_21 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_21 : scan edge (search edge 3) 4 [101] 101 innerTail23_21 best = best := by
  rw [innerTail23_21, scan, if_neg (by decide), innerCandidates23_21_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_22
    (if 3 ≤ innerCandidates23_21.length then search edge 3 [101, 4337] innerCandidates23_21 4438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_22

def innerTail23_20 : List Nat := 4229 :: innerTail23_21
def innerCandidates23_20 : List Nat := [5039, 6689]
set_option maxRecDepth 4096 in
theorem innerCandidates23_20_checked : candidates edge 4 101 4229 innerTail23_21 best.bound = innerCandidates23_20 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_20 : scan edge (search edge 3) 4 [101] 101 innerTail23_20 best = best := by
  rw [innerTail23_20, scan, if_neg (by decide), innerCandidates23_20_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_21
    (if 3 ≤ innerCandidates23_20.length then search edge 3 [101, 4229] innerCandidates23_20 4330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_21

def innerTail23_19 : List Nat := 3851 :: innerTail23_20
def innerCandidates23_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_19_checked : candidates edge 4 101 3851 innerTail23_20 best.bound = innerCandidates23_19 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_19 : scan edge (search edge 3) 4 [101] 101 innerTail23_19 best = best := by
  rw [innerTail23_19, scan, if_neg (by decide), innerCandidates23_19_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_20
    (if 3 ≤ innerCandidates23_19.length then search edge 3 [101, 3851] innerCandidates23_19 3952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_20

def innerTail23_18 : List Nat := 3833 :: innerTail23_19
def innerCandidates23_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_18_checked : candidates edge 4 101 3833 innerTail23_19 best.bound = innerCandidates23_18 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_18 : scan edge (search edge 3) 4 [101] 101 innerTail23_18 best = best := by
  rw [innerTail23_18, scan, if_neg (by decide), innerCandidates23_18_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_19
    (if 3 ≤ innerCandidates23_18.length then search edge 3 [101, 3833] innerCandidates23_18 3934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_19

def innerTail23_17 : List Nat := 3527 :: innerTail23_18
def innerCandidates23_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_17_checked : candidates edge 4 101 3527 innerTail23_18 best.bound = innerCandidates23_17 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_17 : scan edge (search edge 3) 4 [101] 101 innerTail23_17 best = best := by
  rw [innerTail23_17, scan, if_neg (by decide), innerCandidates23_17_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_18
    (if 3 ≤ innerCandidates23_17.length then search edge 3 [101, 3527] innerCandidates23_17 3628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_18

def innerTail23_16 : List Nat := 3329 :: innerTail23_17
def innerCandidates23_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_16_checked : candidates edge 4 101 3329 innerTail23_17 best.bound = innerCandidates23_16 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_16 : scan edge (search edge 3) 4 [101] 101 innerTail23_16 best = best := by
  rw [innerTail23_16, scan, if_neg (by decide), innerCandidates23_16_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_17
    (if 3 ≤ innerCandidates23_16.length then search edge 3 [101, 3329] innerCandidates23_16 3430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_17

def innerTail23_15 : List Nat := 2903 :: innerTail23_16
def innerCandidates23_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_15_checked : candidates edge 4 101 2903 innerTail23_16 best.bound = innerCandidates23_15 := by
  unfold candidates
  have row : edge 2903 = SearchData.row2903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_15 : scan edge (search edge 3) 4 [101] 101 innerTail23_15 best = best := by
  rw [innerTail23_15, scan, if_neg (by decide), innerCandidates23_15_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_16
    (if 3 ≤ innerCandidates23_15.length then search edge 3 [101, 2903] innerCandidates23_15 3004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_16

def innerTail23_14 : List Nat := 2699 :: innerTail23_15
def innerCandidates23_14 : List Nat := [4337]
set_option maxRecDepth 4096 in
theorem innerCandidates23_14_checked : candidates edge 4 101 2699 innerTail23_15 best.bound = innerCandidates23_14 := by
  unfold candidates
  have row : edge 2699 = SearchData.row2699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_14 : scan edge (search edge 3) 4 [101] 101 innerTail23_14 best = best := by
  rw [innerTail23_14, scan, if_neg (by decide), innerCandidates23_14_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_15
    (if 3 ≤ innerCandidates23_14.length then search edge 3 [101, 2699] innerCandidates23_14 2800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_15

def innerTail23_13 : List Nat := 2657 :: innerTail23_14
def innerCandidates23_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_13_checked : candidates edge 4 101 2657 innerTail23_14 best.bound = innerCandidates23_13 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_13 : scan edge (search edge 3) 4 [101] 101 innerTail23_13 best = best := by
  rw [innerTail23_13, scan, if_neg (by decide), innerCandidates23_13_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_14
    (if 3 ≤ innerCandidates23_13.length then search edge 3 [101, 2657] innerCandidates23_13 2758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_14

def innerTail23_12 : List Nat := 2633 :: innerTail23_13
def innerCandidates23_12 : List Nat := [2699, 4337, 7193]
set_option maxRecDepth 4096 in
theorem innerCandidates23_12_checked : candidates edge 4 101 2633 innerTail23_13 best.bound = innerCandidates23_12 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild23_12 : search edge 3 [101, 2633] innerCandidates23_12 2734 best = best := by decide

theorem innerScan23_12 : scan edge (search edge 3) 4 [101] 101 innerTail23_12 best = best := by
  rw [innerTail23_12, scan, if_neg (by decide), innerCandidates23_12_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_13
    (if 3 ≤ innerCandidates23_12.length then search edge 3 [101, 2633] innerCandidates23_12 2734 best else best) = best
  rw [if_pos (by decide), innerChild23_12]
  exact innerScan23_13

def innerTail23_11 : List Nat := 2549 :: innerTail23_12
def innerCandidates23_11 : List Nat := [3329, 5471]
set_option maxRecDepth 4096 in
theorem innerCandidates23_11_checked : candidates edge 4 101 2549 innerTail23_12 best.bound = innerCandidates23_11 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_11 : scan edge (search edge 3) 4 [101] 101 innerTail23_11 best = best := by
  rw [innerTail23_11, scan, if_neg (by decide), innerCandidates23_11_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_12
    (if 3 ≤ innerCandidates23_11.length then search edge 3 [101, 2549] innerCandidates23_11 2650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_12

def innerTail23_10 : List Nat := 1979 :: innerTail23_11
def innerCandidates23_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_10_checked : candidates edge 4 101 1979 innerTail23_11 best.bound = innerCandidates23_10 := by
  unfold candidates
  have row : edge 1979 = PrimePairCache.smallRow1979.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_10 : scan edge (search edge 3) 4 [101] 101 innerTail23_10 best = best := by
  rw [innerTail23_10, scan, if_neg (by decide), innerCandidates23_10_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_11
    (if 3 ≤ innerCandidates23_10.length then search edge 3 [101, 1979] innerCandidates23_10 2080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_11

def innerTail23_9 : List Nat := 1889 :: innerTail23_10
def innerCandidates23_9 : List Nat := [2633]
set_option maxRecDepth 4096 in
theorem innerCandidates23_9_checked : candidates edge 4 101 1889 innerTail23_10 best.bound = innerCandidates23_9 := by
  unfold candidates
  have row : edge 1889 = PrimePairCache.smallRow1889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_9 : scan edge (search edge 3) 4 [101] 101 innerTail23_9 best = best := by
  rw [innerTail23_9, scan, if_neg (by decide), innerCandidates23_9_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_10
    (if 3 ≤ innerCandidates23_9.length then search edge 3 [101, 1889] innerCandidates23_9 1990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_10

def innerTail23_8 : List Nat := 1733 :: innerTail23_9
def innerCandidates23_8 : List Nat := [2657, 4817]
set_option maxRecDepth 4096 in
theorem innerCandidates23_8_checked : candidates edge 4 101 1733 innerTail23_9 best.bound = innerCandidates23_8 := by
  unfold candidates
  have row : edge 1733 = PrimePairCache.smallRow1733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_8 : scan edge (search edge 3) 4 [101] 101 innerTail23_8 best = best := by
  rw [innerTail23_8, scan, if_neg (by decide), innerCandidates23_8_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_9
    (if 3 ≤ innerCandidates23_8.length then search edge 3 [101, 1733] innerCandidates23_8 1834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_9

def innerTail23_7 : List Nat := 719 :: innerTail23_8
def innerCandidates23_7 : List Nat := [3833, 5039]
set_option maxRecDepth 4096 in
theorem innerCandidates23_7_checked : candidates edge 4 101 719 innerTail23_8 best.bound = innerCandidates23_7 := by
  unfold candidates
  have row : edge 719 = PrimePairCache.smallRow719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_7 : scan edge (search edge 3) 4 [101] 101 innerTail23_7 best = best := by
  rw [innerTail23_7, scan, if_neg (by decide), innerCandidates23_7_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_8
    (if 3 ≤ innerCandidates23_7.length then search edge 3 [101, 719] innerCandidates23_7 820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_8

def innerTail23_6 : List Nat := 641 :: innerTail23_7
def innerCandidates23_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates23_6_checked : candidates edge 4 101 641 innerTail23_7 best.bound = innerCandidates23_6 := by
  unfold candidates
  have row : edge 641 = PrimePairCache.smallRow641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_6 : scan edge (search edge 3) 4 [101] 101 innerTail23_6 best = best := by
  rw [innerTail23_6, scan, if_neg (by decide), innerCandidates23_6_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_7
    (if 3 ≤ innerCandidates23_6.length then search edge 3 [101, 641] innerCandidates23_6 742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_7

def innerTail23_5 : List Nat := 467 :: innerTail23_6
def innerCandidates23_5 : List Nat := [641]
set_option maxRecDepth 4096 in
theorem innerCandidates23_5_checked : candidates edge 4 101 467 innerTail23_6 best.bound = innerCandidates23_5 := by
  unfold candidates
  have row : edge 467 = PrimePairCache.smallRow467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_5 : scan edge (search edge 3) 4 [101] 101 innerTail23_5 best = best := by
  rw [innerTail23_5, scan, if_neg (by decide), innerCandidates23_5_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_6
    (if 3 ≤ innerCandidates23_5.length then search edge 3 [101, 467] innerCandidates23_5 568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_6

def innerTail23_4 : List Nat := 383 :: innerTail23_5
def innerCandidates23_4 : List Nat := [1733, 1889, 3527]
set_option maxRecDepth 4096 in
theorem innerCandidates23_4_checked : candidates edge 4 101 383 innerTail23_5 best.bound = innerCandidates23_4 := by
  unfold candidates
  have row : edge 383 = PrimePairCache.smallRow383.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild23_4 : search edge 3 [101, 383] innerCandidates23_4 484 best = best := by decide

theorem innerScan23_4 : scan edge (search edge 3) 4 [101] 101 innerTail23_4 best = best := by
  rw [innerTail23_4, scan, if_neg (by decide), innerCandidates23_4_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_5
    (if 3 ≤ innerCandidates23_4.length then search edge 3 [101, 383] innerCandidates23_4 484 best else best) = best
  rw [if_pos (by decide), innerChild23_4]
  exact innerScan23_5

def innerTail23_3 : List Nat := 359 :: innerTail23_4
def innerCandidates23_3 : List Nat := [3833, 6689]
set_option maxRecDepth 4096 in
theorem innerCandidates23_3_checked : candidates edge 4 101 359 innerTail23_4 best.bound = innerCandidates23_3 := by
  unfold candidates
  have row : edge 359 = PrimePairCache.smallRow359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_3 : scan edge (search edge 3) 4 [101] 101 innerTail23_3 best = best := by
  rw [innerTail23_3, scan, if_neg (by decide), innerCandidates23_3_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_4
    (if 3 ≤ innerCandidates23_3.length then search edge 3 [101, 359] innerCandidates23_3 460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_4

def innerTail23_2 : List Nat := 197 :: innerTail23_3
def innerCandidates23_2 : List Nat := [641, 2657]
set_option maxRecDepth 4096 in
theorem innerCandidates23_2_checked : candidates edge 4 101 197 innerTail23_3 best.bound = innerCandidates23_2 := by
  unfold candidates
  have row : edge 197 = PrimePairCache.smallRow197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_2 : scan edge (search edge 3) 4 [101] 101 innerTail23_2 best = best := by
  rw [innerTail23_2, scan, if_neg (by decide), innerCandidates23_2_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_3
    (if 3 ≤ innerCandidates23_2.length then search edge 3 [101, 197] innerCandidates23_2 298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_3

def innerTail23_1 : List Nat := 149 :: innerTail23_2
def innerCandidates23_1 : List Nat := [1979, 7193]
set_option maxRecDepth 4096 in
theorem innerCandidates23_1_checked : candidates edge 4 101 149 innerTail23_2 best.bound = innerCandidates23_1 := by
  unfold candidates
  have row : edge 149 = PrimePairCache.smallRow149.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_1 : scan edge (search edge 3) 4 [101] 101 innerTail23_1 best = best := by
  rw [innerTail23_1, scan, if_neg (by decide), innerCandidates23_1_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_2
    (if 3 ≤ innerCandidates23_1.length then search edge 3 [101, 149] innerCandidates23_1 250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_2

def innerTail23_0 : List Nat := 107 :: innerTail23_1
def innerCandidates23_0 : List Nat := [2633]
set_option maxRecDepth 4096 in
theorem innerCandidates23_0_checked : candidates edge 4 101 107 innerTail23_1 best.bound = innerCandidates23_0 := by
  unfold candidates
  have row : edge 107 = PrimePairCache.smallRow107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan23_0 : scan edge (search edge 3) 4 [101] 101 innerTail23_0 best = best := by
  rw [innerTail23_0, scan, if_neg (by decide), innerCandidates23_0_checked]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_1
    (if 3 ≤ innerCandidates23_0.length then search edge 3 [101, 107] innerCandidates23_0 208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan23_1

theorem rootBranch23 : search edge 4 [101] rootCandidates23 101 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [101] 101 innerTail23_0 best = best
  exact innerScan23_0

end ProjectEulerSolutions.PrimePairUpper.Search
