import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates241 : List Nat := [1933, 2143, 2221, 2311, 2707, 2833, 3019, 3253, 3463, 3529, 3637, 3733, 3823, 4057, 4561, 4783, 4789, 5059, 5209, 5479, 5881, 6361, 6577, 6997, 7333, 7393]
set_option maxRecDepth 4096 in
theorem rootCandidates241_checked : candidates edge 5 0 1549 tail242 best.bound = rootCandidates241 := by
  unfold candidates
  have row : edge 1549 = PrimePairCache.smallRow1549.contains := by funext q; rfl
  rw [row]
  decide

def innerTail241_26 : List Nat := []
theorem innerScan241_26 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_26 best = best := rfl

def innerTail241_25 : List Nat := 7393 :: innerTail241_26
def innerCandidates241_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_25_checked : candidates edge 4 1549 7393 innerTail241_26 best.bound = innerCandidates241_25 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_25 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_25 best = best := by
  rw [innerTail241_25, scan, if_neg (by decide), innerCandidates241_25_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_26
    (if 3 ≤ innerCandidates241_25.length then search edge 3 [1549, 7393] innerCandidates241_25 8942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_26

def innerTail241_24 : List Nat := 7333 :: innerTail241_25
def innerCandidates241_24 : List Nat := [7393]
set_option maxRecDepth 4096 in
theorem innerCandidates241_24_checked : candidates edge 4 1549 7333 innerTail241_25 best.bound = innerCandidates241_24 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_24 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_24 best = best := by
  rw [innerTail241_24, scan, if_neg (by decide), innerCandidates241_24_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_25
    (if 3 ≤ innerCandidates241_24.length then search edge 3 [1549, 7333] innerCandidates241_24 8882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_25

def innerTail241_23 : List Nat := 6997 :: innerTail241_24
def innerCandidates241_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_23_checked : candidates edge 4 1549 6997 innerTail241_24 best.bound = innerCandidates241_23 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_23 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_23 best = best := by
  rw [innerTail241_23, scan, if_neg (by decide), innerCandidates241_23_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_24
    (if 3 ≤ innerCandidates241_23.length then search edge 3 [1549, 6997] innerCandidates241_23 8546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_24

def innerTail241_22 : List Nat := 6577 :: innerTail241_23
def innerCandidates241_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_22_checked : candidates edge 4 1549 6577 innerTail241_23 best.bound = innerCandidates241_22 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_22 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_22 best = best := by
  rw [innerTail241_22, scan, if_neg (by decide), innerCandidates241_22_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_23
    (if 3 ≤ innerCandidates241_22.length then search edge 3 [1549, 6577] innerCandidates241_22 8126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_23

def innerTail241_21 : List Nat := 6361 :: innerTail241_22
def innerCandidates241_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_21_checked : candidates edge 4 1549 6361 innerTail241_22 best.bound = innerCandidates241_21 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_21 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_21 best = best := by
  rw [innerTail241_21, scan, if_neg (by decide), innerCandidates241_21_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_22
    (if 3 ≤ innerCandidates241_21.length then search edge 3 [1549, 6361] innerCandidates241_21 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_22

def innerTail241_20 : List Nat := 5881 :: innerTail241_21
def innerCandidates241_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_20_checked : candidates edge 4 1549 5881 innerTail241_21 best.bound = innerCandidates241_20 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_20 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_20 best = best := by
  rw [innerTail241_20, scan, if_neg (by decide), innerCandidates241_20_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_21
    (if 3 ≤ innerCandidates241_20.length then search edge 3 [1549, 5881] innerCandidates241_20 7430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_21

def innerTail241_19 : List Nat := 5479 :: innerTail241_20
def innerCandidates241_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_19_checked : candidates edge 4 1549 5479 innerTail241_20 best.bound = innerCandidates241_19 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_19 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_19 best = best := by
  rw [innerTail241_19, scan, if_neg (by decide), innerCandidates241_19_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_20
    (if 3 ≤ innerCandidates241_19.length then search edge 3 [1549, 5479] innerCandidates241_19 7028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_20

def innerTail241_18 : List Nat := 5209 :: innerTail241_19
def innerCandidates241_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_18_checked : candidates edge 4 1549 5209 innerTail241_19 best.bound = innerCandidates241_18 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_18 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_18 best = best := by
  rw [innerTail241_18, scan, if_neg (by decide), innerCandidates241_18_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_19
    (if 3 ≤ innerCandidates241_18.length then search edge 3 [1549, 5209] innerCandidates241_18 6758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_19

def innerTail241_17 : List Nat := 5059 :: innerTail241_18
def innerCandidates241_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_17_checked : candidates edge 4 1549 5059 innerTail241_18 best.bound = innerCandidates241_17 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_17 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_17 best = best := by
  rw [innerTail241_17, scan, if_neg (by decide), innerCandidates241_17_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_18
    (if 3 ≤ innerCandidates241_17.length then search edge 3 [1549, 5059] innerCandidates241_17 6608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_18

def innerTail241_16 : List Nat := 4789 :: innerTail241_17
def innerCandidates241_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_16_checked : candidates edge 4 1549 4789 innerTail241_17 best.bound = innerCandidates241_16 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_16 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_16 best = best := by
  rw [innerTail241_16, scan, if_neg (by decide), innerCandidates241_16_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_17
    (if 3 ≤ innerCandidates241_16.length then search edge 3 [1549, 4789] innerCandidates241_16 6338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_17

def innerTail241_15 : List Nat := 4783 :: innerTail241_16
def innerCandidates241_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_15_checked : candidates edge 4 1549 4783 innerTail241_16 best.bound = innerCandidates241_15 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_15 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_15 best = best := by
  rw [innerTail241_15, scan, if_neg (by decide), innerCandidates241_15_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_16
    (if 3 ≤ innerCandidates241_15.length then search edge 3 [1549, 4783] innerCandidates241_15 6332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_16

def innerTail241_14 : List Nat := 4561 :: innerTail241_15
def innerCandidates241_14 : List Nat := [4783]
set_option maxRecDepth 4096 in
theorem innerCandidates241_14_checked : candidates edge 4 1549 4561 innerTail241_15 best.bound = innerCandidates241_14 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_14 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_14 best = best := by
  rw [innerTail241_14, scan, if_neg (by decide), innerCandidates241_14_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_15
    (if 3 ≤ innerCandidates241_14.length then search edge 3 [1549, 4561] innerCandidates241_14 6110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_15

def innerTail241_13 : List Nat := 4057 :: innerTail241_14
def innerCandidates241_13 : List Nat := [7393]
set_option maxRecDepth 4096 in
theorem innerCandidates241_13_checked : candidates edge 4 1549 4057 innerTail241_14 best.bound = innerCandidates241_13 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_13 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_13 best = best := by
  rw [innerTail241_13, scan, if_neg (by decide), innerCandidates241_13_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_14
    (if 3 ≤ innerCandidates241_13.length then search edge 3 [1549, 4057] innerCandidates241_13 5606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_14

def innerTail241_12 : List Nat := 3823 :: innerTail241_13
def innerCandidates241_12 : List Nat := [5209, 7393]
set_option maxRecDepth 4096 in
theorem innerCandidates241_12_checked : candidates edge 4 1549 3823 innerTail241_13 best.bound = innerCandidates241_12 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_12 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_12 best = best := by
  rw [innerTail241_12, scan, if_neg (by decide), innerCandidates241_12_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_13
    (if 3 ≤ innerCandidates241_12.length then search edge 3 [1549, 3823] innerCandidates241_12 5372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_13

def innerTail241_11 : List Nat := 3733 :: innerTail241_12
def innerCandidates241_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_11_checked : candidates edge 4 1549 3733 innerTail241_12 best.bound = innerCandidates241_11 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_11 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_11 best = best := by
  rw [innerTail241_11, scan, if_neg (by decide), innerCandidates241_11_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_12
    (if 3 ≤ innerCandidates241_11.length then search edge 3 [1549, 3733] innerCandidates241_11 5282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_12

def innerTail241_10 : List Nat := 3637 :: innerTail241_11
def innerCandidates241_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_10_checked : candidates edge 4 1549 3637 innerTail241_11 best.bound = innerCandidates241_10 := by
  unfold candidates
  have row : edge 3637 = SearchData.row3637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_10 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_10 best = best := by
  rw [innerTail241_10, scan, if_neg (by decide), innerCandidates241_10_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_11
    (if 3 ≤ innerCandidates241_10.length then search edge 3 [1549, 3637] innerCandidates241_10 5186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_11

def innerTail241_9 : List Nat := 3529 :: innerTail241_10
def innerCandidates241_9 : List Nat := [7393]
set_option maxRecDepth 4096 in
theorem innerCandidates241_9_checked : candidates edge 4 1549 3529 innerTail241_10 best.bound = innerCandidates241_9 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_9 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_9 best = best := by
  rw [innerTail241_9, scan, if_neg (by decide), innerCandidates241_9_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_10
    (if 3 ≤ innerCandidates241_9.length then search edge 3 [1549, 3529] innerCandidates241_9 5078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_10

def innerTail241_8 : List Nat := 3463 :: innerTail241_9
def innerCandidates241_8 : List Nat := [5479]
set_option maxRecDepth 4096 in
theorem innerCandidates241_8_checked : candidates edge 4 1549 3463 innerTail241_9 best.bound = innerCandidates241_8 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_8 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_8 best = best := by
  rw [innerTail241_8, scan, if_neg (by decide), innerCandidates241_8_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_9
    (if 3 ≤ innerCandidates241_8.length then search edge 3 [1549, 3463] innerCandidates241_8 5012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_9

def innerTail241_7 : List Nat := 3253 :: innerTail241_8
def innerCandidates241_7 : List Nat := [5479]
set_option maxRecDepth 4096 in
theorem innerCandidates241_7_checked : candidates edge 4 1549 3253 innerTail241_8 best.bound = innerCandidates241_7 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_7 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_7 best = best := by
  rw [innerTail241_7, scan, if_neg (by decide), innerCandidates241_7_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_8
    (if 3 ≤ innerCandidates241_7.length then search edge 3 [1549, 3253] innerCandidates241_7 4802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_8

def innerTail241_6 : List Nat := 3019 :: innerTail241_7
def innerCandidates241_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_6_checked : candidates edge 4 1549 3019 innerTail241_7 best.bound = innerCandidates241_6 := by
  unfold candidates
  have row : edge 3019 = SearchData.row3019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_6 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_6 best = best := by
  rw [innerTail241_6, scan, if_neg (by decide), innerCandidates241_6_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_7
    (if 3 ≤ innerCandidates241_6.length then search edge 3 [1549, 3019] innerCandidates241_6 4568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_7

def innerTail241_5 : List Nat := 2833 :: innerTail241_6
def innerCandidates241_5 : List Nat := [6361]
set_option maxRecDepth 4096 in
theorem innerCandidates241_5_checked : candidates edge 4 1549 2833 innerTail241_6 best.bound = innerCandidates241_5 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_5 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_5 best = best := by
  rw [innerTail241_5, scan, if_neg (by decide), innerCandidates241_5_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_6
    (if 3 ≤ innerCandidates241_5.length then search edge 3 [1549, 2833] innerCandidates241_5 4382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_6

def innerTail241_4 : List Nat := 2707 :: innerTail241_5
def innerCandidates241_4 : List Nat := [3253, 3463]
set_option maxRecDepth 4096 in
theorem innerCandidates241_4_checked : candidates edge 4 1549 2707 innerTail241_5 best.bound = innerCandidates241_4 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_4 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_4 best = best := by
  rw [innerTail241_4, scan, if_neg (by decide), innerCandidates241_4_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_5
    (if 3 ≤ innerCandidates241_4.length then search edge 3 [1549, 2707] innerCandidates241_4 4256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_5

def innerTail241_3 : List Nat := 2311 :: innerTail241_4
def innerCandidates241_3 : List Nat := [3529]
set_option maxRecDepth 4096 in
theorem innerCandidates241_3_checked : candidates edge 4 1549 2311 innerTail241_4 best.bound = innerCandidates241_3 := by
  unfold candidates
  have row : edge 2311 = PrimePairCache.smallRow2311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_3 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_3 best = best := by
  rw [innerTail241_3, scan, if_neg (by decide), innerCandidates241_3_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_4
    (if 3 ≤ innerCandidates241_3.length then search edge 3 [1549, 2311] innerCandidates241_3 3860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_4

def innerTail241_2 : List Nat := 2221 :: innerTail241_3
def innerCandidates241_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_2_checked : candidates edge 4 1549 2221 innerTail241_3 best.bound = innerCandidates241_2 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_2 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_2 best = best := by
  rw [innerTail241_2, scan, if_neg (by decide), innerCandidates241_2_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_3
    (if 3 ≤ innerCandidates241_2.length then search edge 3 [1549, 2221] innerCandidates241_2 3770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_3

def innerTail241_1 : List Nat := 2143 :: innerTail241_2
def innerCandidates241_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates241_1_checked : candidates edge 4 1549 2143 innerTail241_2 best.bound = innerCandidates241_1 := by
  unfold candidates
  have row : edge 2143 = PrimePairCache.smallRow2143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_1 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_1 best = best := by
  rw [innerTail241_1, scan, if_neg (by decide), innerCandidates241_1_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_2
    (if 3 ≤ innerCandidates241_1.length then search edge 3 [1549, 2143] innerCandidates241_1 3692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_2

def innerTail241_0 : List Nat := 1933 :: innerTail241_1
def innerCandidates241_0 : List Nat := [3733, 7333]
set_option maxRecDepth 4096 in
theorem innerCandidates241_0_checked : candidates edge 4 1549 1933 innerTail241_1 best.bound = innerCandidates241_0 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan241_0 : scan edge (search edge 3) 4 [1549] 1549 innerTail241_0 best = best := by
  rw [innerTail241_0, scan, if_neg (by decide), innerCandidates241_0_checked]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_1
    (if 3 ≤ innerCandidates241_0.length then search edge 3 [1549, 1933] innerCandidates241_0 3482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan241_1

theorem rootBranch241 : search edge 4 [1549] rootCandidates241 1549 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1549] 1549 innerTail241_0 best = best
  exact innerScan241_0

def rootCandidates240 : List Nat := [1579, 1693, 1777, 3217, 3463, 3511, 3931, 3967, 4093, 4129, 4273, 4423, 5443, 5689, 5857, 6451, 6571, 6823, 7159, 7879, 8053]
set_option maxRecDepth 4096 in
theorem rootCandidates240_checked : candidates edge 5 0 1543 tail241 best.bound = rootCandidates240 := by
  unfold candidates
  have row : edge 1543 = PrimePairCache.smallRow1543.contains := by funext q; rfl
  rw [row]
  decide

def innerTail240_21 : List Nat := []
theorem innerScan240_21 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_21 best = best := rfl

def innerTail240_20 : List Nat := 8053 :: innerTail240_21
def innerCandidates240_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates240_20_checked : candidates edge 4 1543 8053 innerTail240_21 best.bound = innerCandidates240_20 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_20 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_20 best = best := by
  rw [innerTail240_20, scan, if_neg (by decide), innerCandidates240_20_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_21
    (if 3 ≤ innerCandidates240_20.length then search edge 3 [1543, 8053] innerCandidates240_20 9596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_21

def innerTail240_19 : List Nat := 7879 :: innerTail240_20
def innerCandidates240_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates240_19_checked : candidates edge 4 1543 7879 innerTail240_20 best.bound = innerCandidates240_19 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_19 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_19 best = best := by
  rw [innerTail240_19, scan, if_neg (by decide), innerCandidates240_19_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_20
    (if 3 ≤ innerCandidates240_19.length then search edge 3 [1543, 7879] innerCandidates240_19 9422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_20

def innerTail240_18 : List Nat := 7159 :: innerTail240_19
def innerCandidates240_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates240_18_checked : candidates edge 4 1543 7159 innerTail240_19 best.bound = innerCandidates240_18 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_18 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_18 best = best := by
  rw [innerTail240_18, scan, if_neg (by decide), innerCandidates240_18_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_19
    (if 3 ≤ innerCandidates240_18.length then search edge 3 [1543, 7159] innerCandidates240_18 8702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_19

def innerTail240_17 : List Nat := 6823 :: innerTail240_18
def innerCandidates240_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates240_17_checked : candidates edge 4 1543 6823 innerTail240_18 best.bound = innerCandidates240_17 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_17 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_17 best = best := by
  rw [innerTail240_17, scan, if_neg (by decide), innerCandidates240_17_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_18
    (if 3 ≤ innerCandidates240_17.length then search edge 3 [1543, 6823] innerCandidates240_17 8366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_18

def innerTail240_16 : List Nat := 6571 :: innerTail240_17
def innerCandidates240_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates240_16_checked : candidates edge 4 1543 6571 innerTail240_17 best.bound = innerCandidates240_16 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_16 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_16 best = best := by
  rw [innerTail240_16, scan, if_neg (by decide), innerCandidates240_16_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_17
    (if 3 ≤ innerCandidates240_16.length then search edge 3 [1543, 6571] innerCandidates240_16 8114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_17

def innerTail240_15 : List Nat := 6451 :: innerTail240_16
def innerCandidates240_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates240_15_checked : candidates edge 4 1543 6451 innerTail240_16 best.bound = innerCandidates240_15 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_15 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_15 best = best := by
  rw [innerTail240_15, scan, if_neg (by decide), innerCandidates240_15_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_16
    (if 3 ≤ innerCandidates240_15.length then search edge 3 [1543, 6451] innerCandidates240_15 7994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_16

def innerTail240_14 : List Nat := 5857 :: innerTail240_15
def innerCandidates240_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates240_14_checked : candidates edge 4 1543 5857 innerTail240_15 best.bound = innerCandidates240_14 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_14 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_14 best = best := by
  rw [innerTail240_14, scan, if_neg (by decide), innerCandidates240_14_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_15
    (if 3 ≤ innerCandidates240_14.length then search edge 3 [1543, 5857] innerCandidates240_14 7400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_15

def innerTail240_13 : List Nat := 5689 :: innerTail240_14
def innerCandidates240_13 : List Nat := [6823]
set_option maxRecDepth 4096 in
theorem innerCandidates240_13_checked : candidates edge 4 1543 5689 innerTail240_14 best.bound = innerCandidates240_13 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_13 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_13 best = best := by
  rw [innerTail240_13, scan, if_neg (by decide), innerCandidates240_13_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_14
    (if 3 ≤ innerCandidates240_13.length then search edge 3 [1543, 5689] innerCandidates240_13 7232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_14

def innerTail240_12 : List Nat := 5443 :: innerTail240_13
def innerCandidates240_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates240_12_checked : candidates edge 4 1543 5443 innerTail240_13 best.bound = innerCandidates240_12 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_12 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_12 best = best := by
  rw [innerTail240_12, scan, if_neg (by decide), innerCandidates240_12_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_13
    (if 3 ≤ innerCandidates240_12.length then search edge 3 [1543, 5443] innerCandidates240_12 6986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_13

def innerTail240_11 : List Nat := 4423 :: innerTail240_12
def innerCandidates240_11 : List Nat := [5689]
set_option maxRecDepth 4096 in
theorem innerCandidates240_11_checked : candidates edge 4 1543 4423 innerTail240_12 best.bound = innerCandidates240_11 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_11 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_11 best = best := by
  rw [innerTail240_11, scan, if_neg (by decide), innerCandidates240_11_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_12
    (if 3 ≤ innerCandidates240_11.length then search edge 3 [1543, 4423] innerCandidates240_11 5966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_12

def innerTail240_10 : List Nat := 4273 :: innerTail240_11
def innerCandidates240_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates240_10_checked : candidates edge 4 1543 4273 innerTail240_11 best.bound = innerCandidates240_10 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_10 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_10 best = best := by
  rw [innerTail240_10, scan, if_neg (by decide), innerCandidates240_10_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_11
    (if 3 ≤ innerCandidates240_10.length then search edge 3 [1543, 4273] innerCandidates240_10 5816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_11

def innerTail240_9 : List Nat := 4129 :: innerTail240_10
def innerCandidates240_9 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates240_9_checked : candidates edge 4 1543 4129 innerTail240_10 best.bound = innerCandidates240_9 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_9 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_9 best = best := by
  rw [innerTail240_9, scan, if_neg (by decide), innerCandidates240_9_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_10
    (if 3 ≤ innerCandidates240_9.length then search edge 3 [1543, 4129] innerCandidates240_9 5672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_10

def innerTail240_8 : List Nat := 4093 :: innerTail240_9
def innerCandidates240_8 : List Nat := [6823]
set_option maxRecDepth 4096 in
theorem innerCandidates240_8_checked : candidates edge 4 1543 4093 innerTail240_9 best.bound = innerCandidates240_8 := by
  unfold candidates
  have row : edge 4093 = SearchData.row4093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_8 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_8 best = best := by
  rw [innerTail240_8, scan, if_neg (by decide), innerCandidates240_8_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_9
    (if 3 ≤ innerCandidates240_8.length then search edge 3 [1543, 4093] innerCandidates240_8 5636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_9

def innerTail240_7 : List Nat := 3967 :: innerTail240_8
def innerCandidates240_7 : List Nat := [7159, 8053]
set_option maxRecDepth 4096 in
theorem innerCandidates240_7_checked : candidates edge 4 1543 3967 innerTail240_8 best.bound = innerCandidates240_7 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_7 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_7 best = best := by
  rw [innerTail240_7, scan, if_neg (by decide), innerCandidates240_7_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_8
    (if 3 ≤ innerCandidates240_7.length then search edge 3 [1543, 3967] innerCandidates240_7 5510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_8

def innerTail240_6 : List Nat := 3931 :: innerTail240_7
def innerCandidates240_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates240_6_checked : candidates edge 4 1543 3931 innerTail240_7 best.bound = innerCandidates240_6 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_6 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_6 best = best := by
  rw [innerTail240_6, scan, if_neg (by decide), innerCandidates240_6_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_7
    (if 3 ≤ innerCandidates240_6.length then search edge 3 [1543, 3931] innerCandidates240_6 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_7

def innerTail240_5 : List Nat := 3511 :: innerTail240_6
def innerCandidates240_5 : List Nat := [7879]
set_option maxRecDepth 4096 in
theorem innerCandidates240_5_checked : candidates edge 4 1543 3511 innerTail240_6 best.bound = innerCandidates240_5 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_5 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_5 best = best := by
  rw [innerTail240_5, scan, if_neg (by decide), innerCandidates240_5_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_6
    (if 3 ≤ innerCandidates240_5.length then search edge 3 [1543, 3511] innerCandidates240_5 5054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_6

def innerTail240_4 : List Nat := 3463 :: innerTail240_5
def innerCandidates240_4 : List Nat := [8053]
set_option maxRecDepth 4096 in
theorem innerCandidates240_4_checked : candidates edge 4 1543 3463 innerTail240_5 best.bound = innerCandidates240_4 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_4 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_4 best = best := by
  rw [innerTail240_4, scan, if_neg (by decide), innerCandidates240_4_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_5
    (if 3 ≤ innerCandidates240_4.length then search edge 3 [1543, 3463] innerCandidates240_4 5006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_5

def innerTail240_3 : List Nat := 3217 :: innerTail240_4
def innerCandidates240_3 : List Nat := [3511, 3931]
set_option maxRecDepth 4096 in
theorem innerCandidates240_3_checked : candidates edge 4 1543 3217 innerTail240_4 best.bound = innerCandidates240_3 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_3 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_3 best = best := by
  rw [innerTail240_3, scan, if_neg (by decide), innerCandidates240_3_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_4
    (if 3 ≤ innerCandidates240_3.length then search edge 3 [1543, 3217] innerCandidates240_3 4760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_4

def innerTail240_2 : List Nat := 1777 :: innerTail240_3
def innerCandidates240_2 : List Nat := [3463]
set_option maxRecDepth 4096 in
theorem innerCandidates240_2_checked : candidates edge 4 1543 1777 innerTail240_3 best.bound = innerCandidates240_2 := by
  unfold candidates
  have row : edge 1777 = PrimePairCache.smallRow1777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_2 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_2 best = best := by
  rw [innerTail240_2, scan, if_neg (by decide), innerCandidates240_2_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_3
    (if 3 ≤ innerCandidates240_2.length then search edge 3 [1543, 1777] innerCandidates240_2 3320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_3

def innerTail240_1 : List Nat := 1693 :: innerTail240_2
def innerCandidates240_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates240_1_checked : candidates edge 4 1543 1693 innerTail240_2 best.bound = innerCandidates240_1 := by
  unfold candidates
  have row : edge 1693 = PrimePairCache.smallRow1693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_1 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_1 best = best := by
  rw [innerTail240_1, scan, if_neg (by decide), innerCandidates240_1_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_2
    (if 3 ≤ innerCandidates240_1.length then search edge 3 [1543, 1693] innerCandidates240_1 3236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_2

def innerTail240_0 : List Nat := 1579 :: innerTail240_1
def innerCandidates240_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates240_0_checked : candidates edge 4 1543 1579 innerTail240_1 best.bound = innerCandidates240_0 := by
  unfold candidates
  have row : edge 1579 = PrimePairCache.smallRow1579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan240_0 : scan edge (search edge 3) 4 [1543] 1543 innerTail240_0 best = best := by
  rw [innerTail240_0, scan, if_neg (by decide), innerCandidates240_0_checked]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_1
    (if 3 ≤ innerCandidates240_0.length then search edge 3 [1543, 1579] innerCandidates240_0 3122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan240_1

theorem rootBranch240 : search edge 4 [1543] rootCandidates240 1543 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1543] 1543 innerTail240_0 best = best
  exact innerScan240_0

end ProjectEulerSolutions.PrimePairUpper.Search
