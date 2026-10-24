import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates405 : List Nat := [2843, 3221, 3527, 3803, 3851, 3917, 4073, 4229, 4733, 5051, 5417, 5573, 5693, 5987, 6101, 6311, 6689, 6803, 7043, 7193, 7331, 7919, 8087, 8171, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates405_checked : candidates edge 5 0 2801 tail406 best.bound = rootCandidates405 := by
  unfold candidates
  have row : edge 2801 = SearchData.row2801.contains := by funext q; rfl
  rw [row]
  decide

def innerTail405_25 : List Nat := []
theorem innerScan405_25 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_25 best = best := rfl

def innerTail405_24 : List Nat := 8387 :: innerTail405_25
def innerCandidates405_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_24_checked : candidates edge 4 2801 8387 innerTail405_25 best.bound = innerCandidates405_24 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_24 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_24 best = best := by
  rw [innerTail405_24, scan, if_neg (by decide), innerCandidates405_24_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_25
    (if 3 ≤ innerCandidates405_24.length then search edge 3 [2801, 8387] innerCandidates405_24 11188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_25

def innerTail405_23 : List Nat := 8171 :: innerTail405_24
def innerCandidates405_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_23_checked : candidates edge 4 2801 8171 innerTail405_24 best.bound = innerCandidates405_23 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_23 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_23 best = best := by
  rw [innerTail405_23, scan, if_neg (by decide), innerCandidates405_23_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_24
    (if 3 ≤ innerCandidates405_23.length then search edge 3 [2801, 8171] innerCandidates405_23 10972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_24

def innerTail405_22 : List Nat := 8087 :: innerTail405_23
def innerCandidates405_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_22_checked : candidates edge 4 2801 8087 innerTail405_23 best.bound = innerCandidates405_22 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_22 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_22 best = best := by
  rw [innerTail405_22, scan, if_neg (by decide), innerCandidates405_22_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_23
    (if 3 ≤ innerCandidates405_22.length then search edge 3 [2801, 8087] innerCandidates405_22 10888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_23

def innerTail405_21 : List Nat := 7919 :: innerTail405_22
def innerCandidates405_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_21_checked : candidates edge 4 2801 7919 innerTail405_22 best.bound = innerCandidates405_21 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_21 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_21 best = best := by
  rw [innerTail405_21, scan, if_neg (by decide), innerCandidates405_21_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_22
    (if 3 ≤ innerCandidates405_21.length then search edge 3 [2801, 7919] innerCandidates405_21 10720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_22

def innerTail405_20 : List Nat := 7331 :: innerTail405_21
def innerCandidates405_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_20_checked : candidates edge 4 2801 7331 innerTail405_21 best.bound = innerCandidates405_20 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_20 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_20 best = best := by
  rw [innerTail405_20, scan, if_neg (by decide), innerCandidates405_20_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_21
    (if 3 ≤ innerCandidates405_20.length then search edge 3 [2801, 7331] innerCandidates405_20 10132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_21

def innerTail405_19 : List Nat := 7193 :: innerTail405_20
def innerCandidates405_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_19_checked : candidates edge 4 2801 7193 innerTail405_20 best.bound = innerCandidates405_19 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_19 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_19 best = best := by
  rw [innerTail405_19, scan, if_neg (by decide), innerCandidates405_19_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_20
    (if 3 ≤ innerCandidates405_19.length then search edge 3 [2801, 7193] innerCandidates405_19 9994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_20

def innerTail405_18 : List Nat := 7043 :: innerTail405_19
def innerCandidates405_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_18_checked : candidates edge 4 2801 7043 innerTail405_19 best.bound = innerCandidates405_18 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_18 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_18 best = best := by
  rw [innerTail405_18, scan, if_neg (by decide), innerCandidates405_18_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_19
    (if 3 ≤ innerCandidates405_18.length then search edge 3 [2801, 7043] innerCandidates405_18 9844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_19

def innerTail405_17 : List Nat := 6803 :: innerTail405_18
def innerCandidates405_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_17_checked : candidates edge 4 2801 6803 innerTail405_18 best.bound = innerCandidates405_17 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_17 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_17 best = best := by
  rw [innerTail405_17, scan, if_neg (by decide), innerCandidates405_17_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_18
    (if 3 ≤ innerCandidates405_17.length then search edge 3 [2801, 6803] innerCandidates405_17 9604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_18

def innerTail405_16 : List Nat := 6689 :: innerTail405_17
def innerCandidates405_16 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates405_16_checked : candidates edge 4 2801 6689 innerTail405_17 best.bound = innerCandidates405_16 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_16 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_16 best = best := by
  rw [innerTail405_16, scan, if_neg (by decide), innerCandidates405_16_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_17
    (if 3 ≤ innerCandidates405_16.length then search edge 3 [2801, 6689] innerCandidates405_16 9490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_17

def innerTail405_15 : List Nat := 6311 :: innerTail405_16
def innerCandidates405_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_15_checked : candidates edge 4 2801 6311 innerTail405_16 best.bound = innerCandidates405_15 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_15 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_15 best = best := by
  rw [innerTail405_15, scan, if_neg (by decide), innerCandidates405_15_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_16
    (if 3 ≤ innerCandidates405_15.length then search edge 3 [2801, 6311] innerCandidates405_15 9112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_16

def innerTail405_14 : List Nat := 6101 :: innerTail405_15
def innerCandidates405_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_14_checked : candidates edge 4 2801 6101 innerTail405_15 best.bound = innerCandidates405_14 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_14 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_14 best = best := by
  rw [innerTail405_14, scan, if_neg (by decide), innerCandidates405_14_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_15
    (if 3 ≤ innerCandidates405_14.length then search edge 3 [2801, 6101] innerCandidates405_14 8902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_15

def innerTail405_13 : List Nat := 5987 :: innerTail405_14
def innerCandidates405_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_13_checked : candidates edge 4 2801 5987 innerTail405_14 best.bound = innerCandidates405_13 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_13 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_13 best = best := by
  rw [innerTail405_13, scan, if_neg (by decide), innerCandidates405_13_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_14
    (if 3 ≤ innerCandidates405_13.length then search edge 3 [2801, 5987] innerCandidates405_13 8788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_14

def innerTail405_12 : List Nat := 5693 :: innerTail405_13
def innerCandidates405_12 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates405_12_checked : candidates edge 4 2801 5693 innerTail405_13 best.bound = innerCandidates405_12 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_12 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_12 best = best := by
  rw [innerTail405_12, scan, if_neg (by decide), innerCandidates405_12_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_13
    (if 3 ≤ innerCandidates405_12.length then search edge 3 [2801, 5693] innerCandidates405_12 8494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_13

def innerTail405_11 : List Nat := 5573 :: innerTail405_12
def innerCandidates405_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_11_checked : candidates edge 4 2801 5573 innerTail405_12 best.bound = innerCandidates405_11 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_11 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_11 best = best := by
  rw [innerTail405_11, scan, if_neg (by decide), innerCandidates405_11_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_12
    (if 3 ≤ innerCandidates405_11.length then search edge 3 [2801, 5573] innerCandidates405_11 8374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_12

def innerTail405_10 : List Nat := 5417 :: innerTail405_11
def innerCandidates405_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_10_checked : candidates edge 4 2801 5417 innerTail405_11 best.bound = innerCandidates405_10 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_10 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_10 best = best := by
  rw [innerTail405_10, scan, if_neg (by decide), innerCandidates405_10_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_11
    (if 3 ≤ innerCandidates405_10.length then search edge 3 [2801, 5417] innerCandidates405_10 8218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_11

def innerTail405_9 : List Nat := 5051 :: innerTail405_10
def innerCandidates405_9 : List Nat := [6101]
set_option maxRecDepth 4096 in
theorem innerCandidates405_9_checked : candidates edge 4 2801 5051 innerTail405_10 best.bound = innerCandidates405_9 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_9 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_9 best = best := by
  rw [innerTail405_9, scan, if_neg (by decide), innerCandidates405_9_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_10
    (if 3 ≤ innerCandidates405_9.length then search edge 3 [2801, 5051] innerCandidates405_9 7852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_10

def innerTail405_8 : List Nat := 4733 :: innerTail405_9
def innerCandidates405_8 : List Nat := [6689, 7331, 7919]
set_option maxRecDepth 4096 in
theorem innerCandidates405_8_checked : candidates edge 4 2801 4733 innerTail405_9 best.bound = innerCandidates405_8 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild405_8 : search edge 3 [2801, 4733] innerCandidates405_8 7534 best = best := by decide

theorem innerScan405_8 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_8 best = best := by
  rw [innerTail405_8, scan, if_neg (by decide), innerCandidates405_8_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_9
    (if 3 ≤ innerCandidates405_8.length then search edge 3 [2801, 4733] innerCandidates405_8 7534 best else best) = best
  rw [if_pos (by decide), innerChild405_8]
  exact innerScan405_9

def innerTail405_7 : List Nat := 4229 :: innerTail405_8
def innerCandidates405_7 : List Nat := [6689]
set_option maxRecDepth 4096 in
theorem innerCandidates405_7_checked : candidates edge 4 2801 4229 innerTail405_8 best.bound = innerCandidates405_7 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_7 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_7 best = best := by
  rw [innerTail405_7, scan, if_neg (by decide), innerCandidates405_7_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_8
    (if 3 ≤ innerCandidates405_7.length then search edge 3 [2801, 4229] innerCandidates405_7 7030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_8

def innerTail405_6 : List Nat := 4073 :: innerTail405_7
def innerCandidates405_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates405_6_checked : candidates edge 4 2801 4073 innerTail405_7 best.bound = innerCandidates405_6 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_6 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_6 best = best := by
  rw [innerTail405_6, scan, if_neg (by decide), innerCandidates405_6_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_7
    (if 3 ≤ innerCandidates405_6.length then search edge 3 [2801, 4073] innerCandidates405_6 6874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_7

def innerTail405_5 : List Nat := 3917 :: innerTail405_6
def innerCandidates405_5 : List Nat := [5987, 7331]
set_option maxRecDepth 4096 in
theorem innerCandidates405_5_checked : candidates edge 4 2801 3917 innerTail405_6 best.bound = innerCandidates405_5 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_5 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_5 best = best := by
  rw [innerTail405_5, scan, if_neg (by decide), innerCandidates405_5_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_6
    (if 3 ≤ innerCandidates405_5.length then search edge 3 [2801, 3917] innerCandidates405_5 6718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_6

def innerTail405_4 : List Nat := 3851 :: innerTail405_5
def innerCandidates405_4 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates405_4_checked : candidates edge 4 2801 3851 innerTail405_5 best.bound = innerCandidates405_4 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_4 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_4 best = best := by
  rw [innerTail405_4, scan, if_neg (by decide), innerCandidates405_4_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_5
    (if 3 ≤ innerCandidates405_4.length then search edge 3 [2801, 3851] innerCandidates405_4 6652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_5

def innerTail405_3 : List Nat := 3803 :: innerTail405_4
def innerCandidates405_3 : List Nat := [5051]
set_option maxRecDepth 4096 in
theorem innerCandidates405_3_checked : candidates edge 4 2801 3803 innerTail405_4 best.bound = innerCandidates405_3 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_3 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_3 best = best := by
  rw [innerTail405_3, scan, if_neg (by decide), innerCandidates405_3_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_4
    (if 3 ≤ innerCandidates405_3.length then search edge 3 [2801, 3803] innerCandidates405_3 6604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_4

def innerTail405_2 : List Nat := 3527 :: innerTail405_3
def innerCandidates405_2 : List Nat := [6803]
set_option maxRecDepth 4096 in
theorem innerCandidates405_2_checked : candidates edge 4 2801 3527 innerTail405_3 best.bound = innerCandidates405_2 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_2 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_2 best = best := by
  rw [innerTail405_2, scan, if_neg (by decide), innerCandidates405_2_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_3
    (if 3 ≤ innerCandidates405_2.length then search edge 3 [2801, 3527] innerCandidates405_2 6328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_3

def innerTail405_1 : List Nat := 3221 :: innerTail405_2
def innerCandidates405_1 : List Nat := [3851, 6311]
set_option maxRecDepth 4096 in
theorem innerCandidates405_1_checked : candidates edge 4 2801 3221 innerTail405_2 best.bound = innerCandidates405_1 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_1 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_1 best = best := by
  rw [innerTail405_1, scan, if_neg (by decide), innerCandidates405_1_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_2
    (if 3 ≤ innerCandidates405_1.length then search edge 3 [2801, 3221] innerCandidates405_1 6022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_2

def innerTail405_0 : List Nat := 2843 :: innerTail405_1
def innerCandidates405_0 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates405_0_checked : candidates edge 4 2801 2843 innerTail405_1 best.bound = innerCandidates405_0 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan405_0 : scan edge (search edge 3) 4 [2801] 2801 innerTail405_0 best = best := by
  rw [innerTail405_0, scan, if_neg (by decide), innerCandidates405_0_checked]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_1
    (if 3 ≤ innerCandidates405_0.length then search edge 3 [2801, 2843] innerCandidates405_0 5644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan405_1

theorem rootBranch405 : search edge 4 [2801] rootCandidates405 2801 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2801] 2801 innerTail405_0 best = best
  exact innerScan405_0

def rootCandidates404 : List Nat := [4561, 4567, 4663, 4861, 5023, 5449, 5953, 6163, 6367, 6841, 7207, 7681, 7759, 8017]
set_option maxRecDepth 4096 in
theorem rootCandidates404_checked : candidates edge 5 0 2797 tail405 best.bound = rootCandidates404 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext q; rfl
  rw [row]
  decide

def innerTail404_14 : List Nat := []
theorem innerScan404_14 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_14 best = best := rfl

def innerTail404_13 : List Nat := 8017 :: innerTail404_14
def innerCandidates404_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates404_13_checked : candidates edge 4 2797 8017 innerTail404_14 best.bound = innerCandidates404_13 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_13 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_13 best = best := by
  rw [innerTail404_13, scan, if_neg (by decide), innerCandidates404_13_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_14
    (if 3 ≤ innerCandidates404_13.length then search edge 3 [2797, 8017] innerCandidates404_13 10814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_14

def innerTail404_12 : List Nat := 7759 :: innerTail404_13
def innerCandidates404_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates404_12_checked : candidates edge 4 2797 7759 innerTail404_13 best.bound = innerCandidates404_12 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_12 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_12 best = best := by
  rw [innerTail404_12, scan, if_neg (by decide), innerCandidates404_12_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_13
    (if 3 ≤ innerCandidates404_12.length then search edge 3 [2797, 7759] innerCandidates404_12 10556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_13

def innerTail404_11 : List Nat := 7681 :: innerTail404_12
def innerCandidates404_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates404_11_checked : candidates edge 4 2797 7681 innerTail404_12 best.bound = innerCandidates404_11 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_11 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_11 best = best := by
  rw [innerTail404_11, scan, if_neg (by decide), innerCandidates404_11_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_12
    (if 3 ≤ innerCandidates404_11.length then search edge 3 [2797, 7681] innerCandidates404_11 10478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_12

def innerTail404_10 : List Nat := 7207 :: innerTail404_11
def innerCandidates404_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates404_10_checked : candidates edge 4 2797 7207 innerTail404_11 best.bound = innerCandidates404_10 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_10 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_10 best = best := by
  rw [innerTail404_10, scan, if_neg (by decide), innerCandidates404_10_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_11
    (if 3 ≤ innerCandidates404_10.length then search edge 3 [2797, 7207] innerCandidates404_10 10004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_11

def innerTail404_9 : List Nat := 6841 :: innerTail404_10
def innerCandidates404_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates404_9_checked : candidates edge 4 2797 6841 innerTail404_10 best.bound = innerCandidates404_9 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_9 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_9 best = best := by
  rw [innerTail404_9, scan, if_neg (by decide), innerCandidates404_9_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_10
    (if 3 ≤ innerCandidates404_9.length then search edge 3 [2797, 6841] innerCandidates404_9 9638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_10

def innerTail404_8 : List Nat := 6367 :: innerTail404_9
def innerCandidates404_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates404_8_checked : candidates edge 4 2797 6367 innerTail404_9 best.bound = innerCandidates404_8 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_8 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_8 best = best := by
  rw [innerTail404_8, scan, if_neg (by decide), innerCandidates404_8_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_9
    (if 3 ≤ innerCandidates404_8.length then search edge 3 [2797, 6367] innerCandidates404_8 9164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_9

def innerTail404_7 : List Nat := 6163 :: innerTail404_8
def innerCandidates404_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates404_7_checked : candidates edge 4 2797 6163 innerTail404_8 best.bound = innerCandidates404_7 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_7 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_7 best = best := by
  rw [innerTail404_7, scan, if_neg (by decide), innerCandidates404_7_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_8
    (if 3 ≤ innerCandidates404_7.length then search edge 3 [2797, 6163] innerCandidates404_7 8960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_8

def innerTail404_6 : List Nat := 5953 :: innerTail404_7
def innerCandidates404_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates404_6_checked : candidates edge 4 2797 5953 innerTail404_7 best.bound = innerCandidates404_6 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_6 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_6 best = best := by
  rw [innerTail404_6, scan, if_neg (by decide), innerCandidates404_6_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_7
    (if 3 ≤ innerCandidates404_6.length then search edge 3 [2797, 5953] innerCandidates404_6 8750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_7

def innerTail404_5 : List Nat := 5449 :: innerTail404_6
def innerCandidates404_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates404_5_checked : candidates edge 4 2797 5449 innerTail404_6 best.bound = innerCandidates404_5 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_5 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_5 best = best := by
  rw [innerTail404_5, scan, if_neg (by decide), innerCandidates404_5_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_6
    (if 3 ≤ innerCandidates404_5.length then search edge 3 [2797, 5449] innerCandidates404_5 8246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_6

def innerTail404_4 : List Nat := 5023 :: innerTail404_5
def innerCandidates404_4 : List Nat := [6841]
set_option maxRecDepth 4096 in
theorem innerCandidates404_4_checked : candidates edge 4 2797 5023 innerTail404_5 best.bound = innerCandidates404_4 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_4 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_4 best = best := by
  rw [innerTail404_4, scan, if_neg (by decide), innerCandidates404_4_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_5
    (if 3 ≤ innerCandidates404_4.length then search edge 3 [2797, 5023] innerCandidates404_4 7820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_5

def innerTail404_3 : List Nat := 4861 :: innerTail404_4
def innerCandidates404_3 : List Nat := [5953, 6163]
set_option maxRecDepth 4096 in
theorem innerCandidates404_3_checked : candidates edge 4 2797 4861 innerTail404_4 best.bound = innerCandidates404_3 := by
  unfold candidates
  have row : edge 4861 = SearchData.row4861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_3 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_3 best = best := by
  rw [innerTail404_3, scan, if_neg (by decide), innerCandidates404_3_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_4
    (if 3 ≤ innerCandidates404_3.length then search edge 3 [2797, 4861] innerCandidates404_3 7658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_4

def innerTail404_2 : List Nat := 4663 :: innerTail404_3
def innerCandidates404_2 : List Nat := [8017]
set_option maxRecDepth 4096 in
theorem innerCandidates404_2_checked : candidates edge 4 2797 4663 innerTail404_3 best.bound = innerCandidates404_2 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_2 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_2 best = best := by
  rw [innerTail404_2, scan, if_neg (by decide), innerCandidates404_2_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_3
    (if 3 ≤ innerCandidates404_2.length then search edge 3 [2797, 4663] innerCandidates404_2 7460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_3

def innerTail404_1 : List Nat := 4567 :: innerTail404_2
def innerCandidates404_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates404_1_checked : candidates edge 4 2797 4567 innerTail404_2 best.bound = innerCandidates404_1 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_1 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_1 best = best := by
  rw [innerTail404_1, scan, if_neg (by decide), innerCandidates404_1_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_2
    (if 3 ≤ innerCandidates404_1.length then search edge 3 [2797, 4567] innerCandidates404_1 7364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_2

def innerTail404_0 : List Nat := 4561 :: innerTail404_1
def innerCandidates404_0 : List Nat := [8017]
set_option maxRecDepth 4096 in
theorem innerCandidates404_0_checked : candidates edge 4 2797 4561 innerTail404_1 best.bound = innerCandidates404_0 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan404_0 : scan edge (search edge 3) 4 [2797] 2797 innerTail404_0 best = best := by
  rw [innerTail404_0, scan, if_neg (by decide), innerCandidates404_0_checked]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_1
    (if 3 ≤ innerCandidates404_0.length then search edge 3 [2797, 4561] innerCandidates404_0 7358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan404_1

theorem rootBranch404 : search edge 4 [2797] rootCandidates404 2797 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2797] 2797 innerTail404_0 best = best
  exact innerScan404_0

def rootCandidates403 : List Nat := [3001, 3499, 3967, 4297, 5113, 5179, 5281, 6379, 6607, 6883, 6961, 7369, 7489, 8179, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates403_checked : candidates edge 5 0 2791 tail404 best.bound = rootCandidates403 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext q; rfl
  rw [row]
  decide

def innerTail403_15 : List Nat := []
theorem innerScan403_15 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_15 best = best := rfl

def innerTail403_14 : List Nat := 8287 :: innerTail403_15
def innerCandidates403_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates403_14_checked : candidates edge 4 2791 8287 innerTail403_15 best.bound = innerCandidates403_14 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_14 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_14 best = best := by
  rw [innerTail403_14, scan, if_neg (by decide), innerCandidates403_14_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_15
    (if 3 ≤ innerCandidates403_14.length then search edge 3 [2791, 8287] innerCandidates403_14 11078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_15

def innerTail403_13 : List Nat := 8179 :: innerTail403_14
def innerCandidates403_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates403_13_checked : candidates edge 4 2791 8179 innerTail403_14 best.bound = innerCandidates403_13 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_13 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_13 best = best := by
  rw [innerTail403_13, scan, if_neg (by decide), innerCandidates403_13_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_14
    (if 3 ≤ innerCandidates403_13.length then search edge 3 [2791, 8179] innerCandidates403_13 10970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_14

def innerTail403_12 : List Nat := 7489 :: innerTail403_13
def innerCandidates403_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates403_12_checked : candidates edge 4 2791 7489 innerTail403_13 best.bound = innerCandidates403_12 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_12 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_12 best = best := by
  rw [innerTail403_12, scan, if_neg (by decide), innerCandidates403_12_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_13
    (if 3 ≤ innerCandidates403_12.length then search edge 3 [2791, 7489] innerCandidates403_12 10280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_13

def innerTail403_11 : List Nat := 7369 :: innerTail403_12
def innerCandidates403_11 : List Nat := [7489]
set_option maxRecDepth 4096 in
theorem innerCandidates403_11_checked : candidates edge 4 2791 7369 innerTail403_12 best.bound = innerCandidates403_11 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_11 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_11 best = best := by
  rw [innerTail403_11, scan, if_neg (by decide), innerCandidates403_11_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_12
    (if 3 ≤ innerCandidates403_11.length then search edge 3 [2791, 7369] innerCandidates403_11 10160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_12

def innerTail403_10 : List Nat := 6961 :: innerTail403_11
def innerCandidates403_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates403_10_checked : candidates edge 4 2791 6961 innerTail403_11 best.bound = innerCandidates403_10 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_10 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_10 best = best := by
  rw [innerTail403_10, scan, if_neg (by decide), innerCandidates403_10_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_11
    (if 3 ≤ innerCandidates403_10.length then search edge 3 [2791, 6961] innerCandidates403_10 9752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_11

def innerTail403_9 : List Nat := 6883 :: innerTail403_10
def innerCandidates403_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates403_9_checked : candidates edge 4 2791 6883 innerTail403_10 best.bound = innerCandidates403_9 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_9 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_9 best = best := by
  rw [innerTail403_9, scan, if_neg (by decide), innerCandidates403_9_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_10
    (if 3 ≤ innerCandidates403_9.length then search edge 3 [2791, 6883] innerCandidates403_9 9674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_10

def innerTail403_8 : List Nat := 6607 :: innerTail403_9
def innerCandidates403_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates403_8_checked : candidates edge 4 2791 6607 innerTail403_9 best.bound = innerCandidates403_8 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_8 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_8 best = best := by
  rw [innerTail403_8, scan, if_neg (by decide), innerCandidates403_8_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_9
    (if 3 ≤ innerCandidates403_8.length then search edge 3 [2791, 6607] innerCandidates403_8 9398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_9

def innerTail403_7 : List Nat := 6379 :: innerTail403_8
def innerCandidates403_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates403_7_checked : candidates edge 4 2791 6379 innerTail403_8 best.bound = innerCandidates403_7 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_7 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_7 best = best := by
  rw [innerTail403_7, scan, if_neg (by decide), innerCandidates403_7_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_8
    (if 3 ≤ innerCandidates403_7.length then search edge 3 [2791, 6379] innerCandidates403_7 9170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_8

def innerTail403_6 : List Nat := 5281 :: innerTail403_7
def innerCandidates403_6 : List Nat := [7489, 8287]
set_option maxRecDepth 4096 in
theorem innerCandidates403_6_checked : candidates edge 4 2791 5281 innerTail403_7 best.bound = innerCandidates403_6 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_6 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_6 best = best := by
  rw [innerTail403_6, scan, if_neg (by decide), innerCandidates403_6_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_7
    (if 3 ≤ innerCandidates403_6.length then search edge 3 [2791, 5281] innerCandidates403_6 8072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_7

def innerTail403_5 : List Nat := 5179 :: innerTail403_6
def innerCandidates403_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates403_5_checked : candidates edge 4 2791 5179 innerTail403_6 best.bound = innerCandidates403_5 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_5 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_5 best = best := by
  rw [innerTail403_5, scan, if_neg (by decide), innerCandidates403_5_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_6
    (if 3 ≤ innerCandidates403_5.length then search edge 3 [2791, 5179] innerCandidates403_5 7970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_6

def innerTail403_4 : List Nat := 5113 :: innerTail403_5
def innerCandidates403_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates403_4_checked : candidates edge 4 2791 5113 innerTail403_5 best.bound = innerCandidates403_4 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_4 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_4 best = best := by
  rw [innerTail403_4, scan, if_neg (by decide), innerCandidates403_4_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_5
    (if 3 ≤ innerCandidates403_4.length then search edge 3 [2791, 5113] innerCandidates403_4 7904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_5

def innerTail403_3 : List Nat := 4297 :: innerTail403_4
def innerCandidates403_3 : List Nat := [6607]
set_option maxRecDepth 4096 in
theorem innerCandidates403_3_checked : candidates edge 4 2791 4297 innerTail403_4 best.bound = innerCandidates403_3 := by
  unfold candidates
  have row : edge 4297 = SearchData.row4297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_3 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_3 best = best := by
  rw [innerTail403_3, scan, if_neg (by decide), innerCandidates403_3_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_4
    (if 3 ≤ innerCandidates403_3.length then search edge 3 [2791, 4297] innerCandidates403_3 7088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_4

def innerTail403_2 : List Nat := 3967 :: innerTail403_3
def innerCandidates403_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates403_2_checked : candidates edge 4 2791 3967 innerTail403_3 best.bound = innerCandidates403_2 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_2 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_2 best = best := by
  rw [innerTail403_2, scan, if_neg (by decide), innerCandidates403_2_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_3
    (if 3 ≤ innerCandidates403_2.length then search edge 3 [2791, 3967] innerCandidates403_2 6758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_3

def innerTail403_1 : List Nat := 3499 :: innerTail403_2
def innerCandidates403_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates403_1_checked : candidates edge 4 2791 3499 innerTail403_2 best.bound = innerCandidates403_1 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan403_1 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_1 best = best := by
  rw [innerTail403_1, scan, if_neg (by decide), innerCandidates403_1_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_2
    (if 3 ≤ innerCandidates403_1.length then search edge 3 [2791, 3499] innerCandidates403_1 6290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan403_2

def innerTail403_0 : List Nat := 3001 :: innerTail403_1
def innerCandidates403_0 : List Nat := [3499, 3967, 6607]
set_option maxRecDepth 4096 in
theorem innerCandidates403_0_checked : candidates edge 4 2791 3001 innerTail403_1 best.bound = innerCandidates403_0 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild403_0 : search edge 3 [2791, 3001] innerCandidates403_0 5792 best = best := by decide

theorem innerScan403_0 : scan edge (search edge 3) 4 [2791] 2791 innerTail403_0 best = best := by
  rw [innerTail403_0, scan, if_neg (by decide), innerCandidates403_0_checked]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_1
    (if 3 ≤ innerCandidates403_0.length then search edge 3 [2791, 3001] innerCandidates403_0 5792 best else best) = best
  rw [if_pos (by decide), innerChild403_0]
  exact innerScan403_1

theorem rootBranch403 : search edge 4 [2791] rootCandidates403 2791 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2791] 2791 innerTail403_0 best = best
  exact innerScan403_0

end ProjectEulerSolutions.PrimePairUpper.Search
