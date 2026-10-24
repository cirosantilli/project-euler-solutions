import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates226 : List Nat := [1543, 1549, 1753, 1993, 2011, 2347, 2719, 2803, 2833, 3229, 3271, 3313, 3709, 3769, 4051, 4093, 4441, 5641, 5683, 5689, 6949, 6967, 6991, 7759, 8059]
set_option maxRecDepth 4096 in
theorem rootCandidates226_checked : candidates edge 5 0 1447 tail227 best.bound = rootCandidates226 := by
  unfold candidates
  have row : edge 1447 = PrimePairCache.smallRow1447.contains := by funext q; rfl
  rw [row]
  decide

def innerTail226_25 : List Nat := []
theorem innerScan226_25 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_25 best = best := rfl

def innerTail226_24 : List Nat := 8059 :: innerTail226_25
def innerCandidates226_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates226_24_checked : candidates edge 4 1447 8059 innerTail226_25 best.bound = innerCandidates226_24 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_24 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_24 best = best := by
  rw [innerTail226_24, scan, if_neg (by decide), innerCandidates226_24_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_25
    (if 3 ≤ innerCandidates226_24.length then search edge 3 [1447, 8059] innerCandidates226_24 9506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_25

def innerTail226_23 : List Nat := 7759 :: innerTail226_24
def innerCandidates226_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates226_23_checked : candidates edge 4 1447 7759 innerTail226_24 best.bound = innerCandidates226_23 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_23 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_23 best = best := by
  rw [innerTail226_23, scan, if_neg (by decide), innerCandidates226_23_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_24
    (if 3 ≤ innerCandidates226_23.length then search edge 3 [1447, 7759] innerCandidates226_23 9206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_24

def innerTail226_22 : List Nat := 6991 :: innerTail226_23
def innerCandidates226_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates226_22_checked : candidates edge 4 1447 6991 innerTail226_23 best.bound = innerCandidates226_22 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_22 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_22 best = best := by
  rw [innerTail226_22, scan, if_neg (by decide), innerCandidates226_22_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_23
    (if 3 ≤ innerCandidates226_22.length then search edge 3 [1447, 6991] innerCandidates226_22 8438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_23

def innerTail226_21 : List Nat := 6967 :: innerTail226_22
def innerCandidates226_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates226_21_checked : candidates edge 4 1447 6967 innerTail226_22 best.bound = innerCandidates226_21 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_21 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_21 best = best := by
  rw [innerTail226_21, scan, if_neg (by decide), innerCandidates226_21_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_22
    (if 3 ≤ innerCandidates226_21.length then search edge 3 [1447, 6967] innerCandidates226_21 8414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_22

def innerTail226_20 : List Nat := 6949 :: innerTail226_21
def innerCandidates226_20 : List Nat := [6967]
set_option maxRecDepth 4096 in
theorem innerCandidates226_20_checked : candidates edge 4 1447 6949 innerTail226_21 best.bound = innerCandidates226_20 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_20 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_20 best = best := by
  rw [innerTail226_20, scan, if_neg (by decide), innerCandidates226_20_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_21
    (if 3 ≤ innerCandidates226_20.length then search edge 3 [1447, 6949] innerCandidates226_20 8396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_21

def innerTail226_19 : List Nat := 5689 :: innerTail226_20
def innerCandidates226_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates226_19_checked : candidates edge 4 1447 5689 innerTail226_20 best.bound = innerCandidates226_19 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_19 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_19 best = best := by
  rw [innerTail226_19, scan, if_neg (by decide), innerCandidates226_19_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_20
    (if 3 ≤ innerCandidates226_19.length then search edge 3 [1447, 5689] innerCandidates226_19 7136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_20

def innerTail226_18 : List Nat := 5683 :: innerTail226_19
def innerCandidates226_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates226_18_checked : candidates edge 4 1447 5683 innerTail226_19 best.bound = innerCandidates226_18 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_18 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_18 best = best := by
  rw [innerTail226_18, scan, if_neg (by decide), innerCandidates226_18_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_19
    (if 3 ≤ innerCandidates226_18.length then search edge 3 [1447, 5683] innerCandidates226_18 7130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_19

def innerTail226_17 : List Nat := 5641 :: innerTail226_18
def innerCandidates226_17 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates226_17_checked : candidates edge 4 1447 5641 innerTail226_18 best.bound = innerCandidates226_17 := by
  unfold candidates
  have row : edge 5641 = SearchData.row5641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_17 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_17 best = best := by
  rw [innerTail226_17, scan, if_neg (by decide), innerCandidates226_17_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_18
    (if 3 ≤ innerCandidates226_17.length then search edge 3 [1447, 5641] innerCandidates226_17 7088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_18

def innerTail226_16 : List Nat := 4441 :: innerTail226_17
def innerCandidates226_16 : List Nat := [6949, 7759]
set_option maxRecDepth 4096 in
theorem innerCandidates226_16_checked : candidates edge 4 1447 4441 innerTail226_17 best.bound = innerCandidates226_16 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_16 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_16 best = best := by
  rw [innerTail226_16, scan, if_neg (by decide), innerCandidates226_16_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_17
    (if 3 ≤ innerCandidates226_16.length then search edge 3 [1447, 4441] innerCandidates226_16 5888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_17

def innerTail226_15 : List Nat := 4093 :: innerTail226_16
def innerCandidates226_15 : List Nat := [6991]
set_option maxRecDepth 4096 in
theorem innerCandidates226_15_checked : candidates edge 4 1447 4093 innerTail226_16 best.bound = innerCandidates226_15 := by
  unfold candidates
  have row : edge 4093 = SearchData.row4093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_15 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_15 best = best := by
  rw [innerTail226_15, scan, if_neg (by decide), innerCandidates226_15_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_16
    (if 3 ≤ innerCandidates226_15.length then search edge 3 [1447, 4093] innerCandidates226_15 5540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_16

def innerTail226_14 : List Nat := 4051 :: innerTail226_15
def innerCandidates226_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates226_14_checked : candidates edge 4 1447 4051 innerTail226_15 best.bound = innerCandidates226_14 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_14 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_14 best = best := by
  rw [innerTail226_14, scan, if_neg (by decide), innerCandidates226_14_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_15
    (if 3 ≤ innerCandidates226_14.length then search edge 3 [1447, 4051] innerCandidates226_14 5498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_15

def innerTail226_13 : List Nat := 3769 :: innerTail226_14
def innerCandidates226_13 : List Nat := [4051, 6949]
set_option maxRecDepth 4096 in
theorem innerCandidates226_13_checked : candidates edge 4 1447 3769 innerTail226_14 best.bound = innerCandidates226_13 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_13 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_13 best = best := by
  rw [innerTail226_13, scan, if_neg (by decide), innerCandidates226_13_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_14
    (if 3 ≤ innerCandidates226_13.length then search edge 3 [1447, 3769] innerCandidates226_13 5216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_14

def innerTail226_12 : List Nat := 3709 :: innerTail226_13
def innerCandidates226_12 : List Nat := [6991]
set_option maxRecDepth 4096 in
theorem innerCandidates226_12_checked : candidates edge 4 1447 3709 innerTail226_13 best.bound = innerCandidates226_12 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_12 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_12 best = best := by
  rw [innerTail226_12, scan, if_neg (by decide), innerCandidates226_12_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_13
    (if 3 ≤ innerCandidates226_12.length then search edge 3 [1447, 3709] innerCandidates226_12 5156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_13

def innerTail226_11 : List Nat := 3313 :: innerTail226_12
def innerCandidates226_11 : List Nat := [8059]
set_option maxRecDepth 4096 in
theorem innerCandidates226_11_checked : candidates edge 4 1447 3313 innerTail226_12 best.bound = innerCandidates226_11 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_11 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_11 best = best := by
  rw [innerTail226_11, scan, if_neg (by decide), innerCandidates226_11_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_12
    (if 3 ≤ innerCandidates226_11.length then search edge 3 [1447, 3313] innerCandidates226_11 4760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_12

def innerTail226_10 : List Nat := 3271 :: innerTail226_11
def innerCandidates226_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates226_10_checked : candidates edge 4 1447 3271 innerTail226_11 best.bound = innerCandidates226_10 := by
  unfold candidates
  have row : edge 3271 = SearchData.row3271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_10 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_10 best = best := by
  rw [innerTail226_10, scan, if_neg (by decide), innerCandidates226_10_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_11
    (if 3 ≤ innerCandidates226_10.length then search edge 3 [1447, 3271] innerCandidates226_10 4718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_11

def innerTail226_9 : List Nat := 3229 :: innerTail226_10
def innerCandidates226_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates226_9_checked : candidates edge 4 1447 3229 innerTail226_10 best.bound = innerCandidates226_9 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_9 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_9 best = best := by
  rw [innerTail226_9, scan, if_neg (by decide), innerCandidates226_9_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_10
    (if 3 ≤ innerCandidates226_9.length then search edge 3 [1447, 3229] innerCandidates226_9 4676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_10

def innerTail226_8 : List Nat := 2833 :: innerTail226_9
def innerCandidates226_8 : List Nat := [3229, 5683]
set_option maxRecDepth 4096 in
theorem innerCandidates226_8_checked : candidates edge 4 1447 2833 innerTail226_9 best.bound = innerCandidates226_8 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_8 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_8 best = best := by
  rw [innerTail226_8, scan, if_neg (by decide), innerCandidates226_8_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_9
    (if 3 ≤ innerCandidates226_8.length then search edge 3 [1447, 2833] innerCandidates226_8 4280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_9

def innerTail226_7 : List Nat := 2803 :: innerTail226_8
def innerCandidates226_7 : List Nat := [3769, 4051]
set_option maxRecDepth 4096 in
theorem innerCandidates226_7_checked : candidates edge 4 1447 2803 innerTail226_8 best.bound = innerCandidates226_7 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_7 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_7 best = best := by
  rw [innerTail226_7, scan, if_neg (by decide), innerCandidates226_7_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_8
    (if 3 ≤ innerCandidates226_7.length then search edge 3 [1447, 2803] innerCandidates226_7 4250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_8

def innerTail226_6 : List Nat := 2719 :: innerTail226_7
def innerCandidates226_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates226_6_checked : candidates edge 4 1447 2719 innerTail226_7 best.bound = innerCandidates226_6 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_6 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_6 best = best := by
  rw [innerTail226_6, scan, if_neg (by decide), innerCandidates226_6_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_7
    (if 3 ≤ innerCandidates226_6.length then search edge 3 [1447, 2719] innerCandidates226_6 4166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_7

def innerTail226_5 : List Nat := 2347 :: innerTail226_6
def innerCandidates226_5 : List Nat := [2833]
set_option maxRecDepth 4096 in
theorem innerCandidates226_5_checked : candidates edge 4 1447 2347 innerTail226_6 best.bound = innerCandidates226_5 := by
  unfold candidates
  have row : edge 2347 = PrimePairCache.smallRow2347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_5 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_5 best = best := by
  rw [innerTail226_5, scan, if_neg (by decide), innerCandidates226_5_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_6
    (if 3 ≤ innerCandidates226_5.length then search edge 3 [1447, 2347] innerCandidates226_5 3794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_6

def innerTail226_4 : List Nat := 2011 :: innerTail226_5
def innerCandidates226_4 : List Nat := [4093]
set_option maxRecDepth 4096 in
theorem innerCandidates226_4_checked : candidates edge 4 1447 2011 innerTail226_5 best.bound = innerCandidates226_4 := by
  unfold candidates
  have row : edge 2011 = PrimePairCache.smallRow2011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_4 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_4 best = best := by
  rw [innerTail226_4, scan, if_neg (by decide), innerCandidates226_4_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_5
    (if 3 ≤ innerCandidates226_4.length then search edge 3 [1447, 2011] innerCandidates226_4 3458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_5

def innerTail226_3 : List Nat := 1993 :: innerTail226_4
def innerCandidates226_3 : List Nat := [3271, 4093]
set_option maxRecDepth 4096 in
theorem innerCandidates226_3_checked : candidates edge 4 1447 1993 innerTail226_4 best.bound = innerCandidates226_3 := by
  unfold candidates
  have row : edge 1993 = PrimePairCache.smallRow1993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_3 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_3 best = best := by
  rw [innerTail226_3, scan, if_neg (by decide), innerCandidates226_3_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_4
    (if 3 ≤ innerCandidates226_3.length then search edge 3 [1447, 1993] innerCandidates226_3 3440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_4

def innerTail226_2 : List Nat := 1753 :: innerTail226_3
def innerCandidates226_2 : List Nat := [3709]
set_option maxRecDepth 4096 in
theorem innerCandidates226_2_checked : candidates edge 4 1447 1753 innerTail226_3 best.bound = innerCandidates226_2 := by
  unfold candidates
  have row : edge 1753 = PrimePairCache.smallRow1753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_2 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_2 best = best := by
  rw [innerTail226_2, scan, if_neg (by decide), innerCandidates226_2_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_3
    (if 3 ≤ innerCandidates226_2.length then search edge 3 [1447, 1753] innerCandidates226_2 3200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_3

def innerTail226_1 : List Nat := 1549 :: innerTail226_2
def innerCandidates226_1 : List Nat := [2833]
set_option maxRecDepth 4096 in
theorem innerCandidates226_1_checked : candidates edge 4 1447 1549 innerTail226_2 best.bound = innerCandidates226_1 := by
  unfold candidates
  have row : edge 1549 = PrimePairCache.smallRow1549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_1 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_1 best = best := by
  rw [innerTail226_1, scan, if_neg (by decide), innerCandidates226_1_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_2
    (if 3 ≤ innerCandidates226_1.length then search edge 3 [1447, 1549] innerCandidates226_1 2996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_2

def innerTail226_0 : List Nat := 1543 :: innerTail226_1
def innerCandidates226_0 : List Nat := [4093, 5689]
set_option maxRecDepth 4096 in
theorem innerCandidates226_0_checked : candidates edge 4 1447 1543 innerTail226_1 best.bound = innerCandidates226_0 := by
  unfold candidates
  have row : edge 1543 = PrimePairCache.smallRow1543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan226_0 : scan edge (search edge 3) 4 [1447] 1447 innerTail226_0 best = best := by
  rw [innerTail226_0, scan, if_neg (by decide), innerCandidates226_0_checked]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_1
    (if 3 ≤ innerCandidates226_0.length then search edge 3 [1447, 1543] innerCandidates226_0 2990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan226_1

theorem rootBranch226 : search edge 4 [1447] rootCandidates226 1447 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1447] 1447 innerTail226_0 best = best
  exact innerScan226_0

def rootCandidates225 : List Nat := [2003, 2243, 2579, 2837, 3413, 3719, 3833, 4007, 4091, 4271, 4337, 4463, 4931, 5021, 5231, 5261, 5273, 5279, 5657, 5783, 6581, 6659, 6869, 6983, 7127, 7583, 8039, 8081]
set_option maxRecDepth 4096 in
theorem rootCandidates225_checked : candidates edge 5 0 1439 tail226 best.bound = rootCandidates225 := by
  unfold candidates
  have row : edge 1439 = PrimePairCache.smallRow1439.contains := by funext q; rfl
  rw [row]
  decide

def innerTail225_28 : List Nat := []
theorem innerScan225_28 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_28 best = best := rfl

def innerTail225_27 : List Nat := 8081 :: innerTail225_28
def innerCandidates225_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_27_checked : candidates edge 4 1439 8081 innerTail225_28 best.bound = innerCandidates225_27 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_27 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_27 best = best := by
  rw [innerTail225_27, scan, if_neg (by decide), innerCandidates225_27_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_28
    (if 3 ≤ innerCandidates225_27.length then search edge 3 [1439, 8081] innerCandidates225_27 9520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_28

def innerTail225_26 : List Nat := 8039 :: innerTail225_27
def innerCandidates225_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_26_checked : candidates edge 4 1439 8039 innerTail225_27 best.bound = innerCandidates225_26 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_26 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_26 best = best := by
  rw [innerTail225_26, scan, if_neg (by decide), innerCandidates225_26_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_27
    (if 3 ≤ innerCandidates225_26.length then search edge 3 [1439, 8039] innerCandidates225_26 9478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_27

def innerTail225_25 : List Nat := 7583 :: innerTail225_26
def innerCandidates225_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_25_checked : candidates edge 4 1439 7583 innerTail225_26 best.bound = innerCandidates225_25 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_25 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_25 best = best := by
  rw [innerTail225_25, scan, if_neg (by decide), innerCandidates225_25_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_26
    (if 3 ≤ innerCandidates225_25.length then search edge 3 [1439, 7583] innerCandidates225_25 9022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_26

def innerTail225_24 : List Nat := 7127 :: innerTail225_25
def innerCandidates225_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_24_checked : candidates edge 4 1439 7127 innerTail225_25 best.bound = innerCandidates225_24 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_24 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_24 best = best := by
  rw [innerTail225_24, scan, if_neg (by decide), innerCandidates225_24_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_25
    (if 3 ≤ innerCandidates225_24.length then search edge 3 [1439, 7127] innerCandidates225_24 8566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_25

def innerTail225_23 : List Nat := 6983 :: innerTail225_24
def innerCandidates225_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_23_checked : candidates edge 4 1439 6983 innerTail225_24 best.bound = innerCandidates225_23 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_23 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_23 best = best := by
  rw [innerTail225_23, scan, if_neg (by decide), innerCandidates225_23_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_24
    (if 3 ≤ innerCandidates225_23.length then search edge 3 [1439, 6983] innerCandidates225_23 8422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_24

def innerTail225_22 : List Nat := 6869 :: innerTail225_23
def innerCandidates225_22 : List Nat := [7583]
set_option maxRecDepth 4096 in
theorem innerCandidates225_22_checked : candidates edge 4 1439 6869 innerTail225_23 best.bound = innerCandidates225_22 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_22 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_22 best = best := by
  rw [innerTail225_22, scan, if_neg (by decide), innerCandidates225_22_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_23
    (if 3 ≤ innerCandidates225_22.length then search edge 3 [1439, 6869] innerCandidates225_22 8308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_23

def innerTail225_21 : List Nat := 6659 :: innerTail225_22
def innerCandidates225_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_21_checked : candidates edge 4 1439 6659 innerTail225_22 best.bound = innerCandidates225_21 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_21 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_21 best = best := by
  rw [innerTail225_21, scan, if_neg (by decide), innerCandidates225_21_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_22
    (if 3 ≤ innerCandidates225_21.length then search edge 3 [1439, 6659] innerCandidates225_21 8098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_22

def innerTail225_20 : List Nat := 6581 :: innerTail225_21
def innerCandidates225_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_20_checked : candidates edge 4 1439 6581 innerTail225_21 best.bound = innerCandidates225_20 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_20 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_20 best = best := by
  rw [innerTail225_20, scan, if_neg (by decide), innerCandidates225_20_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_21
    (if 3 ≤ innerCandidates225_20.length then search edge 3 [1439, 6581] innerCandidates225_20 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_21

def innerTail225_19 : List Nat := 5783 :: innerTail225_20
def innerCandidates225_19 : List Nat := [6983]
set_option maxRecDepth 4096 in
theorem innerCandidates225_19_checked : candidates edge 4 1439 5783 innerTail225_20 best.bound = innerCandidates225_19 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_19 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_19 best = best := by
  rw [innerTail225_19, scan, if_neg (by decide), innerCandidates225_19_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_20
    (if 3 ≤ innerCandidates225_19.length then search edge 3 [1439, 5783] innerCandidates225_19 7222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_20

def innerTail225_18 : List Nat := 5657 :: innerTail225_19
def innerCandidates225_18 : List Nat := [6983]
set_option maxRecDepth 4096 in
theorem innerCandidates225_18_checked : candidates edge 4 1439 5657 innerTail225_19 best.bound = innerCandidates225_18 := by
  unfold candidates
  have row : edge 5657 = SearchData.row5657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_18 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_18 best = best := by
  rw [innerTail225_18, scan, if_neg (by decide), innerCandidates225_18_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_19
    (if 3 ≤ innerCandidates225_18.length then search edge 3 [1439, 5657] innerCandidates225_18 7096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_19

def innerTail225_17 : List Nat := 5279 :: innerTail225_18
def innerCandidates225_17 : List Nat := [5783]
set_option maxRecDepth 4096 in
theorem innerCandidates225_17_checked : candidates edge 4 1439 5279 innerTail225_18 best.bound = innerCandidates225_17 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_17 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_17 best = best := by
  rw [innerTail225_17, scan, if_neg (by decide), innerCandidates225_17_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_18
    (if 3 ≤ innerCandidates225_17.length then search edge 3 [1439, 5279] innerCandidates225_17 6718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_18

def innerTail225_16 : List Nat := 5273 :: innerTail225_17
def innerCandidates225_16 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates225_16_checked : candidates edge 4 1439 5273 innerTail225_17 best.bound = innerCandidates225_16 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_16 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_16 best = best := by
  rw [innerTail225_16, scan, if_neg (by decide), innerCandidates225_16_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_17
    (if 3 ≤ innerCandidates225_16.length then search edge 3 [1439, 5273] innerCandidates225_16 6712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_17

def innerTail225_15 : List Nat := 5261 :: innerTail225_16
def innerCandidates225_15 : List Nat := [6983]
set_option maxRecDepth 4096 in
theorem innerCandidates225_15_checked : candidates edge 4 1439 5261 innerTail225_16 best.bound = innerCandidates225_15 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_15 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_15 best = best := by
  rw [innerTail225_15, scan, if_neg (by decide), innerCandidates225_15_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_16
    (if 3 ≤ innerCandidates225_15.length then search edge 3 [1439, 5261] innerCandidates225_15 6700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_16

def innerTail225_14 : List Nat := 5231 :: innerTail225_15
def innerCandidates225_14 : List Nat := [6581]
set_option maxRecDepth 4096 in
theorem innerCandidates225_14_checked : candidates edge 4 1439 5231 innerTail225_15 best.bound = innerCandidates225_14 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_14 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_14 best = best := by
  rw [innerTail225_14, scan, if_neg (by decide), innerCandidates225_14_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_15
    (if 3 ≤ innerCandidates225_14.length then search edge 3 [1439, 5231] innerCandidates225_14 6670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_15

def innerTail225_13 : List Nat := 5021 :: innerTail225_14
def innerCandidates225_13 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates225_13_checked : candidates edge 4 1439 5021 innerTail225_14 best.bound = innerCandidates225_13 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_13 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_13 best = best := by
  rw [innerTail225_13, scan, if_neg (by decide), innerCandidates225_13_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_14
    (if 3 ≤ innerCandidates225_13.length then search edge 3 [1439, 5021] innerCandidates225_13 6460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_14

def innerTail225_12 : List Nat := 4931 :: innerTail225_13
def innerCandidates225_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_12_checked : candidates edge 4 1439 4931 innerTail225_13 best.bound = innerCandidates225_12 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_12 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_12 best = best := by
  rw [innerTail225_12, scan, if_neg (by decide), innerCandidates225_12_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_13
    (if 3 ≤ innerCandidates225_12.length then search edge 3 [1439, 4931] innerCandidates225_12 6370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_13

def innerTail225_11 : List Nat := 4463 :: innerTail225_12
def innerCandidates225_11 : List Nat := [5273]
set_option maxRecDepth 4096 in
theorem innerCandidates225_11_checked : candidates edge 4 1439 4463 innerTail225_12 best.bound = innerCandidates225_11 := by
  unfold candidates
  have row : edge 4463 = SearchData.row4463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_11 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_11 best = best := by
  rw [innerTail225_11, scan, if_neg (by decide), innerCandidates225_11_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_12
    (if 3 ≤ innerCandidates225_11.length then search edge 3 [1439, 4463] innerCandidates225_11 5902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_12

def innerTail225_10 : List Nat := 4337 :: innerTail225_11
def innerCandidates225_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_10_checked : candidates edge 4 1439 4337 innerTail225_11 best.bound = innerCandidates225_10 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_10 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_10 best = best := by
  rw [innerTail225_10, scan, if_neg (by decide), innerCandidates225_10_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_11
    (if 3 ≤ innerCandidates225_10.length then search edge 3 [1439, 4337] innerCandidates225_10 5776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_11

def innerTail225_9 : List Nat := 4271 :: innerTail225_10
def innerCandidates225_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_9_checked : candidates edge 4 1439 4271 innerTail225_10 best.bound = innerCandidates225_9 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_9 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_9 best = best := by
  rw [innerTail225_9, scan, if_neg (by decide), innerCandidates225_9_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_10
    (if 3 ≤ innerCandidates225_9.length then search edge 3 [1439, 4271] innerCandidates225_9 5710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_10

def innerTail225_8 : List Nat := 4091 :: innerTail225_9
def innerCandidates225_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_8_checked : candidates edge 4 1439 4091 innerTail225_9 best.bound = innerCandidates225_8 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_8 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_8 best = best := by
  rw [innerTail225_8, scan, if_neg (by decide), innerCandidates225_8_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_9
    (if 3 ≤ innerCandidates225_8.length then search edge 3 [1439, 4091] innerCandidates225_8 5530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_9

def innerTail225_7 : List Nat := 4007 :: innerTail225_8
def innerCandidates225_7 : List Nat := [4091, 4337, 8081]
set_option maxRecDepth 4096 in
theorem innerCandidates225_7_checked : candidates edge 4 1439 4007 innerTail225_8 best.bound = innerCandidates225_7 := by
  unfold candidates
  have row : edge 4007 = SearchData.row4007.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild225_7 : search edge 3 [1439, 4007] innerCandidates225_7 5446 best = best := by decide

theorem innerScan225_7 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_7 best = best := by
  rw [innerTail225_7, scan, if_neg (by decide), innerCandidates225_7_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_8
    (if 3 ≤ innerCandidates225_7.length then search edge 3 [1439, 4007] innerCandidates225_7 5446 best else best) = best
  rw [if_pos (by decide), innerChild225_7]
  exact innerScan225_8

def innerTail225_6 : List Nat := 3833 :: innerTail225_7
def innerCandidates225_6 : List Nat := [5657, 6983]
set_option maxRecDepth 4096 in
theorem innerCandidates225_6_checked : candidates edge 4 1439 3833 innerTail225_7 best.bound = innerCandidates225_6 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_6 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_6 best = best := by
  rw [innerTail225_6, scan, if_neg (by decide), innerCandidates225_6_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_7
    (if 3 ≤ innerCandidates225_6.length then search edge 3 [1439, 3833] innerCandidates225_6 5272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_7

def innerTail225_5 : List Nat := 3719 :: innerTail225_6
def innerCandidates225_5 : List Nat := [4271, 5273]
set_option maxRecDepth 4096 in
theorem innerCandidates225_5_checked : candidates edge 4 1439 3719 innerTail225_6 best.bound = innerCandidates225_5 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_5 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_5 best = best := by
  rw [innerTail225_5, scan, if_neg (by decide), innerCandidates225_5_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_6
    (if 3 ≤ innerCandidates225_5.length then search edge 3 [1439, 3719] innerCandidates225_5 5158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_6

def innerTail225_4 : List Nat := 3413 :: innerTail225_5
def innerCandidates225_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_4_checked : candidates edge 4 1439 3413 innerTail225_5 best.bound = innerCandidates225_4 := by
  unfold candidates
  have row : edge 3413 = SearchData.row3413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_4 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_4 best = best := by
  rw [innerTail225_4, scan, if_neg (by decide), innerCandidates225_4_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_5
    (if 3 ≤ innerCandidates225_4.length then search edge 3 [1439, 3413] innerCandidates225_4 4852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_5

def innerTail225_3 : List Nat := 2837 :: innerTail225_4
def innerCandidates225_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates225_3_checked : candidates edge 4 1439 2837 innerTail225_4 best.bound = innerCandidates225_3 := by
  unfold candidates
  have row : edge 2837 = SearchData.row2837.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_3 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_3 best = best := by
  rw [innerTail225_3, scan, if_neg (by decide), innerCandidates225_3_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_4
    (if 3 ≤ innerCandidates225_3.length then search edge 3 [1439, 2837] innerCandidates225_3 4276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_4

def innerTail225_2 : List Nat := 2579 :: innerTail225_3
def innerCandidates225_2 : List Nat := [4007, 4931]
set_option maxRecDepth 4096 in
theorem innerCandidates225_2_checked : candidates edge 4 1439 2579 innerTail225_3 best.bound = innerCandidates225_2 := by
  unfold candidates
  have row : edge 2579 = PrimePairCache.smallRow2579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_2 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_2 best = best := by
  rw [innerTail225_2, scan, if_neg (by decide), innerCandidates225_2_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_3
    (if 3 ≤ innerCandidates225_2.length then search edge 3 [1439, 2579] innerCandidates225_2 4018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_3

def innerTail225_1 : List Nat := 2243 :: innerTail225_2
def innerCandidates225_1 : List Nat := [8081]
set_option maxRecDepth 4096 in
theorem innerCandidates225_1_checked : candidates edge 4 1439 2243 innerTail225_2 best.bound = innerCandidates225_1 := by
  unfold candidates
  have row : edge 2243 = PrimePairCache.smallRow2243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan225_1 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_1 best = best := by
  rw [innerTail225_1, scan, if_neg (by decide), innerCandidates225_1_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_2
    (if 3 ≤ innerCandidates225_1.length then search edge 3 [1439, 2243] innerCandidates225_1 3682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan225_2

def innerTail225_0 : List Nat := 2003 :: innerTail225_1
def innerCandidates225_0 : List Nat := [3413, 4271, 5657, 6983]
set_option maxRecDepth 4096 in
theorem innerCandidates225_0_checked : candidates edge 4 1439 2003 innerTail225_1 best.bound = innerCandidates225_0 := by
  unfold candidates
  have row : edge 2003 = PrimePairCache.smallRow2003.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild225_0 : search edge 3 [1439, 2003] innerCandidates225_0 3442 best = best := by decide

theorem innerScan225_0 : scan edge (search edge 3) 4 [1439] 1439 innerTail225_0 best = best := by
  rw [innerTail225_0, scan, if_neg (by decide), innerCandidates225_0_checked]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_1
    (if 3 ≤ innerCandidates225_0.length then search edge 3 [1439, 2003] innerCandidates225_0 3442 best else best) = best
  rw [if_pos (by decide), innerChild225_0]
  exact innerScan225_1

theorem rootBranch225 : search edge 4 [1439] rootCandidates225 1439 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1439] 1439 innerTail225_0 best = best
  exact innerScan225_0

end ProjectEulerSolutions.PrimePairUpper.Search
