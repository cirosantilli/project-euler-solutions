import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates50 : List Nat := [271, 313, 421, 601, 691, 727, 739, 811, 823, 883, 1867, 2029, 2671, 2731, 2797, 2833, 3217, 3253, 3511, 3823, 4129, 4549, 5431, 5503, 5689, 6361, 6607, 6763, 7741, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates50_checked : candidates edge 5 0 241 tail51 best.bound = rootCandidates50 := by
  unfold candidates
  have row : edge 241 = PrimePairCache.smallRow241.contains := by funext q; rfl
  rw [row]
  decide

def innerTail50_30 : List Nat := []
theorem innerScan50_30 : scan edge (search edge 3) 4 [241] 241 innerTail50_30 best = best := rfl

def innerTail50_29 : List Nat := 8329 :: innerTail50_30
def innerCandidates50_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_29_checked : candidates edge 4 241 8329 innerTail50_30 best.bound = innerCandidates50_29 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_29 : scan edge (search edge 3) 4 [241] 241 innerTail50_29 best = best := by
  rw [innerTail50_29, scan, if_neg (by decide), innerCandidates50_29_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_30
    (if 3 ≤ innerCandidates50_29.length then search edge 3 [241, 8329] innerCandidates50_29 8570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_30

def innerTail50_28 : List Nat := 7741 :: innerTail50_29
def innerCandidates50_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_28_checked : candidates edge 4 241 7741 innerTail50_29 best.bound = innerCandidates50_28 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_28 : scan edge (search edge 3) 4 [241] 241 innerTail50_28 best = best := by
  rw [innerTail50_28, scan, if_neg (by decide), innerCandidates50_28_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_29
    (if 3 ≤ innerCandidates50_28.length then search edge 3 [241, 7741] innerCandidates50_28 7982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_29

def innerTail50_27 : List Nat := 6763 :: innerTail50_28
def innerCandidates50_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_27_checked : candidates edge 4 241 6763 innerTail50_28 best.bound = innerCandidates50_27 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_27 : scan edge (search edge 3) 4 [241] 241 innerTail50_27 best = best := by
  rw [innerTail50_27, scan, if_neg (by decide), innerCandidates50_27_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_28
    (if 3 ≤ innerCandidates50_27.length then search edge 3 [241, 6763] innerCandidates50_27 7004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_28

def innerTail50_26 : List Nat := 6607 :: innerTail50_27
def innerCandidates50_26 : List Nat := [6763]
set_option maxRecDepth 4096 in
theorem innerCandidates50_26_checked : candidates edge 4 241 6607 innerTail50_27 best.bound = innerCandidates50_26 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_26 : scan edge (search edge 3) 4 [241] 241 innerTail50_26 best = best := by
  rw [innerTail50_26, scan, if_neg (by decide), innerCandidates50_26_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_27
    (if 3 ≤ innerCandidates50_26.length then search edge 3 [241, 6607] innerCandidates50_26 6848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_27

def innerTail50_25 : List Nat := 6361 :: innerTail50_26
def innerCandidates50_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_25_checked : candidates edge 4 241 6361 innerTail50_26 best.bound = innerCandidates50_25 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_25 : scan edge (search edge 3) 4 [241] 241 innerTail50_25 best = best := by
  rw [innerTail50_25, scan, if_neg (by decide), innerCandidates50_25_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_26
    (if 3 ≤ innerCandidates50_25.length then search edge 3 [241, 6361] innerCandidates50_25 6602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_26

def innerTail50_24 : List Nat := 5689 :: innerTail50_25
def innerCandidates50_24 : List Nat := [8329]
set_option maxRecDepth 4096 in
theorem innerCandidates50_24_checked : candidates edge 4 241 5689 innerTail50_25 best.bound = innerCandidates50_24 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_24 : scan edge (search edge 3) 4 [241] 241 innerTail50_24 best = best := by
  rw [innerTail50_24, scan, if_neg (by decide), innerCandidates50_24_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_25
    (if 3 ≤ innerCandidates50_24.length then search edge 3 [241, 5689] innerCandidates50_24 5930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_25

def innerTail50_23 : List Nat := 5503 :: innerTail50_24
def innerCandidates50_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_23_checked : candidates edge 4 241 5503 innerTail50_24 best.bound = innerCandidates50_23 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_23 : scan edge (search edge 3) 4 [241] 241 innerTail50_23 best = best := by
  rw [innerTail50_23, scan, if_neg (by decide), innerCandidates50_23_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_24
    (if 3 ≤ innerCandidates50_23.length then search edge 3 [241, 5503] innerCandidates50_23 5744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_24

def innerTail50_22 : List Nat := 5431 :: innerTail50_23
def innerCandidates50_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_22_checked : candidates edge 4 241 5431 innerTail50_23 best.bound = innerCandidates50_22 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_22 : scan edge (search edge 3) 4 [241] 241 innerTail50_22 best = best := by
  rw [innerTail50_22, scan, if_neg (by decide), innerCandidates50_22_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_23
    (if 3 ≤ innerCandidates50_22.length then search edge 3 [241, 5431] innerCandidates50_22 5672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_23

def innerTail50_21 : List Nat := 4549 :: innerTail50_22
def innerCandidates50_21 : List Nat := [6361]
set_option maxRecDepth 4096 in
theorem innerCandidates50_21_checked : candidates edge 4 241 4549 innerTail50_22 best.bound = innerCandidates50_21 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_21 : scan edge (search edge 3) 4 [241] 241 innerTail50_21 best = best := by
  rw [innerTail50_21, scan, if_neg (by decide), innerCandidates50_21_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_22
    (if 3 ≤ innerCandidates50_21.length then search edge 3 [241, 4549] innerCandidates50_21 4790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_22

def innerTail50_20 : List Nat := 4129 :: innerTail50_21
def innerCandidates50_20 : List Nat := [6361]
set_option maxRecDepth 4096 in
theorem innerCandidates50_20_checked : candidates edge 4 241 4129 innerTail50_21 best.bound = innerCandidates50_20 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_20 : scan edge (search edge 3) 4 [241] 241 innerTail50_20 best = best := by
  rw [innerTail50_20, scan, if_neg (by decide), innerCandidates50_20_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_21
    (if 3 ≤ innerCandidates50_20.length then search edge 3 [241, 4129] innerCandidates50_20 4370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_21

def innerTail50_19 : List Nat := 3823 :: innerTail50_20
def innerCandidates50_19 : List Nat := [4549]
set_option maxRecDepth 4096 in
theorem innerCandidates50_19_checked : candidates edge 4 241 3823 innerTail50_20 best.bound = innerCandidates50_19 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_19 : scan edge (search edge 3) 4 [241] 241 innerTail50_19 best = best := by
  rw [innerTail50_19, scan, if_neg (by decide), innerCandidates50_19_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_20
    (if 3 ≤ innerCandidates50_19.length then search edge 3 [241, 3823] innerCandidates50_19 4064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_20

def innerTail50_18 : List Nat := 3511 :: innerTail50_19
def innerCandidates50_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_18_checked : candidates edge 4 241 3511 innerTail50_19 best.bound = innerCandidates50_18 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_18 : scan edge (search edge 3) 4 [241] 241 innerTail50_18 best = best := by
  rw [innerTail50_18, scan, if_neg (by decide), innerCandidates50_18_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_19
    (if 3 ≤ innerCandidates50_18.length then search edge 3 [241, 3511] innerCandidates50_18 3752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_19

def innerTail50_17 : List Nat := 3253 :: innerTail50_18
def innerCandidates50_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_17_checked : candidates edge 4 241 3253 innerTail50_18 best.bound = innerCandidates50_17 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_17 : scan edge (search edge 3) 4 [241] 241 innerTail50_17 best = best := by
  rw [innerTail50_17, scan, if_neg (by decide), innerCandidates50_17_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_18
    (if 3 ≤ innerCandidates50_17.length then search edge 3 [241, 3253] innerCandidates50_17 3494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_18

def innerTail50_16 : List Nat := 3217 :: innerTail50_17
def innerCandidates50_16 : List Nat := [3511]
set_option maxRecDepth 4096 in
theorem innerCandidates50_16_checked : candidates edge 4 241 3217 innerTail50_17 best.bound = innerCandidates50_16 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_16 : scan edge (search edge 3) 4 [241] 241 innerTail50_16 best = best := by
  rw [innerTail50_16, scan, if_neg (by decide), innerCandidates50_16_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_17
    (if 3 ≤ innerCandidates50_16.length then search edge 3 [241, 3217] innerCandidates50_16 3458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_17

def innerTail50_15 : List Nat := 2833 :: innerTail50_16
def innerCandidates50_15 : List Nat := [6361]
set_option maxRecDepth 4096 in
theorem innerCandidates50_15_checked : candidates edge 4 241 2833 innerTail50_16 best.bound = innerCandidates50_15 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_15 : scan edge (search edge 3) 4 [241] 241 innerTail50_15 best = best := by
  rw [innerTail50_15, scan, if_neg (by decide), innerCandidates50_15_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_16
    (if 3 ≤ innerCandidates50_15.length then search edge 3 [241, 2833] innerCandidates50_15 3074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_16

def innerTail50_14 : List Nat := 2797 :: innerTail50_15
def innerCandidates50_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_14_checked : candidates edge 4 241 2797 innerTail50_15 best.bound = innerCandidates50_14 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_14 : scan edge (search edge 3) 4 [241] 241 innerTail50_14 best = best := by
  rw [innerTail50_14, scan, if_neg (by decide), innerCandidates50_14_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_15
    (if 3 ≤ innerCandidates50_14.length then search edge 3 [241, 2797] innerCandidates50_14 3038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_15

def innerTail50_13 : List Nat := 2731 :: innerTail50_14
def innerCandidates50_13 : List Nat := [4129]
set_option maxRecDepth 4096 in
theorem innerCandidates50_13_checked : candidates edge 4 241 2731 innerTail50_14 best.bound = innerCandidates50_13 := by
  unfold candidates
  have row : edge 2731 = SearchData.row2731.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_13 : scan edge (search edge 3) 4 [241] 241 innerTail50_13 best = best := by
  rw [innerTail50_13, scan, if_neg (by decide), innerCandidates50_13_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_14
    (if 3 ≤ innerCandidates50_13.length then search edge 3 [241, 2731] innerCandidates50_13 2972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_14

def innerTail50_12 : List Nat := 2671 :: innerTail50_13
def innerCandidates50_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_12_checked : candidates edge 4 241 2671 innerTail50_13 best.bound = innerCandidates50_12 := by
  unfold candidates
  have row : edge 2671 = SearchData.row2671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_12 : scan edge (search edge 3) 4 [241] 241 innerTail50_12 best = best := by
  rw [innerTail50_12, scan, if_neg (by decide), innerCandidates50_12_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_13
    (if 3 ≤ innerCandidates50_12.length then search edge 3 [241, 2671] innerCandidates50_12 2912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_13

def innerTail50_11 : List Nat := 2029 :: innerTail50_12
def innerCandidates50_11 : List Nat := [5689]
set_option maxRecDepth 4096 in
theorem innerCandidates50_11_checked : candidates edge 4 241 2029 innerTail50_12 best.bound = innerCandidates50_11 := by
  unfold candidates
  have row : edge 2029 = PrimePairCache.smallRow2029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_11 : scan edge (search edge 3) 4 [241] 241 innerTail50_11 best = best := by
  rw [innerTail50_11, scan, if_neg (by decide), innerCandidates50_11_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_12
    (if 3 ≤ innerCandidates50_11.length then search edge 3 [241, 2029] innerCandidates50_11 2270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_12

def innerTail50_10 : List Nat := 1867 :: innerTail50_11
def innerCandidates50_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_10_checked : candidates edge 4 241 1867 innerTail50_11 best.bound = innerCandidates50_10 := by
  unfold candidates
  have row : edge 1867 = PrimePairCache.smallRow1867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_10 : scan edge (search edge 3) 4 [241] 241 innerTail50_10 best = best := by
  rw [innerTail50_10, scan, if_neg (by decide), innerCandidates50_10_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_11
    (if 3 ≤ innerCandidates50_10.length then search edge 3 [241, 1867] innerCandidates50_10 2108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_11

def innerTail50_9 : List Nat := 883 :: innerTail50_10
def innerCandidates50_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_9_checked : candidates edge 4 241 883 innerTail50_10 best.bound = innerCandidates50_9 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_9 : scan edge (search edge 3) 4 [241] 241 innerTail50_9 best = best := by
  rw [innerTail50_9, scan, if_neg (by decide), innerCandidates50_9_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_10
    (if 3 ≤ innerCandidates50_9.length then search edge 3 [241, 883] innerCandidates50_9 1124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_10

def innerTail50_8 : List Nat := 823 :: innerTail50_9
def innerCandidates50_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_8_checked : candidates edge 4 241 823 innerTail50_9 best.bound = innerCandidates50_8 := by
  unfold candidates
  have row : edge 823 = PrimePairCache.smallRow823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_8 : scan edge (search edge 3) 4 [241] 241 innerTail50_8 best = best := by
  rw [innerTail50_8, scan, if_neg (by decide), innerCandidates50_8_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_9
    (if 3 ≤ innerCandidates50_8.length then search edge 3 [241, 823] innerCandidates50_8 1064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_9

def innerTail50_7 : List Nat := 811 :: innerTail50_8
def innerCandidates50_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_7_checked : candidates edge 4 241 811 innerTail50_8 best.bound = innerCandidates50_7 := by
  unfold candidates
  have row : edge 811 = PrimePairCache.smallRow811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_7 : scan edge (search edge 3) 4 [241] 241 innerTail50_7 best = best := by
  rw [innerTail50_7, scan, if_neg (by decide), innerCandidates50_7_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_8
    (if 3 ≤ innerCandidates50_7.length then search edge 3 [241, 811] innerCandidates50_7 1052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_8

def innerTail50_6 : List Nat := 739 :: innerTail50_7
def innerCandidates50_6 : List Nat := [4549]
set_option maxRecDepth 4096 in
theorem innerCandidates50_6_checked : candidates edge 4 241 739 innerTail50_7 best.bound = innerCandidates50_6 := by
  unfold candidates
  have row : edge 739 = PrimePairCache.smallRow739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_6 : scan edge (search edge 3) 4 [241] 241 innerTail50_6 best = best := by
  rw [innerTail50_6, scan, if_neg (by decide), innerCandidates50_6_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_7
    (if 3 ≤ innerCandidates50_6.length then search edge 3 [241, 739] innerCandidates50_6 980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_7

def innerTail50_5 : List Nat := 727 :: innerTail50_6
def innerCandidates50_5 : List Nat := [823, 1867, 5503]
set_option maxRecDepth 4096 in
theorem innerCandidates50_5_checked : candidates edge 4 241 727 innerTail50_6 best.bound = innerCandidates50_5 := by
  unfold candidates
  have row : edge 727 = PrimePairCache.smallRow727.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild50_5 : search edge 3 [241, 727] innerCandidates50_5 968 best = best := by decide

theorem innerScan50_5 : scan edge (search edge 3) 4 [241] 241 innerTail50_5 best = best := by
  rw [innerTail50_5, scan, if_neg (by decide), innerCandidates50_5_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_6
    (if 3 ≤ innerCandidates50_5.length then search edge 3 [241, 727] innerCandidates50_5 968 best else best) = best
  rw [if_pos (by decide), innerChild50_5]
  exact innerScan50_6

def innerTail50_4 : List Nat := 691 :: innerTail50_5
def innerCandidates50_4 : List Nat := [727, 5431]
set_option maxRecDepth 4096 in
theorem innerCandidates50_4_checked : candidates edge 4 241 691 innerTail50_5 best.bound = innerCandidates50_4 := by
  unfold candidates
  have row : edge 691 = PrimePairCache.smallRow691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_4 : scan edge (search edge 3) 4 [241] 241 innerTail50_4 best = best := by
  rw [innerTail50_4, scan, if_neg (by decide), innerCandidates50_4_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_5
    (if 3 ≤ innerCandidates50_4.length then search edge 3 [241, 691] innerCandidates50_4 932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_5

def innerTail50_3 : List Nat := 601 :: innerTail50_4
def innerCandidates50_3 : List Nat := [823, 1867]
set_option maxRecDepth 4096 in
theorem innerCandidates50_3_checked : candidates edge 4 241 601 innerTail50_4 best.bound = innerCandidates50_3 := by
  unfold candidates
  have row : edge 601 = PrimePairCache.smallRow601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_3 : scan edge (search edge 3) 4 [241] 241 innerTail50_3 best = best := by
  rw [innerTail50_3, scan, if_neg (by decide), innerCandidates50_3_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_4
    (if 3 ≤ innerCandidates50_3.length then search edge 3 [241, 601] innerCandidates50_3 842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_4

def innerTail50_2 : List Nat := 421 :: innerTail50_3
def innerCandidates50_2 : List Nat := [3217]
set_option maxRecDepth 4096 in
theorem innerCandidates50_2_checked : candidates edge 4 241 421 innerTail50_3 best.bound = innerCandidates50_2 := by
  unfold candidates
  have row : edge 421 = PrimePairCache.smallRow421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_2 : scan edge (search edge 3) 4 [241] 241 innerTail50_2 best = best := by
  rw [innerTail50_2, scan, if_neg (by decide), innerCandidates50_2_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_3
    (if 3 ≤ innerCandidates50_2.length then search edge 3 [241, 421] innerCandidates50_2 662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_3

def innerTail50_1 : List Nat := 313 :: innerTail50_2
def innerCandidates50_1 : List Nat := [727, 8329]
set_option maxRecDepth 4096 in
theorem innerCandidates50_1_checked : candidates edge 4 241 313 innerTail50_2 best.bound = innerCandidates50_1 := by
  unfold candidates
  have row : edge 313 = PrimePairCache.smallRow313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_1 : scan edge (search edge 3) 4 [241] 241 innerTail50_1 best = best := by
  rw [innerTail50_1, scan, if_neg (by decide), innerCandidates50_1_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_2
    (if 3 ≤ innerCandidates50_1.length then search edge 3 [241, 313] innerCandidates50_1 554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_2

def innerTail50_0 : List Nat := 271 :: innerTail50_1
def innerCandidates50_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates50_0_checked : candidates edge 4 241 271 innerTail50_1 best.bound = innerCandidates50_0 := by
  unfold candidates
  have row : edge 271 = PrimePairCache.smallRow271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan50_0 : scan edge (search edge 3) 4 [241] 241 innerTail50_0 best = best := by
  rw [innerTail50_0, scan, if_neg (by decide), innerCandidates50_0_checked]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_1
    (if 3 ≤ innerCandidates50_0.length then search edge 3 [241, 271] innerCandidates50_0 512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan50_1

theorem rootBranch50 : search edge 4 [241] rootCandidates50 241 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [241] 241 innerTail50_0 best = best
  exact innerScan50_0

end ProjectEulerSolutions.PrimePairUpper.Search
