import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates176 : List Nat := [1471, 1531, 1609, 2551, 2719, 2749, 2953, 3001, 3067, 3307, 3463, 3499, 3547, 3919, 4339, 5197, 5701, 6247, 6793, 7213, 7507]
set_option maxRecDepth 4096 in
theorem rootCandidates176_checked : candidates edge 5 0 1063 tail177 best.bound = rootCandidates176 := by
  unfold candidates
  have row : edge 1063 = PrimePairCache.smallRow1063.contains := by funext q; rfl
  rw [row]
  decide

def innerTail176_21 : List Nat := []
theorem innerScan176_21 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_21 best = best := rfl

def innerTail176_20 : List Nat := 7507 :: innerTail176_21
def innerCandidates176_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_20_checked : candidates edge 4 1063 7507 innerTail176_21 best.bound = innerCandidates176_20 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_20 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_20 best = best := by
  rw [innerTail176_20, scan, if_neg (by decide), innerCandidates176_20_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_21
    (if 3 ≤ innerCandidates176_20.length then search edge 3 [1063, 7507] innerCandidates176_20 8570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_21

def innerTail176_19 : List Nat := 7213 :: innerTail176_20
def innerCandidates176_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_19_checked : candidates edge 4 1063 7213 innerTail176_20 best.bound = innerCandidates176_19 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_19 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_19 best = best := by
  rw [innerTail176_19, scan, if_neg (by decide), innerCandidates176_19_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_20
    (if 3 ≤ innerCandidates176_19.length then search edge 3 [1063, 7213] innerCandidates176_19 8276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_20

def innerTail176_18 : List Nat := 6793 :: innerTail176_19
def innerCandidates176_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_18_checked : candidates edge 4 1063 6793 innerTail176_19 best.bound = innerCandidates176_18 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_18 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_18 best = best := by
  rw [innerTail176_18, scan, if_neg (by decide), innerCandidates176_18_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_19
    (if 3 ≤ innerCandidates176_18.length then search edge 3 [1063, 6793] innerCandidates176_18 7856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_19

def innerTail176_17 : List Nat := 6247 :: innerTail176_18
def innerCandidates176_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_17_checked : candidates edge 4 1063 6247 innerTail176_18 best.bound = innerCandidates176_17 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_17 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_17 best = best := by
  rw [innerTail176_17, scan, if_neg (by decide), innerCandidates176_17_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_18
    (if 3 ≤ innerCandidates176_17.length then search edge 3 [1063, 6247] innerCandidates176_17 7310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_18

def innerTail176_16 : List Nat := 5701 :: innerTail176_17
def innerCandidates176_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_16_checked : candidates edge 4 1063 5701 innerTail176_17 best.bound = innerCandidates176_16 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_16 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_16 best = best := by
  rw [innerTail176_16, scan, if_neg (by decide), innerCandidates176_16_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_17
    (if 3 ≤ innerCandidates176_16.length then search edge 3 [1063, 5701] innerCandidates176_16 6764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_17

def innerTail176_15 : List Nat := 5197 :: innerTail176_16
def innerCandidates176_15 : List Nat := [5701]
set_option maxRecDepth 4096 in
theorem innerCandidates176_15_checked : candidates edge 4 1063 5197 innerTail176_16 best.bound = innerCandidates176_15 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_15 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_15 best = best := by
  rw [innerTail176_15, scan, if_neg (by decide), innerCandidates176_15_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_16
    (if 3 ≤ innerCandidates176_15.length then search edge 3 [1063, 5197] innerCandidates176_15 6260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_16

def innerTail176_14 : List Nat := 4339 :: innerTail176_15
def innerCandidates176_14 : List Nat := [5701]
set_option maxRecDepth 4096 in
theorem innerCandidates176_14_checked : candidates edge 4 1063 4339 innerTail176_15 best.bound = innerCandidates176_14 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_14 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_14 best = best := by
  rw [innerTail176_14, scan, if_neg (by decide), innerCandidates176_14_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_15
    (if 3 ≤ innerCandidates176_14.length then search edge 3 [1063, 4339] innerCandidates176_14 5402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_15

def innerTail176_13 : List Nat := 3919 :: innerTail176_14
def innerCandidates176_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_13_checked : candidates edge 4 1063 3919 innerTail176_14 best.bound = innerCandidates176_13 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_13 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_13 best = best := by
  rw [innerTail176_13, scan, if_neg (by decide), innerCandidates176_13_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_14
    (if 3 ≤ innerCandidates176_13.length then search edge 3 [1063, 3919] innerCandidates176_13 4982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_14

def innerTail176_12 : List Nat := 3547 :: innerTail176_13
def innerCandidates176_12 : List Nat := [7213]
set_option maxRecDepth 4096 in
theorem innerCandidates176_12_checked : candidates edge 4 1063 3547 innerTail176_13 best.bound = innerCandidates176_12 := by
  unfold candidates
  have row : edge 3547 = SearchData.row3547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_12 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_12 best = best := by
  rw [innerTail176_12, scan, if_neg (by decide), innerCandidates176_12_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_13
    (if 3 ≤ innerCandidates176_12.length then search edge 3 [1063, 3547] innerCandidates176_12 4610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_13

def innerTail176_11 : List Nat := 3499 :: innerTail176_12
def innerCandidates176_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_11_checked : candidates edge 4 1063 3499 innerTail176_12 best.bound = innerCandidates176_11 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_11 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_11 best = best := by
  rw [innerTail176_11, scan, if_neg (by decide), innerCandidates176_11_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_12
    (if 3 ≤ innerCandidates176_11.length then search edge 3 [1063, 3499] innerCandidates176_11 4562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_12

def innerTail176_10 : List Nat := 3463 :: innerTail176_11
def innerCandidates176_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_10_checked : candidates edge 4 1063 3463 innerTail176_11 best.bound = innerCandidates176_10 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_10 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_10 best = best := by
  rw [innerTail176_10, scan, if_neg (by decide), innerCandidates176_10_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_11
    (if 3 ≤ innerCandidates176_10.length then search edge 3 [1063, 3463] innerCandidates176_10 4526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_11

def innerTail176_9 : List Nat := 3307 :: innerTail176_10
def innerCandidates176_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_9_checked : candidates edge 4 1063 3307 innerTail176_10 best.bound = innerCandidates176_9 := by
  unfold candidates
  have row : edge 3307 = SearchData.row3307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_9 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_9 best = best := by
  rw [innerTail176_9, scan, if_neg (by decide), innerCandidates176_9_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_10
    (if 3 ≤ innerCandidates176_9.length then search edge 3 [1063, 3307] innerCandidates176_9 4370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_10

def innerTail176_8 : List Nat := 3067 :: innerTail176_9
def innerCandidates176_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_8_checked : candidates edge 4 1063 3067 innerTail176_9 best.bound = innerCandidates176_8 := by
  unfold candidates
  have row : edge 3067 = SearchData.row3067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_8 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_8 best = best := by
  rw [innerTail176_8, scan, if_neg (by decide), innerCandidates176_8_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_9
    (if 3 ≤ innerCandidates176_8.length then search edge 3 [1063, 3067] innerCandidates176_8 4130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_9

def innerTail176_7 : List Nat := 3001 :: innerTail176_8
def innerCandidates176_7 : List Nat := [3499]
set_option maxRecDepth 4096 in
theorem innerCandidates176_7_checked : candidates edge 4 1063 3001 innerTail176_8 best.bound = innerCandidates176_7 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_7 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_7 best = best := by
  rw [innerTail176_7, scan, if_neg (by decide), innerCandidates176_7_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_8
    (if 3 ≤ innerCandidates176_7.length then search edge 3 [1063, 3001] innerCandidates176_7 4064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_8

def innerTail176_6 : List Nat := 2953 :: innerTail176_7
def innerCandidates176_6 : List Nat := [7213]
set_option maxRecDepth 4096 in
theorem innerCandidates176_6_checked : candidates edge 4 1063 2953 innerTail176_7 best.bound = innerCandidates176_6 := by
  unfold candidates
  have row : edge 2953 = SearchData.row2953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_6 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_6 best = best := by
  rw [innerTail176_6, scan, if_neg (by decide), innerCandidates176_6_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_7
    (if 3 ≤ innerCandidates176_6.length then search edge 3 [1063, 2953] innerCandidates176_6 4016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_7

def innerTail176_5 : List Nat := 2749 :: innerTail176_6
def innerCandidates176_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_5_checked : candidates edge 4 1063 2749 innerTail176_6 best.bound = innerCandidates176_5 := by
  unfold candidates
  have row : edge 2749 = SearchData.row2749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_5 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_5 best = best := by
  rw [innerTail176_5, scan, if_neg (by decide), innerCandidates176_5_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_6
    (if 3 ≤ innerCandidates176_5.length then search edge 3 [1063, 2749] innerCandidates176_5 3812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_6

def innerTail176_4 : List Nat := 2719 :: innerTail176_5
def innerCandidates176_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_4_checked : candidates edge 4 1063 2719 innerTail176_5 best.bound = innerCandidates176_4 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_4 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_4 best = best := by
  rw [innerTail176_4, scan, if_neg (by decide), innerCandidates176_4_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_5
    (if 3 ≤ innerCandidates176_4.length then search edge 3 [1063, 2719] innerCandidates176_4 3782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_5

def innerTail176_3 : List Nat := 2551 :: innerTail176_4
def innerCandidates176_3 : List Nat := [6793]
set_option maxRecDepth 4096 in
theorem innerCandidates176_3_checked : candidates edge 4 1063 2551 innerTail176_4 best.bound = innerCandidates176_3 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_3 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_3 best = best := by
  rw [innerTail176_3, scan, if_neg (by decide), innerCandidates176_3_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_4
    (if 3 ≤ innerCandidates176_3.length then search edge 3 [1063, 2551] innerCandidates176_3 3614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_4

def innerTail176_2 : List Nat := 1609 :: innerTail176_3
def innerCandidates176_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_2_checked : candidates edge 4 1063 1609 innerTail176_3 best.bound = innerCandidates176_2 := by
  unfold candidates
  have row : edge 1609 = PrimePairCache.smallRow1609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_2 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_2 best = best := by
  rw [innerTail176_2, scan, if_neg (by decide), innerCandidates176_2_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_3
    (if 3 ≤ innerCandidates176_2.length then search edge 3 [1063, 1609] innerCandidates176_2 2672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_3

def innerTail176_1 : List Nat := 1531 :: innerTail176_2
def innerCandidates176_1 : List Nat := [7213]
set_option maxRecDepth 4096 in
theorem innerCandidates176_1_checked : candidates edge 4 1063 1531 innerTail176_2 best.bound = innerCandidates176_1 := by
  unfold candidates
  have row : edge 1531 = PrimePairCache.smallRow1531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_1 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_1 best = best := by
  rw [innerTail176_1, scan, if_neg (by decide), innerCandidates176_1_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_2
    (if 3 ≤ innerCandidates176_1.length then search edge 3 [1063, 1531] innerCandidates176_1 2594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_2

def innerTail176_0 : List Nat := 1471 :: innerTail176_1
def innerCandidates176_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates176_0_checked : candidates edge 4 1063 1471 innerTail176_1 best.bound = innerCandidates176_0 := by
  unfold candidates
  have row : edge 1471 = PrimePairCache.smallRow1471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan176_0 : scan edge (search edge 3) 4 [1063] 1063 innerTail176_0 best = best := by
  rw [innerTail176_0, scan, if_neg (by decide), innerCandidates176_0_checked]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_1
    (if 3 ≤ innerCandidates176_0.length then search edge 3 [1063, 1471] innerCandidates176_0 2534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan176_1

theorem rootBranch176 : search edge 4 [1063] rootCandidates176 1063 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1063] 1063 innerTail176_0 best = best
  exact innerScan176_0

def rootCandidates175 : List Nat := [1151, 1229, 1493, 1619, 2027, 2213, 2711, 3041, 3167, 3257, 3767, 4139, 4391, 4463, 4649, 4733, 4973, 5147, 5477, 6197, 6329, 7589, 8117]
set_option maxRecDepth 4096 in
theorem rootCandidates175_checked : candidates edge 5 0 1061 tail176 best.bound = rootCandidates175 := by
  unfold candidates
  have row : edge 1061 = PrimePairCache.smallRow1061.contains := by funext q; rfl
  rw [row]
  decide

def innerTail175_23 : List Nat := []
theorem innerScan175_23 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_23 best = best := rfl

def innerTail175_22 : List Nat := 8117 :: innerTail175_23
def innerCandidates175_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_22_checked : candidates edge 4 1061 8117 innerTail175_23 best.bound = innerCandidates175_22 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_22 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_22 best = best := by
  rw [innerTail175_22, scan, if_neg (by decide), innerCandidates175_22_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_23
    (if 3 ≤ innerCandidates175_22.length then search edge 3 [1061, 8117] innerCandidates175_22 9178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_23

def innerTail175_21 : List Nat := 7589 :: innerTail175_22
def innerCandidates175_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_21_checked : candidates edge 4 1061 7589 innerTail175_22 best.bound = innerCandidates175_21 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_21 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_21 best = best := by
  rw [innerTail175_21, scan, if_neg (by decide), innerCandidates175_21_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_22
    (if 3 ≤ innerCandidates175_21.length then search edge 3 [1061, 7589] innerCandidates175_21 8650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_22

def innerTail175_20 : List Nat := 6329 :: innerTail175_21
def innerCandidates175_20 : List Nat := [7589]
set_option maxRecDepth 4096 in
theorem innerCandidates175_20_checked : candidates edge 4 1061 6329 innerTail175_21 best.bound = innerCandidates175_20 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_20 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_20 best = best := by
  rw [innerTail175_20, scan, if_neg (by decide), innerCandidates175_20_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_21
    (if 3 ≤ innerCandidates175_20.length then search edge 3 [1061, 6329] innerCandidates175_20 7390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_21

def innerTail175_19 : List Nat := 6197 :: innerTail175_20
def innerCandidates175_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_19_checked : candidates edge 4 1061 6197 innerTail175_20 best.bound = innerCandidates175_19 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_19 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_19 best = best := by
  rw [innerTail175_19, scan, if_neg (by decide), innerCandidates175_19_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_20
    (if 3 ≤ innerCandidates175_19.length then search edge 3 [1061, 6197] innerCandidates175_19 7258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_20

def innerTail175_18 : List Nat := 5477 :: innerTail175_19
def innerCandidates175_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_18_checked : candidates edge 4 1061 5477 innerTail175_19 best.bound = innerCandidates175_18 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_18 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_18 best = best := by
  rw [innerTail175_18, scan, if_neg (by decide), innerCandidates175_18_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_19
    (if 3 ≤ innerCandidates175_18.length then search edge 3 [1061, 5477] innerCandidates175_18 6538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_19

def innerTail175_17 : List Nat := 5147 :: innerTail175_18
def innerCandidates175_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_17_checked : candidates edge 4 1061 5147 innerTail175_18 best.bound = innerCandidates175_17 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_17 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_17 best = best := by
  rw [innerTail175_17, scan, if_neg (by decide), innerCandidates175_17_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_18
    (if 3 ≤ innerCandidates175_17.length then search edge 3 [1061, 5147] innerCandidates175_17 6208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_18

def innerTail175_16 : List Nat := 4973 :: innerTail175_17
def innerCandidates175_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_16_checked : candidates edge 4 1061 4973 innerTail175_17 best.bound = innerCandidates175_16 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_16 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_16 best = best := by
  rw [innerTail175_16, scan, if_neg (by decide), innerCandidates175_16_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_17
    (if 3 ≤ innerCandidates175_16.length then search edge 3 [1061, 4973] innerCandidates175_16 6034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_17

def innerTail175_15 : List Nat := 4733 :: innerTail175_16
def innerCandidates175_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_15_checked : candidates edge 4 1061 4733 innerTail175_16 best.bound = innerCandidates175_15 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_15 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_15 best = best := by
  rw [innerTail175_15, scan, if_neg (by decide), innerCandidates175_15_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_16
    (if 3 ≤ innerCandidates175_15.length then search edge 3 [1061, 4733] innerCandidates175_15 5794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_16

def innerTail175_14 : List Nat := 4649 :: innerTail175_15
def innerCandidates175_14 : List Nat := [8117]
set_option maxRecDepth 4096 in
theorem innerCandidates175_14_checked : candidates edge 4 1061 4649 innerTail175_15 best.bound = innerCandidates175_14 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_14 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_14 best = best := by
  rw [innerTail175_14, scan, if_neg (by decide), innerCandidates175_14_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_15
    (if 3 ≤ innerCandidates175_14.length then search edge 3 [1061, 4649] innerCandidates175_14 5710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_15

def innerTail175_13 : List Nat := 4463 :: innerTail175_14
def innerCandidates175_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_13_checked : candidates edge 4 1061 4463 innerTail175_14 best.bound = innerCandidates175_13 := by
  unfold candidates
  have row : edge 4463 = SearchData.row4463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_13 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_13 best = best := by
  rw [innerTail175_13, scan, if_neg (by decide), innerCandidates175_13_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_14
    (if 3 ≤ innerCandidates175_13.length then search edge 3 [1061, 4463] innerCandidates175_13 5524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_14

def innerTail175_12 : List Nat := 4391 :: innerTail175_13
def innerCandidates175_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_12_checked : candidates edge 4 1061 4391 innerTail175_13 best.bound = innerCandidates175_12 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_12 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_12 best = best := by
  rw [innerTail175_12, scan, if_neg (by decide), innerCandidates175_12_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_13
    (if 3 ≤ innerCandidates175_12.length then search edge 3 [1061, 4391] innerCandidates175_12 5452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_13

def innerTail175_11 : List Nat := 4139 :: innerTail175_12
def innerCandidates175_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_11_checked : candidates edge 4 1061 4139 innerTail175_12 best.bound = innerCandidates175_11 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_11 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_11 best = best := by
  rw [innerTail175_11, scan, if_neg (by decide), innerCandidates175_11_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_12
    (if 3 ≤ innerCandidates175_11.length then search edge 3 [1061, 4139] innerCandidates175_11 5200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_12

def innerTail175_10 : List Nat := 3767 :: innerTail175_11
def innerCandidates175_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_10_checked : candidates edge 4 1061 3767 innerTail175_11 best.bound = innerCandidates175_10 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_10 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_10 best = best := by
  rw [innerTail175_10, scan, if_neg (by decide), innerCandidates175_10_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_11
    (if 3 ≤ innerCandidates175_10.length then search edge 3 [1061, 3767] innerCandidates175_10 4828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_11

def innerTail175_9 : List Nat := 3257 :: innerTail175_10
def innerCandidates175_9 : List Nat := [4649]
set_option maxRecDepth 4096 in
theorem innerCandidates175_9_checked : candidates edge 4 1061 3257 innerTail175_10 best.bound = innerCandidates175_9 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_9 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_9 best = best := by
  rw [innerTail175_9, scan, if_neg (by decide), innerCandidates175_9_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_10
    (if 3 ≤ innerCandidates175_9.length then search edge 3 [1061, 3257] innerCandidates175_9 4318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_10

def innerTail175_8 : List Nat := 3167 :: innerTail175_9
def innerCandidates175_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_8_checked : candidates edge 4 1061 3167 innerTail175_9 best.bound = innerCandidates175_8 := by
  unfold candidates
  have row : edge 3167 = SearchData.row3167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_8 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_8 best = best := by
  rw [innerTail175_8, scan, if_neg (by decide), innerCandidates175_8_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_9
    (if 3 ≤ innerCandidates175_8.length then search edge 3 [1061, 3167] innerCandidates175_8 4228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_9

def innerTail175_7 : List Nat := 3041 :: innerTail175_8
def innerCandidates175_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_7_checked : candidates edge 4 1061 3041 innerTail175_8 best.bound = innerCandidates175_7 := by
  unfold candidates
  have row : edge 3041 = SearchData.row3041.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_7 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_7 best = best := by
  rw [innerTail175_7, scan, if_neg (by decide), innerCandidates175_7_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_8
    (if 3 ≤ innerCandidates175_7.length then search edge 3 [1061, 3041] innerCandidates175_7 4102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_8

def innerTail175_6 : List Nat := 2711 :: innerTail175_7
def innerCandidates175_6 : List Nat := [7589]
set_option maxRecDepth 4096 in
theorem innerCandidates175_6_checked : candidates edge 4 1061 2711 innerTail175_7 best.bound = innerCandidates175_6 := by
  unfold candidates
  have row : edge 2711 = SearchData.row2711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_6 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_6 best = best := by
  rw [innerTail175_6, scan, if_neg (by decide), innerCandidates175_6_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_7
    (if 3 ≤ innerCandidates175_6.length then search edge 3 [1061, 2711] innerCandidates175_6 3772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_7

def innerTail175_5 : List Nat := 2213 :: innerTail175_6
def innerCandidates175_5 : List Nat := [2711]
set_option maxRecDepth 4096 in
theorem innerCandidates175_5_checked : candidates edge 4 1061 2213 innerTail175_6 best.bound = innerCandidates175_5 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_5 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_5 best = best := by
  rw [innerTail175_5, scan, if_neg (by decide), innerCandidates175_5_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_6
    (if 3 ≤ innerCandidates175_5.length then search edge 3 [1061, 2213] innerCandidates175_5 3274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_6

def innerTail175_4 : List Nat := 2027 :: innerTail175_5
def innerCandidates175_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_4_checked : candidates edge 4 1061 2027 innerTail175_5 best.bound = innerCandidates175_4 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_4 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_4 best = best := by
  rw [innerTail175_4, scan, if_neg (by decide), innerCandidates175_4_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_5
    (if 3 ≤ innerCandidates175_4.length then search edge 3 [1061, 2027] innerCandidates175_4 3088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_5

def innerTail175_3 : List Nat := 1619 :: innerTail175_4
def innerCandidates175_3 : List Nat := [2213, 3257, 6197]
set_option maxRecDepth 4096 in
theorem innerCandidates175_3_checked : candidates edge 4 1061 1619 innerTail175_4 best.bound = innerCandidates175_3 := by
  unfold candidates
  have row : edge 1619 = PrimePairCache.smallRow1619.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild175_3 : search edge 3 [1061, 1619] innerCandidates175_3 2680 best = best := by decide

theorem innerScan175_3 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_3 best = best := by
  rw [innerTail175_3, scan, if_neg (by decide), innerCandidates175_3_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_4
    (if 3 ≤ innerCandidates175_3.length then search edge 3 [1061, 1619] innerCandidates175_3 2680 best else best) = best
  rw [if_pos (by decide), innerChild175_3]
  exact innerScan175_4

def innerTail175_2 : List Nat := 1493 :: innerTail175_3
def innerCandidates175_2 : List Nat := [4649]
set_option maxRecDepth 4096 in
theorem innerCandidates175_2_checked : candidates edge 4 1061 1493 innerTail175_3 best.bound = innerCandidates175_2 := by
  unfold candidates
  have row : edge 1493 = PrimePairCache.smallRow1493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_2 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_2 best = best := by
  rw [innerTail175_2, scan, if_neg (by decide), innerCandidates175_2_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_3
    (if 3 ≤ innerCandidates175_2.length then search edge 3 [1061, 1493] innerCandidates175_2 2554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_3

def innerTail175_1 : List Nat := 1229 :: innerTail175_2
def innerCandidates175_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_1_checked : candidates edge 4 1061 1229 innerTail175_2 best.bound = innerCandidates175_1 := by
  unfold candidates
  have row : edge 1229 = PrimePairCache.smallRow1229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_1 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_1 best = best := by
  rw [innerTail175_1, scan, if_neg (by decide), innerCandidates175_1_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_2
    (if 3 ≤ innerCandidates175_1.length then search edge 3 [1061, 1229] innerCandidates175_1 2290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_2

def innerTail175_0 : List Nat := 1151 :: innerTail175_1
def innerCandidates175_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates175_0_checked : candidates edge 4 1061 1151 innerTail175_1 best.bound = innerCandidates175_0 := by
  unfold candidates
  have row : edge 1151 = PrimePairCache.smallRow1151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan175_0 : scan edge (search edge 3) 4 [1061] 1061 innerTail175_0 best = best := by
  rw [innerTail175_0, scan, if_neg (by decide), innerCandidates175_0_checked]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_1
    (if 3 ≤ innerCandidates175_0.length then search edge 3 [1061, 1151] innerCandidates175_0 2212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan175_1

theorem rootBranch175 : search edge 4 [1061] rootCandidates175 1061 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1061] 1061 innerTail175_0 best = best
  exact innerScan175_0

end ProjectEulerSolutions.PrimePairUpper.Search
