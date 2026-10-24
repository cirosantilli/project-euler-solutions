import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates126 : List Nat := [823, 997, 1009, 1237, 1279, 1669, 1723, 1867, 2137, 2689, 2803, 3001, 3463, 4327, 5197, 5209, 5227, 5503, 5623, 5857, 5923, 6121, 6277, 6427, 6637, 7639]
set_option maxRecDepth 4096 in
theorem rootCandidates126_checked : candidates edge 5 0 727 tail127 best.bound = rootCandidates126 := by
  unfold candidates
  have row : edge 727 = PrimePairCache.smallRow727.contains := by funext q; rfl
  rw [row]
  decide

def innerTail126_26 : List Nat := []
theorem innerScan126_26 : scan edge (search edge 3) 4 [727] 727 innerTail126_26 best = best := rfl

def innerTail126_25 : List Nat := 7639 :: innerTail126_26
def innerCandidates126_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_25_checked : candidates edge 4 727 7639 innerTail126_26 best.bound = innerCandidates126_25 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_25 : scan edge (search edge 3) 4 [727] 727 innerTail126_25 best = best := by
  rw [innerTail126_25, scan, if_neg (by decide), innerCandidates126_25_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_26
    (if 3 ≤ innerCandidates126_25.length then search edge 3 [727, 7639] innerCandidates126_25 8366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_26

def innerTail126_24 : List Nat := 6637 :: innerTail126_25
def innerCandidates126_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_24_checked : candidates edge 4 727 6637 innerTail126_25 best.bound = innerCandidates126_24 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_24 : scan edge (search edge 3) 4 [727] 727 innerTail126_24 best = best := by
  rw [innerTail126_24, scan, if_neg (by decide), innerCandidates126_24_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_25
    (if 3 ≤ innerCandidates126_24.length then search edge 3 [727, 6637] innerCandidates126_24 7364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_25

def innerTail126_23 : List Nat := 6427 :: innerTail126_24
def innerCandidates126_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_23_checked : candidates edge 4 727 6427 innerTail126_24 best.bound = innerCandidates126_23 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_23 : scan edge (search edge 3) 4 [727] 727 innerTail126_23 best = best := by
  rw [innerTail126_23, scan, if_neg (by decide), innerCandidates126_23_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_24
    (if 3 ≤ innerCandidates126_23.length then search edge 3 [727, 6427] innerCandidates126_23 7154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_24

def innerTail126_22 : List Nat := 6277 :: innerTail126_23
def innerCandidates126_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_22_checked : candidates edge 4 727 6277 innerTail126_23 best.bound = innerCandidates126_22 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_22 : scan edge (search edge 3) 4 [727] 727 innerTail126_22 best = best := by
  rw [innerTail126_22, scan, if_neg (by decide), innerCandidates126_22_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_23
    (if 3 ≤ innerCandidates126_22.length then search edge 3 [727, 6277] innerCandidates126_22 7004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_23

def innerTail126_21 : List Nat := 6121 :: innerTail126_22
def innerCandidates126_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_21_checked : candidates edge 4 727 6121 innerTail126_22 best.bound = innerCandidates126_21 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_21 : scan edge (search edge 3) 4 [727] 727 innerTail126_21 best = best := by
  rw [innerTail126_21, scan, if_neg (by decide), innerCandidates126_21_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_22
    (if 3 ≤ innerCandidates126_21.length then search edge 3 [727, 6121] innerCandidates126_21 6848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_22

def innerTail126_20 : List Nat := 5923 :: innerTail126_21
def innerCandidates126_20 : List Nat := [6427]
set_option maxRecDepth 4096 in
theorem innerCandidates126_20_checked : candidates edge 4 727 5923 innerTail126_21 best.bound = innerCandidates126_20 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_20 : scan edge (search edge 3) 4 [727] 727 innerTail126_20 best = best := by
  rw [innerTail126_20, scan, if_neg (by decide), innerCandidates126_20_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_21
    (if 3 ≤ innerCandidates126_20.length then search edge 3 [727, 5923] innerCandidates126_20 6650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_21

def innerTail126_19 : List Nat := 5857 :: innerTail126_20
def innerCandidates126_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_19_checked : candidates edge 4 727 5857 innerTail126_20 best.bound = innerCandidates126_19 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_19 : scan edge (search edge 3) 4 [727] 727 innerTail126_19 best = best := by
  rw [innerTail126_19, scan, if_neg (by decide), innerCandidates126_19_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_20
    (if 3 ≤ innerCandidates126_19.length then search edge 3 [727, 5857] innerCandidates126_19 6584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_20

def innerTail126_18 : List Nat := 5623 :: innerTail126_19
def innerCandidates126_18 : List Nat := [7639]
set_option maxRecDepth 4096 in
theorem innerCandidates126_18_checked : candidates edge 4 727 5623 innerTail126_19 best.bound = innerCandidates126_18 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_18 : scan edge (search edge 3) 4 [727] 727 innerTail126_18 best = best := by
  rw [innerTail126_18, scan, if_neg (by decide), innerCandidates126_18_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_19
    (if 3 ≤ innerCandidates126_18.length then search edge 3 [727, 5623] innerCandidates126_18 6350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_19

def innerTail126_17 : List Nat := 5503 :: innerTail126_18
def innerCandidates126_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_17_checked : candidates edge 4 727 5503 innerTail126_18 best.bound = innerCandidates126_17 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_17 : scan edge (search edge 3) 4 [727] 727 innerTail126_17 best = best := by
  rw [innerTail126_17, scan, if_neg (by decide), innerCandidates126_17_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_18
    (if 3 ≤ innerCandidates126_17.length then search edge 3 [727, 5503] innerCandidates126_17 6230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_18

def innerTail126_16 : List Nat := 5227 :: innerTail126_17
def innerCandidates126_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_16_checked : candidates edge 4 727 5227 innerTail126_17 best.bound = innerCandidates126_16 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_16 : scan edge (search edge 3) 4 [727] 727 innerTail126_16 best = best := by
  rw [innerTail126_16, scan, if_neg (by decide), innerCandidates126_16_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_17
    (if 3 ≤ innerCandidates126_16.length then search edge 3 [727, 5227] innerCandidates126_16 5954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_17

def innerTail126_15 : List Nat := 5209 :: innerTail126_16
def innerCandidates126_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_15_checked : candidates edge 4 727 5209 innerTail126_16 best.bound = innerCandidates126_15 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_15 : scan edge (search edge 3) 4 [727] 727 innerTail126_15 best = best := by
  rw [innerTail126_15, scan, if_neg (by decide), innerCandidates126_15_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_16
    (if 3 ≤ innerCandidates126_15.length then search edge 3 [727, 5209] innerCandidates126_15 5936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_16

def innerTail126_14 : List Nat := 5197 :: innerTail126_15
def innerCandidates126_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_14_checked : candidates edge 4 727 5197 innerTail126_15 best.bound = innerCandidates126_14 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_14 : scan edge (search edge 3) 4 [727] 727 innerTail126_14 best = best := by
  rw [innerTail126_14, scan, if_neg (by decide), innerCandidates126_14_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_15
    (if 3 ≤ innerCandidates126_14.length then search edge 3 [727, 5197] innerCandidates126_14 5924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_15

def innerTail126_13 : List Nat := 4327 :: innerTail126_14
def innerCandidates126_13 : List Nat := [5503, 7639]
set_option maxRecDepth 4096 in
theorem innerCandidates126_13_checked : candidates edge 4 727 4327 innerTail126_14 best.bound = innerCandidates126_13 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_13 : scan edge (search edge 3) 4 [727] 727 innerTail126_13 best = best := by
  rw [innerTail126_13, scan, if_neg (by decide), innerCandidates126_13_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_14
    (if 3 ≤ innerCandidates126_13.length then search edge 3 [727, 4327] innerCandidates126_13 5054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_14

def innerTail126_12 : List Nat := 3463 :: innerTail126_13
def innerCandidates126_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_12_checked : candidates edge 4 727 3463 innerTail126_13 best.bound = innerCandidates126_12 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_12 : scan edge (search edge 3) 4 [727] 727 innerTail126_12 best = best := by
  rw [innerTail126_12, scan, if_neg (by decide), innerCandidates126_12_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_13
    (if 3 ≤ innerCandidates126_12.length then search edge 3 [727, 3463] innerCandidates126_12 4190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_13

def innerTail126_11 : List Nat := 3001 :: innerTail126_12
def innerCandidates126_11 : List Nat := [6121]
set_option maxRecDepth 4096 in
theorem innerCandidates126_11_checked : candidates edge 4 727 3001 innerTail126_12 best.bound = innerCandidates126_11 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_11 : scan edge (search edge 3) 4 [727] 727 innerTail126_11 best = best := by
  rw [innerTail126_11, scan, if_neg (by decide), innerCandidates126_11_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_12
    (if 3 ≤ innerCandidates126_11.length then search edge 3 [727, 3001] innerCandidates126_11 3728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_12

def innerTail126_10 : List Nat := 2803 :: innerTail126_11
def innerCandidates126_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_10_checked : candidates edge 4 727 2803 innerTail126_11 best.bound = innerCandidates126_10 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_10 : scan edge (search edge 3) 4 [727] 727 innerTail126_10 best = best := by
  rw [innerTail126_10, scan, if_neg (by decide), innerCandidates126_10_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_11
    (if 3 ≤ innerCandidates126_10.length then search edge 3 [727, 2803] innerCandidates126_10 3530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_11

def innerTail126_9 : List Nat := 2689 :: innerTail126_10
def innerCandidates126_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_9_checked : candidates edge 4 727 2689 innerTail126_10 best.bound = innerCandidates126_9 := by
  unfold candidates
  have row : edge 2689 = SearchData.row2689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_9 : scan edge (search edge 3) 4 [727] 727 innerTail126_9 best = best := by
  rw [innerTail126_9, scan, if_neg (by decide), innerCandidates126_9_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_10
    (if 3 ≤ innerCandidates126_9.length then search edge 3 [727, 2689] innerCandidates126_9 3416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_10

def innerTail126_8 : List Nat := 2137 :: innerTail126_9
def innerCandidates126_8 : List Nat := [5503]
set_option maxRecDepth 4096 in
theorem innerCandidates126_8_checked : candidates edge 4 727 2137 innerTail126_9 best.bound = innerCandidates126_8 := by
  unfold candidates
  have row : edge 2137 = PrimePairCache.smallRow2137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_8 : scan edge (search edge 3) 4 [727] 727 innerTail126_8 best = best := by
  rw [innerTail126_8, scan, if_neg (by decide), innerCandidates126_8_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_9
    (if 3 ≤ innerCandidates126_8.length then search edge 3 [727, 2137] innerCandidates126_8 2864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_9

def innerTail126_7 : List Nat := 1867 :: innerTail126_8
def innerCandidates126_7 : List Nat := [6121, 7639]
set_option maxRecDepth 4096 in
theorem innerCandidates126_7_checked : candidates edge 4 727 1867 innerTail126_8 best.bound = innerCandidates126_7 := by
  unfold candidates
  have row : edge 1867 = PrimePairCache.smallRow1867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_7 : scan edge (search edge 3) 4 [727] 727 innerTail126_7 best = best := by
  rw [innerTail126_7, scan, if_neg (by decide), innerCandidates126_7_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_8
    (if 3 ≤ innerCandidates126_7.length then search edge 3 [727, 1867] innerCandidates126_7 2594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_8

def innerTail126_6 : List Nat := 1723 :: innerTail126_7
def innerCandidates126_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_6_checked : candidates edge 4 727 1723 innerTail126_7 best.bound = innerCandidates126_6 := by
  unfold candidates
  have row : edge 1723 = PrimePairCache.smallRow1723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_6 : scan edge (search edge 3) 4 [727] 727 innerTail126_6 best = best := by
  rw [innerTail126_6, scan, if_neg (by decide), innerCandidates126_6_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_7
    (if 3 ≤ innerCandidates126_6.length then search edge 3 [727, 1723] innerCandidates126_6 2450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_7

def innerTail126_5 : List Nat := 1669 :: innerTail126_6
def innerCandidates126_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_5_checked : candidates edge 4 727 1669 innerTail126_6 best.bound = innerCandidates126_5 := by
  unfold candidates
  have row : edge 1669 = PrimePairCache.smallRow1669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_5 : scan edge (search edge 3) 4 [727] 727 innerTail126_5 best = best := by
  rw [innerTail126_5, scan, if_neg (by decide), innerCandidates126_5_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_6
    (if 3 ≤ innerCandidates126_5.length then search edge 3 [727, 1669] innerCandidates126_5 2396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_6

def innerTail126_4 : List Nat := 1279 :: innerTail126_5
def innerCandidates126_4 : List Nat := [2803, 3463]
set_option maxRecDepth 4096 in
theorem innerCandidates126_4_checked : candidates edge 4 727 1279 innerTail126_5 best.bound = innerCandidates126_4 := by
  unfold candidates
  have row : edge 1279 = PrimePairCache.smallRow1279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_4 : scan edge (search edge 3) 4 [727] 727 innerTail126_4 best = best := by
  rw [innerTail126_4, scan, if_neg (by decide), innerCandidates126_4_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_5
    (if 3 ≤ innerCandidates126_4.length then search edge 3 [727, 1279] innerCandidates126_4 2006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_5

def innerTail126_3 : List Nat := 1237 :: innerTail126_4
def innerCandidates126_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_3_checked : candidates edge 4 727 1237 innerTail126_4 best.bound = innerCandidates126_3 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_3 : scan edge (search edge 3) 4 [727] 727 innerTail126_3 best = best := by
  rw [innerTail126_3, scan, if_neg (by decide), innerCandidates126_3_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_4
    (if 3 ≤ innerCandidates126_3.length then search edge 3 [727, 1237] innerCandidates126_3 1964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_4

def innerTail126_2 : List Nat := 1009 :: innerTail126_3
def innerCandidates126_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_2_checked : candidates edge 4 727 1009 innerTail126_3 best.bound = innerCandidates126_2 := by
  unfold candidates
  have row : edge 1009 = PrimePairCache.smallRow1009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_2 : scan edge (search edge 3) 4 [727] 727 innerTail126_2 best = best := by
  rw [innerTail126_2, scan, if_neg (by decide), innerCandidates126_2_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_3
    (if 3 ≤ innerCandidates126_2.length then search edge 3 [727, 1009] innerCandidates126_2 1736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_3

def innerTail126_1 : List Nat := 997 :: innerTail126_2
def innerCandidates126_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates126_1_checked : candidates edge 4 727 997 innerTail126_2 best.bound = innerCandidates126_1 := by
  unfold candidates
  have row : edge 997 = PrimePairCache.smallRow997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_1 : scan edge (search edge 3) 4 [727] 727 innerTail126_1 best = best := by
  rw [innerTail126_1, scan, if_neg (by decide), innerCandidates126_1_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_2
    (if 3 ≤ innerCandidates126_1.length then search edge 3 [727, 997] innerCandidates126_1 1724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_2

def innerTail126_0 : List Nat := 823 :: innerTail126_1
def innerCandidates126_0 : List Nat := [5209]
set_option maxRecDepth 4096 in
theorem innerCandidates126_0_checked : candidates edge 4 727 823 innerTail126_1 best.bound = innerCandidates126_0 := by
  unfold candidates
  have row : edge 823 = PrimePairCache.smallRow823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan126_0 : scan edge (search edge 3) 4 [727] 727 innerTail126_0 best = best := by
  rw [innerTail126_0, scan, if_neg (by decide), innerCandidates126_0_checked]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_1
    (if 3 ≤ innerCandidates126_0.length then search edge 3 [727, 823] innerCandidates126_0 1550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan126_1

theorem rootBranch126 : search edge 4 [727] rootCandidates126 727 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [727] 727 innerTail126_0 best = best
  exact innerScan126_0

def rootCandidates125 : List Nat := [773, 911, 947, 1319, 1439, 1997, 2027, 2333, 2543, 2579, 3257, 3833, 4799, 4973, 5039, 5261, 5387, 5807, 6317, 6323, 6869, 7079, 7283, 7433, 7793, 7937, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates125_checked : candidates edge 5 0 719 tail126 best.bound = rootCandidates125 := by
  unfold candidates
  have row : edge 719 = PrimePairCache.smallRow719.contains := by funext q; rfl
  rw [row]
  decide

def innerTail125_27 : List Nat := []
theorem innerScan125_27 : scan edge (search edge 3) 4 [719] 719 innerTail125_27 best = best := rfl

def innerTail125_26 : List Nat := 8297 :: innerTail125_27
def innerCandidates125_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_26_checked : candidates edge 4 719 8297 innerTail125_27 best.bound = innerCandidates125_26 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_26 : scan edge (search edge 3) 4 [719] 719 innerTail125_26 best = best := by
  rw [innerTail125_26, scan, if_neg (by decide), innerCandidates125_26_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_27
    (if 3 ≤ innerCandidates125_26.length then search edge 3 [719, 8297] innerCandidates125_26 9016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_27

def innerTail125_25 : List Nat := 7937 :: innerTail125_26
def innerCandidates125_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_25_checked : candidates edge 4 719 7937 innerTail125_26 best.bound = innerCandidates125_25 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_25 : scan edge (search edge 3) 4 [719] 719 innerTail125_25 best = best := by
  rw [innerTail125_25, scan, if_neg (by decide), innerCandidates125_25_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_26
    (if 3 ≤ innerCandidates125_25.length then search edge 3 [719, 7937] innerCandidates125_25 8656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_26

def innerTail125_24 : List Nat := 7793 :: innerTail125_25
def innerCandidates125_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_24_checked : candidates edge 4 719 7793 innerTail125_25 best.bound = innerCandidates125_24 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_24 : scan edge (search edge 3) 4 [719] 719 innerTail125_24 best = best := by
  rw [innerTail125_24, scan, if_neg (by decide), innerCandidates125_24_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_25
    (if 3 ≤ innerCandidates125_24.length then search edge 3 [719, 7793] innerCandidates125_24 8512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_25

def innerTail125_23 : List Nat := 7433 :: innerTail125_24
def innerCandidates125_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_23_checked : candidates edge 4 719 7433 innerTail125_24 best.bound = innerCandidates125_23 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_23 : scan edge (search edge 3) 4 [719] 719 innerTail125_23 best = best := by
  rw [innerTail125_23, scan, if_neg (by decide), innerCandidates125_23_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_24
    (if 3 ≤ innerCandidates125_23.length then search edge 3 [719, 7433] innerCandidates125_23 8152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_24

def innerTail125_22 : List Nat := 7283 :: innerTail125_23
def innerCandidates125_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_22_checked : candidates edge 4 719 7283 innerTail125_23 best.bound = innerCandidates125_22 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_22 : scan edge (search edge 3) 4 [719] 719 innerTail125_22 best = best := by
  rw [innerTail125_22, scan, if_neg (by decide), innerCandidates125_22_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_23
    (if 3 ≤ innerCandidates125_22.length then search edge 3 [719, 7283] innerCandidates125_22 8002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_23

def innerTail125_21 : List Nat := 7079 :: innerTail125_22
def innerCandidates125_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_21_checked : candidates edge 4 719 7079 innerTail125_22 best.bound = innerCandidates125_21 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_21 : scan edge (search edge 3) 4 [719] 719 innerTail125_21 best = best := by
  rw [innerTail125_21, scan, if_neg (by decide), innerCandidates125_21_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_22
    (if 3 ≤ innerCandidates125_21.length then search edge 3 [719, 7079] innerCandidates125_21 7798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_22

def innerTail125_20 : List Nat := 6869 :: innerTail125_21
def innerCandidates125_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_20_checked : candidates edge 4 719 6869 innerTail125_21 best.bound = innerCandidates125_20 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_20 : scan edge (search edge 3) 4 [719] 719 innerTail125_20 best = best := by
  rw [innerTail125_20, scan, if_neg (by decide), innerCandidates125_20_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_21
    (if 3 ≤ innerCandidates125_20.length then search edge 3 [719, 6869] innerCandidates125_20 7588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_21

def innerTail125_19 : List Nat := 6323 :: innerTail125_20
def innerCandidates125_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_19_checked : candidates edge 4 719 6323 innerTail125_20 best.bound = innerCandidates125_19 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_19 : scan edge (search edge 3) 4 [719] 719 innerTail125_19 best = best := by
  rw [innerTail125_19, scan, if_neg (by decide), innerCandidates125_19_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_20
    (if 3 ≤ innerCandidates125_19.length then search edge 3 [719, 6323] innerCandidates125_19 7042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_20

def innerTail125_18 : List Nat := 6317 :: innerTail125_19
def innerCandidates125_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_18_checked : candidates edge 4 719 6317 innerTail125_19 best.bound = innerCandidates125_18 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_18 : scan edge (search edge 3) 4 [719] 719 innerTail125_18 best = best := by
  rw [innerTail125_18, scan, if_neg (by decide), innerCandidates125_18_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_19
    (if 3 ≤ innerCandidates125_18.length then search edge 3 [719, 6317] innerCandidates125_18 7036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_19

def innerTail125_17 : List Nat := 5807 :: innerTail125_18
def innerCandidates125_17 : List Nat := [8297]
set_option maxRecDepth 4096 in
theorem innerCandidates125_17_checked : candidates edge 4 719 5807 innerTail125_18 best.bound = innerCandidates125_17 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_17 : scan edge (search edge 3) 4 [719] 719 innerTail125_17 best = best := by
  rw [innerTail125_17, scan, if_neg (by decide), innerCandidates125_17_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_18
    (if 3 ≤ innerCandidates125_17.length then search edge 3 [719, 5807] innerCandidates125_17 6526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_18

def innerTail125_16 : List Nat := 5387 :: innerTail125_17
def innerCandidates125_16 : List Nat := [5807]
set_option maxRecDepth 4096 in
theorem innerCandidates125_16_checked : candidates edge 4 719 5387 innerTail125_17 best.bound = innerCandidates125_16 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_16 : scan edge (search edge 3) 4 [719] 719 innerTail125_16 best = best := by
  rw [innerTail125_16, scan, if_neg (by decide), innerCandidates125_16_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_17
    (if 3 ≤ innerCandidates125_16.length then search edge 3 [719, 5387] innerCandidates125_16 6106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_17

def innerTail125_15 : List Nat := 5261 :: innerTail125_16
def innerCandidates125_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_15_checked : candidates edge 4 719 5261 innerTail125_16 best.bound = innerCandidates125_15 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_15 : scan edge (search edge 3) 4 [719] 719 innerTail125_15 best = best := by
  rw [innerTail125_15, scan, if_neg (by decide), innerCandidates125_15_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_16
    (if 3 ≤ innerCandidates125_15.length then search edge 3 [719, 5261] innerCandidates125_15 5980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_16

def innerTail125_14 : List Nat := 5039 :: innerTail125_15
def innerCandidates125_14 : List Nat := [6317]
set_option maxRecDepth 4096 in
theorem innerCandidates125_14_checked : candidates edge 4 719 5039 innerTail125_15 best.bound = innerCandidates125_14 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_14 : scan edge (search edge 3) 4 [719] 719 innerTail125_14 best = best := by
  rw [innerTail125_14, scan, if_neg (by decide), innerCandidates125_14_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_15
    (if 3 ≤ innerCandidates125_14.length then search edge 3 [719, 5039] innerCandidates125_14 5758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_15

def innerTail125_13 : List Nat := 4973 :: innerTail125_14
def innerCandidates125_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_13_checked : candidates edge 4 719 4973 innerTail125_14 best.bound = innerCandidates125_13 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_13 : scan edge (search edge 3) 4 [719] 719 innerTail125_13 best = best := by
  rw [innerTail125_13, scan, if_neg (by decide), innerCandidates125_13_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_14
    (if 3 ≤ innerCandidates125_13.length then search edge 3 [719, 4973] innerCandidates125_13 5692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_14

def innerTail125_12 : List Nat := 4799 :: innerTail125_13
def innerCandidates125_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_12_checked : candidates edge 4 719 4799 innerTail125_13 best.bound = innerCandidates125_12 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_12 : scan edge (search edge 3) 4 [719] 719 innerTail125_12 best = best := by
  rw [innerTail125_12, scan, if_neg (by decide), innerCandidates125_12_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_13
    (if 3 ≤ innerCandidates125_12.length then search edge 3 [719, 4799] innerCandidates125_12 5518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_13

def innerTail125_11 : List Nat := 3833 :: innerTail125_12
def innerCandidates125_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_11_checked : candidates edge 4 719 3833 innerTail125_12 best.bound = innerCandidates125_11 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_11 : scan edge (search edge 3) 4 [719] 719 innerTail125_11 best = best := by
  rw [innerTail125_11, scan, if_neg (by decide), innerCandidates125_11_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_12
    (if 3 ≤ innerCandidates125_11.length then search edge 3 [719, 3833] innerCandidates125_11 4552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_12

def innerTail125_10 : List Nat := 3257 :: innerTail125_11
def innerCandidates125_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_10_checked : candidates edge 4 719 3257 innerTail125_11 best.bound = innerCandidates125_10 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_10 : scan edge (search edge 3) 4 [719] 719 innerTail125_10 best = best := by
  rw [innerTail125_10, scan, if_neg (by decide), innerCandidates125_10_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_11
    (if 3 ≤ innerCandidates125_10.length then search edge 3 [719, 3257] innerCandidates125_10 3976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_11

def innerTail125_9 : List Nat := 2579 :: innerTail125_10
def innerCandidates125_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_9_checked : candidates edge 4 719 2579 innerTail125_10 best.bound = innerCandidates125_9 := by
  unfold candidates
  have row : edge 2579 = PrimePairCache.smallRow2579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_9 : scan edge (search edge 3) 4 [719] 719 innerTail125_9 best = best := by
  rw [innerTail125_9, scan, if_neg (by decide), innerCandidates125_9_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_10
    (if 3 ≤ innerCandidates125_9.length then search edge 3 [719, 2579] innerCandidates125_9 3298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_10

def innerTail125_8 : List Nat := 2543 :: innerTail125_9
def innerCandidates125_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_8_checked : candidates edge 4 719 2543 innerTail125_9 best.bound = innerCandidates125_8 := by
  unfold candidates
  have row : edge 2543 = PrimePairCache.smallRow2543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_8 : scan edge (search edge 3) 4 [719] 719 innerTail125_8 best = best := by
  rw [innerTail125_8, scan, if_neg (by decide), innerCandidates125_8_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_9
    (if 3 ≤ innerCandidates125_8.length then search edge 3 [719, 2543] innerCandidates125_8 3262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_9

def innerTail125_7 : List Nat := 2333 :: innerTail125_8
def innerCandidates125_7 : List Nat := [5261]
set_option maxRecDepth 4096 in
theorem innerCandidates125_7_checked : candidates edge 4 719 2333 innerTail125_8 best.bound = innerCandidates125_7 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_7 : scan edge (search edge 3) 4 [719] 719 innerTail125_7 best = best := by
  rw [innerTail125_7, scan, if_neg (by decide), innerCandidates125_7_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_8
    (if 3 ≤ innerCandidates125_7.length then search edge 3 [719, 2333] innerCandidates125_7 3052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_8

def innerTail125_6 : List Nat := 2027 :: innerTail125_7
def innerCandidates125_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_6_checked : candidates edge 4 719 2027 innerTail125_7 best.bound = innerCandidates125_6 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_6 : scan edge (search edge 3) 4 [719] 719 innerTail125_6 best = best := by
  rw [innerTail125_6, scan, if_neg (by decide), innerCandidates125_6_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_7
    (if 3 ≤ innerCandidates125_6.length then search edge 3 [719, 2027] innerCandidates125_6 2746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_7

def innerTail125_5 : List Nat := 1997 :: innerTail125_6
def innerCandidates125_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_5_checked : candidates edge 4 719 1997 innerTail125_6 best.bound = innerCandidates125_5 := by
  unfold candidates
  have row : edge 1997 = PrimePairCache.smallRow1997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_5 : scan edge (search edge 3) 4 [719] 719 innerTail125_5 best = best := by
  rw [innerTail125_5, scan, if_neg (by decide), innerCandidates125_5_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_6
    (if 3 ≤ innerCandidates125_5.length then search edge 3 [719, 1997] innerCandidates125_5 2716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_6

def innerTail125_4 : List Nat := 1439 :: innerTail125_5
def innerCandidates125_4 : List Nat := [2579, 3833, 5261, 6869]
set_option maxRecDepth 4096 in
theorem innerCandidates125_4_checked : candidates edge 4 719 1439 innerTail125_5 best.bound = innerCandidates125_4 := by
  unfold candidates
  have row : edge 1439 = PrimePairCache.smallRow1439.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild125_4 : search edge 3 [719, 1439] innerCandidates125_4 2158 best = best := by decide

theorem innerScan125_4 : scan edge (search edge 3) 4 [719] 719 innerTail125_4 best = best := by
  rw [innerTail125_4, scan, if_neg (by decide), innerCandidates125_4_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_5
    (if 3 ≤ innerCandidates125_4.length then search edge 3 [719, 1439] innerCandidates125_4 2158 best else best) = best
  rw [if_pos (by decide), innerChild125_4]
  exact innerScan125_5

def innerTail125_3 : List Nat := 1319 :: innerTail125_4
def innerCandidates125_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates125_3_checked : candidates edge 4 719 1319 innerTail125_4 best.bound = innerCandidates125_3 := by
  unfold candidates
  have row : edge 1319 = PrimePairCache.smallRow1319.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_3 : scan edge (search edge 3) 4 [719] 719 innerTail125_3 best = best := by
  rw [innerTail125_3, scan, if_neg (by decide), innerCandidates125_3_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_4
    (if 3 ≤ innerCandidates125_3.length then search edge 3 [719, 1319] innerCandidates125_3 2038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_4

def innerTail125_2 : List Nat := 947 :: innerTail125_3
def innerCandidates125_2 : List Nat := [1439]
set_option maxRecDepth 4096 in
theorem innerCandidates125_2_checked : candidates edge 4 719 947 innerTail125_3 best.bound = innerCandidates125_2 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_2 : scan edge (search edge 3) 4 [719] 719 innerTail125_2 best = best := by
  rw [innerTail125_2, scan, if_neg (by decide), innerCandidates125_2_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_3
    (if 3 ≤ innerCandidates125_2.length then search edge 3 [719, 947] innerCandidates125_2 1666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_3

def innerTail125_1 : List Nat := 911 :: innerTail125_2
def innerCandidates125_1 : List Nat := [947]
set_option maxRecDepth 4096 in
theorem innerCandidates125_1_checked : candidates edge 4 719 911 innerTail125_2 best.bound = innerCandidates125_1 := by
  unfold candidates
  have row : edge 911 = PrimePairCache.smallRow911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_1 : scan edge (search edge 3) 4 [719] 719 innerTail125_1 best = best := by
  rw [innerTail125_1, scan, if_neg (by decide), innerCandidates125_1_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_2
    (if 3 ≤ innerCandidates125_1.length then search edge 3 [719, 911] innerCandidates125_1 1630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_2

def innerTail125_0 : List Nat := 773 :: innerTail125_1
def innerCandidates125_0 : List Nat := [1439]
set_option maxRecDepth 4096 in
theorem innerCandidates125_0_checked : candidates edge 4 719 773 innerTail125_1 best.bound = innerCandidates125_0 := by
  unfold candidates
  have row : edge 773 = PrimePairCache.smallRow773.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan125_0 : scan edge (search edge 3) 4 [719] 719 innerTail125_0 best = best := by
  rw [innerTail125_0, scan, if_neg (by decide), innerCandidates125_0_checked]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_1
    (if 3 ≤ innerCandidates125_0.length then search edge 3 [719, 773] innerCandidates125_0 1492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan125_1

theorem rootBranch125 : search edge 4 [719] rootCandidates125 719 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [719] 719 innerTail125_0 best = best
  exact innerScan125_0

end ProjectEulerSolutions.PrimePairUpper.Search
