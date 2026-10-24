import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates43 : List Nat := [211, 373, 379, 673, 739, 751, 811, 877, 1153, 1489, 1609, 1933, 2251, 2437, 2917, 2953, 3067, 3109, 3259, 3529, 4327, 5011, 5023, 5107, 6301, 6949]
set_option maxRecDepth 4096 in
theorem rootCandidates43_checked : candidates edge 5 0 199 tail44 best.bound = rootCandidates43 := by
  unfold candidates
  have row : edge 199 = PrimePairCache.smallRow199.contains := by funext q; rfl
  rw [row]
  decide

def innerTail43_26 : List Nat := []
theorem innerScan43_26 : scan edge (search edge 3) 4 [199] 199 innerTail43_26 best = best := rfl

def innerTail43_25 : List Nat := 6949 :: innerTail43_26
def innerCandidates43_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates43_25_checked : candidates edge 4 199 6949 innerTail43_26 best.bound = innerCandidates43_25 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_25 : scan edge (search edge 3) 4 [199] 199 innerTail43_25 best = best := by
  rw [innerTail43_25, scan, if_neg (by decide), innerCandidates43_25_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_26
    (if 3 ≤ innerCandidates43_25.length then search edge 3 [199, 6949] innerCandidates43_25 7148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_26

def innerTail43_24 : List Nat := 6301 :: innerTail43_25
def innerCandidates43_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates43_24_checked : candidates edge 4 199 6301 innerTail43_25 best.bound = innerCandidates43_24 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_24 : scan edge (search edge 3) 4 [199] 199 innerTail43_24 best = best := by
  rw [innerTail43_24, scan, if_neg (by decide), innerCandidates43_24_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_25
    (if 3 ≤ innerCandidates43_24.length then search edge 3 [199, 6301] innerCandidates43_24 6500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_25

def innerTail43_23 : List Nat := 5107 :: innerTail43_24
def innerCandidates43_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates43_23_checked : candidates edge 4 199 5107 innerTail43_24 best.bound = innerCandidates43_23 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_23 : scan edge (search edge 3) 4 [199] 199 innerTail43_23 best = best := by
  rw [innerTail43_23, scan, if_neg (by decide), innerCandidates43_23_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_24
    (if 3 ≤ innerCandidates43_23.length then search edge 3 [199, 5107] innerCandidates43_23 5306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_24

def innerTail43_22 : List Nat := 5023 :: innerTail43_23
def innerCandidates43_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates43_22_checked : candidates edge 4 199 5023 innerTail43_23 best.bound = innerCandidates43_22 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_22 : scan edge (search edge 3) 4 [199] 199 innerTail43_22 best = best := by
  rw [innerTail43_22, scan, if_neg (by decide), innerCandidates43_22_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_23
    (if 3 ≤ innerCandidates43_22.length then search edge 3 [199, 5023] innerCandidates43_22 5222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_23

def innerTail43_21 : List Nat := 5011 :: innerTail43_22
def innerCandidates43_21 : List Nat := [6949]
set_option maxRecDepth 4096 in
theorem innerCandidates43_21_checked : candidates edge 4 199 5011 innerTail43_22 best.bound = innerCandidates43_21 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_21 : scan edge (search edge 3) 4 [199] 199 innerTail43_21 best = best := by
  rw [innerTail43_21, scan, if_neg (by decide), innerCandidates43_21_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_22
    (if 3 ≤ innerCandidates43_21.length then search edge 3 [199, 5011] innerCandidates43_21 5210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_22

def innerTail43_20 : List Nat := 4327 :: innerTail43_21
def innerCandidates43_20 : List Nat := [5107]
set_option maxRecDepth 4096 in
theorem innerCandidates43_20_checked : candidates edge 4 199 4327 innerTail43_21 best.bound = innerCandidates43_20 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_20 : scan edge (search edge 3) 4 [199] 199 innerTail43_20 best = best := by
  rw [innerTail43_20, scan, if_neg (by decide), innerCandidates43_20_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_21
    (if 3 ≤ innerCandidates43_20.length then search edge 3 [199, 4327] innerCandidates43_20 4526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_21

def innerTail43_19 : List Nat := 3529 :: innerTail43_20
def innerCandidates43_19 : List Nat := [5011]
set_option maxRecDepth 4096 in
theorem innerCandidates43_19_checked : candidates edge 4 199 3529 innerTail43_20 best.bound = innerCandidates43_19 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_19 : scan edge (search edge 3) 4 [199] 199 innerTail43_19 best = best := by
  rw [innerTail43_19, scan, if_neg (by decide), innerCandidates43_19_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_20
    (if 3 ≤ innerCandidates43_19.length then search edge 3 [199, 3529] innerCandidates43_19 3728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_20

def innerTail43_18 : List Nat := 3259 :: innerTail43_19
def innerCandidates43_18 : List Nat := [4327, 5107]
set_option maxRecDepth 4096 in
theorem innerCandidates43_18_checked : candidates edge 4 199 3259 innerTail43_19 best.bound = innerCandidates43_18 := by
  unfold candidates
  have row : edge 3259 = SearchData.row3259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_18 : scan edge (search edge 3) 4 [199] 199 innerTail43_18 best = best := by
  rw [innerTail43_18, scan, if_neg (by decide), innerCandidates43_18_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_19
    (if 3 ≤ innerCandidates43_18.length then search edge 3 [199, 3259] innerCandidates43_18 3458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_19

def innerTail43_17 : List Nat := 3109 :: innerTail43_18
def innerCandidates43_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates43_17_checked : candidates edge 4 199 3109 innerTail43_18 best.bound = innerCandidates43_17 := by
  unfold candidates
  have row : edge 3109 = SearchData.row3109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_17 : scan edge (search edge 3) 4 [199] 199 innerTail43_17 best = best := by
  rw [innerTail43_17, scan, if_neg (by decide), innerCandidates43_17_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_18
    (if 3 ≤ innerCandidates43_17.length then search edge 3 [199, 3109] innerCandidates43_17 3308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_18

def innerTail43_16 : List Nat := 3067 :: innerTail43_17
def innerCandidates43_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates43_16_checked : candidates edge 4 199 3067 innerTail43_17 best.bound = innerCandidates43_16 := by
  unfold candidates
  have row : edge 3067 = SearchData.row3067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_16 : scan edge (search edge 3) 4 [199] 199 innerTail43_16 best = best := by
  rw [innerTail43_16, scan, if_neg (by decide), innerCandidates43_16_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_17
    (if 3 ≤ innerCandidates43_16.length then search edge 3 [199, 3067] innerCandidates43_16 3266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_17

def innerTail43_15 : List Nat := 2953 :: innerTail43_16
def innerCandidates43_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates43_15_checked : candidates edge 4 199 2953 innerTail43_16 best.bound = innerCandidates43_15 := by
  unfold candidates
  have row : edge 2953 = SearchData.row2953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_15 : scan edge (search edge 3) 4 [199] 199 innerTail43_15 best = best := by
  rw [innerTail43_15, scan, if_neg (by decide), innerCandidates43_15_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_16
    (if 3 ≤ innerCandidates43_15.length then search edge 3 [199, 2953] innerCandidates43_15 3152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_16

def innerTail43_14 : List Nat := 2917 :: innerTail43_15
def innerCandidates43_14 : List Nat := [5011]
set_option maxRecDepth 4096 in
theorem innerCandidates43_14_checked : candidates edge 4 199 2917 innerTail43_15 best.bound = innerCandidates43_14 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_14 : scan edge (search edge 3) 4 [199] 199 innerTail43_14 best = best := by
  rw [innerTail43_14, scan, if_neg (by decide), innerCandidates43_14_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_15
    (if 3 ≤ innerCandidates43_14.length then search edge 3 [199, 2917] innerCandidates43_14 3116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_15

def innerTail43_13 : List Nat := 2437 :: innerTail43_14
def innerCandidates43_13 : List Nat := [6301]
set_option maxRecDepth 4096 in
theorem innerCandidates43_13_checked : candidates edge 4 199 2437 innerTail43_14 best.bound = innerCandidates43_13 := by
  unfold candidates
  have row : edge 2437 = PrimePairCache.smallRow2437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_13 : scan edge (search edge 3) 4 [199] 199 innerTail43_13 best = best := by
  rw [innerTail43_13, scan, if_neg (by decide), innerCandidates43_13_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_14
    (if 3 ≤ innerCandidates43_13.length then search edge 3 [199, 2437] innerCandidates43_13 2636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_14

def innerTail43_12 : List Nat := 2251 :: innerTail43_13
def innerCandidates43_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates43_12_checked : candidates edge 4 199 2251 innerTail43_13 best.bound = innerCandidates43_12 := by
  unfold candidates
  have row : edge 2251 = PrimePairCache.smallRow2251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_12 : scan edge (search edge 3) 4 [199] 199 innerTail43_12 best = best := by
  rw [innerTail43_12, scan, if_neg (by decide), innerCandidates43_12_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_13
    (if 3 ≤ innerCandidates43_12.length then search edge 3 [199, 2251] innerCandidates43_12 2450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_13

def innerTail43_11 : List Nat := 1933 :: innerTail43_12
def innerCandidates43_11 : List Nat := [5011, 5023]
set_option maxRecDepth 4096 in
theorem innerCandidates43_11_checked : candidates edge 4 199 1933 innerTail43_12 best.bound = innerCandidates43_11 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_11 : scan edge (search edge 3) 4 [199] 199 innerTail43_11 best = best := by
  rw [innerTail43_11, scan, if_neg (by decide), innerCandidates43_11_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_12
    (if 3 ≤ innerCandidates43_11.length then search edge 3 [199, 1933] innerCandidates43_11 2132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_12

def innerTail43_10 : List Nat := 1609 :: innerTail43_11
def innerCandidates43_10 : List Nat := [5011]
set_option maxRecDepth 4096 in
theorem innerCandidates43_10_checked : candidates edge 4 199 1609 innerTail43_11 best.bound = innerCandidates43_10 := by
  unfold candidates
  have row : edge 1609 = PrimePairCache.smallRow1609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_10 : scan edge (search edge 3) 4 [199] 199 innerTail43_10 best = best := by
  rw [innerTail43_10, scan, if_neg (by decide), innerCandidates43_10_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_11
    (if 3 ≤ innerCandidates43_10.length then search edge 3 [199, 1609] innerCandidates43_10 1808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_11

def innerTail43_9 : List Nat := 1489 :: innerTail43_10
def innerCandidates43_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates43_9_checked : candidates edge 4 199 1489 innerTail43_10 best.bound = innerCandidates43_9 := by
  unfold candidates
  have row : edge 1489 = PrimePairCache.smallRow1489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_9 : scan edge (search edge 3) 4 [199] 199 innerTail43_9 best = best := by
  rw [innerTail43_9, scan, if_neg (by decide), innerCandidates43_9_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_10
    (if 3 ≤ innerCandidates43_9.length then search edge 3 [199, 1489] innerCandidates43_9 1688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_10

def innerTail43_8 : List Nat := 1153 :: innerTail43_9
def innerCandidates43_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates43_8_checked : candidates edge 4 199 1153 innerTail43_9 best.bound = innerCandidates43_8 := by
  unfold candidates
  have row : edge 1153 = PrimePairCache.smallRow1153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_8 : scan edge (search edge 3) 4 [199] 199 innerTail43_8 best = best := by
  rw [innerTail43_8, scan, if_neg (by decide), innerCandidates43_8_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_9
    (if 3 ≤ innerCandidates43_8.length then search edge 3 [199, 1153] innerCandidates43_8 1352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_9

def innerTail43_7 : List Nat := 877 :: innerTail43_8
def innerCandidates43_7 : List Nat := [2251, 2917]
set_option maxRecDepth 4096 in
theorem innerCandidates43_7_checked : candidates edge 4 199 877 innerTail43_8 best.bound = innerCandidates43_7 := by
  unfold candidates
  have row : edge 877 = PrimePairCache.smallRow877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_7 : scan edge (search edge 3) 4 [199] 199 innerTail43_7 best = best := by
  rw [innerTail43_7, scan, if_neg (by decide), innerCandidates43_7_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_8
    (if 3 ≤ innerCandidates43_7.length then search edge 3 [199, 877] innerCandidates43_7 1076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_8

def innerTail43_6 : List Nat := 811 :: innerTail43_7
def innerCandidates43_6 : List Nat := [3109]
set_option maxRecDepth 4096 in
theorem innerCandidates43_6_checked : candidates edge 4 199 811 innerTail43_7 best.bound = innerCandidates43_6 := by
  unfold candidates
  have row : edge 811 = PrimePairCache.smallRow811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_6 : scan edge (search edge 3) 4 [199] 199 innerTail43_6 best = best := by
  rw [innerTail43_6, scan, if_neg (by decide), innerCandidates43_6_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_7
    (if 3 ≤ innerCandidates43_6.length then search edge 3 [199, 811] innerCandidates43_6 1010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_7

def innerTail43_5 : List Nat := 751 :: innerTail43_6
def innerCandidates43_5 : List Nat := [2437]
set_option maxRecDepth 4096 in
theorem innerCandidates43_5_checked : candidates edge 4 199 751 innerTail43_6 best.bound = innerCandidates43_5 := by
  unfold candidates
  have row : edge 751 = PrimePairCache.smallRow751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_5 : scan edge (search edge 3) 4 [199] 199 innerTail43_5 best = best := by
  rw [innerTail43_5, scan, if_neg (by decide), innerCandidates43_5_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_6
    (if 3 ≤ innerCandidates43_5.length then search edge 3 [199, 751] innerCandidates43_5 950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_6

def innerTail43_4 : List Nat := 739 :: innerTail43_5
def innerCandidates43_4 : List Nat := [751]
set_option maxRecDepth 4096 in
theorem innerCandidates43_4_checked : candidates edge 4 199 739 innerTail43_5 best.bound = innerCandidates43_4 := by
  unfold candidates
  have row : edge 739 = PrimePairCache.smallRow739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_4 : scan edge (search edge 3) 4 [199] 199 innerTail43_4 best = best := by
  rw [innerTail43_4, scan, if_neg (by decide), innerCandidates43_4_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_5
    (if 3 ≤ innerCandidates43_4.length then search edge 3 [199, 739] innerCandidates43_4 938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_5

def innerTail43_3 : List Nat := 673 :: innerTail43_4
def innerCandidates43_3 : List Nat := [1489]
set_option maxRecDepth 4096 in
theorem innerCandidates43_3_checked : candidates edge 4 199 673 innerTail43_4 best.bound = innerCandidates43_3 := by
  unfold candidates
  have row : edge 673 = PrimePairCache.smallRow673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_3 : scan edge (search edge 3) 4 [199] 199 innerTail43_3 best = best := by
  rw [innerTail43_3, scan, if_neg (by decide), innerCandidates43_3_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_4
    (if 3 ≤ innerCandidates43_3.length then search edge 3 [199, 673] innerCandidates43_3 872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_4

def innerTail43_2 : List Nat := 379 :: innerTail43_3
def innerCandidates43_2 : List Nat := [751, 811, 877, 3259]
set_option maxRecDepth 4096 in
theorem innerCandidates43_2_checked : candidates edge 4 199 379 innerTail43_3 best.bound = innerCandidates43_2 := by
  unfold candidates
  have row : edge 379 = PrimePairCache.smallRow379.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild43_2 : search edge 3 [199, 379] innerCandidates43_2 578 best = best := by decide

theorem innerScan43_2 : scan edge (search edge 3) 4 [199] 199 innerTail43_2 best = best := by
  rw [innerTail43_2, scan, if_neg (by decide), innerCandidates43_2_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_3
    (if 3 ≤ innerCandidates43_2.length then search edge 3 [199, 379] innerCandidates43_2 578 best else best) = best
  rw [if_pos (by decide), innerChild43_2]
  exact innerScan43_3

def innerTail43_1 : List Nat := 373 :: innerTail43_2
def innerCandidates43_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates43_1_checked : candidates edge 4 199 373 innerTail43_2 best.bound = innerCandidates43_1 := by
  unfold candidates
  have row : edge 373 = PrimePairCache.smallRow373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_1 : scan edge (search edge 3) 4 [199] 199 innerTail43_1 best = best := by
  rw [innerTail43_1, scan, if_neg (by decide), innerCandidates43_1_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_2
    (if 3 ≤ innerCandidates43_1.length then search edge 3 [199, 373] innerCandidates43_1 572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_2

def innerTail43_0 : List Nat := 211 :: innerTail43_1
def innerCandidates43_0 : List Nat := [373]
set_option maxRecDepth 4096 in
theorem innerCandidates43_0_checked : candidates edge 4 199 211 innerTail43_1 best.bound = innerCandidates43_0 := by
  unfold candidates
  have row : edge 211 = PrimePairCache.smallRow211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan43_0 : scan edge (search edge 3) 4 [199] 199 innerTail43_0 best = best := by
  rw [innerTail43_0, scan, if_neg (by decide), innerCandidates43_0_checked]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_1
    (if 3 ≤ innerCandidates43_0.length then search edge 3 [199, 211] innerCandidates43_0 410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan43_1

theorem rootBranch43 : search edge 4 [199] rootCandidates43 199 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [199] 199 innerTail43_0 best = best
  exact innerScan43_0

def rootCandidates42 : List Nat := [311, 347, 569, 641, 947, 971, 1181, 1481, 1637, 2099, 2207, 2657, 4073, 4457, 4493, 4547, 4919, 4937, 4967, 5021, 5387, 5399, 6113, 6173, 6761, 7187, 7817, 8297, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates42_checked : candidates edge 5 0 197 tail43 best.bound = rootCandidates42 := by
  unfold candidates
  have row : edge 197 = PrimePairCache.smallRow197.contains := by funext q; rfl
  rw [row]
  decide

def innerTail42_29 : List Nat := []
theorem innerScan42_29 : scan edge (search edge 3) 4 [197] 197 innerTail42_29 best = best := rfl

def innerTail42_28 : List Nat := 8363 :: innerTail42_29
def innerCandidates42_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_28_checked : candidates edge 4 197 8363 innerTail42_29 best.bound = innerCandidates42_28 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_28 : scan edge (search edge 3) 4 [197] 197 innerTail42_28 best = best := by
  rw [innerTail42_28, scan, if_neg (by decide), innerCandidates42_28_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_29
    (if 3 ≤ innerCandidates42_28.length then search edge 3 [197, 8363] innerCandidates42_28 8560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_29

def innerTail42_27 : List Nat := 8297 :: innerTail42_28
def innerCandidates42_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_27_checked : candidates edge 4 197 8297 innerTail42_28 best.bound = innerCandidates42_27 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_27 : scan edge (search edge 3) 4 [197] 197 innerTail42_27 best = best := by
  rw [innerTail42_27, scan, if_neg (by decide), innerCandidates42_27_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_28
    (if 3 ≤ innerCandidates42_27.length then search edge 3 [197, 8297] innerCandidates42_27 8494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_28

def innerTail42_26 : List Nat := 7817 :: innerTail42_27
def innerCandidates42_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_26_checked : candidates edge 4 197 7817 innerTail42_27 best.bound = innerCandidates42_26 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_26 : scan edge (search edge 3) 4 [197] 197 innerTail42_26 best = best := by
  rw [innerTail42_26, scan, if_neg (by decide), innerCandidates42_26_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_27
    (if 3 ≤ innerCandidates42_26.length then search edge 3 [197, 7817] innerCandidates42_26 8014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_27

def innerTail42_25 : List Nat := 7187 :: innerTail42_26
def innerCandidates42_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_25_checked : candidates edge 4 197 7187 innerTail42_26 best.bound = innerCandidates42_25 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_25 : scan edge (search edge 3) 4 [197] 197 innerTail42_25 best = best := by
  rw [innerTail42_25, scan, if_neg (by decide), innerCandidates42_25_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_26
    (if 3 ≤ innerCandidates42_25.length then search edge 3 [197, 7187] innerCandidates42_25 7384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_26

def innerTail42_24 : List Nat := 6761 :: innerTail42_25
def innerCandidates42_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_24_checked : candidates edge 4 197 6761 innerTail42_25 best.bound = innerCandidates42_24 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_24 : scan edge (search edge 3) 4 [197] 197 innerTail42_24 best = best := by
  rw [innerTail42_24, scan, if_neg (by decide), innerCandidates42_24_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_25
    (if 3 ≤ innerCandidates42_24.length then search edge 3 [197, 6761] innerCandidates42_24 6958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_25

def innerTail42_23 : List Nat := 6173 :: innerTail42_24
def innerCandidates42_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_23_checked : candidates edge 4 197 6173 innerTail42_24 best.bound = innerCandidates42_23 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_23 : scan edge (search edge 3) 4 [197] 197 innerTail42_23 best = best := by
  rw [innerTail42_23, scan, if_neg (by decide), innerCandidates42_23_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_24
    (if 3 ≤ innerCandidates42_23.length then search edge 3 [197, 6173] innerCandidates42_23 6370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_24

def innerTail42_22 : List Nat := 6113 :: innerTail42_23
def innerCandidates42_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_22_checked : candidates edge 4 197 6113 innerTail42_23 best.bound = innerCandidates42_22 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_22 : scan edge (search edge 3) 4 [197] 197 innerTail42_22 best = best := by
  rw [innerTail42_22, scan, if_neg (by decide), innerCandidates42_22_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_23
    (if 3 ≤ innerCandidates42_22.length then search edge 3 [197, 6113] innerCandidates42_22 6310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_23

def innerTail42_21 : List Nat := 5399 :: innerTail42_22
def innerCandidates42_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_21_checked : candidates edge 4 197 5399 innerTail42_22 best.bound = innerCandidates42_21 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_21 : scan edge (search edge 3) 4 [197] 197 innerTail42_21 best = best := by
  rw [innerTail42_21, scan, if_neg (by decide), innerCandidates42_21_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_22
    (if 3 ≤ innerCandidates42_21.length then search edge 3 [197, 5399] innerCandidates42_21 5596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_22

def innerTail42_20 : List Nat := 5387 :: innerTail42_21
def innerCandidates42_20 : List Nat := [5399]
set_option maxRecDepth 4096 in
theorem innerCandidates42_20_checked : candidates edge 4 197 5387 innerTail42_21 best.bound = innerCandidates42_20 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_20 : scan edge (search edge 3) 4 [197] 197 innerTail42_20 best = best := by
  rw [innerTail42_20, scan, if_neg (by decide), innerCandidates42_20_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_21
    (if 3 ≤ innerCandidates42_20.length then search edge 3 [197, 5387] innerCandidates42_20 5584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_21

def innerTail42_19 : List Nat := 5021 :: innerTail42_20
def innerCandidates42_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_19_checked : candidates edge 4 197 5021 innerTail42_20 best.bound = innerCandidates42_19 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_19 : scan edge (search edge 3) 4 [197] 197 innerTail42_19 best = best := by
  rw [innerTail42_19, scan, if_neg (by decide), innerCandidates42_19_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_20
    (if 3 ≤ innerCandidates42_19.length then search edge 3 [197, 5021] innerCandidates42_19 5218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_20

def innerTail42_18 : List Nat := 4967 :: innerTail42_19
def innerCandidates42_18 : List Nat := [5387]
set_option maxRecDepth 4096 in
theorem innerCandidates42_18_checked : candidates edge 4 197 4967 innerTail42_19 best.bound = innerCandidates42_18 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_18 : scan edge (search edge 3) 4 [197] 197 innerTail42_18 best = best := by
  rw [innerTail42_18, scan, if_neg (by decide), innerCandidates42_18_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_19
    (if 3 ≤ innerCandidates42_18.length then search edge 3 [197, 4967] innerCandidates42_18 5164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_19

def innerTail42_17 : List Nat := 4937 :: innerTail42_18
def innerCandidates42_17 : List Nat := [5399]
set_option maxRecDepth 4096 in
theorem innerCandidates42_17_checked : candidates edge 4 197 4937 innerTail42_18 best.bound = innerCandidates42_17 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_17 : scan edge (search edge 3) 4 [197] 197 innerTail42_17 best = best := by
  rw [innerTail42_17, scan, if_neg (by decide), innerCandidates42_17_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_18
    (if 3 ≤ innerCandidates42_17.length then search edge 3 [197, 4937] innerCandidates42_17 5134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_18

def innerTail42_16 : List Nat := 4919 :: innerTail42_17
def innerCandidates42_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_16_checked : candidates edge 4 197 4919 innerTail42_17 best.bound = innerCandidates42_16 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_16 : scan edge (search edge 3) 4 [197] 197 innerTail42_16 best = best := by
  rw [innerTail42_16, scan, if_neg (by decide), innerCandidates42_16_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_17
    (if 3 ≤ innerCandidates42_16.length then search edge 3 [197, 4919] innerCandidates42_16 5116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_17

def innerTail42_15 : List Nat := 4547 :: innerTail42_16
def innerCandidates42_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_15_checked : candidates edge 4 197 4547 innerTail42_16 best.bound = innerCandidates42_15 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_15 : scan edge (search edge 3) 4 [197] 197 innerTail42_15 best = best := by
  rw [innerTail42_15, scan, if_neg (by decide), innerCandidates42_15_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_16
    (if 3 ≤ innerCandidates42_15.length then search edge 3 [197, 4547] innerCandidates42_15 4744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_16

def innerTail42_14 : List Nat := 4493 :: innerTail42_15
def innerCandidates42_14 : List Nat := [6173, 7187]
set_option maxRecDepth 4096 in
theorem innerCandidates42_14_checked : candidates edge 4 197 4493 innerTail42_15 best.bound = innerCandidates42_14 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_14 : scan edge (search edge 3) 4 [197] 197 innerTail42_14 best = best := by
  rw [innerTail42_14, scan, if_neg (by decide), innerCandidates42_14_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_15
    (if 3 ≤ innerCandidates42_14.length then search edge 3 [197, 4493] innerCandidates42_14 4690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_15

def innerTail42_13 : List Nat := 4457 :: innerTail42_14
def innerCandidates42_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_13_checked : candidates edge 4 197 4457 innerTail42_14 best.bound = innerCandidates42_13 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_13 : scan edge (search edge 3) 4 [197] 197 innerTail42_13 best = best := by
  rw [innerTail42_13, scan, if_neg (by decide), innerCandidates42_13_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_14
    (if 3 ≤ innerCandidates42_13.length then search edge 3 [197, 4457] innerCandidates42_13 4654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_14

def innerTail42_12 : List Nat := 4073 :: innerTail42_13
def innerCandidates42_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_12_checked : candidates edge 4 197 4073 innerTail42_13 best.bound = innerCandidates42_12 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_12 : scan edge (search edge 3) 4 [197] 197 innerTail42_12 best = best := by
  rw [innerTail42_12, scan, if_neg (by decide), innerCandidates42_12_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_13
    (if 3 ≤ innerCandidates42_12.length then search edge 3 [197, 4073] innerCandidates42_12 4270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_13

def innerTail42_11 : List Nat := 2657 :: innerTail42_12
def innerCandidates42_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_11_checked : candidates edge 4 197 2657 innerTail42_12 best.bound = innerCandidates42_11 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_11 : scan edge (search edge 3) 4 [197] 197 innerTail42_11 best = best := by
  rw [innerTail42_11, scan, if_neg (by decide), innerCandidates42_11_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_12
    (if 3 ≤ innerCandidates42_11.length then search edge 3 [197, 2657] innerCandidates42_11 2854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_12

def innerTail42_10 : List Nat := 2207 :: innerTail42_11
def innerCandidates42_10 : List Nat := [8363]
set_option maxRecDepth 4096 in
theorem innerCandidates42_10_checked : candidates edge 4 197 2207 innerTail42_11 best.bound = innerCandidates42_10 := by
  unfold candidates
  have row : edge 2207 = PrimePairCache.smallRow2207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_10 : scan edge (search edge 3) 4 [197] 197 innerTail42_10 best = best := by
  rw [innerTail42_10, scan, if_neg (by decide), innerCandidates42_10_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_11
    (if 3 ≤ innerCandidates42_10.length then search edge 3 [197, 2207] innerCandidates42_10 2404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_11

def innerTail42_9 : List Nat := 2099 :: innerTail42_10
def innerCandidates42_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_9_checked : candidates edge 4 197 2099 innerTail42_10 best.bound = innerCandidates42_9 := by
  unfold candidates
  have row : edge 2099 = PrimePairCache.smallRow2099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_9 : scan edge (search edge 3) 4 [197] 197 innerTail42_9 best = best := by
  rw [innerTail42_9, scan, if_neg (by decide), innerCandidates42_9_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_10
    (if 3 ≤ innerCandidates42_9.length then search edge 3 [197, 2099] innerCandidates42_9 2296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_10

def innerTail42_8 : List Nat := 1637 :: innerTail42_9
def innerCandidates42_8 : List Nat := [4073, 8363]
set_option maxRecDepth 4096 in
theorem innerCandidates42_8_checked : candidates edge 4 197 1637 innerTail42_9 best.bound = innerCandidates42_8 := by
  unfold candidates
  have row : edge 1637 = PrimePairCache.smallRow1637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_8 : scan edge (search edge 3) 4 [197] 197 innerTail42_8 best = best := by
  rw [innerTail42_8, scan, if_neg (by decide), innerCandidates42_8_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_9
    (if 3 ≤ innerCandidates42_8.length then search edge 3 [197, 1637] innerCandidates42_8 1834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_9

def innerTail42_7 : List Nat := 1481 :: innerTail42_8
def innerCandidates42_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_7_checked : candidates edge 4 197 1481 innerTail42_8 best.bound = innerCandidates42_7 := by
  unfold candidates
  have row : edge 1481 = PrimePairCache.smallRow1481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_7 : scan edge (search edge 3) 4 [197] 197 innerTail42_7 best = best := by
  rw [innerTail42_7, scan, if_neg (by decide), innerCandidates42_7_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_8
    (if 3 ≤ innerCandidates42_7.length then search edge 3 [197, 1481] innerCandidates42_7 1678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_8

def innerTail42_6 : List Nat := 1181 :: innerTail42_7
def innerCandidates42_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_6_checked : candidates edge 4 197 1181 innerTail42_7 best.bound = innerCandidates42_6 := by
  unfold candidates
  have row : edge 1181 = PrimePairCache.smallRow1181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_6 : scan edge (search edge 3) 4 [197] 197 innerTail42_6 best = best := by
  rw [innerTail42_6, scan, if_neg (by decide), innerCandidates42_6_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_7
    (if 3 ≤ innerCandidates42_6.length then search edge 3 [197, 1181] innerCandidates42_6 1378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_7

def innerTail42_5 : List Nat := 971 :: innerTail42_6
def innerCandidates42_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates42_5_checked : candidates edge 4 197 971 innerTail42_6 best.bound = innerCandidates42_5 := by
  unfold candidates
  have row : edge 971 = PrimePairCache.smallRow971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_5 : scan edge (search edge 3) 4 [197] 197 innerTail42_5 best = best := by
  rw [innerTail42_5, scan, if_neg (by decide), innerCandidates42_5_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_6
    (if 3 ≤ innerCandidates42_5.length then search edge 3 [197, 971] innerCandidates42_5 1168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_6

def innerTail42_4 : List Nat := 947 :: innerTail42_5
def innerCandidates42_4 : List Nat := [2657]
set_option maxRecDepth 4096 in
theorem innerCandidates42_4_checked : candidates edge 4 197 947 innerTail42_5 best.bound = innerCandidates42_4 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_4 : scan edge (search edge 3) 4 [197] 197 innerTail42_4 best = best := by
  rw [innerTail42_4, scan, if_neg (by decide), innerCandidates42_4_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_5
    (if 3 ≤ innerCandidates42_4.length then search edge 3 [197, 947] innerCandidates42_4 1144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_5

def innerTail42_3 : List Nat := 641 :: innerTail42_4
def innerCandidates42_3 : List Nat := [4937]
set_option maxRecDepth 4096 in
theorem innerCandidates42_3_checked : candidates edge 4 197 641 innerTail42_4 best.bound = innerCandidates42_3 := by
  unfold candidates
  have row : edge 641 = PrimePairCache.smallRow641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_3 : scan edge (search edge 3) 4 [197] 197 innerTail42_3 best = best := by
  rw [innerTail42_3, scan, if_neg (by decide), innerCandidates42_3_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_4
    (if 3 ≤ innerCandidates42_3.length then search edge 3 [197, 641] innerCandidates42_3 838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_4

def innerTail42_2 : List Nat := 569 :: innerTail42_3
def innerCandidates42_2 : List Nat := [2657, 4919]
set_option maxRecDepth 4096 in
theorem innerCandidates42_2_checked : candidates edge 4 197 569 innerTail42_3 best.bound = innerCandidates42_2 := by
  unfold candidates
  have row : edge 569 = PrimePairCache.smallRow569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_2 : scan edge (search edge 3) 4 [197] 197 innerTail42_2 best = best := by
  rw [innerTail42_2, scan, if_neg (by decide), innerCandidates42_2_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_3
    (if 3 ≤ innerCandidates42_2.length then search edge 3 [197, 569] innerCandidates42_2 766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_3

def innerTail42_1 : List Nat := 347 :: innerTail42_2
def innerCandidates42_1 : List Nat := [6761]
set_option maxRecDepth 4096 in
theorem innerCandidates42_1_checked : candidates edge 4 197 347 innerTail42_2 best.bound = innerCandidates42_1 := by
  unfold candidates
  have row : edge 347 = PrimePairCache.smallRow347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_1 : scan edge (search edge 3) 4 [197] 197 innerTail42_1 best = best := by
  rw [innerTail42_1, scan, if_neg (by decide), innerCandidates42_1_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_2
    (if 3 ≤ innerCandidates42_1.length then search edge 3 [197, 347] innerCandidates42_1 544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_2

def innerTail42_0 : List Nat := 311 :: innerTail42_1
def innerCandidates42_0 : List Nat := [4967, 6761]
set_option maxRecDepth 4096 in
theorem innerCandidates42_0_checked : candidates edge 4 197 311 innerTail42_1 best.bound = innerCandidates42_0 := by
  unfold candidates
  have row : edge 311 = PrimePairCache.smallRow311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan42_0 : scan edge (search edge 3) 4 [197] 197 innerTail42_0 best = best := by
  rw [innerTail42_0, scan, if_neg (by decide), innerCandidates42_0_checked]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_1
    (if 3 ≤ innerCandidates42_0.length then search edge 3 [197, 311] innerCandidates42_0 508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan42_1

theorem rootBranch42 : search edge 4 [197] rootCandidates42 197 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [197] 197 innerTail42_0 best = best
  exact innerScan42_0

end ProjectEulerSolutions.PrimePairUpper.Search
