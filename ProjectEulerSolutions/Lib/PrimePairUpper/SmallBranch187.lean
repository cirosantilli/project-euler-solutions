import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates132 : List Nat := [977, 1187, 1613, 1889, 2351, 2531, 2621, 2663, 3209, 3251, 3329, 3539, 3779, 3923, 4463, 4967, 5009, 5261, 5351, 5417, 5879, 6911, 7283, 7529, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates132_checked : candidates edge 5 0 761 tail133 best.bound = rootCandidates132 := by
  unfold candidates
  have row : edge 761 = PrimePairCache.smallRow761.contains := by funext q; rfl
  rw [row]
  decide

def innerTail132_25 : List Nat := []
theorem innerScan132_25 : scan edge (search edge 3) 4 [761] 761 innerTail132_25 best = best := rfl

def innerTail132_24 : List Nat := 8243 :: innerTail132_25
def innerCandidates132_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_24_checked : candidates edge 4 761 8243 innerTail132_25 best.bound = innerCandidates132_24 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_24 : scan edge (search edge 3) 4 [761] 761 innerTail132_24 best = best := by
  rw [innerTail132_24, scan, if_neg (by decide), innerCandidates132_24_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_25
    (if 3 ≤ innerCandidates132_24.length then search edge 3 [761, 8243] innerCandidates132_24 9004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_25

def innerTail132_23 : List Nat := 7529 :: innerTail132_24
def innerCandidates132_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_23_checked : candidates edge 4 761 7529 innerTail132_24 best.bound = innerCandidates132_23 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_23 : scan edge (search edge 3) 4 [761] 761 innerTail132_23 best = best := by
  rw [innerTail132_23, scan, if_neg (by decide), innerCandidates132_23_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_24
    (if 3 ≤ innerCandidates132_23.length then search edge 3 [761, 7529] innerCandidates132_23 8290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_24

def innerTail132_22 : List Nat := 7283 :: innerTail132_23
def innerCandidates132_22 : List Nat := [8243]
set_option maxRecDepth 4096 in
theorem innerCandidates132_22_checked : candidates edge 4 761 7283 innerTail132_23 best.bound = innerCandidates132_22 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_22 : scan edge (search edge 3) 4 [761] 761 innerTail132_22 best = best := by
  rw [innerTail132_22, scan, if_neg (by decide), innerCandidates132_22_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_23
    (if 3 ≤ innerCandidates132_22.length then search edge 3 [761, 7283] innerCandidates132_22 8044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_23

def innerTail132_21 : List Nat := 6911 :: innerTail132_22
def innerCandidates132_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_21_checked : candidates edge 4 761 6911 innerTail132_22 best.bound = innerCandidates132_21 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_21 : scan edge (search edge 3) 4 [761] 761 innerTail132_21 best = best := by
  rw [innerTail132_21, scan, if_neg (by decide), innerCandidates132_21_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_22
    (if 3 ≤ innerCandidates132_21.length then search edge 3 [761, 6911] innerCandidates132_21 7672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_22

def innerTail132_20 : List Nat := 5879 :: innerTail132_21
def innerCandidates132_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_20_checked : candidates edge 4 761 5879 innerTail132_21 best.bound = innerCandidates132_20 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_20 : scan edge (search edge 3) 4 [761] 761 innerTail132_20 best = best := by
  rw [innerTail132_20, scan, if_neg (by decide), innerCandidates132_20_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_21
    (if 3 ≤ innerCandidates132_20.length then search edge 3 [761, 5879] innerCandidates132_20 6640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_21

def innerTail132_19 : List Nat := 5417 :: innerTail132_20
def innerCandidates132_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_19_checked : candidates edge 4 761 5417 innerTail132_20 best.bound = innerCandidates132_19 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_19 : scan edge (search edge 3) 4 [761] 761 innerTail132_19 best = best := by
  rw [innerTail132_19, scan, if_neg (by decide), innerCandidates132_19_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_20
    (if 3 ≤ innerCandidates132_19.length then search edge 3 [761, 5417] innerCandidates132_19 6178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_20

def innerTail132_18 : List Nat := 5351 :: innerTail132_19
def innerCandidates132_18 : List Nat := [5879]
set_option maxRecDepth 4096 in
theorem innerCandidates132_18_checked : candidates edge 4 761 5351 innerTail132_19 best.bound = innerCandidates132_18 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_18 : scan edge (search edge 3) 4 [761] 761 innerTail132_18 best = best := by
  rw [innerTail132_18, scan, if_neg (by decide), innerCandidates132_18_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_19
    (if 3 ≤ innerCandidates132_18.length then search edge 3 [761, 5351] innerCandidates132_18 6112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_19

def innerTail132_17 : List Nat := 5261 :: innerTail132_18
def innerCandidates132_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_17_checked : candidates edge 4 761 5261 innerTail132_18 best.bound = innerCandidates132_17 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_17 : scan edge (search edge 3) 4 [761] 761 innerTail132_17 best = best := by
  rw [innerTail132_17, scan, if_neg (by decide), innerCandidates132_17_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_18
    (if 3 ≤ innerCandidates132_17.length then search edge 3 [761, 5261] innerCandidates132_17 6022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_18

def innerTail132_16 : List Nat := 5009 :: innerTail132_17
def innerCandidates132_16 : List Nat := [5351]
set_option maxRecDepth 4096 in
theorem innerCandidates132_16_checked : candidates edge 4 761 5009 innerTail132_17 best.bound = innerCandidates132_16 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_16 : scan edge (search edge 3) 4 [761] 761 innerTail132_16 best = best := by
  rw [innerTail132_16, scan, if_neg (by decide), innerCandidates132_16_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_17
    (if 3 ≤ innerCandidates132_16.length then search edge 3 [761, 5009] innerCandidates132_16 5770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_17

def innerTail132_15 : List Nat := 4967 :: innerTail132_16
def innerCandidates132_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_15_checked : candidates edge 4 761 4967 innerTail132_16 best.bound = innerCandidates132_15 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_15 : scan edge (search edge 3) 4 [761] 761 innerTail132_15 best = best := by
  rw [innerTail132_15, scan, if_neg (by decide), innerCandidates132_15_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_16
    (if 3 ≤ innerCandidates132_15.length then search edge 3 [761, 4967] innerCandidates132_15 5728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_16

def innerTail132_14 : List Nat := 4463 :: innerTail132_15
def innerCandidates132_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_14_checked : candidates edge 4 761 4463 innerTail132_15 best.bound = innerCandidates132_14 := by
  unfold candidates
  have row : edge 4463 = SearchData.row4463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_14 : scan edge (search edge 3) 4 [761] 761 innerTail132_14 best = best := by
  rw [innerTail132_14, scan, if_neg (by decide), innerCandidates132_14_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_15
    (if 3 ≤ innerCandidates132_14.length then search edge 3 [761, 4463] innerCandidates132_14 5224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_15

def innerTail132_13 : List Nat := 3923 :: innerTail132_14
def innerCandidates132_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_13_checked : candidates edge 4 761 3923 innerTail132_14 best.bound = innerCandidates132_13 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_13 : scan edge (search edge 3) 4 [761] 761 innerTail132_13 best = best := by
  rw [innerTail132_13, scan, if_neg (by decide), innerCandidates132_13_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_14
    (if 3 ≤ innerCandidates132_13.length then search edge 3 [761, 3923] innerCandidates132_13 4684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_14

def innerTail132_12 : List Nat := 3779 :: innerTail132_13
def innerCandidates132_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_12_checked : candidates edge 4 761 3779 innerTail132_13 best.bound = innerCandidates132_12 := by
  unfold candidates
  have row : edge 3779 = SearchData.row3779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_12 : scan edge (search edge 3) 4 [761] 761 innerTail132_12 best = best := by
  rw [innerTail132_12, scan, if_neg (by decide), innerCandidates132_12_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_13
    (if 3 ≤ innerCandidates132_12.length then search edge 3 [761, 3779] innerCandidates132_12 4540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_13

def innerTail132_11 : List Nat := 3539 :: innerTail132_12
def innerCandidates132_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_11_checked : candidates edge 4 761 3539 innerTail132_12 best.bound = innerCandidates132_11 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_11 : scan edge (search edge 3) 4 [761] 761 innerTail132_11 best = best := by
  rw [innerTail132_11, scan, if_neg (by decide), innerCandidates132_11_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_12
    (if 3 ≤ innerCandidates132_11.length then search edge 3 [761, 3539] innerCandidates132_11 4300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_12

def innerTail132_10 : List Nat := 3329 :: innerTail132_11
def innerCandidates132_10 : List Nat := [4967]
set_option maxRecDepth 4096 in
theorem innerCandidates132_10_checked : candidates edge 4 761 3329 innerTail132_11 best.bound = innerCandidates132_10 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_10 : scan edge (search edge 3) 4 [761] 761 innerTail132_10 best = best := by
  rw [innerTail132_10, scan, if_neg (by decide), innerCandidates132_10_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_11
    (if 3 ≤ innerCandidates132_10.length then search edge 3 [761, 3329] innerCandidates132_10 4090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_11

def innerTail132_9 : List Nat := 3251 :: innerTail132_10
def innerCandidates132_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_9_checked : candidates edge 4 761 3251 innerTail132_10 best.bound = innerCandidates132_9 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_9 : scan edge (search edge 3) 4 [761] 761 innerTail132_9 best = best := by
  rw [innerTail132_9, scan, if_neg (by decide), innerCandidates132_9_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_10
    (if 3 ≤ innerCandidates132_9.length then search edge 3 [761, 3251] innerCandidates132_9 4012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_10

def innerTail132_8 : List Nat := 3209 :: innerTail132_9
def innerCandidates132_8 : List Nat := [5417]
set_option maxRecDepth 4096 in
theorem innerCandidates132_8_checked : candidates edge 4 761 3209 innerTail132_9 best.bound = innerCandidates132_8 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_8 : scan edge (search edge 3) 4 [761] 761 innerTail132_8 best = best := by
  rw [innerTail132_8, scan, if_neg (by decide), innerCandidates132_8_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_9
    (if 3 ≤ innerCandidates132_8.length then search edge 3 [761, 3209] innerCandidates132_8 3970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_9

def innerTail132_7 : List Nat := 2663 :: innerTail132_8
def innerCandidates132_7 : List Nat := [3251, 3779]
set_option maxRecDepth 4096 in
theorem innerCandidates132_7_checked : candidates edge 4 761 2663 innerTail132_8 best.bound = innerCandidates132_7 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_7 : scan edge (search edge 3) 4 [761] 761 innerTail132_7 best = best := by
  rw [innerTail132_7, scan, if_neg (by decide), innerCandidates132_7_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_8
    (if 3 ≤ innerCandidates132_7.length then search edge 3 [761, 2663] innerCandidates132_7 3424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_8

def innerTail132_6 : List Nat := 2621 :: innerTail132_7
def innerCandidates132_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_6_checked : candidates edge 4 761 2621 innerTail132_7 best.bound = innerCandidates132_6 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_6 : scan edge (search edge 3) 4 [761] 761 innerTail132_6 best = best := by
  rw [innerTail132_6, scan, if_neg (by decide), innerCandidates132_6_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_7
    (if 3 ≤ innerCandidates132_6.length then search edge 3 [761, 2621] innerCandidates132_6 3382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_7

def innerTail132_5 : List Nat := 2531 :: innerTail132_6
def innerCandidates132_5 : List Nat := [3539, 4967]
set_option maxRecDepth 4096 in
theorem innerCandidates132_5_checked : candidates edge 4 761 2531 innerTail132_6 best.bound = innerCandidates132_5 := by
  unfold candidates
  have row : edge 2531 = PrimePairCache.smallRow2531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_5 : scan edge (search edge 3) 4 [761] 761 innerTail132_5 best = best := by
  rw [innerTail132_5, scan, if_neg (by decide), innerCandidates132_5_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_6
    (if 3 ≤ innerCandidates132_5.length then search edge 3 [761, 2531] innerCandidates132_5 3292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_6

def innerTail132_4 : List Nat := 2351 :: innerTail132_5
def innerCandidates132_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_4_checked : candidates edge 4 761 2351 innerTail132_5 best.bound = innerCandidates132_4 := by
  unfold candidates
  have row : edge 2351 = PrimePairCache.smallRow2351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_4 : scan edge (search edge 3) 4 [761] 761 innerTail132_4 best = best := by
  rw [innerTail132_4, scan, if_neg (by decide), innerCandidates132_4_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_5
    (if 3 ≤ innerCandidates132_4.length then search edge 3 [761, 2351] innerCandidates132_4 3112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_5

def innerTail132_3 : List Nat := 1889 :: innerTail132_4
def innerCandidates132_3 : List Nat := [5417]
set_option maxRecDepth 4096 in
theorem innerCandidates132_3_checked : candidates edge 4 761 1889 innerTail132_4 best.bound = innerCandidates132_3 := by
  unfold candidates
  have row : edge 1889 = PrimePairCache.smallRow1889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_3 : scan edge (search edge 3) 4 [761] 761 innerTail132_3 best = best := by
  rw [innerTail132_3, scan, if_neg (by decide), innerCandidates132_3_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_4
    (if 3 ≤ innerCandidates132_3.length then search edge 3 [761, 1889] innerCandidates132_3 2650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_4

def innerTail132_2 : List Nat := 1613 :: innerTail132_3
def innerCandidates132_2 : List Nat := [3779, 4967]
set_option maxRecDepth 4096 in
theorem innerCandidates132_2_checked : candidates edge 4 761 1613 innerTail132_3 best.bound = innerCandidates132_2 := by
  unfold candidates
  have row : edge 1613 = PrimePairCache.smallRow1613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_2 : scan edge (search edge 3) 4 [761] 761 innerTail132_2 best = best := by
  rw [innerTail132_2, scan, if_neg (by decide), innerCandidates132_2_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_3
    (if 3 ≤ innerCandidates132_2.length then search edge 3 [761, 1613] innerCandidates132_2 2374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_3

def innerTail132_1 : List Nat := 1187 :: innerTail132_2
def innerCandidates132_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_1_checked : candidates edge 4 761 1187 innerTail132_2 best.bound = innerCandidates132_1 := by
  unfold candidates
  have row : edge 1187 = PrimePairCache.smallRow1187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_1 : scan edge (search edge 3) 4 [761] 761 innerTail132_1 best = best := by
  rw [innerTail132_1, scan, if_neg (by decide), innerCandidates132_1_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_2
    (if 3 ≤ innerCandidates132_1.length then search edge 3 [761, 1187] innerCandidates132_1 1948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_2

def innerTail132_0 : List Nat := 977 :: innerTail132_1
def innerCandidates132_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates132_0_checked : candidates edge 4 761 977 innerTail132_1 best.bound = innerCandidates132_0 := by
  unfold candidates
  have row : edge 977 = PrimePairCache.smallRow977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan132_0 : scan edge (search edge 3) 4 [761] 761 innerTail132_0 best = best := by
  rw [innerTail132_0, scan, if_neg (by decide), innerCandidates132_0_checked]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_1
    (if 3 ≤ innerCandidates132_0.length then search edge 3 [761, 977] innerCandidates132_0 1738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan132_1

theorem rootBranch132 : search edge 4 [761] rootCandidates132 761 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [761] 761 innerTail132_0 best = best
  exact innerScan132_0

def rootCandidates131 : List Nat := [811, 829, 1231, 1423, 3037, 3217, 3733, 3931, 4549, 4993, 5077, 5569, 5653, 5791, 6043, 6211, 6619, 6661, 7321, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates131_checked : candidates edge 5 0 757 tail132 best.bound = rootCandidates131 := by
  unfold candidates
  have row : edge 757 = PrimePairCache.smallRow757.contains := by funext q; rfl
  rw [row]
  decide

def innerTail131_20 : List Nat := []
theorem innerScan131_20 : scan edge (search edge 3) 4 [757] 757 innerTail131_20 best = best := rfl

def innerTail131_19 : List Nat := 8269 :: innerTail131_20
def innerCandidates131_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates131_19_checked : candidates edge 4 757 8269 innerTail131_20 best.bound = innerCandidates131_19 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_19 : scan edge (search edge 3) 4 [757] 757 innerTail131_19 best = best := by
  rw [innerTail131_19, scan, if_neg (by decide), innerCandidates131_19_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_20
    (if 3 ≤ innerCandidates131_19.length then search edge 3 [757, 8269] innerCandidates131_19 9026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_20

def innerTail131_18 : List Nat := 7321 :: innerTail131_19
def innerCandidates131_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates131_18_checked : candidates edge 4 757 7321 innerTail131_19 best.bound = innerCandidates131_18 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_18 : scan edge (search edge 3) 4 [757] 757 innerTail131_18 best = best := by
  rw [innerTail131_18, scan, if_neg (by decide), innerCandidates131_18_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_19
    (if 3 ≤ innerCandidates131_18.length then search edge 3 [757, 7321] innerCandidates131_18 8078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_19

def innerTail131_17 : List Nat := 6661 :: innerTail131_18
def innerCandidates131_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates131_17_checked : candidates edge 4 757 6661 innerTail131_18 best.bound = innerCandidates131_17 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_17 : scan edge (search edge 3) 4 [757] 757 innerTail131_17 best = best := by
  rw [innerTail131_17, scan, if_neg (by decide), innerCandidates131_17_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_18
    (if 3 ≤ innerCandidates131_17.length then search edge 3 [757, 6661] innerCandidates131_17 7418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_18

def innerTail131_16 : List Nat := 6619 :: innerTail131_17
def innerCandidates131_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates131_16_checked : candidates edge 4 757 6619 innerTail131_17 best.bound = innerCandidates131_16 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_16 : scan edge (search edge 3) 4 [757] 757 innerTail131_16 best = best := by
  rw [innerTail131_16, scan, if_neg (by decide), innerCandidates131_16_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_17
    (if 3 ≤ innerCandidates131_16.length then search edge 3 [757, 6619] innerCandidates131_16 7376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_17

def innerTail131_15 : List Nat := 6211 :: innerTail131_16
def innerCandidates131_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates131_15_checked : candidates edge 4 757 6211 innerTail131_16 best.bound = innerCandidates131_15 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_15 : scan edge (search edge 3) 4 [757] 757 innerTail131_15 best = best := by
  rw [innerTail131_15, scan, if_neg (by decide), innerCandidates131_15_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_16
    (if 3 ≤ innerCandidates131_15.length then search edge 3 [757, 6211] innerCandidates131_15 6968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_16

def innerTail131_14 : List Nat := 6043 :: innerTail131_15
def innerCandidates131_14 : List Nat := [6619]
set_option maxRecDepth 4096 in
theorem innerCandidates131_14_checked : candidates edge 4 757 6043 innerTail131_15 best.bound = innerCandidates131_14 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_14 : scan edge (search edge 3) 4 [757] 757 innerTail131_14 best = best := by
  rw [innerTail131_14, scan, if_neg (by decide), innerCandidates131_14_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_15
    (if 3 ≤ innerCandidates131_14.length then search edge 3 [757, 6043] innerCandidates131_14 6800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_15

def innerTail131_13 : List Nat := 5791 :: innerTail131_14
def innerCandidates131_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates131_13_checked : candidates edge 4 757 5791 innerTail131_14 best.bound = innerCandidates131_13 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_13 : scan edge (search edge 3) 4 [757] 757 innerTail131_13 best = best := by
  rw [innerTail131_13, scan, if_neg (by decide), innerCandidates131_13_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_14
    (if 3 ≤ innerCandidates131_13.length then search edge 3 [757, 5791] innerCandidates131_13 6548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_14

def innerTail131_12 : List Nat := 5653 :: innerTail131_13
def innerCandidates131_12 : List Nat := [6211, 6619, 8269]
set_option maxRecDepth 4096 in
theorem innerCandidates131_12_checked : candidates edge 4 757 5653 innerTail131_13 best.bound = innerCandidates131_12 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild131_12 : search edge 3 [757, 5653] innerCandidates131_12 6410 best = best := by decide

theorem innerScan131_12 : scan edge (search edge 3) 4 [757] 757 innerTail131_12 best = best := by
  rw [innerTail131_12, scan, if_neg (by decide), innerCandidates131_12_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_13
    (if 3 ≤ innerCandidates131_12.length then search edge 3 [757, 5653] innerCandidates131_12 6410 best else best) = best
  rw [if_pos (by decide), innerChild131_12]
  exact innerScan131_13

def innerTail131_11 : List Nat := 5569 :: innerTail131_12
def innerCandidates131_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates131_11_checked : candidates edge 4 757 5569 innerTail131_12 best.bound = innerCandidates131_11 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_11 : scan edge (search edge 3) 4 [757] 757 innerTail131_11 best = best := by
  rw [innerTail131_11, scan, if_neg (by decide), innerCandidates131_11_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_12
    (if 3 ≤ innerCandidates131_11.length then search edge 3 [757, 5569] innerCandidates131_11 6326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_12

def innerTail131_10 : List Nat := 5077 :: innerTail131_11
def innerCandidates131_10 : List Nat := [6043]
set_option maxRecDepth 4096 in
theorem innerCandidates131_10_checked : candidates edge 4 757 5077 innerTail131_11 best.bound = innerCandidates131_10 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_10 : scan edge (search edge 3) 4 [757] 757 innerTail131_10 best = best := by
  rw [innerTail131_10, scan, if_neg (by decide), innerCandidates131_10_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_11
    (if 3 ≤ innerCandidates131_10.length then search edge 3 [757, 5077] innerCandidates131_10 5834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_11

def innerTail131_9 : List Nat := 4993 :: innerTail131_10
def innerCandidates131_9 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates131_9_checked : candidates edge 4 757 4993 innerTail131_10 best.bound = innerCandidates131_9 := by
  unfold candidates
  have row : edge 4993 = SearchData.row4993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_9 : scan edge (search edge 3) 4 [757] 757 innerTail131_9 best = best := by
  rw [innerTail131_9, scan, if_neg (by decide), innerCandidates131_9_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_10
    (if 3 ≤ innerCandidates131_9.length then search edge 3 [757, 4993] innerCandidates131_9 5750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_10

def innerTail131_8 : List Nat := 4549 :: innerTail131_9
def innerCandidates131_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates131_8_checked : candidates edge 4 757 4549 innerTail131_9 best.bound = innerCandidates131_8 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_8 : scan edge (search edge 3) 4 [757] 757 innerTail131_8 best = best := by
  rw [innerTail131_8, scan, if_neg (by decide), innerCandidates131_8_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_9
    (if 3 ≤ innerCandidates131_8.length then search edge 3 [757, 4549] innerCandidates131_8 5306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_9

def innerTail131_7 : List Nat := 3931 :: innerTail131_8
def innerCandidates131_7 : List Nat := [6661]
set_option maxRecDepth 4096 in
theorem innerCandidates131_7_checked : candidates edge 4 757 3931 innerTail131_8 best.bound = innerCandidates131_7 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_7 : scan edge (search edge 3) 4 [757] 757 innerTail131_7 best = best := by
  rw [innerTail131_7, scan, if_neg (by decide), innerCandidates131_7_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_8
    (if 3 ≤ innerCandidates131_7.length then search edge 3 [757, 3931] innerCandidates131_7 4688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_8

def innerTail131_6 : List Nat := 3733 :: innerTail131_7
def innerCandidates131_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates131_6_checked : candidates edge 4 757 3733 innerTail131_7 best.bound = innerCandidates131_6 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_6 : scan edge (search edge 3) 4 [757] 757 innerTail131_6 best = best := by
  rw [innerTail131_6, scan, if_neg (by decide), innerCandidates131_6_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_7
    (if 3 ≤ innerCandidates131_6.length then search edge 3 [757, 3733] innerCandidates131_6 4490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_7

def innerTail131_5 : List Nat := 3217 :: innerTail131_6
def innerCandidates131_5 : List Nat := [3931, 6211, 6619]
set_option maxRecDepth 4096 in
theorem innerCandidates131_5_checked : candidates edge 4 757 3217 innerTail131_6 best.bound = innerCandidates131_5 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild131_5 : search edge 3 [757, 3217] innerCandidates131_5 3974 best = best := by decide

theorem innerScan131_5 : scan edge (search edge 3) 4 [757] 757 innerTail131_5 best = best := by
  rw [innerTail131_5, scan, if_neg (by decide), innerCandidates131_5_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_6
    (if 3 ≤ innerCandidates131_5.length then search edge 3 [757, 3217] innerCandidates131_5 3974 best else best) = best
  rw [if_pos (by decide), innerChild131_5]
  exact innerScan131_6

def innerTail131_4 : List Nat := 3037 :: innerTail131_5
def innerCandidates131_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates131_4_checked : candidates edge 4 757 3037 innerTail131_5 best.bound = innerCandidates131_4 := by
  unfold candidates
  have row : edge 3037 = SearchData.row3037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_4 : scan edge (search edge 3) 4 [757] 757 innerTail131_4 best = best := by
  rw [innerTail131_4, scan, if_neg (by decide), innerCandidates131_4_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_5
    (if 3 ≤ innerCandidates131_4.length then search edge 3 [757, 3037] innerCandidates131_4 3794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_5

def innerTail131_3 : List Nat := 1423 :: innerTail131_4
def innerCandidates131_3 : List Nat := [5791]
set_option maxRecDepth 4096 in
theorem innerCandidates131_3_checked : candidates edge 4 757 1423 innerTail131_4 best.bound = innerCandidates131_3 := by
  unfold candidates
  have row : edge 1423 = PrimePairCache.smallRow1423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_3 : scan edge (search edge 3) 4 [757] 757 innerTail131_3 best = best := by
  rw [innerTail131_3, scan, if_neg (by decide), innerCandidates131_3_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_4
    (if 3 ≤ innerCandidates131_3.length then search edge 3 [757, 1423] innerCandidates131_3 2180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_4

def innerTail131_2 : List Nat := 1231 :: innerTail131_3
def innerCandidates131_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates131_2_checked : candidates edge 4 757 1231 innerTail131_3 best.bound = innerCandidates131_2 := by
  unfold candidates
  have row : edge 1231 = PrimePairCache.smallRow1231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_2 : scan edge (search edge 3) 4 [757] 757 innerTail131_2 best = best := by
  rw [innerTail131_2, scan, if_neg (by decide), innerCandidates131_2_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_3
    (if 3 ≤ innerCandidates131_2.length then search edge 3 [757, 1231] innerCandidates131_2 1988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_3

def innerTail131_1 : List Nat := 829 :: innerTail131_2
def innerCandidates131_1 : List Nat := [7321]
set_option maxRecDepth 4096 in
theorem innerCandidates131_1_checked : candidates edge 4 757 829 innerTail131_2 best.bound = innerCandidates131_1 := by
  unfold candidates
  have row : edge 829 = PrimePairCache.smallRow829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_1 : scan edge (search edge 3) 4 [757] 757 innerTail131_1 best = best := by
  rw [innerTail131_1, scan, if_neg (by decide), innerCandidates131_1_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_2
    (if 3 ≤ innerCandidates131_1.length then search edge 3 [757, 829] innerCandidates131_1 1586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_2

def innerTail131_0 : List Nat := 811 :: innerTail131_1
def innerCandidates131_0 : List Nat := [5791]
set_option maxRecDepth 4096 in
theorem innerCandidates131_0_checked : candidates edge 4 757 811 innerTail131_1 best.bound = innerCandidates131_0 := by
  unfold candidates
  have row : edge 811 = PrimePairCache.smallRow811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan131_0 : scan edge (search edge 3) 4 [757] 757 innerTail131_0 best = best := by
  rw [innerTail131_0, scan, if_neg (by decide), innerCandidates131_0_checked]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_1
    (if 3 ≤ innerCandidates131_0.length then search edge 3 [757, 811] innerCandidates131_0 1568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan131_1

theorem rootBranch131 : search edge 4 [757] rootCandidates131 757 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [757] 757 innerTail131_0 best = best
  exact innerScan131_0

def rootCandidates130 : List Nat := [787, 997, 1033, 1789, 2377, 2383, 2437, 2671, 3343, 3361, 3547, 4231, 4597, 5281, 7591, 7669]
set_option maxRecDepth 4096 in
theorem rootCandidates130_checked : candidates edge 5 0 751 tail131 best.bound = rootCandidates130 := by
  unfold candidates
  have row : edge 751 = PrimePairCache.smallRow751.contains := by funext q; rfl
  rw [row]
  decide

def innerTail130_16 : List Nat := []
theorem innerScan130_16 : scan edge (search edge 3) 4 [751] 751 innerTail130_16 best = best := rfl

def innerTail130_15 : List Nat := 7669 :: innerTail130_16
def innerCandidates130_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates130_15_checked : candidates edge 4 751 7669 innerTail130_16 best.bound = innerCandidates130_15 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_15 : scan edge (search edge 3) 4 [751] 751 innerTail130_15 best = best := by
  rw [innerTail130_15, scan, if_neg (by decide), innerCandidates130_15_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_16
    (if 3 ≤ innerCandidates130_15.length then search edge 3 [751, 7669] innerCandidates130_15 8420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_16

def innerTail130_14 : List Nat := 7591 :: innerTail130_15
def innerCandidates130_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates130_14_checked : candidates edge 4 751 7591 innerTail130_15 best.bound = innerCandidates130_14 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_14 : scan edge (search edge 3) 4 [751] 751 innerTail130_14 best = best := by
  rw [innerTail130_14, scan, if_neg (by decide), innerCandidates130_14_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_15
    (if 3 ≤ innerCandidates130_14.length then search edge 3 [751, 7591] innerCandidates130_14 8342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_15

def innerTail130_13 : List Nat := 5281 :: innerTail130_14
def innerCandidates130_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates130_13_checked : candidates edge 4 751 5281 innerTail130_14 best.bound = innerCandidates130_13 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_13 : scan edge (search edge 3) 4 [751] 751 innerTail130_13 best = best := by
  rw [innerTail130_13, scan, if_neg (by decide), innerCandidates130_13_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_14
    (if 3 ≤ innerCandidates130_13.length then search edge 3 [751, 5281] innerCandidates130_13 6032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_14

def innerTail130_12 : List Nat := 4597 :: innerTail130_13
def innerCandidates130_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates130_12_checked : candidates edge 4 751 4597 innerTail130_13 best.bound = innerCandidates130_12 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_12 : scan edge (search edge 3) 4 [751] 751 innerTail130_12 best = best := by
  rw [innerTail130_12, scan, if_neg (by decide), innerCandidates130_12_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_13
    (if 3 ≤ innerCandidates130_12.length then search edge 3 [751, 4597] innerCandidates130_12 5348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_13

def innerTail130_11 : List Nat := 4231 :: innerTail130_12
def innerCandidates130_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates130_11_checked : candidates edge 4 751 4231 innerTail130_12 best.bound = innerCandidates130_11 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_11 : scan edge (search edge 3) 4 [751] 751 innerTail130_11 best = best := by
  rw [innerTail130_11, scan, if_neg (by decide), innerCandidates130_11_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_12
    (if 3 ≤ innerCandidates130_11.length then search edge 3 [751, 4231] innerCandidates130_11 4982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_12

def innerTail130_10 : List Nat := 3547 :: innerTail130_11
def innerCandidates130_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates130_10_checked : candidates edge 4 751 3547 innerTail130_11 best.bound = innerCandidates130_10 := by
  unfold candidates
  have row : edge 3547 = SearchData.row3547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_10 : scan edge (search edge 3) 4 [751] 751 innerTail130_10 best = best := by
  rw [innerTail130_10, scan, if_neg (by decide), innerCandidates130_10_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_11
    (if 3 ≤ innerCandidates130_10.length then search edge 3 [751, 3547] innerCandidates130_10 4298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_11

def innerTail130_9 : List Nat := 3361 :: innerTail130_10
def innerCandidates130_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates130_9_checked : candidates edge 4 751 3361 innerTail130_10 best.bound = innerCandidates130_9 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_9 : scan edge (search edge 3) 4 [751] 751 innerTail130_9 best = best := by
  rw [innerTail130_9, scan, if_neg (by decide), innerCandidates130_9_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_10
    (if 3 ≤ innerCandidates130_9.length then search edge 3 [751, 3361] innerCandidates130_9 4112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_10

def innerTail130_8 : List Nat := 3343 :: innerTail130_9
def innerCandidates130_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates130_8_checked : candidates edge 4 751 3343 innerTail130_9 best.bound = innerCandidates130_8 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_8 : scan edge (search edge 3) 4 [751] 751 innerTail130_8 best = best := by
  rw [innerTail130_8, scan, if_neg (by decide), innerCandidates130_8_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_9
    (if 3 ≤ innerCandidates130_8.length then search edge 3 [751, 3343] innerCandidates130_8 4094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_9

def innerTail130_7 : List Nat := 2671 :: innerTail130_8
def innerCandidates130_7 : List Nat := [3361, 4231]
set_option maxRecDepth 4096 in
theorem innerCandidates130_7_checked : candidates edge 4 751 2671 innerTail130_8 best.bound = innerCandidates130_7 := by
  unfold candidates
  have row : edge 2671 = SearchData.row2671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_7 : scan edge (search edge 3) 4 [751] 751 innerTail130_7 best = best := by
  rw [innerTail130_7, scan, if_neg (by decide), innerCandidates130_7_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_8
    (if 3 ≤ innerCandidates130_7.length then search edge 3 [751, 2671] innerCandidates130_7 3422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_8

def innerTail130_6 : List Nat := 2437 :: innerTail130_7
def innerCandidates130_6 : List Nat := [3361]
set_option maxRecDepth 4096 in
theorem innerCandidates130_6_checked : candidates edge 4 751 2437 innerTail130_7 best.bound = innerCandidates130_6 := by
  unfold candidates
  have row : edge 2437 = PrimePairCache.smallRow2437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_6 : scan edge (search edge 3) 4 [751] 751 innerTail130_6 best = best := by
  rw [innerTail130_6, scan, if_neg (by decide), innerCandidates130_6_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_7
    (if 3 ≤ innerCandidates130_6.length then search edge 3 [751, 2437] innerCandidates130_6 3188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_7

def innerTail130_5 : List Nat := 2383 :: innerTail130_6
def innerCandidates130_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates130_5_checked : candidates edge 4 751 2383 innerTail130_6 best.bound = innerCandidates130_5 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_5 : scan edge (search edge 3) 4 [751] 751 innerTail130_5 best = best := by
  rw [innerTail130_5, scan, if_neg (by decide), innerCandidates130_5_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_6
    (if 3 ≤ innerCandidates130_5.length then search edge 3 [751, 2383] innerCandidates130_5 3134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_6

def innerTail130_4 : List Nat := 2377 :: innerTail130_5
def innerCandidates130_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates130_4_checked : candidates edge 4 751 2377 innerTail130_5 best.bound = innerCandidates130_4 := by
  unfold candidates
  have row : edge 2377 = PrimePairCache.smallRow2377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_4 : scan edge (search edge 3) 4 [751] 751 innerTail130_4 best = best := by
  rw [innerTail130_4, scan, if_neg (by decide), innerCandidates130_4_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_5
    (if 3 ≤ innerCandidates130_4.length then search edge 3 [751, 2377] innerCandidates130_4 3128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_5

def innerTail130_3 : List Nat := 1789 :: innerTail130_4
def innerCandidates130_3 : List Nat := [2377]
set_option maxRecDepth 4096 in
theorem innerCandidates130_3_checked : candidates edge 4 751 1789 innerTail130_4 best.bound = innerCandidates130_3 := by
  unfold candidates
  have row : edge 1789 = PrimePairCache.smallRow1789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_3 : scan edge (search edge 3) 4 [751] 751 innerTail130_3 best = best := by
  rw [innerTail130_3, scan, if_neg (by decide), innerCandidates130_3_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_4
    (if 3 ≤ innerCandidates130_3.length then search edge 3 [751, 1789] innerCandidates130_3 2540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_4

def innerTail130_2 : List Nat := 1033 :: innerTail130_3
def innerCandidates130_2 : List Nat := [4231]
set_option maxRecDepth 4096 in
theorem innerCandidates130_2_checked : candidates edge 4 751 1033 innerTail130_3 best.bound = innerCandidates130_2 := by
  unfold candidates
  have row : edge 1033 = PrimePairCache.smallRow1033.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_2 : scan edge (search edge 3) 4 [751] 751 innerTail130_2 best = best := by
  rw [innerTail130_2, scan, if_neg (by decide), innerCandidates130_2_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_3
    (if 3 ≤ innerCandidates130_2.length then search edge 3 [751, 1033] innerCandidates130_2 1784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_3

def innerTail130_1 : List Nat := 997 :: innerTail130_2
def innerCandidates130_1 : List Nat := [2377]
set_option maxRecDepth 4096 in
theorem innerCandidates130_1_checked : candidates edge 4 751 997 innerTail130_2 best.bound = innerCandidates130_1 := by
  unfold candidates
  have row : edge 997 = PrimePairCache.smallRow997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_1 : scan edge (search edge 3) 4 [751] 751 innerTail130_1 best = best := by
  rw [innerTail130_1, scan, if_neg (by decide), innerCandidates130_1_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_2
    (if 3 ≤ innerCandidates130_1.length then search edge 3 [751, 997] innerCandidates130_1 1748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_2

def innerTail130_0 : List Nat := 787 :: innerTail130_1
def innerCandidates130_0 : List Nat := [2671]
set_option maxRecDepth 4096 in
theorem innerCandidates130_0_checked : candidates edge 4 751 787 innerTail130_1 best.bound = innerCandidates130_0 := by
  unfold candidates
  have row : edge 787 = PrimePairCache.smallRow787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan130_0 : scan edge (search edge 3) 4 [751] 751 innerTail130_0 best = best := by
  rw [innerTail130_0, scan, if_neg (by decide), innerCandidates130_0_checked]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_1
    (if 3 ≤ innerCandidates130_0.length then search edge 3 [751, 787] innerCandidates130_0 1538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan130_1

theorem rootBranch130 : search edge 4 [751] rootCandidates130 751 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [751] 751 innerTail130_0 best = best
  exact innerScan130_0

end ProjectEulerSolutions.PrimePairUpper.Search
