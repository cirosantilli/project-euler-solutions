import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates148 : List Nat := [883, 937, 997, 1327, 1621, 2251, 2371, 2551, 2917, 3469, 3727, 3823, 4483, 4723, 5419, 5527, 5659, 5743, 6121, 6373, 6529, 6637, 6991, 7243, 7669, 7759, 8191, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates148_checked : candidates edge 5 0 877 tail149 best.bound = rootCandidates148 := by
  unfold candidates
  have row : edge 877 = PrimePairCache.smallRow877.contains := by funext q; rfl
  rw [row]
  decide

def innerTail148_28 : List Nat := []
theorem innerScan148_28 : scan edge (search edge 3) 4 [877] 877 innerTail148_28 best = best := rfl

def innerTail148_27 : List Nat := 8293 :: innerTail148_28
def innerCandidates148_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_27_checked : candidates edge 4 877 8293 innerTail148_28 best.bound = innerCandidates148_27 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_27 : scan edge (search edge 3) 4 [877] 877 innerTail148_27 best = best := by
  rw [innerTail148_27, scan, if_neg (by decide), innerCandidates148_27_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_28
    (if 3 ≤ innerCandidates148_27.length then search edge 3 [877, 8293] innerCandidates148_27 9170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_28

def innerTail148_26 : List Nat := 8191 :: innerTail148_27
def innerCandidates148_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_26_checked : candidates edge 4 877 8191 innerTail148_27 best.bound = innerCandidates148_26 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_26 : scan edge (search edge 3) 4 [877] 877 innerTail148_26 best = best := by
  rw [innerTail148_26, scan, if_neg (by decide), innerCandidates148_26_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_27
    (if 3 ≤ innerCandidates148_26.length then search edge 3 [877, 8191] innerCandidates148_26 9068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_27

def innerTail148_25 : List Nat := 7759 :: innerTail148_26
def innerCandidates148_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_25_checked : candidates edge 4 877 7759 innerTail148_26 best.bound = innerCandidates148_25 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_25 : scan edge (search edge 3) 4 [877] 877 innerTail148_25 best = best := by
  rw [innerTail148_25, scan, if_neg (by decide), innerCandidates148_25_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_26
    (if 3 ≤ innerCandidates148_25.length then search edge 3 [877, 7759] innerCandidates148_25 8636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_26

def innerTail148_24 : List Nat := 7669 :: innerTail148_25
def innerCandidates148_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_24_checked : candidates edge 4 877 7669 innerTail148_25 best.bound = innerCandidates148_24 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_24 : scan edge (search edge 3) 4 [877] 877 innerTail148_24 best = best := by
  rw [innerTail148_24, scan, if_neg (by decide), innerCandidates148_24_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_25
    (if 3 ≤ innerCandidates148_24.length then search edge 3 [877, 7669] innerCandidates148_24 8546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_25

def innerTail148_23 : List Nat := 7243 :: innerTail148_24
def innerCandidates148_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_23_checked : candidates edge 4 877 7243 innerTail148_24 best.bound = innerCandidates148_23 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_23 : scan edge (search edge 3) 4 [877] 877 innerTail148_23 best = best := by
  rw [innerTail148_23, scan, if_neg (by decide), innerCandidates148_23_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_24
    (if 3 ≤ innerCandidates148_23.length then search edge 3 [877, 7243] innerCandidates148_23 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_24

def innerTail148_22 : List Nat := 6991 :: innerTail148_23
def innerCandidates148_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_22_checked : candidates edge 4 877 6991 innerTail148_23 best.bound = innerCandidates148_22 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_22 : scan edge (search edge 3) 4 [877] 877 innerTail148_22 best = best := by
  rw [innerTail148_22, scan, if_neg (by decide), innerCandidates148_22_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_23
    (if 3 ≤ innerCandidates148_22.length then search edge 3 [877, 6991] innerCandidates148_22 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_23

def innerTail148_21 : List Nat := 6637 :: innerTail148_22
def innerCandidates148_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_21_checked : candidates edge 4 877 6637 innerTail148_22 best.bound = innerCandidates148_21 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_21 : scan edge (search edge 3) 4 [877] 877 innerTail148_21 best = best := by
  rw [innerTail148_21, scan, if_neg (by decide), innerCandidates148_21_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_22
    (if 3 ≤ innerCandidates148_21.length then search edge 3 [877, 6637] innerCandidates148_21 7514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_22

def innerTail148_20 : List Nat := 6529 :: innerTail148_21
def innerCandidates148_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_20_checked : candidates edge 4 877 6529 innerTail148_21 best.bound = innerCandidates148_20 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_20 : scan edge (search edge 3) 4 [877] 877 innerTail148_20 best = best := by
  rw [innerTail148_20, scan, if_neg (by decide), innerCandidates148_20_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_21
    (if 3 ≤ innerCandidates148_20.length then search edge 3 [877, 6529] innerCandidates148_20 7406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_21

def innerTail148_19 : List Nat := 6373 :: innerTail148_20
def innerCandidates148_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_19_checked : candidates edge 4 877 6373 innerTail148_20 best.bound = innerCandidates148_19 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_19 : scan edge (search edge 3) 4 [877] 877 innerTail148_19 best = best := by
  rw [innerTail148_19, scan, if_neg (by decide), innerCandidates148_19_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_20
    (if 3 ≤ innerCandidates148_19.length then search edge 3 [877, 6373] innerCandidates148_19 7250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_20

def innerTail148_18 : List Nat := 6121 :: innerTail148_19
def innerCandidates148_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_18_checked : candidates edge 4 877 6121 innerTail148_19 best.bound = innerCandidates148_18 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_18 : scan edge (search edge 3) 4 [877] 877 innerTail148_18 best = best := by
  rw [innerTail148_18, scan, if_neg (by decide), innerCandidates148_18_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_19
    (if 3 ≤ innerCandidates148_18.length then search edge 3 [877, 6121] innerCandidates148_18 6998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_19

def innerTail148_17 : List Nat := 5743 :: innerTail148_18
def innerCandidates148_17 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates148_17_checked : candidates edge 4 877 5743 innerTail148_18 best.bound = innerCandidates148_17 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_17 : scan edge (search edge 3) 4 [877] 877 innerTail148_17 best = best := by
  rw [innerTail148_17, scan, if_neg (by decide), innerCandidates148_17_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_18
    (if 3 ≤ innerCandidates148_17.length then search edge 3 [877, 5743] innerCandidates148_17 6620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_18

def innerTail148_16 : List Nat := 5659 :: innerTail148_17
def innerCandidates148_16 : List Nat := [6121, 7759]
set_option maxRecDepth 4096 in
theorem innerCandidates148_16_checked : candidates edge 4 877 5659 innerTail148_17 best.bound = innerCandidates148_16 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_16 : scan edge (search edge 3) 4 [877] 877 innerTail148_16 best = best := by
  rw [innerTail148_16, scan, if_neg (by decide), innerCandidates148_16_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_17
    (if 3 ≤ innerCandidates148_16.length then search edge 3 [877, 5659] innerCandidates148_16 6536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_17

def innerTail148_15 : List Nat := 5527 :: innerTail148_16
def innerCandidates148_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_15_checked : candidates edge 4 877 5527 innerTail148_16 best.bound = innerCandidates148_15 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_15 : scan edge (search edge 3) 4 [877] 877 innerTail148_15 best = best := by
  rw [innerTail148_15, scan, if_neg (by decide), innerCandidates148_15_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_16
    (if 3 ≤ innerCandidates148_15.length then search edge 3 [877, 5527] innerCandidates148_15 6404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_16

def innerTail148_14 : List Nat := 5419 :: innerTail148_15
def innerCandidates148_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_14_checked : candidates edge 4 877 5419 innerTail148_15 best.bound = innerCandidates148_14 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_14 : scan edge (search edge 3) 4 [877] 877 innerTail148_14 best = best := by
  rw [innerTail148_14, scan, if_neg (by decide), innerCandidates148_14_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_15
    (if 3 ≤ innerCandidates148_14.length then search edge 3 [877, 5419] innerCandidates148_14 6296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_15

def innerTail148_13 : List Nat := 4723 :: innerTail148_14
def innerCandidates148_13 : List Nat := [7243]
set_option maxRecDepth 4096 in
theorem innerCandidates148_13_checked : candidates edge 4 877 4723 innerTail148_14 best.bound = innerCandidates148_13 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_13 : scan edge (search edge 3) 4 [877] 877 innerTail148_13 best = best := by
  rw [innerTail148_13, scan, if_neg (by decide), innerCandidates148_13_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_14
    (if 3 ≤ innerCandidates148_13.length then search edge 3 [877, 4723] innerCandidates148_13 5600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_14

def innerTail148_12 : List Nat := 4483 :: innerTail148_13
def innerCandidates148_12 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates148_12_checked : candidates edge 4 877 4483 innerTail148_13 best.bound = innerCandidates148_12 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_12 : scan edge (search edge 3) 4 [877] 877 innerTail148_12 best = best := by
  rw [innerTail148_12, scan, if_neg (by decide), innerCandidates148_12_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_13
    (if 3 ≤ innerCandidates148_12.length then search edge 3 [877, 4483] innerCandidates148_12 5360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_13

def innerTail148_11 : List Nat := 3823 :: innerTail148_12
def innerCandidates148_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_11_checked : candidates edge 4 877 3823 innerTail148_12 best.bound = innerCandidates148_11 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_11 : scan edge (search edge 3) 4 [877] 877 innerTail148_11 best = best := by
  rw [innerTail148_11, scan, if_neg (by decide), innerCandidates148_11_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_12
    (if 3 ≤ innerCandidates148_11.length then search edge 3 [877, 3823] innerCandidates148_11 4700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_12

def innerTail148_10 : List Nat := 3727 :: innerTail148_11
def innerCandidates148_10 : List Nat := [5659]
set_option maxRecDepth 4096 in
theorem innerCandidates148_10_checked : candidates edge 4 877 3727 innerTail148_11 best.bound = innerCandidates148_10 := by
  unfold candidates
  have row : edge 3727 = SearchData.row3727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_10 : scan edge (search edge 3) 4 [877] 877 innerTail148_10 best = best := by
  rw [innerTail148_10, scan, if_neg (by decide), innerCandidates148_10_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_11
    (if 3 ≤ innerCandidates148_10.length then search edge 3 [877, 3727] innerCandidates148_10 4604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_11

def innerTail148_9 : List Nat := 3469 :: innerTail148_10
def innerCandidates148_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_9_checked : candidates edge 4 877 3469 innerTail148_10 best.bound = innerCandidates148_9 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_9 : scan edge (search edge 3) 4 [877] 877 innerTail148_9 best = best := by
  rw [innerTail148_9, scan, if_neg (by decide), innerCandidates148_9_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_10
    (if 3 ≤ innerCandidates148_9.length then search edge 3 [877, 3469] innerCandidates148_9 4346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_10

def innerTail148_8 : List Nat := 2917 :: innerTail148_9
def innerCandidates148_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_8_checked : candidates edge 4 877 2917 innerTail148_9 best.bound = innerCandidates148_8 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_8 : scan edge (search edge 3) 4 [877] 877 innerTail148_8 best = best := by
  rw [innerTail148_8, scan, if_neg (by decide), innerCandidates148_8_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_9
    (if 3 ≤ innerCandidates148_8.length then search edge 3 [877, 2917] innerCandidates148_8 3794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_9

def innerTail148_7 : List Nat := 2551 :: innerTail148_8
def innerCandidates148_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_7_checked : candidates edge 4 877 2551 innerTail148_8 best.bound = innerCandidates148_7 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_7 : scan edge (search edge 3) 4 [877] 877 innerTail148_7 best = best := by
  rw [innerTail148_7, scan, if_neg (by decide), innerCandidates148_7_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_8
    (if 3 ≤ innerCandidates148_7.length then search edge 3 [877, 2551] innerCandidates148_7 3428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_8

def innerTail148_6 : List Nat := 2371 :: innerTail148_7
def innerCandidates148_6 : List Nat := [2917]
set_option maxRecDepth 4096 in
theorem innerCandidates148_6_checked : candidates edge 4 877 2371 innerTail148_7 best.bound = innerCandidates148_6 := by
  unfold candidates
  have row : edge 2371 = PrimePairCache.smallRow2371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_6 : scan edge (search edge 3) 4 [877] 877 innerTail148_6 best = best := by
  rw [innerTail148_6, scan, if_neg (by decide), innerCandidates148_6_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_7
    (if 3 ≤ innerCandidates148_6.length then search edge 3 [877, 2371] innerCandidates148_6 3248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_7

def innerTail148_5 : List Nat := 2251 :: innerTail148_6
def innerCandidates148_5 : List Nat := [7669, 8191]
set_option maxRecDepth 4096 in
theorem innerCandidates148_5_checked : candidates edge 4 877 2251 innerTail148_6 best.bound = innerCandidates148_5 := by
  unfold candidates
  have row : edge 2251 = PrimePairCache.smallRow2251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_5 : scan edge (search edge 3) 4 [877] 877 innerTail148_5 best = best := by
  rw [innerTail148_5, scan, if_neg (by decide), innerCandidates148_5_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_6
    (if 3 ≤ innerCandidates148_5.length then search edge 3 [877, 2251] innerCandidates148_5 3128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_6

def innerTail148_4 : List Nat := 1621 :: innerTail148_5
def innerCandidates148_4 : List Nat := [3823, 5527]
set_option maxRecDepth 4096 in
theorem innerCandidates148_4_checked : candidates edge 4 877 1621 innerTail148_5 best.bound = innerCandidates148_4 := by
  unfold candidates
  have row : edge 1621 = PrimePairCache.smallRow1621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_4 : scan edge (search edge 3) 4 [877] 877 innerTail148_4 best = best := by
  rw [innerTail148_4, scan, if_neg (by decide), innerCandidates148_4_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_5
    (if 3 ≤ innerCandidates148_4.length then search edge 3 [877, 1621] innerCandidates148_4 2498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_5

def innerTail148_3 : List Nat := 1327 :: innerTail148_4
def innerCandidates148_3 : List Nat := [2551]
set_option maxRecDepth 4096 in
theorem innerCandidates148_3_checked : candidates edge 4 877 1327 innerTail148_4 best.bound = innerCandidates148_3 := by
  unfold candidates
  have row : edge 1327 = PrimePairCache.smallRow1327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_3 : scan edge (search edge 3) 4 [877] 877 innerTail148_3 best = best := by
  rw [innerTail148_3, scan, if_neg (by decide), innerCandidates148_3_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_4
    (if 3 ≤ innerCandidates148_3.length then search edge 3 [877, 1327] innerCandidates148_3 2204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_4

def innerTail148_2 : List Nat := 997 :: innerTail148_3
def innerCandidates148_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_2_checked : candidates edge 4 877 997 innerTail148_3 best.bound = innerCandidates148_2 := by
  unfold candidates
  have row : edge 997 = PrimePairCache.smallRow997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_2 : scan edge (search edge 3) 4 [877] 877 innerTail148_2 best = best := by
  rw [innerTail148_2, scan, if_neg (by decide), innerCandidates148_2_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_3
    (if 3 ≤ innerCandidates148_2.length then search edge 3 [877, 997] innerCandidates148_2 1874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_3

def innerTail148_1 : List Nat := 937 :: innerTail148_2
def innerCandidates148_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_1_checked : candidates edge 4 877 937 innerTail148_2 best.bound = innerCandidates148_1 := by
  unfold candidates
  have row : edge 937 = PrimePairCache.smallRow937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_1 : scan edge (search edge 3) 4 [877] 877 innerTail148_1 best = best := by
  rw [innerTail148_1, scan, if_neg (by decide), innerCandidates148_1_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_2
    (if 3 ≤ innerCandidates148_1.length then search edge 3 [877, 937] innerCandidates148_1 1814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_2

def innerTail148_0 : List Nat := 883 :: innerTail148_1
def innerCandidates148_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates148_0_checked : candidates edge 4 877 883 innerTail148_1 best.bound = innerCandidates148_0 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan148_0 : scan edge (search edge 3) 4 [877] 877 innerTail148_0 best = best := by
  rw [innerTail148_0, scan, if_neg (by decide), innerCandidates148_0_checked]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_1
    (if 3 ≤ innerCandidates148_0.length then search edge 3 [877, 883] innerCandidates148_0 1760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan148_1

theorem rootBranch148 : search edge 4 [877] rootCandidates148 877 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [877] 877 innerTail148_0 best = best
  exact innerScan148_0

def rootCandidates147 : List Nat := [983, 1277, 1307, 1823, 1907, 2003, 2297, 2357, 2621, 2801, 2927, 3137, 4253, 4259, 4397, 4421, 5381, 5717, 5849, 6779, 6899, 6959, 7523]
set_option maxRecDepth 4096 in
theorem rootCandidates147_checked : candidates edge 5 0 863 tail148 best.bound = rootCandidates147 := by
  unfold candidates
  have row : edge 863 = PrimePairCache.smallRow863.contains := by funext q; rfl
  rw [row]
  decide

def innerTail147_23 : List Nat := []
theorem innerScan147_23 : scan edge (search edge 3) 4 [863] 863 innerTail147_23 best = best := rfl

def innerTail147_22 : List Nat := 7523 :: innerTail147_23
def innerCandidates147_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_22_checked : candidates edge 4 863 7523 innerTail147_23 best.bound = innerCandidates147_22 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_22 : scan edge (search edge 3) 4 [863] 863 innerTail147_22 best = best := by
  rw [innerTail147_22, scan, if_neg (by decide), innerCandidates147_22_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_23
    (if 3 ≤ innerCandidates147_22.length then search edge 3 [863, 7523] innerCandidates147_22 8386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_23

def innerTail147_21 : List Nat := 6959 :: innerTail147_22
def innerCandidates147_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_21_checked : candidates edge 4 863 6959 innerTail147_22 best.bound = innerCandidates147_21 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_21 : scan edge (search edge 3) 4 [863] 863 innerTail147_21 best = best := by
  rw [innerTail147_21, scan, if_neg (by decide), innerCandidates147_21_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_22
    (if 3 ≤ innerCandidates147_21.length then search edge 3 [863, 6959] innerCandidates147_21 7822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_22

def innerTail147_20 : List Nat := 6899 :: innerTail147_21
def innerCandidates147_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_20_checked : candidates edge 4 863 6899 innerTail147_21 best.bound = innerCandidates147_20 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_20 : scan edge (search edge 3) 4 [863] 863 innerTail147_20 best = best := by
  rw [innerTail147_20, scan, if_neg (by decide), innerCandidates147_20_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_21
    (if 3 ≤ innerCandidates147_20.length then search edge 3 [863, 6899] innerCandidates147_20 7762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_21

def innerTail147_19 : List Nat := 6779 :: innerTail147_20
def innerCandidates147_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_19_checked : candidates edge 4 863 6779 innerTail147_20 best.bound = innerCandidates147_19 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_19 : scan edge (search edge 3) 4 [863] 863 innerTail147_19 best = best := by
  rw [innerTail147_19, scan, if_neg (by decide), innerCandidates147_19_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_20
    (if 3 ≤ innerCandidates147_19.length then search edge 3 [863, 6779] innerCandidates147_19 7642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_20

def innerTail147_18 : List Nat := 5849 :: innerTail147_19
def innerCandidates147_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_18_checked : candidates edge 4 863 5849 innerTail147_19 best.bound = innerCandidates147_18 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_18 : scan edge (search edge 3) 4 [863] 863 innerTail147_18 best = best := by
  rw [innerTail147_18, scan, if_neg (by decide), innerCandidates147_18_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_19
    (if 3 ≤ innerCandidates147_18.length then search edge 3 [863, 5849] innerCandidates147_18 6712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_19

def innerTail147_17 : List Nat := 5717 :: innerTail147_18
def innerCandidates147_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_17_checked : candidates edge 4 863 5717 innerTail147_18 best.bound = innerCandidates147_17 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_17 : scan edge (search edge 3) 4 [863] 863 innerTail147_17 best = best := by
  rw [innerTail147_17, scan, if_neg (by decide), innerCandidates147_17_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_18
    (if 3 ≤ innerCandidates147_17.length then search edge 3 [863, 5717] innerCandidates147_17 6580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_18

def innerTail147_16 : List Nat := 5381 :: innerTail147_17
def innerCandidates147_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_16_checked : candidates edge 4 863 5381 innerTail147_17 best.bound = innerCandidates147_16 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_16 : scan edge (search edge 3) 4 [863] 863 innerTail147_16 best = best := by
  rw [innerTail147_16, scan, if_neg (by decide), innerCandidates147_16_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_17
    (if 3 ≤ innerCandidates147_16.length then search edge 3 [863, 5381] innerCandidates147_16 6244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_17

def innerTail147_15 : List Nat := 4421 :: innerTail147_16
def innerCandidates147_15 : List Nat := [6899]
set_option maxRecDepth 4096 in
theorem innerCandidates147_15_checked : candidates edge 4 863 4421 innerTail147_16 best.bound = innerCandidates147_15 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_15 : scan edge (search edge 3) 4 [863] 863 innerTail147_15 best = best := by
  rw [innerTail147_15, scan, if_neg (by decide), innerCandidates147_15_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_16
    (if 3 ≤ innerCandidates147_15.length then search edge 3 [863, 4421] innerCandidates147_15 5284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_16

def innerTail147_14 : List Nat := 4397 :: innerTail147_15
def innerCandidates147_14 : List Nat := [4421]
set_option maxRecDepth 4096 in
theorem innerCandidates147_14_checked : candidates edge 4 863 4397 innerTail147_15 best.bound = innerCandidates147_14 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_14 : scan edge (search edge 3) 4 [863] 863 innerTail147_14 best = best := by
  rw [innerTail147_14, scan, if_neg (by decide), innerCandidates147_14_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_15
    (if 3 ≤ innerCandidates147_14.length then search edge 3 [863, 4397] innerCandidates147_14 5260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_15

def innerTail147_13 : List Nat := 4259 :: innerTail147_14
def innerCandidates147_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_13_checked : candidates edge 4 863 4259 innerTail147_14 best.bound = innerCandidates147_13 := by
  unfold candidates
  have row : edge 4259 = SearchData.row4259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_13 : scan edge (search edge 3) 4 [863] 863 innerTail147_13 best = best := by
  rw [innerTail147_13, scan, if_neg (by decide), innerCandidates147_13_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_14
    (if 3 ≤ innerCandidates147_13.length then search edge 3 [863, 4259] innerCandidates147_13 5122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_14

def innerTail147_12 : List Nat := 4253 :: innerTail147_13
def innerCandidates147_12 : List Nat := [4397]
set_option maxRecDepth 4096 in
theorem innerCandidates147_12_checked : candidates edge 4 863 4253 innerTail147_13 best.bound = innerCandidates147_12 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_12 : scan edge (search edge 3) 4 [863] 863 innerTail147_12 best = best := by
  rw [innerTail147_12, scan, if_neg (by decide), innerCandidates147_12_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_13
    (if 3 ≤ innerCandidates147_12.length then search edge 3 [863, 4253] innerCandidates147_12 5116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_13

def innerTail147_11 : List Nat := 3137 :: innerTail147_12
def innerCandidates147_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_11_checked : candidates edge 4 863 3137 innerTail147_12 best.bound = innerCandidates147_11 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_11 : scan edge (search edge 3) 4 [863] 863 innerTail147_11 best = best := by
  rw [innerTail147_11, scan, if_neg (by decide), innerCandidates147_11_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_12
    (if 3 ≤ innerCandidates147_11.length then search edge 3 [863, 3137] innerCandidates147_11 4000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_12

def innerTail147_10 : List Nat := 2927 :: innerTail147_11
def innerCandidates147_10 : List Nat := [3137]
set_option maxRecDepth 4096 in
theorem innerCandidates147_10_checked : candidates edge 4 863 2927 innerTail147_11 best.bound = innerCandidates147_10 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_10 : scan edge (search edge 3) 4 [863] 863 innerTail147_10 best = best := by
  rw [innerTail147_10, scan, if_neg (by decide), innerCandidates147_10_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_11
    (if 3 ≤ innerCandidates147_10.length then search edge 3 [863, 2927] innerCandidates147_10 3790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_11

def innerTail147_9 : List Nat := 2801 :: innerTail147_10
def innerCandidates147_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_9_checked : candidates edge 4 863 2801 innerTail147_10 best.bound = innerCandidates147_9 := by
  unfold candidates
  have row : edge 2801 = SearchData.row2801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_9 : scan edge (search edge 3) 4 [863] 863 innerTail147_9 best = best := by
  rw [innerTail147_9, scan, if_neg (by decide), innerCandidates147_9_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_10
    (if 3 ≤ innerCandidates147_9.length then search edge 3 [863, 2801] innerCandidates147_9 3664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_10

def innerTail147_8 : List Nat := 2621 :: innerTail147_9
def innerCandidates147_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_8_checked : candidates edge 4 863 2621 innerTail147_9 best.bound = innerCandidates147_8 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_8 : scan edge (search edge 3) 4 [863] 863 innerTail147_8 best = best := by
  rw [innerTail147_8, scan, if_neg (by decide), innerCandidates147_8_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_9
    (if 3 ≤ innerCandidates147_8.length then search edge 3 [863, 2621] innerCandidates147_8 3484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_9

def innerTail147_7 : List Nat := 2357 :: innerTail147_8
def innerCandidates147_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_7_checked : candidates edge 4 863 2357 innerTail147_8 best.bound = innerCandidates147_7 := by
  unfold candidates
  have row : edge 2357 = PrimePairCache.smallRow2357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_7 : scan edge (search edge 3) 4 [863] 863 innerTail147_7 best = best := by
  rw [innerTail147_7, scan, if_neg (by decide), innerCandidates147_7_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_8
    (if 3 ≤ innerCandidates147_7.length then search edge 3 [863, 2357] innerCandidates147_7 3220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_8

def innerTail147_6 : List Nat := 2297 :: innerTail147_7
def innerCandidates147_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_6_checked : candidates edge 4 863 2297 innerTail147_7 best.bound = innerCandidates147_6 := by
  unfold candidates
  have row : edge 2297 = PrimePairCache.smallRow2297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_6 : scan edge (search edge 3) 4 [863] 863 innerTail147_6 best = best := by
  rw [innerTail147_6, scan, if_neg (by decide), innerCandidates147_6_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_7
    (if 3 ≤ innerCandidates147_6.length then search edge 3 [863, 2297] innerCandidates147_6 3160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_7

def innerTail147_5 : List Nat := 2003 :: innerTail147_6
def innerCandidates147_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_5_checked : candidates edge 4 863 2003 innerTail147_6 best.bound = innerCandidates147_5 := by
  unfold candidates
  have row : edge 2003 = PrimePairCache.smallRow2003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_5 : scan edge (search edge 3) 4 [863] 863 innerTail147_5 best = best := by
  rw [innerTail147_5, scan, if_neg (by decide), innerCandidates147_5_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_6
    (if 3 ≤ innerCandidates147_5.length then search edge 3 [863, 2003] innerCandidates147_5 2866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_6

def innerTail147_4 : List Nat := 1907 :: innerTail147_5
def innerCandidates147_4 : List Nat := [2621]
set_option maxRecDepth 4096 in
theorem innerCandidates147_4_checked : candidates edge 4 863 1907 innerTail147_5 best.bound = innerCandidates147_4 := by
  unfold candidates
  have row : edge 1907 = PrimePairCache.smallRow1907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_4 : scan edge (search edge 3) 4 [863] 863 innerTail147_4 best = best := by
  rw [innerTail147_4, scan, if_neg (by decide), innerCandidates147_4_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_5
    (if 3 ≤ innerCandidates147_4.length then search edge 3 [863, 1907] innerCandidates147_4 2770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_5

def innerTail147_3 : List Nat := 1823 :: innerTail147_4
def innerCandidates147_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_3_checked : candidates edge 4 863 1823 innerTail147_4 best.bound = innerCandidates147_3 := by
  unfold candidates
  have row : edge 1823 = PrimePairCache.smallRow1823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_3 : scan edge (search edge 3) 4 [863] 863 innerTail147_3 best = best := by
  rw [innerTail147_3, scan, if_neg (by decide), innerCandidates147_3_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_4
    (if 3 ≤ innerCandidates147_3.length then search edge 3 [863, 1823] innerCandidates147_3 2686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_4

def innerTail147_2 : List Nat := 1307 :: innerTail147_3
def innerCandidates147_2 : List Nat := [2357]
set_option maxRecDepth 4096 in
theorem innerCandidates147_2_checked : candidates edge 4 863 1307 innerTail147_3 best.bound = innerCandidates147_2 := by
  unfold candidates
  have row : edge 1307 = PrimePairCache.smallRow1307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_2 : scan edge (search edge 3) 4 [863] 863 innerTail147_2 best = best := by
  rw [innerTail147_2, scan, if_neg (by decide), innerCandidates147_2_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_3
    (if 3 ≤ innerCandidates147_2.length then search edge 3 [863, 1307] innerCandidates147_2 2170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_3

def innerTail147_1 : List Nat := 1277 :: innerTail147_2
def innerCandidates147_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_1_checked : candidates edge 4 863 1277 innerTail147_2 best.bound = innerCandidates147_1 := by
  unfold candidates
  have row : edge 1277 = PrimePairCache.smallRow1277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_1 : scan edge (search edge 3) 4 [863] 863 innerTail147_1 best = best := by
  rw [innerTail147_1, scan, if_neg (by decide), innerCandidates147_1_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_2
    (if 3 ≤ innerCandidates147_1.length then search edge 3 [863, 1277] innerCandidates147_1 2140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_2

def innerTail147_0 : List Nat := 983 :: innerTail147_1
def innerCandidates147_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates147_0_checked : candidates edge 4 863 983 innerTail147_1 best.bound = innerCandidates147_0 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan147_0 : scan edge (search edge 3) 4 [863] 863 innerTail147_0 best = best := by
  rw [innerTail147_0, scan, if_neg (by decide), innerCandidates147_0_checked]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_1
    (if 3 ≤ innerCandidates147_0.length then search edge 3 [863, 983] innerCandidates147_0 1846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan147_1

theorem rootBranch147 : search edge 4 [863] rootCandidates147 863 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [863] 863 innerTail147_0 best = best
  exact innerScan147_0

end ProjectEulerSolutions.PrimePairUpper.Search
