import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates196 : List Nat := [1361, 1427, 1433, 1583, 1721, 2003, 2087, 2273, 2309, 2393, 2423, 2477, 2591, 2843, 2903, 3671, 4931, 5003, 5081, 5153, 5393, 5477, 5573, 5879, 6173, 7001, 7019, 7817, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates196_checked : candidates edge 5 0 1217 tail197 best.bound = rootCandidates196 := by
  unfold candidates
  have row : edge 1217 = PrimePairCache.smallRow1217.contains := by funext q; rfl
  rw [row]
  decide

def innerTail196_29 : List Nat := []
theorem innerScan196_29 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_29 best = best := rfl

def innerTail196_28 : List Nat := 8093 :: innerTail196_29
def innerCandidates196_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_28_checked : candidates edge 4 1217 8093 innerTail196_29 best.bound = innerCandidates196_28 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_28 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_28 best = best := by
  rw [innerTail196_28, scan, if_neg (by decide), innerCandidates196_28_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_29
    (if 3 ≤ innerCandidates196_28.length then search edge 3 [1217, 8093] innerCandidates196_28 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_29

def innerTail196_27 : List Nat := 7817 :: innerTail196_28
def innerCandidates196_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_27_checked : candidates edge 4 1217 7817 innerTail196_28 best.bound = innerCandidates196_27 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_27 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_27 best = best := by
  rw [innerTail196_27, scan, if_neg (by decide), innerCandidates196_27_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_28
    (if 3 ≤ innerCandidates196_27.length then search edge 3 [1217, 7817] innerCandidates196_27 9034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_28

def innerTail196_26 : List Nat := 7019 :: innerTail196_27
def innerCandidates196_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_26_checked : candidates edge 4 1217 7019 innerTail196_27 best.bound = innerCandidates196_26 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_26 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_26 best = best := by
  rw [innerTail196_26, scan, if_neg (by decide), innerCandidates196_26_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_27
    (if 3 ≤ innerCandidates196_26.length then search edge 3 [1217, 7019] innerCandidates196_26 8236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_27

def innerTail196_25 : List Nat := 7001 :: innerTail196_26
def innerCandidates196_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_25_checked : candidates edge 4 1217 7001 innerTail196_26 best.bound = innerCandidates196_25 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_25 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_25 best = best := by
  rw [innerTail196_25, scan, if_neg (by decide), innerCandidates196_25_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_26
    (if 3 ≤ innerCandidates196_25.length then search edge 3 [1217, 7001] innerCandidates196_25 8218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_26

def innerTail196_24 : List Nat := 6173 :: innerTail196_25
def innerCandidates196_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_24_checked : candidates edge 4 1217 6173 innerTail196_25 best.bound = innerCandidates196_24 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_24 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_24 best = best := by
  rw [innerTail196_24, scan, if_neg (by decide), innerCandidates196_24_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_25
    (if 3 ≤ innerCandidates196_24.length then search edge 3 [1217, 6173] innerCandidates196_24 7390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_25

def innerTail196_23 : List Nat := 5879 :: innerTail196_24
def innerCandidates196_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_23_checked : candidates edge 4 1217 5879 innerTail196_24 best.bound = innerCandidates196_23 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_23 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_23 best = best := by
  rw [innerTail196_23, scan, if_neg (by decide), innerCandidates196_23_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_24
    (if 3 ≤ innerCandidates196_23.length then search edge 3 [1217, 5879] innerCandidates196_23 7096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_24

def innerTail196_22 : List Nat := 5573 :: innerTail196_23
def innerCandidates196_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_22_checked : candidates edge 4 1217 5573 innerTail196_23 best.bound = innerCandidates196_22 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_22 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_22 best = best := by
  rw [innerTail196_22, scan, if_neg (by decide), innerCandidates196_22_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_23
    (if 3 ≤ innerCandidates196_22.length then search edge 3 [1217, 5573] innerCandidates196_22 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_23

def innerTail196_21 : List Nat := 5477 :: innerTail196_22
def innerCandidates196_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_21_checked : candidates edge 4 1217 5477 innerTail196_22 best.bound = innerCandidates196_21 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_21 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_21 best = best := by
  rw [innerTail196_21, scan, if_neg (by decide), innerCandidates196_21_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_22
    (if 3 ≤ innerCandidates196_21.length then search edge 3 [1217, 5477] innerCandidates196_21 6694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_22

def innerTail196_20 : List Nat := 5393 :: innerTail196_21
def innerCandidates196_20 : List Nat := [7817]
set_option maxRecDepth 4096 in
theorem innerCandidates196_20_checked : candidates edge 4 1217 5393 innerTail196_21 best.bound = innerCandidates196_20 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_20 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_20 best = best := by
  rw [innerTail196_20, scan, if_neg (by decide), innerCandidates196_20_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_21
    (if 3 ≤ innerCandidates196_20.length then search edge 3 [1217, 5393] innerCandidates196_20 6610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_21

def innerTail196_19 : List Nat := 5153 :: innerTail196_20
def innerCandidates196_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_19_checked : candidates edge 4 1217 5153 innerTail196_20 best.bound = innerCandidates196_19 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_19 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_19 best = best := by
  rw [innerTail196_19, scan, if_neg (by decide), innerCandidates196_19_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_20
    (if 3 ≤ innerCandidates196_19.length then search edge 3 [1217, 5153] innerCandidates196_19 6370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_20

def innerTail196_18 : List Nat := 5081 :: innerTail196_19
def innerCandidates196_18 : List Nat := [7817]
set_option maxRecDepth 4096 in
theorem innerCandidates196_18_checked : candidates edge 4 1217 5081 innerTail196_19 best.bound = innerCandidates196_18 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_18 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_18 best = best := by
  rw [innerTail196_18, scan, if_neg (by decide), innerCandidates196_18_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_19
    (if 3 ≤ innerCandidates196_18.length then search edge 3 [1217, 5081] innerCandidates196_18 6298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_19

def innerTail196_17 : List Nat := 5003 :: innerTail196_18
def innerCandidates196_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_17_checked : candidates edge 4 1217 5003 innerTail196_18 best.bound = innerCandidates196_17 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_17 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_17 best = best := by
  rw [innerTail196_17, scan, if_neg (by decide), innerCandidates196_17_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_18
    (if 3 ≤ innerCandidates196_17.length then search edge 3 [1217, 5003] innerCandidates196_17 6220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_18

def innerTail196_16 : List Nat := 4931 :: innerTail196_17
def innerCandidates196_16 : List Nat := [7817]
set_option maxRecDepth 4096 in
theorem innerCandidates196_16_checked : candidates edge 4 1217 4931 innerTail196_17 best.bound = innerCandidates196_16 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_16 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_16 best = best := by
  rw [innerTail196_16, scan, if_neg (by decide), innerCandidates196_16_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_17
    (if 3 ≤ innerCandidates196_16.length then search edge 3 [1217, 4931] innerCandidates196_16 6148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_17

def innerTail196_15 : List Nat := 3671 :: innerTail196_16
def innerCandidates196_15 : List Nat := [7019]
set_option maxRecDepth 4096 in
theorem innerCandidates196_15_checked : candidates edge 4 1217 3671 innerTail196_16 best.bound = innerCandidates196_15 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_15 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_15 best = best := by
  rw [innerTail196_15, scan, if_neg (by decide), innerCandidates196_15_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_16
    (if 3 ≤ innerCandidates196_15.length then search edge 3 [1217, 3671] innerCandidates196_15 4888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_16

def innerTail196_14 : List Nat := 2903 :: innerTail196_15
def innerCandidates196_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_14_checked : candidates edge 4 1217 2903 innerTail196_15 best.bound = innerCandidates196_14 := by
  unfold candidates
  have row : edge 2903 = SearchData.row2903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_14 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_14 best = best := by
  rw [innerTail196_14, scan, if_neg (by decide), innerCandidates196_14_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_15
    (if 3 ≤ innerCandidates196_14.length then search edge 3 [1217, 2903] innerCandidates196_14 4120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_15

def innerTail196_13 : List Nat := 2843 :: innerTail196_14
def innerCandidates196_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_13_checked : candidates edge 4 1217 2843 innerTail196_14 best.bound = innerCandidates196_13 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_13 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_13 best = best := by
  rw [innerTail196_13, scan, if_neg (by decide), innerCandidates196_13_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_14
    (if 3 ≤ innerCandidates196_13.length then search edge 3 [1217, 2843] innerCandidates196_13 4060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_14

def innerTail196_12 : List Nat := 2591 :: innerTail196_13
def innerCandidates196_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_12_checked : candidates edge 4 1217 2591 innerTail196_13 best.bound = innerCandidates196_12 := by
  unfold candidates
  have row : edge 2591 = PrimePairCache.smallRow2591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_12 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_12 best = best := by
  rw [innerTail196_12, scan, if_neg (by decide), innerCandidates196_12_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_13
    (if 3 ≤ innerCandidates196_12.length then search edge 3 [1217, 2591] innerCandidates196_12 3808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_13

def innerTail196_11 : List Nat := 2477 :: innerTail196_12
def innerCandidates196_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_11_checked : candidates edge 4 1217 2477 innerTail196_12 best.bound = innerCandidates196_11 := by
  unfold candidates
  have row : edge 2477 = PrimePairCache.smallRow2477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_11 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_11 best = best := by
  rw [innerTail196_11, scan, if_neg (by decide), innerCandidates196_11_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_12
    (if 3 ≤ innerCandidates196_11.length then search edge 3 [1217, 2477] innerCandidates196_11 3694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_12

def innerTail196_10 : List Nat := 2423 :: innerTail196_11
def innerCandidates196_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_10_checked : candidates edge 4 1217 2423 innerTail196_11 best.bound = innerCandidates196_10 := by
  unfold candidates
  have row : edge 2423 = PrimePairCache.smallRow2423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_10 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_10 best = best := by
  rw [innerTail196_10, scan, if_neg (by decide), innerCandidates196_10_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_11
    (if 3 ≤ innerCandidates196_10.length then search edge 3 [1217, 2423] innerCandidates196_10 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_11

def innerTail196_9 : List Nat := 2393 :: innerTail196_10
def innerCandidates196_9 : List Nat := [2423]
set_option maxRecDepth 4096 in
theorem innerCandidates196_9_checked : candidates edge 4 1217 2393 innerTail196_10 best.bound = innerCandidates196_9 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_9 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_9 best = best := by
  rw [innerTail196_9, scan, if_neg (by decide), innerCandidates196_9_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_10
    (if 3 ≤ innerCandidates196_9.length then search edge 3 [1217, 2393] innerCandidates196_9 3610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_10

def innerTail196_8 : List Nat := 2309 :: innerTail196_9
def innerCandidates196_8 : List Nat := [5081, 7817]
set_option maxRecDepth 4096 in
theorem innerCandidates196_8_checked : candidates edge 4 1217 2309 innerTail196_9 best.bound = innerCandidates196_8 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_8 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_8 best = best := by
  rw [innerTail196_8, scan, if_neg (by decide), innerCandidates196_8_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_9
    (if 3 ≤ innerCandidates196_8.length then search edge 3 [1217, 2309] innerCandidates196_8 3526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_9

def innerTail196_7 : List Nat := 2273 :: innerTail196_8
def innerCandidates196_7 : List Nat := [2903]
set_option maxRecDepth 4096 in
theorem innerCandidates196_7_checked : candidates edge 4 1217 2273 innerTail196_8 best.bound = innerCandidates196_7 := by
  unfold candidates
  have row : edge 2273 = PrimePairCache.smallRow2273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_7 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_7 best = best := by
  rw [innerTail196_7, scan, if_neg (by decide), innerCandidates196_7_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_8
    (if 3 ≤ innerCandidates196_7.length then search edge 3 [1217, 2273] innerCandidates196_7 3490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_8

def innerTail196_6 : List Nat := 2087 :: innerTail196_7
def innerCandidates196_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_6_checked : candidates edge 4 1217 2087 innerTail196_7 best.bound = innerCandidates196_6 := by
  unfold candidates
  have row : edge 2087 = PrimePairCache.smallRow2087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_6 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_6 best = best := by
  rw [innerTail196_6, scan, if_neg (by decide), innerCandidates196_6_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_7
    (if 3 ≤ innerCandidates196_6.length then search edge 3 [1217, 2087] innerCandidates196_6 3304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_7

def innerTail196_5 : List Nat := 2003 :: innerTail196_6
def innerCandidates196_5 : List Nat := [2087, 5153]
set_option maxRecDepth 4096 in
theorem innerCandidates196_5_checked : candidates edge 4 1217 2003 innerTail196_6 best.bound = innerCandidates196_5 := by
  unfold candidates
  have row : edge 2003 = PrimePairCache.smallRow2003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_5 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_5 best = best := by
  rw [innerTail196_5, scan, if_neg (by decide), innerCandidates196_5_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_6
    (if 3 ≤ innerCandidates196_5.length then search edge 3 [1217, 2003] innerCandidates196_5 3220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_6

def innerTail196_4 : List Nat := 1721 :: innerTail196_5
def innerCandidates196_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_4_checked : candidates edge 4 1217 1721 innerTail196_5 best.bound = innerCandidates196_4 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_4 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_4 best = best := by
  rw [innerTail196_4, scan, if_neg (by decide), innerCandidates196_4_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_5
    (if 3 ≤ innerCandidates196_4.length then search edge 3 [1217, 1721] innerCandidates196_4 2938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_5

def innerTail196_3 : List Nat := 1583 :: innerTail196_4
def innerCandidates196_3 : List Nat := [5879]
set_option maxRecDepth 4096 in
theorem innerCandidates196_3_checked : candidates edge 4 1217 1583 innerTail196_4 best.bound = innerCandidates196_3 := by
  unfold candidates
  have row : edge 1583 = PrimePairCache.smallRow1583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_3 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_3 best = best := by
  rw [innerTail196_3, scan, if_neg (by decide), innerCandidates196_3_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_4
    (if 3 ≤ innerCandidates196_3.length then search edge 3 [1217, 1583] innerCandidates196_3 2800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_4

def innerTail196_2 : List Nat := 1433 :: innerTail196_3
def innerCandidates196_2 : List Nat := [5393]
set_option maxRecDepth 4096 in
theorem innerCandidates196_2_checked : candidates edge 4 1217 1433 innerTail196_3 best.bound = innerCandidates196_2 := by
  unfold candidates
  have row : edge 1433 = PrimePairCache.smallRow1433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_2 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_2 best = best := by
  rw [innerTail196_2, scan, if_neg (by decide), innerCandidates196_2_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_3
    (if 3 ≤ innerCandidates196_2.length then search edge 3 [1217, 1433] innerCandidates196_2 2650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_3

def innerTail196_1 : List Nat := 1427 :: innerTail196_2
def innerCandidates196_1 : List Nat := [5477]
set_option maxRecDepth 4096 in
theorem innerCandidates196_1_checked : candidates edge 4 1217 1427 innerTail196_2 best.bound = innerCandidates196_1 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_1 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_1 best = best := by
  rw [innerTail196_1, scan, if_neg (by decide), innerCandidates196_1_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_2
    (if 3 ≤ innerCandidates196_1.length then search edge 3 [1217, 1427] innerCandidates196_1 2644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_2

def innerTail196_0 : List Nat := 1361 :: innerTail196_1
def innerCandidates196_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates196_0_checked : candidates edge 4 1217 1361 innerTail196_1 best.bound = innerCandidates196_0 := by
  unfold candidates
  have row : edge 1361 = PrimePairCache.smallRow1361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan196_0 : scan edge (search edge 3) 4 [1217] 1217 innerTail196_0 best = best := by
  rw [innerTail196_0, scan, if_neg (by decide), innerCandidates196_0_checked]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_1
    (if 3 ≤ innerCandidates196_0.length then search edge 3 [1217, 1361] innerCandidates196_0 2578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan196_1

theorem rootBranch196 : search edge 4 [1217] rootCandidates196 1217 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1217] 1217 innerTail196_0 best = best
  exact innerScan196_0

def rootCandidates195 : List Nat := [1297, 1621, 1789, 2011, 2053, 2371, 2797, 2851, 3061, 3301, 3463, 4201, 5443, 5479, 6043, 6073, 6151, 6211, 6547, 7243, 8209, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates195_checked : candidates edge 5 0 1213 tail196 best.bound = rootCandidates195 := by
  unfold candidates
  have row : edge 1213 = PrimePairCache.smallRow1213.contains := by funext q; rfl
  rw [row]
  decide

def innerTail195_22 : List Nat := []
theorem innerScan195_22 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_22 best = best := rfl

def innerTail195_21 : List Nat := 8353 :: innerTail195_22
def innerCandidates195_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_21_checked : candidates edge 4 1213 8353 innerTail195_22 best.bound = innerCandidates195_21 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_21 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_21 best = best := by
  rw [innerTail195_21, scan, if_neg (by decide), innerCandidates195_21_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_22
    (if 3 ≤ innerCandidates195_21.length then search edge 3 [1213, 8353] innerCandidates195_21 9566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_22

def innerTail195_20 : List Nat := 8209 :: innerTail195_21
def innerCandidates195_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_20_checked : candidates edge 4 1213 8209 innerTail195_21 best.bound = innerCandidates195_20 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_20 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_20 best = best := by
  rw [innerTail195_20, scan, if_neg (by decide), innerCandidates195_20_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_21
    (if 3 ≤ innerCandidates195_20.length then search edge 3 [1213, 8209] innerCandidates195_20 9422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_21

def innerTail195_19 : List Nat := 7243 :: innerTail195_20
def innerCandidates195_19 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem innerCandidates195_19_checked : candidates edge 4 1213 7243 innerTail195_20 best.bound = innerCandidates195_19 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_19 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_19 best = best := by
  rw [innerTail195_19, scan, if_neg (by decide), innerCandidates195_19_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_20
    (if 3 ≤ innerCandidates195_19.length then search edge 3 [1213, 7243] innerCandidates195_19 8456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_20

def innerTail195_18 : List Nat := 6547 :: innerTail195_19
def innerCandidates195_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_18_checked : candidates edge 4 1213 6547 innerTail195_19 best.bound = innerCandidates195_18 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_18 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_18 best = best := by
  rw [innerTail195_18, scan, if_neg (by decide), innerCandidates195_18_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_19
    (if 3 ≤ innerCandidates195_18.length then search edge 3 [1213, 6547] innerCandidates195_18 7760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_19

def innerTail195_17 : List Nat := 6211 :: innerTail195_18
def innerCandidates195_17 : List Nat := [6547]
set_option maxRecDepth 4096 in
theorem innerCandidates195_17_checked : candidates edge 4 1213 6211 innerTail195_18 best.bound = innerCandidates195_17 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_17 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_17 best = best := by
  rw [innerTail195_17, scan, if_neg (by decide), innerCandidates195_17_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_18
    (if 3 ≤ innerCandidates195_17.length then search edge 3 [1213, 6211] innerCandidates195_17 7424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_18

def innerTail195_16 : List Nat := 6151 :: innerTail195_17
def innerCandidates195_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_16_checked : candidates edge 4 1213 6151 innerTail195_17 best.bound = innerCandidates195_16 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_16 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_16 best = best := by
  rw [innerTail195_16, scan, if_neg (by decide), innerCandidates195_16_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_17
    (if 3 ≤ innerCandidates195_16.length then search edge 3 [1213, 6151] innerCandidates195_16 7364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_17

def innerTail195_15 : List Nat := 6073 :: innerTail195_16
def innerCandidates195_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_15_checked : candidates edge 4 1213 6073 innerTail195_16 best.bound = innerCandidates195_15 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_15 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_15 best = best := by
  rw [innerTail195_15, scan, if_neg (by decide), innerCandidates195_15_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_16
    (if 3 ≤ innerCandidates195_15.length then search edge 3 [1213, 6073] innerCandidates195_15 7286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_16

def innerTail195_14 : List Nat := 6043 :: innerTail195_15
def innerCandidates195_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_14_checked : candidates edge 4 1213 6043 innerTail195_15 best.bound = innerCandidates195_14 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_14 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_14 best = best := by
  rw [innerTail195_14, scan, if_neg (by decide), innerCandidates195_14_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_15
    (if 3 ≤ innerCandidates195_14.length then search edge 3 [1213, 6043] innerCandidates195_14 7256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_15

def innerTail195_13 : List Nat := 5479 :: innerTail195_14
def innerCandidates195_13 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem innerCandidates195_13_checked : candidates edge 4 1213 5479 innerTail195_14 best.bound = innerCandidates195_13 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_13 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_13 best = best := by
  rw [innerTail195_13, scan, if_neg (by decide), innerCandidates195_13_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_14
    (if 3 ≤ innerCandidates195_13.length then search edge 3 [1213, 5479] innerCandidates195_13 6692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_14

def innerTail195_12 : List Nat := 5443 :: innerTail195_13
def innerCandidates195_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_12_checked : candidates edge 4 1213 5443 innerTail195_13 best.bound = innerCandidates195_12 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_12 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_12 best = best := by
  rw [innerTail195_12, scan, if_neg (by decide), innerCandidates195_12_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_13
    (if 3 ≤ innerCandidates195_12.length then search edge 3 [1213, 5443] innerCandidates195_12 6656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_13

def innerTail195_11 : List Nat := 4201 :: innerTail195_12
def innerCandidates195_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_11_checked : candidates edge 4 1213 4201 innerTail195_12 best.bound = innerCandidates195_11 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_11 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_11 best = best := by
  rw [innerTail195_11, scan, if_neg (by decide), innerCandidates195_11_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_12
    (if 3 ≤ innerCandidates195_11.length then search edge 3 [1213, 4201] innerCandidates195_11 5414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_12

def innerTail195_10 : List Nat := 3463 :: innerTail195_11
def innerCandidates195_10 : List Nat := [5479, 6211]
set_option maxRecDepth 4096 in
theorem innerCandidates195_10_checked : candidates edge 4 1213 3463 innerTail195_11 best.bound = innerCandidates195_10 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_10 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_10 best = best := by
  rw [innerTail195_10, scan, if_neg (by decide), innerCandidates195_10_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_11
    (if 3 ≤ innerCandidates195_10.length then search edge 3 [1213, 3463] innerCandidates195_10 4676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_11

def innerTail195_9 : List Nat := 3301 :: innerTail195_10
def innerCandidates195_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_9_checked : candidates edge 4 1213 3301 innerTail195_10 best.bound = innerCandidates195_9 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_9 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_9 best = best := by
  rw [innerTail195_9, scan, if_neg (by decide), innerCandidates195_9_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_10
    (if 3 ≤ innerCandidates195_9.length then search edge 3 [1213, 3301] innerCandidates195_9 4514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_10

def innerTail195_8 : List Nat := 3061 :: innerTail195_9
def innerCandidates195_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_8_checked : candidates edge 4 1213 3061 innerTail195_9 best.bound = innerCandidates195_8 := by
  unfold candidates
  have row : edge 3061 = SearchData.row3061.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_8 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_8 best = best := by
  rw [innerTail195_8, scan, if_neg (by decide), innerCandidates195_8_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_9
    (if 3 ≤ innerCandidates195_8.length then search edge 3 [1213, 3061] innerCandidates195_8 4274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_9

def innerTail195_7 : List Nat := 2851 :: innerTail195_8
def innerCandidates195_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_7_checked : candidates edge 4 1213 2851 innerTail195_8 best.bound = innerCandidates195_7 := by
  unfold candidates
  have row : edge 2851 = SearchData.row2851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_7 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_7 best = best := by
  rw [innerTail195_7, scan, if_neg (by decide), innerCandidates195_7_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_8
    (if 3 ≤ innerCandidates195_7.length then search edge 3 [1213, 2851] innerCandidates195_7 4064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_8

def innerTail195_6 : List Nat := 2797 :: innerTail195_7
def innerCandidates195_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_6_checked : candidates edge 4 1213 2797 innerTail195_7 best.bound = innerCandidates195_6 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_6 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_6 best = best := by
  rw [innerTail195_6, scan, if_neg (by decide), innerCandidates195_6_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_7
    (if 3 ≤ innerCandidates195_6.length then search edge 3 [1213, 2797] innerCandidates195_6 4010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_7

def innerTail195_5 : List Nat := 2371 :: innerTail195_6
def innerCandidates195_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_5_checked : candidates edge 4 1213 2371 innerTail195_6 best.bound = innerCandidates195_5 := by
  unfold candidates
  have row : edge 2371 = PrimePairCache.smallRow2371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_5 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_5 best = best := by
  rw [innerTail195_5, scan, if_neg (by decide), innerCandidates195_5_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_6
    (if 3 ≤ innerCandidates195_5.length then search edge 3 [1213, 2371] innerCandidates195_5 3584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_6

def innerTail195_4 : List Nat := 2053 :: innerTail195_5
def innerCandidates195_4 : List Nat := [2797]
set_option maxRecDepth 4096 in
theorem innerCandidates195_4_checked : candidates edge 4 1213 2053 innerTail195_5 best.bound = innerCandidates195_4 := by
  unfold candidates
  have row : edge 2053 = PrimePairCache.smallRow2053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_4 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_4 best = best := by
  rw [innerTail195_4, scan, if_neg (by decide), innerCandidates195_4_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_5
    (if 3 ≤ innerCandidates195_4.length then search edge 3 [1213, 2053] innerCandidates195_4 3266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_5

def innerTail195_3 : List Nat := 2011 :: innerTail195_4
def innerCandidates195_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_3_checked : candidates edge 4 1213 2011 innerTail195_4 best.bound = innerCandidates195_3 := by
  unfold candidates
  have row : edge 2011 = PrimePairCache.smallRow2011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_3 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_3 best = best := by
  rw [innerTail195_3, scan, if_neg (by decide), innerCandidates195_3_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_4
    (if 3 ≤ innerCandidates195_3.length then search edge 3 [1213, 2011] innerCandidates195_3 3224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_4

def innerTail195_2 : List Nat := 1789 :: innerTail195_3
def innerCandidates195_2 : List Nat := [6547]
set_option maxRecDepth 4096 in
theorem innerCandidates195_2_checked : candidates edge 4 1213 1789 innerTail195_3 best.bound = innerCandidates195_2 := by
  unfold candidates
  have row : edge 1789 = PrimePairCache.smallRow1789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_2 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_2 best = best := by
  rw [innerTail195_2, scan, if_neg (by decide), innerCandidates195_2_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_3
    (if 3 ≤ innerCandidates195_2.length then search edge 3 [1213, 1789] innerCandidates195_2 3002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_3

def innerTail195_1 : List Nat := 1621 :: innerTail195_2
def innerCandidates195_1 : List Nat := [2851, 5443, 6073]
set_option maxRecDepth 4096 in
theorem innerCandidates195_1_checked : candidates edge 4 1213 1621 innerTail195_2 best.bound = innerCandidates195_1 := by
  unfold candidates
  have row : edge 1621 = PrimePairCache.smallRow1621.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild195_1 : search edge 3 [1213, 1621] innerCandidates195_1 2834 best = best := by decide

theorem innerScan195_1 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_1 best = best := by
  rw [innerTail195_1, scan, if_neg (by decide), innerCandidates195_1_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_2
    (if 3 ≤ innerCandidates195_1.length then search edge 3 [1213, 1621] innerCandidates195_1 2834 best else best) = best
  rw [if_pos (by decide), innerChild195_1]
  exact innerScan195_2

def innerTail195_0 : List Nat := 1297 :: innerTail195_1
def innerCandidates195_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates195_0_checked : candidates edge 4 1213 1297 innerTail195_1 best.bound = innerCandidates195_0 := by
  unfold candidates
  have row : edge 1297 = PrimePairCache.smallRow1297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan195_0 : scan edge (search edge 3) 4 [1213] 1213 innerTail195_0 best = best := by
  rw [innerTail195_0, scan, if_neg (by decide), innerCandidates195_0_checked]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_1
    (if 3 ≤ innerCandidates195_0.length then search edge 3 [1213, 1297] innerCandidates195_0 2510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan195_1

theorem rootBranch195 : search edge 4 [1213] rootCandidates195 1213 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1213] 1213 innerTail195_0 best = best
  exact innerScan195_0

end ProjectEulerSolutions.PrimePairUpper.Search
