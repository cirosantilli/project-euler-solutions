import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates45 : List Nat := [229, 277, 337, 547, 829, 919, 1093, 1447, 2179, 2437, 2551, 3547, 4159, 4261, 5647, 6007, 6121, 6133, 6373, 6781, 6907, 7159, 7309, 7537, 7927]
set_option maxRecDepth 4096 in
theorem rootCandidates45_checked : candidates edge 5 0 223 tail46 best.bound = rootCandidates45 := by
  unfold candidates
  have row : edge 223 = PrimePairCache.smallRow223.contains := by funext q; rfl
  rw [row]
  decide

def innerTail45_25 : List Nat := []
theorem innerScan45_25 : scan edge (search edge 3) 4 [223] 223 innerTail45_25 best = best := rfl

def innerTail45_24 : List Nat := 7927 :: innerTail45_25
def innerCandidates45_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_24_checked : candidates edge 4 223 7927 innerTail45_25 best.bound = innerCandidates45_24 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_24 : scan edge (search edge 3) 4 [223] 223 innerTail45_24 best = best := by
  rw [innerTail45_24, scan, if_neg (by decide), innerCandidates45_24_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_25
    (if 3 ≤ innerCandidates45_24.length then search edge 3 [223, 7927] innerCandidates45_24 8150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_25

def innerTail45_23 : List Nat := 7537 :: innerTail45_24
def innerCandidates45_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_23_checked : candidates edge 4 223 7537 innerTail45_24 best.bound = innerCandidates45_23 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_23 : scan edge (search edge 3) 4 [223] 223 innerTail45_23 best = best := by
  rw [innerTail45_23, scan, if_neg (by decide), innerCandidates45_23_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_24
    (if 3 ≤ innerCandidates45_23.length then search edge 3 [223, 7537] innerCandidates45_23 7760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_24

def innerTail45_22 : List Nat := 7309 :: innerTail45_23
def innerCandidates45_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_22_checked : candidates edge 4 223 7309 innerTail45_23 best.bound = innerCandidates45_22 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_22 : scan edge (search edge 3) 4 [223] 223 innerTail45_22 best = best := by
  rw [innerTail45_22, scan, if_neg (by decide), innerCandidates45_22_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_23
    (if 3 ≤ innerCandidates45_22.length then search edge 3 [223, 7309] innerCandidates45_22 7532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_23

def innerTail45_21 : List Nat := 7159 :: innerTail45_22
def innerCandidates45_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_21_checked : candidates edge 4 223 7159 innerTail45_22 best.bound = innerCandidates45_21 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_21 : scan edge (search edge 3) 4 [223] 223 innerTail45_21 best = best := by
  rw [innerTail45_21, scan, if_neg (by decide), innerCandidates45_21_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_22
    (if 3 ≤ innerCandidates45_21.length then search edge 3 [223, 7159] innerCandidates45_21 7382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_22

def innerTail45_20 : List Nat := 6907 :: innerTail45_21
def innerCandidates45_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_20_checked : candidates edge 4 223 6907 innerTail45_21 best.bound = innerCandidates45_20 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_20 : scan edge (search edge 3) 4 [223] 223 innerTail45_20 best = best := by
  rw [innerTail45_20, scan, if_neg (by decide), innerCandidates45_20_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_21
    (if 3 ≤ innerCandidates45_20.length then search edge 3 [223, 6907] innerCandidates45_20 7130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_21

def innerTail45_19 : List Nat := 6781 :: innerTail45_20
def innerCandidates45_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_19_checked : candidates edge 4 223 6781 innerTail45_20 best.bound = innerCandidates45_19 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_19 : scan edge (search edge 3) 4 [223] 223 innerTail45_19 best = best := by
  rw [innerTail45_19, scan, if_neg (by decide), innerCandidates45_19_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_20
    (if 3 ≤ innerCandidates45_19.length then search edge 3 [223, 6781] innerCandidates45_19 7004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_20

def innerTail45_18 : List Nat := 6373 :: innerTail45_19
def innerCandidates45_18 : List Nat := [6781]
set_option maxRecDepth 4096 in
theorem innerCandidates45_18_checked : candidates edge 4 223 6373 innerTail45_19 best.bound = innerCandidates45_18 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_18 : scan edge (search edge 3) 4 [223] 223 innerTail45_18 best = best := by
  rw [innerTail45_18, scan, if_neg (by decide), innerCandidates45_18_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_19
    (if 3 ≤ innerCandidates45_18.length then search edge 3 [223, 6373] innerCandidates45_18 6596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_19

def innerTail45_17 : List Nat := 6133 :: innerTail45_18
def innerCandidates45_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_17_checked : candidates edge 4 223 6133 innerTail45_18 best.bound = innerCandidates45_17 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_17 : scan edge (search edge 3) 4 [223] 223 innerTail45_17 best = best := by
  rw [innerTail45_17, scan, if_neg (by decide), innerCandidates45_17_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_18
    (if 3 ≤ innerCandidates45_17.length then search edge 3 [223, 6133] innerCandidates45_17 6356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_18

def innerTail45_16 : List Nat := 6121 :: innerTail45_17
def innerCandidates45_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_16_checked : candidates edge 4 223 6121 innerTail45_17 best.bound = innerCandidates45_16 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_16 : scan edge (search edge 3) 4 [223] 223 innerTail45_16 best = best := by
  rw [innerTail45_16, scan, if_neg (by decide), innerCandidates45_16_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_17
    (if 3 ≤ innerCandidates45_16.length then search edge 3 [223, 6121] innerCandidates45_16 6344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_17

def innerTail45_15 : List Nat := 6007 :: innerTail45_16
def innerCandidates45_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_15_checked : candidates edge 4 223 6007 innerTail45_16 best.bound = innerCandidates45_15 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_15 : scan edge (search edge 3) 4 [223] 223 innerTail45_15 best = best := by
  rw [innerTail45_15, scan, if_neg (by decide), innerCandidates45_15_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_16
    (if 3 ≤ innerCandidates45_15.length then search edge 3 [223, 6007] innerCandidates45_15 6230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_16

def innerTail45_14 : List Nat := 5647 :: innerTail45_15
def innerCandidates45_14 : List Nat := [6373]
set_option maxRecDepth 4096 in
theorem innerCandidates45_14_checked : candidates edge 4 223 5647 innerTail45_15 best.bound = innerCandidates45_14 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_14 : scan edge (search edge 3) 4 [223] 223 innerTail45_14 best = best := by
  rw [innerTail45_14, scan, if_neg (by decide), innerCandidates45_14_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_15
    (if 3 ≤ innerCandidates45_14.length then search edge 3 [223, 5647] innerCandidates45_14 5870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_15

def innerTail45_13 : List Nat := 4261 :: innerTail45_14
def innerCandidates45_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_13_checked : candidates edge 4 223 4261 innerTail45_14 best.bound = innerCandidates45_13 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_13 : scan edge (search edge 3) 4 [223] 223 innerTail45_13 best = best := by
  rw [innerTail45_13, scan, if_neg (by decide), innerCandidates45_13_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_14
    (if 3 ≤ innerCandidates45_13.length then search edge 3 [223, 4261] innerCandidates45_13 4484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_14

def innerTail45_12 : List Nat := 4159 :: innerTail45_13
def innerCandidates45_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_12_checked : candidates edge 4 223 4159 innerTail45_13 best.bound = innerCandidates45_12 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_12 : scan edge (search edge 3) 4 [223] 223 innerTail45_12 best = best := by
  rw [innerTail45_12, scan, if_neg (by decide), innerCandidates45_12_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_13
    (if 3 ≤ innerCandidates45_12.length then search edge 3 [223, 4159] innerCandidates45_12 4382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_13

def innerTail45_11 : List Nat := 3547 :: innerTail45_12
def innerCandidates45_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_11_checked : candidates edge 4 223 3547 innerTail45_12 best.bound = innerCandidates45_11 := by
  unfold candidates
  have row : edge 3547 = SearchData.row3547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_11 : scan edge (search edge 3) 4 [223] 223 innerTail45_11 best = best := by
  rw [innerTail45_11, scan, if_neg (by decide), innerCandidates45_11_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_12
    (if 3 ≤ innerCandidates45_11.length then search edge 3 [223, 3547] innerCandidates45_11 3770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_12

def innerTail45_10 : List Nat := 2551 :: innerTail45_11
def innerCandidates45_10 : List Nat := [7537]
set_option maxRecDepth 4096 in
theorem innerCandidates45_10_checked : candidates edge 4 223 2551 innerTail45_11 best.bound = innerCandidates45_10 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_10 : scan edge (search edge 3) 4 [223] 223 innerTail45_10 best = best := by
  rw [innerTail45_10, scan, if_neg (by decide), innerCandidates45_10_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_11
    (if 3 ≤ innerCandidates45_10.length then search edge 3 [223, 2551] innerCandidates45_10 2774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_11

def innerTail45_9 : List Nat := 2437 :: innerTail45_10
def innerCandidates45_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_9_checked : candidates edge 4 223 2437 innerTail45_10 best.bound = innerCandidates45_9 := by
  unfold candidates
  have row : edge 2437 = PrimePairCache.smallRow2437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_9 : scan edge (search edge 3) 4 [223] 223 innerTail45_9 best = best := by
  rw [innerTail45_9, scan, if_neg (by decide), innerCandidates45_9_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_10
    (if 3 ≤ innerCandidates45_9.length then search edge 3 [223, 2437] innerCandidates45_9 2660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_10

def innerTail45_8 : List Nat := 2179 :: innerTail45_9
def innerCandidates45_8 : List Nat := [4261, 5647]
set_option maxRecDepth 4096 in
theorem innerCandidates45_8_checked : candidates edge 4 223 2179 innerTail45_9 best.bound = innerCandidates45_8 := by
  unfold candidates
  have row : edge 2179 = PrimePairCache.smallRow2179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_8 : scan edge (search edge 3) 4 [223] 223 innerTail45_8 best = best := by
  rw [innerTail45_8, scan, if_neg (by decide), innerCandidates45_8_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_9
    (if 3 ≤ innerCandidates45_8.length then search edge 3 [223, 2179] innerCandidates45_8 2402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_9

def innerTail45_7 : List Nat := 1447 :: innerTail45_8
def innerCandidates45_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_7_checked : candidates edge 4 223 1447 innerTail45_8 best.bound = innerCandidates45_7 := by
  unfold candidates
  have row : edge 1447 = PrimePairCache.smallRow1447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_7 : scan edge (search edge 3) 4 [223] 223 innerTail45_7 best = best := by
  rw [innerTail45_7, scan, if_neg (by decide), innerCandidates45_7_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_8
    (if 3 ≤ innerCandidates45_7.length then search edge 3 [223, 1447] innerCandidates45_7 1670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_8

def innerTail45_6 : List Nat := 1093 :: innerTail45_7
def innerCandidates45_6 : List Nat := [2437, 7309]
set_option maxRecDepth 4096 in
theorem innerCandidates45_6_checked : candidates edge 4 223 1093 innerTail45_7 best.bound = innerCandidates45_6 := by
  unfold candidates
  have row : edge 1093 = PrimePairCache.smallRow1093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_6 : scan edge (search edge 3) 4 [223] 223 innerTail45_6 best = best := by
  rw [innerTail45_6, scan, if_neg (by decide), innerCandidates45_6_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_7
    (if 3 ≤ innerCandidates45_6.length then search edge 3 [223, 1093] innerCandidates45_6 1316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_7

def innerTail45_5 : List Nat := 919 :: innerTail45_6
def innerCandidates45_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_5_checked : candidates edge 4 223 919 innerTail45_6 best.bound = innerCandidates45_5 := by
  unfold candidates
  have row : edge 919 = PrimePairCache.smallRow919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_5 : scan edge (search edge 3) 4 [223] 223 innerTail45_5 best = best := by
  rw [innerTail45_5, scan, if_neg (by decide), innerCandidates45_5_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_6
    (if 3 ≤ innerCandidates45_5.length then search edge 3 [223, 919] innerCandidates45_5 1142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_6

def innerTail45_4 : List Nat := 829 :: innerTail45_5
def innerCandidates45_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_4_checked : candidates edge 4 223 829 innerTail45_5 best.bound = innerCandidates45_4 := by
  unfold candidates
  have row : edge 829 = PrimePairCache.smallRow829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_4 : scan edge (search edge 3) 4 [223] 223 innerTail45_4 best = best := by
  rw [innerTail45_4, scan, if_neg (by decide), innerCandidates45_4_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_5
    (if 3 ≤ innerCandidates45_4.length then search edge 3 [223, 829] innerCandidates45_4 1052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_5

def innerTail45_3 : List Nat := 547 :: innerTail45_4
def innerCandidates45_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates45_3_checked : candidates edge 4 223 547 innerTail45_4 best.bound = innerCandidates45_3 := by
  unfold candidates
  have row : edge 547 = PrimePairCache.smallRow547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_3 : scan edge (search edge 3) 4 [223] 223 innerTail45_3 best = best := by
  rw [innerTail45_3, scan, if_neg (by decide), innerCandidates45_3_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_4
    (if 3 ≤ innerCandidates45_3.length then search edge 3 [223, 547] innerCandidates45_3 770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_4

def innerTail45_2 : List Nat := 337 :: innerTail45_3
def innerCandidates45_2 : List Nat := [919, 6781]
set_option maxRecDepth 4096 in
theorem innerCandidates45_2_checked : candidates edge 4 223 337 innerTail45_3 best.bound = innerCandidates45_2 := by
  unfold candidates
  have row : edge 337 = PrimePairCache.smallRow337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_2 : scan edge (search edge 3) 4 [223] 223 innerTail45_2 best = best := by
  rw [innerTail45_2, scan, if_neg (by decide), innerCandidates45_2_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_3
    (if 3 ≤ innerCandidates45_2.length then search edge 3 [223, 337] innerCandidates45_2 560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_3

def innerTail45_1 : List Nat := 277 :: innerTail45_2
def innerCandidates45_1 : List Nat := [2437]
set_option maxRecDepth 4096 in
theorem innerCandidates45_1_checked : candidates edge 4 223 277 innerTail45_2 best.bound = innerCandidates45_1 := by
  unfold candidates
  have row : edge 277 = PrimePairCache.smallRow277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_1 : scan edge (search edge 3) 4 [223] 223 innerTail45_1 best = best := by
  rw [innerTail45_1, scan, if_neg (by decide), innerCandidates45_1_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_2
    (if 3 ≤ innerCandidates45_1.length then search edge 3 [223, 277] innerCandidates45_1 500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_2

def innerTail45_0 : List Nat := 229 :: innerTail45_1
def innerCandidates45_0 : List Nat := [547]
set_option maxRecDepth 4096 in
theorem innerCandidates45_0_checked : candidates edge 4 223 229 innerTail45_1 best.bound = innerCandidates45_0 := by
  unfold candidates
  have row : edge 229 = PrimePairCache.smallRow229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan45_0 : scan edge (search edge 3) 4 [223] 223 innerTail45_0 best = best := by
  rw [innerTail45_0, scan, if_neg (by decide), innerCandidates45_0_checked]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_1
    (if 3 ≤ innerCandidates45_0.length then search edge 3 [223, 229] innerCandidates45_0 452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan45_1

theorem rootBranch45 : search edge 4 [223] rootCandidates45 223 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [223] 223 innerTail45_0 best = best
  exact innerScan45_0

def rootCandidates44 : List Nat := [271, 283, 313, 349, 373, 499, 571, 727, 1129, 1783, 1873, 2053, 2683, 2833, 3733, 4027, 4507, 4549, 5059, 5101, 5653, 5791, 6199, 6397, 7039, 8209, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates44_checked : candidates edge 5 0 211 tail45 best.bound = rootCandidates44 := by
  unfold candidates
  have row : edge 211 = PrimePairCache.smallRow211.contains := by funext q; rfl
  rw [row]
  decide

def innerTail44_27 : List Nat := []
theorem innerScan44_27 : scan edge (search edge 3) 4 [211] 211 innerTail44_27 best = best := rfl

def innerTail44_26 : List Nat := 8269 :: innerTail44_27
def innerCandidates44_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_26_checked : candidates edge 4 211 8269 innerTail44_27 best.bound = innerCandidates44_26 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_26 : scan edge (search edge 3) 4 [211] 211 innerTail44_26 best = best := by
  rw [innerTail44_26, scan, if_neg (by decide), innerCandidates44_26_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_27
    (if 3 ≤ innerCandidates44_26.length then search edge 3 [211, 8269] innerCandidates44_26 8480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_27

def innerTail44_25 : List Nat := 8209 :: innerTail44_26
def innerCandidates44_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_25_checked : candidates edge 4 211 8209 innerTail44_26 best.bound = innerCandidates44_25 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_25 : scan edge (search edge 3) 4 [211] 211 innerTail44_25 best = best := by
  rw [innerTail44_25, scan, if_neg (by decide), innerCandidates44_25_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_26
    (if 3 ≤ innerCandidates44_25.length then search edge 3 [211, 8209] innerCandidates44_25 8420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_26

def innerTail44_24 : List Nat := 7039 :: innerTail44_25
def innerCandidates44_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_24_checked : candidates edge 4 211 7039 innerTail44_25 best.bound = innerCandidates44_24 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_24 : scan edge (search edge 3) 4 [211] 211 innerTail44_24 best = best := by
  rw [innerTail44_24, scan, if_neg (by decide), innerCandidates44_24_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_25
    (if 3 ≤ innerCandidates44_24.length then search edge 3 [211, 7039] innerCandidates44_24 7250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_25

def innerTail44_23 : List Nat := 6397 :: innerTail44_24
def innerCandidates44_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_23_checked : candidates edge 4 211 6397 innerTail44_24 best.bound = innerCandidates44_23 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_23 : scan edge (search edge 3) 4 [211] 211 innerTail44_23 best = best := by
  rw [innerTail44_23, scan, if_neg (by decide), innerCandidates44_23_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_24
    (if 3 ≤ innerCandidates44_23.length then search edge 3 [211, 6397] innerCandidates44_23 6608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_24

def innerTail44_22 : List Nat := 6199 :: innerTail44_23
def innerCandidates44_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_22_checked : candidates edge 4 211 6199 innerTail44_23 best.bound = innerCandidates44_22 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_22 : scan edge (search edge 3) 4 [211] 211 innerTail44_22 best = best := by
  rw [innerTail44_22, scan, if_neg (by decide), innerCandidates44_22_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_23
    (if 3 ≤ innerCandidates44_22.length then search edge 3 [211, 6199] innerCandidates44_22 6410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_23

def innerTail44_21 : List Nat := 5791 :: innerTail44_22
def innerCandidates44_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_21_checked : candidates edge 4 211 5791 innerTail44_22 best.bound = innerCandidates44_21 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_21 : scan edge (search edge 3) 4 [211] 211 innerTail44_21 best = best := by
  rw [innerTail44_21, scan, if_neg (by decide), innerCandidates44_21_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_22
    (if 3 ≤ innerCandidates44_21.length then search edge 3 [211, 5791] innerCandidates44_21 6002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_22

def innerTail44_20 : List Nat := 5653 :: innerTail44_21
def innerCandidates44_20 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates44_20_checked : candidates edge 4 211 5653 innerTail44_21 best.bound = innerCandidates44_20 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_20 : scan edge (search edge 3) 4 [211] 211 innerTail44_20 best = best := by
  rw [innerTail44_20, scan, if_neg (by decide), innerCandidates44_20_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_21
    (if 3 ≤ innerCandidates44_20.length then search edge 3 [211, 5653] innerCandidates44_20 5864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_21

def innerTail44_19 : List Nat := 5101 :: innerTail44_20
def innerCandidates44_19 : List Nat := [6397]
set_option maxRecDepth 4096 in
theorem innerCandidates44_19_checked : candidates edge 4 211 5101 innerTail44_20 best.bound = innerCandidates44_19 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_19 : scan edge (search edge 3) 4 [211] 211 innerTail44_19 best = best := by
  rw [innerTail44_19, scan, if_neg (by decide), innerCandidates44_19_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_20
    (if 3 ≤ innerCandidates44_19.length then search edge 3 [211, 5101] innerCandidates44_19 5312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_20

def innerTail44_18 : List Nat := 5059 :: innerTail44_19
def innerCandidates44_18 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates44_18_checked : candidates edge 4 211 5059 innerTail44_19 best.bound = innerCandidates44_18 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_18 : scan edge (search edge 3) 4 [211] 211 innerTail44_18 best = best := by
  rw [innerTail44_18, scan, if_neg (by decide), innerCandidates44_18_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_19
    (if 3 ≤ innerCandidates44_18.length then search edge 3 [211, 5059] innerCandidates44_18 5270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_19

def innerTail44_17 : List Nat := 4549 :: innerTail44_18
def innerCandidates44_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_17_checked : candidates edge 4 211 4549 innerTail44_18 best.bound = innerCandidates44_17 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_17 : scan edge (search edge 3) 4 [211] 211 innerTail44_17 best = best := by
  rw [innerTail44_17, scan, if_neg (by decide), innerCandidates44_17_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_18
    (if 3 ≤ innerCandidates44_17.length then search edge 3 [211, 4549] innerCandidates44_17 4760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_18

def innerTail44_16 : List Nat := 4507 :: innerTail44_17
def innerCandidates44_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_16_checked : candidates edge 4 211 4507 innerTail44_17 best.bound = innerCandidates44_16 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_16 : scan edge (search edge 3) 4 [211] 211 innerTail44_16 best = best := by
  rw [innerTail44_16, scan, if_neg (by decide), innerCandidates44_16_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_17
    (if 3 ≤ innerCandidates44_16.length then search edge 3 [211, 4507] innerCandidates44_16 4718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_17

def innerTail44_15 : List Nat := 4027 :: innerTail44_16
def innerCandidates44_15 : List Nat := [6199]
set_option maxRecDepth 4096 in
theorem innerCandidates44_15_checked : candidates edge 4 211 4027 innerTail44_16 best.bound = innerCandidates44_15 := by
  unfold candidates
  have row : edge 4027 = SearchData.row4027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_15 : scan edge (search edge 3) 4 [211] 211 innerTail44_15 best = best := by
  rw [innerTail44_15, scan, if_neg (by decide), innerCandidates44_15_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_16
    (if 3 ≤ innerCandidates44_15.length then search edge 3 [211, 4027] innerCandidates44_15 4238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_16

def innerTail44_14 : List Nat := 3733 :: innerTail44_15
def innerCandidates44_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_14_checked : candidates edge 4 211 3733 innerTail44_15 best.bound = innerCandidates44_14 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_14 : scan edge (search edge 3) 4 [211] 211 innerTail44_14 best = best := by
  rw [innerTail44_14, scan, if_neg (by decide), innerCandidates44_14_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_15
    (if 3 ≤ innerCandidates44_14.length then search edge 3 [211, 3733] innerCandidates44_14 3944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_15

def innerTail44_13 : List Nat := 2833 :: innerTail44_14
def innerCandidates44_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_13_checked : candidates edge 4 211 2833 innerTail44_14 best.bound = innerCandidates44_13 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_13 : scan edge (search edge 3) 4 [211] 211 innerTail44_13 best = best := by
  rw [innerTail44_13, scan, if_neg (by decide), innerCandidates44_13_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_14
    (if 3 ≤ innerCandidates44_13.length then search edge 3 [211, 2833] innerCandidates44_13 3044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_14

def innerTail44_12 : List Nat := 2683 :: innerTail44_13
def innerCandidates44_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_12_checked : candidates edge 4 211 2683 innerTail44_13 best.bound = innerCandidates44_12 := by
  unfold candidates
  have row : edge 2683 = SearchData.row2683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_12 : scan edge (search edge 3) 4 [211] 211 innerTail44_12 best = best := by
  rw [innerTail44_12, scan, if_neg (by decide), innerCandidates44_12_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_13
    (if 3 ≤ innerCandidates44_12.length then search edge 3 [211, 2683] innerCandidates44_12 2894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_13

def innerTail44_11 : List Nat := 2053 :: innerTail44_12
def innerCandidates44_11 : List Nat := [5653]
set_option maxRecDepth 4096 in
theorem innerCandidates44_11_checked : candidates edge 4 211 2053 innerTail44_12 best.bound = innerCandidates44_11 := by
  unfold candidates
  have row : edge 2053 = PrimePairCache.smallRow2053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_11 : scan edge (search edge 3) 4 [211] 211 innerTail44_11 best = best := by
  rw [innerTail44_11, scan, if_neg (by decide), innerCandidates44_11_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_12
    (if 3 ≤ innerCandidates44_11.length then search edge 3 [211, 2053] innerCandidates44_11 2264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_12

def innerTail44_10 : List Nat := 1873 :: innerTail44_11
def innerCandidates44_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_10_checked : candidates edge 4 211 1873 innerTail44_11 best.bound = innerCandidates44_10 := by
  unfold candidates
  have row : edge 1873 = PrimePairCache.smallRow1873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_10 : scan edge (search edge 3) 4 [211] 211 innerTail44_10 best = best := by
  rw [innerTail44_10, scan, if_neg (by decide), innerCandidates44_10_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_11
    (if 3 ≤ innerCandidates44_10.length then search edge 3 [211, 1873] innerCandidates44_10 2084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_11

def innerTail44_9 : List Nat := 1783 :: innerTail44_10
def innerCandidates44_9 : List Nat := [2053, 8209]
set_option maxRecDepth 4096 in
theorem innerCandidates44_9_checked : candidates edge 4 211 1783 innerTail44_10 best.bound = innerCandidates44_9 := by
  unfold candidates
  have row : edge 1783 = PrimePairCache.smallRow1783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_9 : scan edge (search edge 3) 4 [211] 211 innerTail44_9 best = best := by
  rw [innerTail44_9, scan, if_neg (by decide), innerCandidates44_9_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_10
    (if 3 ≤ innerCandidates44_9.length then search edge 3 [211, 1783] innerCandidates44_9 1994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_10

def innerTail44_8 : List Nat := 1129 :: innerTail44_9
def innerCandidates44_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_8_checked : candidates edge 4 211 1129 innerTail44_9 best.bound = innerCandidates44_8 := by
  unfold candidates
  have row : edge 1129 = PrimePairCache.smallRow1129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_8 : scan edge (search edge 3) 4 [211] 211 innerTail44_8 best = best := by
  rw [innerTail44_8, scan, if_neg (by decide), innerCandidates44_8_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_9
    (if 3 ≤ innerCandidates44_8.length then search edge 3 [211, 1129] innerCandidates44_8 1340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_9

def innerTail44_7 : List Nat := 727 :: innerTail44_8
def innerCandidates44_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_7_checked : candidates edge 4 211 727 innerTail44_8 best.bound = innerCandidates44_7 := by
  unfold candidates
  have row : edge 727 = PrimePairCache.smallRow727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_7 : scan edge (search edge 3) 4 [211] 211 innerTail44_7 best = best := by
  rw [innerTail44_7, scan, if_neg (by decide), innerCandidates44_7_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_8
    (if 3 ≤ innerCandidates44_7.length then search edge 3 [211, 727] innerCandidates44_7 938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_8

def innerTail44_6 : List Nat := 571 :: innerTail44_7
def innerCandidates44_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_6_checked : candidates edge 4 211 571 innerTail44_7 best.bound = innerCandidates44_6 := by
  unfold candidates
  have row : edge 571 = PrimePairCache.smallRow571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_6 : scan edge (search edge 3) 4 [211] 211 innerTail44_6 best = best := by
  rw [innerTail44_6, scan, if_neg (by decide), innerCandidates44_6_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_7
    (if 3 ≤ innerCandidates44_6.length then search edge 3 [211, 571] innerCandidates44_6 782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_7

def innerTail44_5 : List Nat := 499 :: innerTail44_6
def innerCandidates44_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates44_5_checked : candidates edge 4 211 499 innerTail44_6 best.bound = innerCandidates44_5 := by
  unfold candidates
  have row : edge 499 = PrimePairCache.smallRow499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_5 : scan edge (search edge 3) 4 [211] 211 innerTail44_5 best = best := by
  rw [innerTail44_5, scan, if_neg (by decide), innerCandidates44_5_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_6
    (if 3 ≤ innerCandidates44_5.length then search edge 3 [211, 499] innerCandidates44_5 710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_6

def innerTail44_4 : List Nat := 373 :: innerTail44_5
def innerCandidates44_4 : List Nat := [5101]
set_option maxRecDepth 4096 in
theorem innerCandidates44_4_checked : candidates edge 4 211 373 innerTail44_5 best.bound = innerCandidates44_4 := by
  unfold candidates
  have row : edge 373 = PrimePairCache.smallRow373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_4 : scan edge (search edge 3) 4 [211] 211 innerTail44_4 best = best := by
  rw [innerTail44_4, scan, if_neg (by decide), innerCandidates44_4_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_5
    (if 3 ≤ innerCandidates44_4.length then search edge 3 [211, 373] innerCandidates44_4 584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_5

def innerTail44_3 : List Nat := 349 :: innerTail44_4
def innerCandidates44_3 : List Nat := [373, 499]
set_option maxRecDepth 4096 in
theorem innerCandidates44_3_checked : candidates edge 4 211 349 innerTail44_4 best.bound = innerCandidates44_3 := by
  unfold candidates
  have row : edge 349 = PrimePairCache.smallRow349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_3 : scan edge (search edge 3) 4 [211] 211 innerTail44_3 best = best := by
  rw [innerTail44_3, scan, if_neg (by decide), innerCandidates44_3_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_4
    (if 3 ≤ innerCandidates44_3.length then search edge 3 [211, 349] innerCandidates44_3 560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_4

def innerTail44_2 : List Nat := 313 :: innerTail44_3
def innerCandidates44_2 : List Nat := [727, 3733]
set_option maxRecDepth 4096 in
theorem innerCandidates44_2_checked : candidates edge 4 211 313 innerTail44_3 best.bound = innerCandidates44_2 := by
  unfold candidates
  have row : edge 313 = PrimePairCache.smallRow313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_2 : scan edge (search edge 3) 4 [211] 211 innerTail44_2 best = best := by
  rw [innerTail44_2, scan, if_neg (by decide), innerCandidates44_2_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_3
    (if 3 ≤ innerCandidates44_2.length then search edge 3 [211, 313] innerCandidates44_2 524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_3

def innerTail44_1 : List Nat := 283 :: innerTail44_2
def innerCandidates44_1 : List Nat := [5653, 6397, 7039]
set_option maxRecDepth 4096 in
theorem innerCandidates44_1_checked : candidates edge 4 211 283 innerTail44_2 best.bound = innerCandidates44_1 := by
  unfold candidates
  have row : edge 283 = PrimePairCache.smallRow283.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild44_1 : search edge 3 [211, 283] innerCandidates44_1 494 best = best := by decide

theorem innerScan44_1 : scan edge (search edge 3) 4 [211] 211 innerTail44_1 best = best := by
  rw [innerTail44_1, scan, if_neg (by decide), innerCandidates44_1_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_2
    (if 3 ≤ innerCandidates44_1.length then search edge 3 [211, 283] innerCandidates44_1 494 best else best) = best
  rw [if_pos (by decide), innerChild44_1]
  exact innerScan44_2

def innerTail44_0 : List Nat := 271 :: innerTail44_1
def innerCandidates44_0 : List Nat := [1873, 8269]
set_option maxRecDepth 4096 in
theorem innerCandidates44_0_checked : candidates edge 4 211 271 innerTail44_1 best.bound = innerCandidates44_0 := by
  unfold candidates
  have row : edge 271 = PrimePairCache.smallRow271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan44_0 : scan edge (search edge 3) 4 [211] 211 innerTail44_0 best = best := by
  rw [innerTail44_0, scan, if_neg (by decide), innerCandidates44_0_checked]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_1
    (if 3 ≤ innerCandidates44_0.length then search edge 3 [211, 271] innerCandidates44_0 482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan44_1

theorem rootBranch44 : search edge 4 [211] rootCandidates44 211 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [211] 211 innerTail44_0 best = best
  exact innerScan44_0

end ProjectEulerSolutions.PrimePairUpper.Search
