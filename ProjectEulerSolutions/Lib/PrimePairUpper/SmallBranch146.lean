import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates239 : List Nat := [1987, 2467, 2677, 3169, 4153, 4177, 5059, 5119, 5281, 5647, 6211, 6361, 6421, 7213, 7537, 7681, 7717, 7789, 7927, 8311]
set_option maxRecDepth 4096 in
theorem rootCandidates239_checked : candidates edge 5 0 1531 tail240 best.bound = rootCandidates239 := by
  unfold candidates
  have row : edge 1531 = PrimePairCache.smallRow1531.contains := by funext q; rfl
  rw [row]
  decide

def innerTail239_20 : List Nat := []
theorem innerScan239_20 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_20 best = best := rfl

def innerTail239_19 : List Nat := 8311 :: innerTail239_20
def innerCandidates239_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_19_checked : candidates edge 4 1531 8311 innerTail239_20 best.bound = innerCandidates239_19 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_19 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_19 best = best := by
  rw [innerTail239_19, scan, if_neg (by decide), innerCandidates239_19_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_20
    (if 3 ≤ innerCandidates239_19.length then search edge 3 [1531, 8311] innerCandidates239_19 9842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_20

def innerTail239_18 : List Nat := 7927 :: innerTail239_19
def innerCandidates239_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_18_checked : candidates edge 4 1531 7927 innerTail239_19 best.bound = innerCandidates239_18 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_18 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_18 best = best := by
  rw [innerTail239_18, scan, if_neg (by decide), innerCandidates239_18_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_19
    (if 3 ≤ innerCandidates239_18.length then search edge 3 [1531, 7927] innerCandidates239_18 9458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_19

def innerTail239_17 : List Nat := 7789 :: innerTail239_18
def innerCandidates239_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_17_checked : candidates edge 4 1531 7789 innerTail239_18 best.bound = innerCandidates239_17 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_17 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_17 best = best := by
  rw [innerTail239_17, scan, if_neg (by decide), innerCandidates239_17_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_18
    (if 3 ≤ innerCandidates239_17.length then search edge 3 [1531, 7789] innerCandidates239_17 9320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_18

def innerTail239_16 : List Nat := 7717 :: innerTail239_17
def innerCandidates239_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_16_checked : candidates edge 4 1531 7717 innerTail239_17 best.bound = innerCandidates239_16 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_16 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_16 best = best := by
  rw [innerTail239_16, scan, if_neg (by decide), innerCandidates239_16_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_17
    (if 3 ≤ innerCandidates239_16.length then search edge 3 [1531, 7717] innerCandidates239_16 9248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_17

def innerTail239_15 : List Nat := 7681 :: innerTail239_16
def innerCandidates239_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_15_checked : candidates edge 4 1531 7681 innerTail239_16 best.bound = innerCandidates239_15 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_15 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_15 best = best := by
  rw [innerTail239_15, scan, if_neg (by decide), innerCandidates239_15_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_16
    (if 3 ≤ innerCandidates239_15.length then search edge 3 [1531, 7681] innerCandidates239_15 9212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_16

def innerTail239_14 : List Nat := 7537 :: innerTail239_15
def innerCandidates239_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_14_checked : candidates edge 4 1531 7537 innerTail239_15 best.bound = innerCandidates239_14 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_14 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_14 best = best := by
  rw [innerTail239_14, scan, if_neg (by decide), innerCandidates239_14_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_15
    (if 3 ≤ innerCandidates239_14.length then search edge 3 [1531, 7537] innerCandidates239_14 9068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_15

def innerTail239_13 : List Nat := 7213 :: innerTail239_14
def innerCandidates239_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_13_checked : candidates edge 4 1531 7213 innerTail239_14 best.bound = innerCandidates239_13 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_13 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_13 best = best := by
  rw [innerTail239_13, scan, if_neg (by decide), innerCandidates239_13_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_14
    (if 3 ≤ innerCandidates239_13.length then search edge 3 [1531, 7213] innerCandidates239_13 8744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_14

def innerTail239_12 : List Nat := 6421 :: innerTail239_13
def innerCandidates239_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_12_checked : candidates edge 4 1531 6421 innerTail239_13 best.bound = innerCandidates239_12 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_12 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_12 best = best := by
  rw [innerTail239_12, scan, if_neg (by decide), innerCandidates239_12_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_13
    (if 3 ≤ innerCandidates239_12.length then search edge 3 [1531, 6421] innerCandidates239_12 7952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_13

def innerTail239_11 : List Nat := 6361 :: innerTail239_12
def innerCandidates239_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_11_checked : candidates edge 4 1531 6361 innerTail239_12 best.bound = innerCandidates239_11 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_11 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_11 best = best := by
  rw [innerTail239_11, scan, if_neg (by decide), innerCandidates239_11_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_12
    (if 3 ≤ innerCandidates239_11.length then search edge 3 [1531, 6361] innerCandidates239_11 7892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_12

def innerTail239_10 : List Nat := 6211 :: innerTail239_11
def innerCandidates239_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_10_checked : candidates edge 4 1531 6211 innerTail239_11 best.bound = innerCandidates239_10 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_10 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_10 best = best := by
  rw [innerTail239_10, scan, if_neg (by decide), innerCandidates239_10_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_11
    (if 3 ≤ innerCandidates239_10.length then search edge 3 [1531, 6211] innerCandidates239_10 7742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_11

def innerTail239_9 : List Nat := 5647 :: innerTail239_10
def innerCandidates239_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_9_checked : candidates edge 4 1531 5647 innerTail239_10 best.bound = innerCandidates239_9 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_9 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_9 best = best := by
  rw [innerTail239_9, scan, if_neg (by decide), innerCandidates239_9_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_10
    (if 3 ≤ innerCandidates239_9.length then search edge 3 [1531, 5647] innerCandidates239_9 7178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_10

def innerTail239_8 : List Nat := 5281 :: innerTail239_9
def innerCandidates239_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_8_checked : candidates edge 4 1531 5281 innerTail239_9 best.bound = innerCandidates239_8 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_8 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_8 best = best := by
  rw [innerTail239_8, scan, if_neg (by decide), innerCandidates239_8_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_9
    (if 3 ≤ innerCandidates239_8.length then search edge 3 [1531, 5281] innerCandidates239_8 6812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_9

def innerTail239_7 : List Nat := 5119 :: innerTail239_8
def innerCandidates239_7 : List Nat := [7537]
set_option maxRecDepth 4096 in
theorem innerCandidates239_7_checked : candidates edge 4 1531 5119 innerTail239_8 best.bound = innerCandidates239_7 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_7 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_7 best = best := by
  rw [innerTail239_7, scan, if_neg (by decide), innerCandidates239_7_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_8
    (if 3 ≤ innerCandidates239_7.length then search edge 3 [1531, 5119] innerCandidates239_7 6650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_8

def innerTail239_6 : List Nat := 5059 :: innerTail239_7
def innerCandidates239_6 : List Nat := [7213]
set_option maxRecDepth 4096 in
theorem innerCandidates239_6_checked : candidates edge 4 1531 5059 innerTail239_7 best.bound = innerCandidates239_6 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_6 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_6 best = best := by
  rw [innerTail239_6, scan, if_neg (by decide), innerCandidates239_6_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_7
    (if 3 ≤ innerCandidates239_6.length then search edge 3 [1531, 5059] innerCandidates239_6 6590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_7

def innerTail239_5 : List Nat := 4177 :: innerTail239_6
def innerCandidates239_5 : List Nat := [7537, 8311]
set_option maxRecDepth 4096 in
theorem innerCandidates239_5_checked : candidates edge 4 1531 4177 innerTail239_6 best.bound = innerCandidates239_5 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_5 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_5 best = best := by
  rw [innerTail239_5, scan, if_neg (by decide), innerCandidates239_5_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_6
    (if 3 ≤ innerCandidates239_5.length then search edge 3 [1531, 4177] innerCandidates239_5 5708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_6

def innerTail239_4 : List Nat := 4153 :: innerTail239_5
def innerCandidates239_4 : List Nat := [8311]
set_option maxRecDepth 4096 in
theorem innerCandidates239_4_checked : candidates edge 4 1531 4153 innerTail239_5 best.bound = innerCandidates239_4 := by
  unfold candidates
  have row : edge 4153 = SearchData.row4153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_4 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_4 best = best := by
  rw [innerTail239_4, scan, if_neg (by decide), innerCandidates239_4_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_5
    (if 3 ≤ innerCandidates239_4.length then search edge 3 [1531, 4153] innerCandidates239_4 5684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_5

def innerTail239_3 : List Nat := 3169 :: innerTail239_4
def innerCandidates239_3 : List Nat := [5119, 6361, 7213]
set_option maxRecDepth 4096 in
theorem innerCandidates239_3_checked : candidates edge 4 1531 3169 innerTail239_4 best.bound = innerCandidates239_3 := by
  unfold candidates
  have row : edge 3169 = SearchData.row3169.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild239_3 : search edge 3 [1531, 3169] innerCandidates239_3 4700 best = best := by decide

theorem innerScan239_3 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_3 best = best := by
  rw [innerTail239_3, scan, if_neg (by decide), innerCandidates239_3_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_4
    (if 3 ≤ innerCandidates239_3.length then search edge 3 [1531, 3169] innerCandidates239_3 4700 best else best) = best
  rw [if_pos (by decide), innerChild239_3]
  exact innerScan239_4

def innerTail239_2 : List Nat := 2677 :: innerTail239_3
def innerCandidates239_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_2_checked : candidates edge 4 1531 2677 innerTail239_3 best.bound = innerCandidates239_2 := by
  unfold candidates
  have row : edge 2677 = SearchData.row2677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_2 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_2 best = best := by
  rw [innerTail239_2, scan, if_neg (by decide), innerCandidates239_2_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_3
    (if 3 ≤ innerCandidates239_2.length then search edge 3 [1531, 2677] innerCandidates239_2 4208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_3

def innerTail239_1 : List Nat := 2467 :: innerTail239_2
def innerCandidates239_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates239_1_checked : candidates edge 4 1531 2467 innerTail239_2 best.bound = innerCandidates239_1 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_1 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_1 best = best := by
  rw [innerTail239_1, scan, if_neg (by decide), innerCandidates239_1_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_2
    (if 3 ≤ innerCandidates239_1.length then search edge 3 [1531, 2467] innerCandidates239_1 3998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_2

def innerTail239_0 : List Nat := 1987 :: innerTail239_1
def innerCandidates239_0 : List Nat := [2467]
set_option maxRecDepth 4096 in
theorem innerCandidates239_0_checked : candidates edge 4 1531 1987 innerTail239_1 best.bound = innerCandidates239_0 := by
  unfold candidates
  have row : edge 1987 = PrimePairCache.smallRow1987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan239_0 : scan edge (search edge 3) 4 [1531] 1531 innerTail239_0 best = best := by
  rw [innerTail239_0, scan, if_neg (by decide), innerCandidates239_0_checked]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_1
    (if 3 ≤ innerCandidates239_0.length then search edge 3 [1531, 1987] innerCandidates239_0 3518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan239_1

theorem rootBranch239 : search edge 4 [1531] rootCandidates239 1531 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1531] 1531 innerTail239_0 best = best
  exact innerScan239_0

def rootCandidates238 : List Nat := [1913, 2111, 2447, 2579, 2909, 2957, 2999, 3203, 3299, 3461, 3761, 3833, 3929, 4013, 4139, 4253, 4517, 4679, 5279, 5303, 5309, 5483, 5573, 5813, 6311, 6827, 7043, 7583, 7907]
set_option maxRecDepth 4096 in
theorem rootCandidates238_checked : candidates edge 5 0 1523 tail239 best.bound = rootCandidates238 := by
  unfold candidates
  have row : edge 1523 = PrimePairCache.smallRow1523.contains := by funext q; rfl
  rw [row]
  decide

def innerTail238_29 : List Nat := []
theorem innerScan238_29 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_29 best = best := rfl

def innerTail238_28 : List Nat := 7907 :: innerTail238_29
def innerCandidates238_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_28_checked : candidates edge 4 1523 7907 innerTail238_29 best.bound = innerCandidates238_28 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_28 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_28 best = best := by
  rw [innerTail238_28, scan, if_neg (by decide), innerCandidates238_28_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_29
    (if 3 ≤ innerCandidates238_28.length then search edge 3 [1523, 7907] innerCandidates238_28 9430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_29

def innerTail238_27 : List Nat := 7583 :: innerTail238_28
def innerCandidates238_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_27_checked : candidates edge 4 1523 7583 innerTail238_28 best.bound = innerCandidates238_27 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_27 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_27 best = best := by
  rw [innerTail238_27, scan, if_neg (by decide), innerCandidates238_27_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_28
    (if 3 ≤ innerCandidates238_27.length then search edge 3 [1523, 7583] innerCandidates238_27 9106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_28

def innerTail238_26 : List Nat := 7043 :: innerTail238_27
def innerCandidates238_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_26_checked : candidates edge 4 1523 7043 innerTail238_27 best.bound = innerCandidates238_26 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_26 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_26 best = best := by
  rw [innerTail238_26, scan, if_neg (by decide), innerCandidates238_26_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_27
    (if 3 ≤ innerCandidates238_26.length then search edge 3 [1523, 7043] innerCandidates238_26 8566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_27

def innerTail238_25 : List Nat := 6827 :: innerTail238_26
def innerCandidates238_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_25_checked : candidates edge 4 1523 6827 innerTail238_26 best.bound = innerCandidates238_25 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_25 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_25 best = best := by
  rw [innerTail238_25, scan, if_neg (by decide), innerCandidates238_25_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_26
    (if 3 ≤ innerCandidates238_25.length then search edge 3 [1523, 6827] innerCandidates238_25 8350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_26

def innerTail238_24 : List Nat := 6311 :: innerTail238_25
def innerCandidates238_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_24_checked : candidates edge 4 1523 6311 innerTail238_25 best.bound = innerCandidates238_24 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_24 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_24 best = best := by
  rw [innerTail238_24, scan, if_neg (by decide), innerCandidates238_24_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_25
    (if 3 ≤ innerCandidates238_24.length then search edge 3 [1523, 6311] innerCandidates238_24 7834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_25

def innerTail238_23 : List Nat := 5813 :: innerTail238_24
def innerCandidates238_23 : List Nat := [7907]
set_option maxRecDepth 4096 in
theorem innerCandidates238_23_checked : candidates edge 4 1523 5813 innerTail238_24 best.bound = innerCandidates238_23 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_23 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_23 best = best := by
  rw [innerTail238_23, scan, if_neg (by decide), innerCandidates238_23_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_24
    (if 3 ≤ innerCandidates238_23.length then search edge 3 [1523, 5813] innerCandidates238_23 7336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_24

def innerTail238_22 : List Nat := 5573 :: innerTail238_23
def innerCandidates238_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_22_checked : candidates edge 4 1523 5573 innerTail238_23 best.bound = innerCandidates238_22 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_22 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_22 best = best := by
  rw [innerTail238_22, scan, if_neg (by decide), innerCandidates238_22_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_23
    (if 3 ≤ innerCandidates238_22.length then search edge 3 [1523, 5573] innerCandidates238_22 7096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_23

def innerTail238_21 : List Nat := 5483 :: innerTail238_22
def innerCandidates238_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_21_checked : candidates edge 4 1523 5483 innerTail238_22 best.bound = innerCandidates238_21 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_21 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_21 best = best := by
  rw [innerTail238_21, scan, if_neg (by decide), innerCandidates238_21_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_22
    (if 3 ≤ innerCandidates238_21.length then search edge 3 [1523, 5483] innerCandidates238_21 7006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_22

def innerTail238_20 : List Nat := 5309 :: innerTail238_21
def innerCandidates238_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_20_checked : candidates edge 4 1523 5309 innerTail238_21 best.bound = innerCandidates238_20 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_20 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_20 best = best := by
  rw [innerTail238_20, scan, if_neg (by decide), innerCandidates238_20_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_21
    (if 3 ≤ innerCandidates238_20.length then search edge 3 [1523, 5309] innerCandidates238_20 6832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_21

def innerTail238_19 : List Nat := 5303 :: innerTail238_20
def innerCandidates238_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_19_checked : candidates edge 4 1523 5303 innerTail238_20 best.bound = innerCandidates238_19 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_19 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_19 best = best := by
  rw [innerTail238_19, scan, if_neg (by decide), innerCandidates238_19_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_20
    (if 3 ≤ innerCandidates238_19.length then search edge 3 [1523, 5303] innerCandidates238_19 6826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_20

def innerTail238_18 : List Nat := 5279 :: innerTail238_19
def innerCandidates238_18 : List Nat := [6827]
set_option maxRecDepth 4096 in
theorem innerCandidates238_18_checked : candidates edge 4 1523 5279 innerTail238_19 best.bound = innerCandidates238_18 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_18 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_18 best = best := by
  rw [innerTail238_18, scan, if_neg (by decide), innerCandidates238_18_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_19
    (if 3 ≤ innerCandidates238_18.length then search edge 3 [1523, 5279] innerCandidates238_18 6802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_19

def innerTail238_17 : List Nat := 4679 :: innerTail238_18
def innerCandidates238_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_17_checked : candidates edge 4 1523 4679 innerTail238_18 best.bound = innerCandidates238_17 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_17 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_17 best = best := by
  rw [innerTail238_17, scan, if_neg (by decide), innerCandidates238_17_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_18
    (if 3 ≤ innerCandidates238_17.length then search edge 3 [1523, 4679] innerCandidates238_17 6202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_18

def innerTail238_16 : List Nat := 4517 :: innerTail238_17
def innerCandidates238_16 : List Nat := [5279]
set_option maxRecDepth 4096 in
theorem innerCandidates238_16_checked : candidates edge 4 1523 4517 innerTail238_17 best.bound = innerCandidates238_16 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_16 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_16 best = best := by
  rw [innerTail238_16, scan, if_neg (by decide), innerCandidates238_16_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_17
    (if 3 ≤ innerCandidates238_16.length then search edge 3 [1523, 4517] innerCandidates238_16 6040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_17

def innerTail238_15 : List Nat := 4253 :: innerTail238_16
def innerCandidates238_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_15_checked : candidates edge 4 1523 4253 innerTail238_16 best.bound = innerCandidates238_15 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_15 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_15 best = best := by
  rw [innerTail238_15, scan, if_neg (by decide), innerCandidates238_15_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_16
    (if 3 ≤ innerCandidates238_15.length then search edge 3 [1523, 4253] innerCandidates238_15 5776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_16

def innerTail238_14 : List Nat := 4139 :: innerTail238_15
def innerCandidates238_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_14_checked : candidates edge 4 1523 4139 innerTail238_15 best.bound = innerCandidates238_14 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_14 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_14 best = best := by
  rw [innerTail238_14, scan, if_neg (by decide), innerCandidates238_14_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_15
    (if 3 ≤ innerCandidates238_14.length then search edge 3 [1523, 4139] innerCandidates238_14 5662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_15

def innerTail238_13 : List Nat := 4013 :: innerTail238_14
def innerCandidates238_13 : List Nat := [5573]
set_option maxRecDepth 4096 in
theorem innerCandidates238_13_checked : candidates edge 4 1523 4013 innerTail238_14 best.bound = innerCandidates238_13 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_13 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_13 best = best := by
  rw [innerTail238_13, scan, if_neg (by decide), innerCandidates238_13_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_14
    (if 3 ≤ innerCandidates238_13.length then search edge 3 [1523, 4013] innerCandidates238_13 5536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_14

def innerTail238_12 : List Nat := 3929 :: innerTail238_13
def innerCandidates238_12 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates238_12_checked : candidates edge 4 1523 3929 innerTail238_13 best.bound = innerCandidates238_12 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_12 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_12 best = best := by
  rw [innerTail238_12, scan, if_neg (by decide), innerCandidates238_12_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_13
    (if 3 ≤ innerCandidates238_12.length then search edge 3 [1523, 3929] innerCandidates238_12 5452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_13

def innerTail238_11 : List Nat := 3833 :: innerTail238_12
def innerCandidates238_11 : List Nat := [4253, 5813, 6827]
set_option maxRecDepth 4096 in
theorem innerCandidates238_11_checked : candidates edge 4 1523 3833 innerTail238_12 best.bound = innerCandidates238_11 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild238_11 : search edge 3 [1523, 3833] innerCandidates238_11 5356 best = best := by decide

theorem innerScan238_11 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_11 best = best := by
  rw [innerTail238_11, scan, if_neg (by decide), innerCandidates238_11_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_12
    (if 3 ≤ innerCandidates238_11.length then search edge 3 [1523, 3833] innerCandidates238_11 5356 best else best) = best
  rw [if_pos (by decide), innerChild238_11]
  exact innerScan238_12

def innerTail238_10 : List Nat := 3761 :: innerTail238_11
def innerCandidates238_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_10_checked : candidates edge 4 1523 3761 innerTail238_11 best.bound = innerCandidates238_10 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_10 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_10 best = best := by
  rw [innerTail238_10, scan, if_neg (by decide), innerCandidates238_10_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_11
    (if 3 ≤ innerCandidates238_10.length then search edge 3 [1523, 3761] innerCandidates238_10 5284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_11

def innerTail238_9 : List Nat := 3461 :: innerTail238_10
def innerCandidates238_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_9_checked : candidates edge 4 1523 3461 innerTail238_10 best.bound = innerCandidates238_9 := by
  unfold candidates
  have row : edge 3461 = SearchData.row3461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_9 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_9 best = best := by
  rw [innerTail238_9, scan, if_neg (by decide), innerCandidates238_9_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_10
    (if 3 ≤ innerCandidates238_9.length then search edge 3 [1523, 3461] innerCandidates238_9 4984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_10

def innerTail238_8 : List Nat := 3299 :: innerTail238_9
def innerCandidates238_8 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates238_8_checked : candidates edge 4 1523 3299 innerTail238_9 best.bound = innerCandidates238_8 := by
  unfold candidates
  have row : edge 3299 = SearchData.row3299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_8 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_8 best = best := by
  rw [innerTail238_8, scan, if_neg (by decide), innerCandidates238_8_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_9
    (if 3 ≤ innerCandidates238_8.length then search edge 3 [1523, 3299] innerCandidates238_8 4822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_9

def innerTail238_7 : List Nat := 3203 :: innerTail238_8
def innerCandidates238_7 : List Nat := [5279, 6827]
set_option maxRecDepth 4096 in
theorem innerCandidates238_7_checked : candidates edge 4 1523 3203 innerTail238_8 best.bound = innerCandidates238_7 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_7 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_7 best = best := by
  rw [innerTail238_7, scan, if_neg (by decide), innerCandidates238_7_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_8
    (if 3 ≤ innerCandidates238_7.length then search edge 3 [1523, 3203] innerCandidates238_7 4726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_8

def innerTail238_6 : List Nat := 2999 :: innerTail238_7
def innerCandidates238_6 : List Nat := [6311, 7907]
set_option maxRecDepth 4096 in
theorem innerCandidates238_6_checked : candidates edge 4 1523 2999 innerTail238_7 best.bound = innerCandidates238_6 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_6 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_6 best = best := by
  rw [innerTail238_6, scan, if_neg (by decide), innerCandidates238_6_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_7
    (if 3 ≤ innerCandidates238_6.length then search edge 3 [1523, 2999] innerCandidates238_6 4522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_7

def innerTail238_5 : List Nat := 2957 :: innerTail238_6
def innerCandidates238_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_5_checked : candidates edge 4 1523 2957 innerTail238_6 best.bound = innerCandidates238_5 := by
  unfold candidates
  have row : edge 2957 = SearchData.row2957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_5 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_5 best = best := by
  rw [innerTail238_5, scan, if_neg (by decide), innerCandidates238_5_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_6
    (if 3 ≤ innerCandidates238_5.length then search edge 3 [1523, 2957] innerCandidates238_5 4480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_6

def innerTail238_4 : List Nat := 2909 :: innerTail238_5
def innerCandidates238_4 : List Nat := [4253, 5813]
set_option maxRecDepth 4096 in
theorem innerCandidates238_4_checked : candidates edge 4 1523 2909 innerTail238_5 best.bound = innerCandidates238_4 := by
  unfold candidates
  have row : edge 2909 = SearchData.row2909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_4 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_4 best = best := by
  rw [innerTail238_4, scan, if_neg (by decide), innerCandidates238_4_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_5
    (if 3 ≤ innerCandidates238_4.length then search edge 3 [1523, 2909] innerCandidates238_4 4432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_5

def innerTail238_3 : List Nat := 2579 :: innerTail238_4
def innerCandidates238_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates238_3_checked : candidates edge 4 1523 2579 innerTail238_4 best.bound = innerCandidates238_3 := by
  unfold candidates
  have row : edge 2579 = PrimePairCache.smallRow2579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_3 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_3 best = best := by
  rw [innerTail238_3, scan, if_neg (by decide), innerCandidates238_3_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_4
    (if 3 ≤ innerCandidates238_3.length then search edge 3 [1523, 2579] innerCandidates238_3 4102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_4

def innerTail238_2 : List Nat := 2447 :: innerTail238_3
def innerCandidates238_2 : List Nat := [5813, 7907]
set_option maxRecDepth 4096 in
theorem innerCandidates238_2_checked : candidates edge 4 1523 2447 innerTail238_3 best.bound = innerCandidates238_2 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_2 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_2 best = best := by
  rw [innerTail238_2, scan, if_neg (by decide), innerCandidates238_2_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_3
    (if 3 ≤ innerCandidates238_2.length then search edge 3 [1523, 2447] innerCandidates238_2 3970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_3

def innerTail238_1 : List Nat := 2111 :: innerTail238_2
def innerCandidates238_1 : List Nat := [2999]
set_option maxRecDepth 4096 in
theorem innerCandidates238_1_checked : candidates edge 4 1523 2111 innerTail238_2 best.bound = innerCandidates238_1 := by
  unfold candidates
  have row : edge 2111 = PrimePairCache.smallRow2111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan238_1 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_1 best = best := by
  rw [innerTail238_1, scan, if_neg (by decide), innerCandidates238_1_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_2
    (if 3 ≤ innerCandidates238_1.length then search edge 3 [1523, 2111] innerCandidates238_1 3634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan238_2

def innerTail238_0 : List Nat := 1913 :: innerTail238_1
def innerCandidates238_0 : List Nat := [3203, 5483, 7583]
set_option maxRecDepth 4096 in
theorem innerCandidates238_0_checked : candidates edge 4 1523 1913 innerTail238_1 best.bound = innerCandidates238_0 := by
  unfold candidates
  have row : edge 1913 = PrimePairCache.smallRow1913.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild238_0 : search edge 3 [1523, 1913] innerCandidates238_0 3436 best = best := by decide

theorem innerScan238_0 : scan edge (search edge 3) 4 [1523] 1523 innerTail238_0 best = best := by
  rw [innerTail238_0, scan, if_neg (by decide), innerCandidates238_0_checked]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_1
    (if 3 ≤ innerCandidates238_0.length then search edge 3 [1523, 1913] innerCandidates238_0 3436 best else best) = best
  rw [if_pos (by decide), innerChild238_0]
  exact innerScan238_1

theorem rootBranch238 : search edge 4 [1523] rootCandidates238 1523 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1523] 1523 innerTail238_0 best = best
  exact innerScan238_0

def rootCandidates237 : List Nat := [1571, 2129, 2267, 3011, 3389, 4373, 4889, 5039, 5099, 5843, 5981, 7043, 7853]
set_option maxRecDepth 4096 in
theorem rootCandidates237_checked : candidates edge 5 0 1511 tail238 best.bound = rootCandidates237 := by
  unfold candidates
  have row : edge 1511 = PrimePairCache.smallRow1511.contains := by funext q; rfl
  rw [row]
  decide

def innerTail237_13 : List Nat := []
theorem innerScan237_13 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_13 best = best := rfl

def innerTail237_12 : List Nat := 7853 :: innerTail237_13
def innerCandidates237_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates237_12_checked : candidates edge 4 1511 7853 innerTail237_13 best.bound = innerCandidates237_12 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_12 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_12 best = best := by
  rw [innerTail237_12, scan, if_neg (by decide), innerCandidates237_12_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_13
    (if 3 ≤ innerCandidates237_12.length then search edge 3 [1511, 7853] innerCandidates237_12 9364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_13

def innerTail237_11 : List Nat := 7043 :: innerTail237_12
def innerCandidates237_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates237_11_checked : candidates edge 4 1511 7043 innerTail237_12 best.bound = innerCandidates237_11 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_11 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_11 best = best := by
  rw [innerTail237_11, scan, if_neg (by decide), innerCandidates237_11_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_12
    (if 3 ≤ innerCandidates237_11.length then search edge 3 [1511, 7043] innerCandidates237_11 8554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_12

def innerTail237_10 : List Nat := 5981 :: innerTail237_11
def innerCandidates237_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates237_10_checked : candidates edge 4 1511 5981 innerTail237_11 best.bound = innerCandidates237_10 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_10 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_10 best = best := by
  rw [innerTail237_10, scan, if_neg (by decide), innerCandidates237_10_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_11
    (if 3 ≤ innerCandidates237_10.length then search edge 3 [1511, 5981] innerCandidates237_10 7492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_11

def innerTail237_9 : List Nat := 5843 :: innerTail237_10
def innerCandidates237_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates237_9_checked : candidates edge 4 1511 5843 innerTail237_10 best.bound = innerCandidates237_9 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_9 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_9 best = best := by
  rw [innerTail237_9, scan, if_neg (by decide), innerCandidates237_9_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_10
    (if 3 ≤ innerCandidates237_9.length then search edge 3 [1511, 5843] innerCandidates237_9 7354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_10

def innerTail237_8 : List Nat := 5099 :: innerTail237_9
def innerCandidates237_8 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates237_8_checked : candidates edge 4 1511 5099 innerTail237_9 best.bound = innerCandidates237_8 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_8 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_8 best = best := by
  rw [innerTail237_8, scan, if_neg (by decide), innerCandidates237_8_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_9
    (if 3 ≤ innerCandidates237_8.length then search edge 3 [1511, 5099] innerCandidates237_8 6610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_9

def innerTail237_7 : List Nat := 5039 :: innerTail237_8
def innerCandidates237_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates237_7_checked : candidates edge 4 1511 5039 innerTail237_8 best.bound = innerCandidates237_7 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_7 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_7 best = best := by
  rw [innerTail237_7, scan, if_neg (by decide), innerCandidates237_7_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_8
    (if 3 ≤ innerCandidates237_7.length then search edge 3 [1511, 5039] innerCandidates237_7 6550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_8

def innerTail237_6 : List Nat := 4889 :: innerTail237_7
def innerCandidates237_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates237_6_checked : candidates edge 4 1511 4889 innerTail237_7 best.bound = innerCandidates237_6 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_6 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_6 best = best := by
  rw [innerTail237_6, scan, if_neg (by decide), innerCandidates237_6_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_7
    (if 3 ≤ innerCandidates237_6.length then search edge 3 [1511, 4889] innerCandidates237_6 6400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_7

def innerTail237_5 : List Nat := 4373 :: innerTail237_6
def innerCandidates237_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates237_5_checked : candidates edge 4 1511 4373 innerTail237_6 best.bound = innerCandidates237_5 := by
  unfold candidates
  have row : edge 4373 = SearchData.row4373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_5 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_5 best = best := by
  rw [innerTail237_5, scan, if_neg (by decide), innerCandidates237_5_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_6
    (if 3 ≤ innerCandidates237_5.length then search edge 3 [1511, 4373] innerCandidates237_5 5884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_6

def innerTail237_4 : List Nat := 3389 :: innerTail237_5
def innerCandidates237_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates237_4_checked : candidates edge 4 1511 3389 innerTail237_5 best.bound = innerCandidates237_4 := by
  unfold candidates
  have row : edge 3389 = SearchData.row3389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_4 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_4 best = best := by
  rw [innerTail237_4, scan, if_neg (by decide), innerCandidates237_4_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_5
    (if 3 ≤ innerCandidates237_4.length then search edge 3 [1511, 3389] innerCandidates237_4 4900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_5

def innerTail237_3 : List Nat := 3011 :: innerTail237_4
def innerCandidates237_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates237_3_checked : candidates edge 4 1511 3011 innerTail237_4 best.bound = innerCandidates237_3 := by
  unfold candidates
  have row : edge 3011 = SearchData.row3011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_3 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_3 best = best := by
  rw [innerTail237_3, scan, if_neg (by decide), innerCandidates237_3_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_4
    (if 3 ≤ innerCandidates237_3.length then search edge 3 [1511, 3011] innerCandidates237_3 4522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_4

def innerTail237_2 : List Nat := 2267 :: innerTail237_3
def innerCandidates237_2 : List Nat := [5843]
set_option maxRecDepth 4096 in
theorem innerCandidates237_2_checked : candidates edge 4 1511 2267 innerTail237_3 best.bound = innerCandidates237_2 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_2 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_2 best = best := by
  rw [innerTail237_2, scan, if_neg (by decide), innerCandidates237_2_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_3
    (if 3 ≤ innerCandidates237_2.length then search edge 3 [1511, 2267] innerCandidates237_2 3778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_3

def innerTail237_1 : List Nat := 2129 :: innerTail237_2
def innerCandidates237_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates237_1_checked : candidates edge 4 1511 2129 innerTail237_2 best.bound = innerCandidates237_1 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_1 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_1 best = best := by
  rw [innerTail237_1, scan, if_neg (by decide), innerCandidates237_1_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_2
    (if 3 ≤ innerCandidates237_1.length then search edge 3 [1511, 2129] innerCandidates237_1 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_2

def innerTail237_0 : List Nat := 1571 :: innerTail237_1
def innerCandidates237_0 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates237_0_checked : candidates edge 4 1511 1571 innerTail237_1 best.bound = innerCandidates237_0 := by
  unfold candidates
  have row : edge 1571 = PrimePairCache.smallRow1571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan237_0 : scan edge (search edge 3) 4 [1511] 1511 innerTail237_0 best = best := by
  rw [innerTail237_0, scan, if_neg (by decide), innerCandidates237_0_checked]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_1
    (if 3 ≤ innerCandidates237_0.length then search edge 3 [1511, 1571] innerCandidates237_0 3082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan237_1

theorem rootBranch237 : search edge 4 [1511] rootCandidates237 1511 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1511] 1511 innerTail237_0 best = best
  exact innerScan237_0

end ProjectEulerSolutions.PrimePairUpper.Search
