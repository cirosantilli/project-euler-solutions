import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates211 : List Nat := [1787, 1979, 2357, 2663, 2837, 3023, 3491, 3701, 3719, 3989, 4079, 4133, 4583, 5861, 5927, 5939, 6053, 6863, 7499, 7541, 7583, 7901, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates211_checked : candidates edge 5 0 1307 tail212 best.bound = rootCandidates211 := by
  unfold candidates
  have row : edge 1307 = PrimePairCache.smallRow1307.contains := by funext q; rfl
  rw [row]
  decide

def innerTail211_23 : List Nat := []
theorem innerScan211_23 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_23 best = best := rfl

def innerTail211_22 : List Nat := 8171 :: innerTail211_23
def innerCandidates211_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_22_checked : candidates edge 4 1307 8171 innerTail211_23 best.bound = innerCandidates211_22 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_22 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_22 best = best := by
  rw [innerTail211_22, scan, if_neg (by decide), innerCandidates211_22_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_23
    (if 3 ≤ innerCandidates211_22.length then search edge 3 [1307, 8171] innerCandidates211_22 9478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_23

def innerTail211_21 : List Nat := 7901 :: innerTail211_22
def innerCandidates211_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_21_checked : candidates edge 4 1307 7901 innerTail211_22 best.bound = innerCandidates211_21 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_21 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_21 best = best := by
  rw [innerTail211_21, scan, if_neg (by decide), innerCandidates211_21_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_22
    (if 3 ≤ innerCandidates211_21.length then search edge 3 [1307, 7901] innerCandidates211_21 9208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_22

def innerTail211_20 : List Nat := 7583 :: innerTail211_21
def innerCandidates211_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_20_checked : candidates edge 4 1307 7583 innerTail211_21 best.bound = innerCandidates211_20 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_20 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_20 best = best := by
  rw [innerTail211_20, scan, if_neg (by decide), innerCandidates211_20_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_21
    (if 3 ≤ innerCandidates211_20.length then search edge 3 [1307, 7583] innerCandidates211_20 8890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_21

def innerTail211_19 : List Nat := 7541 :: innerTail211_20
def innerCandidates211_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_19_checked : candidates edge 4 1307 7541 innerTail211_20 best.bound = innerCandidates211_19 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_19 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_19 best = best := by
  rw [innerTail211_19, scan, if_neg (by decide), innerCandidates211_19_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_20
    (if 3 ≤ innerCandidates211_19.length then search edge 3 [1307, 7541] innerCandidates211_19 8848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_20

def innerTail211_18 : List Nat := 7499 :: innerTail211_19
def innerCandidates211_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_18_checked : candidates edge 4 1307 7499 innerTail211_19 best.bound = innerCandidates211_18 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_18 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_18 best = best := by
  rw [innerTail211_18, scan, if_neg (by decide), innerCandidates211_18_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_19
    (if 3 ≤ innerCandidates211_18.length then search edge 3 [1307, 7499] innerCandidates211_18 8806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_19

def innerTail211_17 : List Nat := 6863 :: innerTail211_18
def innerCandidates211_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_17_checked : candidates edge 4 1307 6863 innerTail211_18 best.bound = innerCandidates211_17 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_17 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_17 best = best := by
  rw [innerTail211_17, scan, if_neg (by decide), innerCandidates211_17_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_18
    (if 3 ≤ innerCandidates211_17.length then search edge 3 [1307, 6863] innerCandidates211_17 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_18

def innerTail211_16 : List Nat := 6053 :: innerTail211_17
def innerCandidates211_16 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates211_16_checked : candidates edge 4 1307 6053 innerTail211_17 best.bound = innerCandidates211_16 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_16 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_16 best = best := by
  rw [innerTail211_16, scan, if_neg (by decide), innerCandidates211_16_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_17
    (if 3 ≤ innerCandidates211_16.length then search edge 3 [1307, 6053] innerCandidates211_16 7360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_17

def innerTail211_15 : List Nat := 5939 :: innerTail211_16
def innerCandidates211_15 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates211_15_checked : candidates edge 4 1307 5939 innerTail211_16 best.bound = innerCandidates211_15 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_15 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_15 best = best := by
  rw [innerTail211_15, scan, if_neg (by decide), innerCandidates211_15_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_16
    (if 3 ≤ innerCandidates211_15.length then search edge 3 [1307, 5939] innerCandidates211_15 7246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_16

def innerTail211_14 : List Nat := 5927 :: innerTail211_15
def innerCandidates211_14 : List Nat := [5939, 7499, 7583]
set_option maxRecDepth 4096 in
theorem innerCandidates211_14_checked : candidates edge 4 1307 5927 innerTail211_15 best.bound = innerCandidates211_14 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild211_14 : search edge 3 [1307, 5927] innerCandidates211_14 7234 best = best := by decide

theorem innerScan211_14 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_14 best = best := by
  rw [innerTail211_14, scan, if_neg (by decide), innerCandidates211_14_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_15
    (if 3 ≤ innerCandidates211_14.length then search edge 3 [1307, 5927] innerCandidates211_14 7234 best else best) = best
  rw [if_pos (by decide), innerChild211_14]
  exact innerScan211_15

def innerTail211_13 : List Nat := 5861 :: innerTail211_14
def innerCandidates211_13 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates211_13_checked : candidates edge 4 1307 5861 innerTail211_14 best.bound = innerCandidates211_13 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_13 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_13 best = best := by
  rw [innerTail211_13, scan, if_neg (by decide), innerCandidates211_13_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_14
    (if 3 ≤ innerCandidates211_13.length then search edge 3 [1307, 5861] innerCandidates211_13 7168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_14

def innerTail211_12 : List Nat := 4583 :: innerTail211_13
def innerCandidates211_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_12_checked : candidates edge 4 1307 4583 innerTail211_13 best.bound = innerCandidates211_12 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_12 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_12 best = best := by
  rw [innerTail211_12, scan, if_neg (by decide), innerCandidates211_12_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_13
    (if 3 ≤ innerCandidates211_12.length then search edge 3 [1307, 4583] innerCandidates211_12 5890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_13

def innerTail211_11 : List Nat := 4133 :: innerTail211_12
def innerCandidates211_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_11_checked : candidates edge 4 1307 4133 innerTail211_12 best.bound = innerCandidates211_11 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_11 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_11 best = best := by
  rw [innerTail211_11, scan, if_neg (by decide), innerCandidates211_11_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_12
    (if 3 ≤ innerCandidates211_11.length then search edge 3 [1307, 4133] innerCandidates211_11 5440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_12

def innerTail211_10 : List Nat := 4079 :: innerTail211_11
def innerCandidates211_10 : List Nat := [5927, 6053, 6863]
set_option maxRecDepth 4096 in
theorem innerCandidates211_10_checked : candidates edge 4 1307 4079 innerTail211_11 best.bound = innerCandidates211_10 := by
  unfold candidates
  have row : edge 4079 = SearchData.row4079.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild211_10 : search edge 3 [1307, 4079] innerCandidates211_10 5386 best = best := by decide

theorem innerScan211_10 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_10 best = best := by
  rw [innerTail211_10, scan, if_neg (by decide), innerCandidates211_10_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_11
    (if 3 ≤ innerCandidates211_10.length then search edge 3 [1307, 4079] innerCandidates211_10 5386 best else best) = best
  rw [if_pos (by decide), innerChild211_10]
  exact innerScan211_11

def innerTail211_9 : List Nat := 3989 :: innerTail211_10
def innerCandidates211_9 : List Nat := [5927]
set_option maxRecDepth 4096 in
theorem innerCandidates211_9_checked : candidates edge 4 1307 3989 innerTail211_10 best.bound = innerCandidates211_9 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_9 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_9 best = best := by
  rw [innerTail211_9, scan, if_neg (by decide), innerCandidates211_9_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_10
    (if 3 ≤ innerCandidates211_9.length then search edge 3 [1307, 3989] innerCandidates211_9 5296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_10

def innerTail211_8 : List Nat := 3719 :: innerTail211_9
def innerCandidates211_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_8_checked : candidates edge 4 1307 3719 innerTail211_9 best.bound = innerCandidates211_8 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_8 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_8 best = best := by
  rw [innerTail211_8, scan, if_neg (by decide), innerCandidates211_8_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_9
    (if 3 ≤ innerCandidates211_8.length then search edge 3 [1307, 3719] innerCandidates211_8 5026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_9

def innerTail211_7 : List Nat := 3701 :: innerTail211_8
def innerCandidates211_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_7_checked : candidates edge 4 1307 3701 innerTail211_8 best.bound = innerCandidates211_7 := by
  unfold candidates
  have row : edge 3701 = SearchData.row3701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_7 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_7 best = best := by
  rw [innerTail211_7, scan, if_neg (by decide), innerCandidates211_7_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_8
    (if 3 ≤ innerCandidates211_7.length then search edge 3 [1307, 3701] innerCandidates211_7 5008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_8

def innerTail211_6 : List Nat := 3491 :: innerTail211_7
def innerCandidates211_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_6_checked : candidates edge 4 1307 3491 innerTail211_7 best.bound = innerCandidates211_6 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_6 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_6 best = best := by
  rw [innerTail211_6, scan, if_neg (by decide), innerCandidates211_6_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_7
    (if 3 ≤ innerCandidates211_6.length then search edge 3 [1307, 3491] innerCandidates211_6 4798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_7

def innerTail211_5 : List Nat := 3023 :: innerTail211_6
def innerCandidates211_5 : List Nat := [3989]
set_option maxRecDepth 4096 in
theorem innerCandidates211_5_checked : candidates edge 4 1307 3023 innerTail211_6 best.bound = innerCandidates211_5 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_5 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_5 best = best := by
  rw [innerTail211_5, scan, if_neg (by decide), innerCandidates211_5_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_6
    (if 3 ≤ innerCandidates211_5.length then search edge 3 [1307, 3023] innerCandidates211_5 4330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_6

def innerTail211_4 : List Nat := 2837 :: innerTail211_5
def innerCandidates211_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_4_checked : candidates edge 4 1307 2837 innerTail211_5 best.bound = innerCandidates211_4 := by
  unfold candidates
  have row : edge 2837 = SearchData.row2837.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_4 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_4 best = best := by
  rw [innerTail211_4, scan, if_neg (by decide), innerCandidates211_4_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_5
    (if 3 ≤ innerCandidates211_4.length then search edge 3 [1307, 2837] innerCandidates211_4 4144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_5

def innerTail211_3 : List Nat := 2663 :: innerTail211_4
def innerCandidates211_3 : List Nat := [7901]
set_option maxRecDepth 4096 in
theorem innerCandidates211_3_checked : candidates edge 4 1307 2663 innerTail211_4 best.bound = innerCandidates211_3 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_3 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_3 best = best := by
  rw [innerTail211_3, scan, if_neg (by decide), innerCandidates211_3_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_4
    (if 3 ≤ innerCandidates211_3.length then search edge 3 [1307, 2663] innerCandidates211_3 3970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_4

def innerTail211_2 : List Nat := 2357 :: innerTail211_3
def innerCandidates211_2 : List Nat := [7901]
set_option maxRecDepth 4096 in
theorem innerCandidates211_2_checked : candidates edge 4 1307 2357 innerTail211_3 best.bound = innerCandidates211_2 := by
  unfold candidates
  have row : edge 2357 = PrimePairCache.smallRow2357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_2 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_2 best = best := by
  rw [innerTail211_2, scan, if_neg (by decide), innerCandidates211_2_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_3
    (if 3 ≤ innerCandidates211_2.length then search edge 3 [1307, 2357] innerCandidates211_2 3664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_3

def innerTail211_1 : List Nat := 1979 :: innerTail211_2
def innerCandidates211_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates211_1_checked : candidates edge 4 1307 1979 innerTail211_2 best.bound = innerCandidates211_1 := by
  unfold candidates
  have row : edge 1979 = PrimePairCache.smallRow1979.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_1 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_1 best = best := by
  rw [innerTail211_1, scan, if_neg (by decide), innerCandidates211_1_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_2
    (if 3 ≤ innerCandidates211_1.length then search edge 3 [1307, 1979] innerCandidates211_1 3286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_2

def innerTail211_0 : List Nat := 1787 :: innerTail211_1
def innerCandidates211_0 : List Nat := [5927, 7901]
set_option maxRecDepth 4096 in
theorem innerCandidates211_0_checked : candidates edge 4 1307 1787 innerTail211_1 best.bound = innerCandidates211_0 := by
  unfold candidates
  have row : edge 1787 = PrimePairCache.smallRow1787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan211_0 : scan edge (search edge 3) 4 [1307] 1307 innerTail211_0 best = best := by
  rw [innerTail211_0, scan, if_neg (by decide), innerCandidates211_0_checked]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_1
    (if 3 ≤ innerCandidates211_0.length then search edge 3 [1307, 1787] innerCandidates211_0 3094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan211_1

theorem rootBranch211 : search edge 4 [1307] rootCandidates211 1307 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1307] 1307 innerTail211_0 best = best
  exact innerScan211_0

def rootCandidates210 : List Nat := [1723, 2143, 2857, 2953, 3109, 3469, 4363, 4519, 4561, 4759, 4993, 5023, 5119, 5443, 6043, 6871, 7027, 7207, 7309, 7333, 8101]
set_option maxRecDepth 4096 in
theorem rootCandidates210_checked : candidates edge 5 0 1303 tail211 best.bound = rootCandidates210 := by
  unfold candidates
  have row : edge 1303 = PrimePairCache.smallRow1303.contains := by funext q; rfl
  rw [row]
  decide

def innerTail210_21 : List Nat := []
theorem innerScan210_21 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_21 best = best := rfl

def innerTail210_20 : List Nat := 8101 :: innerTail210_21
def innerCandidates210_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates210_20_checked : candidates edge 4 1303 8101 innerTail210_21 best.bound = innerCandidates210_20 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_20 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_20 best = best := by
  rw [innerTail210_20, scan, if_neg (by decide), innerCandidates210_20_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_21
    (if 3 ≤ innerCandidates210_20.length then search edge 3 [1303, 8101] innerCandidates210_20 9404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_21

def innerTail210_19 : List Nat := 7333 :: innerTail210_20
def innerCandidates210_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates210_19_checked : candidates edge 4 1303 7333 innerTail210_20 best.bound = innerCandidates210_19 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_19 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_19 best = best := by
  rw [innerTail210_19, scan, if_neg (by decide), innerCandidates210_19_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_20
    (if 3 ≤ innerCandidates210_19.length then search edge 3 [1303, 7333] innerCandidates210_19 8636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_20

def innerTail210_18 : List Nat := 7309 :: innerTail210_19
def innerCandidates210_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates210_18_checked : candidates edge 4 1303 7309 innerTail210_19 best.bound = innerCandidates210_18 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_18 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_18 best = best := by
  rw [innerTail210_18, scan, if_neg (by decide), innerCandidates210_18_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_19
    (if 3 ≤ innerCandidates210_18.length then search edge 3 [1303, 7309] innerCandidates210_18 8612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_19

def innerTail210_17 : List Nat := 7207 :: innerTail210_18
def innerCandidates210_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates210_17_checked : candidates edge 4 1303 7207 innerTail210_18 best.bound = innerCandidates210_17 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_17 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_17 best = best := by
  rw [innerTail210_17, scan, if_neg (by decide), innerCandidates210_17_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_18
    (if 3 ≤ innerCandidates210_17.length then search edge 3 [1303, 7207] innerCandidates210_17 8510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_18

def innerTail210_16 : List Nat := 7027 :: innerTail210_17
def innerCandidates210_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates210_16_checked : candidates edge 4 1303 7027 innerTail210_17 best.bound = innerCandidates210_16 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_16 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_16 best = best := by
  rw [innerTail210_16, scan, if_neg (by decide), innerCandidates210_16_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_17
    (if 3 ≤ innerCandidates210_16.length then search edge 3 [1303, 7027] innerCandidates210_16 8330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_17

def innerTail210_15 : List Nat := 6871 :: innerTail210_16
def innerCandidates210_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates210_15_checked : candidates edge 4 1303 6871 innerTail210_16 best.bound = innerCandidates210_15 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_15 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_15 best = best := by
  rw [innerTail210_15, scan, if_neg (by decide), innerCandidates210_15_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_16
    (if 3 ≤ innerCandidates210_15.length then search edge 3 [1303, 6871] innerCandidates210_15 8174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_16

def innerTail210_14 : List Nat := 6043 :: innerTail210_15
def innerCandidates210_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates210_14_checked : candidates edge 4 1303 6043 innerTail210_15 best.bound = innerCandidates210_14 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_14 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_14 best = best := by
  rw [innerTail210_14, scan, if_neg (by decide), innerCandidates210_14_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_15
    (if 3 ≤ innerCandidates210_14.length then search edge 3 [1303, 6043] innerCandidates210_14 7346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_15

def innerTail210_13 : List Nat := 5443 :: innerTail210_14
def innerCandidates210_13 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates210_13_checked : candidates edge 4 1303 5443 innerTail210_14 best.bound = innerCandidates210_13 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_13 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_13 best = best := by
  rw [innerTail210_13, scan, if_neg (by decide), innerCandidates210_13_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_14
    (if 3 ≤ innerCandidates210_13.length then search edge 3 [1303, 5443] innerCandidates210_13 6746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_14

def innerTail210_12 : List Nat := 5119 :: innerTail210_13
def innerCandidates210_12 : List Nat := [7207]
set_option maxRecDepth 4096 in
theorem innerCandidates210_12_checked : candidates edge 4 1303 5119 innerTail210_13 best.bound = innerCandidates210_12 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_12 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_12 best = best := by
  rw [innerTail210_12, scan, if_neg (by decide), innerCandidates210_12_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_13
    (if 3 ≤ innerCandidates210_12.length then search edge 3 [1303, 5119] innerCandidates210_12 6422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_13

def innerTail210_11 : List Nat := 5023 :: innerTail210_12
def innerCandidates210_11 : List Nat := [5443]
set_option maxRecDepth 4096 in
theorem innerCandidates210_11_checked : candidates edge 4 1303 5023 innerTail210_12 best.bound = innerCandidates210_11 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_11 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_11 best = best := by
  rw [innerTail210_11, scan, if_neg (by decide), innerCandidates210_11_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_12
    (if 3 ≤ innerCandidates210_11.length then search edge 3 [1303, 5023] innerCandidates210_11 6326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_12

def innerTail210_10 : List Nat := 4993 :: innerTail210_11
def innerCandidates210_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates210_10_checked : candidates edge 4 1303 4993 innerTail210_11 best.bound = innerCandidates210_10 := by
  unfold candidates
  have row : edge 4993 = SearchData.row4993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_10 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_10 best = best := by
  rw [innerTail210_10, scan, if_neg (by decide), innerCandidates210_10_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_11
    (if 3 ≤ innerCandidates210_10.length then search edge 3 [1303, 4993] innerCandidates210_10 6296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_11

def innerTail210_9 : List Nat := 4759 :: innerTail210_10
def innerCandidates210_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates210_9_checked : candidates edge 4 1303 4759 innerTail210_10 best.bound = innerCandidates210_9 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_9 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_9 best = best := by
  rw [innerTail210_9, scan, if_neg (by decide), innerCandidates210_9_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_10
    (if 3 ≤ innerCandidates210_9.length then search edge 3 [1303, 4759] innerCandidates210_9 6062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_10

def innerTail210_8 : List Nat := 4561 :: innerTail210_9
def innerCandidates210_8 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates210_8_checked : candidates edge 4 1303 4561 innerTail210_9 best.bound = innerCandidates210_8 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_8 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_8 best = best := by
  rw [innerTail210_8, scan, if_neg (by decide), innerCandidates210_8_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_9
    (if 3 ≤ innerCandidates210_8.length then search edge 3 [1303, 4561] innerCandidates210_8 5864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_9

def innerTail210_7 : List Nat := 4519 :: innerTail210_8
def innerCandidates210_7 : List Nat := [6043]
set_option maxRecDepth 4096 in
theorem innerCandidates210_7_checked : candidates edge 4 1303 4519 innerTail210_8 best.bound = innerCandidates210_7 := by
  unfold candidates
  have row : edge 4519 = SearchData.row4519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_7 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_7 best = best := by
  rw [innerTail210_7, scan, if_neg (by decide), innerCandidates210_7_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_8
    (if 3 ≤ innerCandidates210_7.length then search edge 3 [1303, 4519] innerCandidates210_7 5822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_8

def innerTail210_6 : List Nat := 4363 :: innerTail210_7
def innerCandidates210_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates210_6_checked : candidates edge 4 1303 4363 innerTail210_7 best.bound = innerCandidates210_6 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_6 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_6 best = best := by
  rw [innerTail210_6, scan, if_neg (by decide), innerCandidates210_6_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_7
    (if 3 ≤ innerCandidates210_6.length then search edge 3 [1303, 4363] innerCandidates210_6 5666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_7

def innerTail210_5 : List Nat := 3469 :: innerTail210_6
def innerCandidates210_5 : List Nat := [7309]
set_option maxRecDepth 4096 in
theorem innerCandidates210_5_checked : candidates edge 4 1303 3469 innerTail210_6 best.bound = innerCandidates210_5 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_5 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_5 best = best := by
  rw [innerTail210_5, scan, if_neg (by decide), innerCandidates210_5_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_6
    (if 3 ≤ innerCandidates210_5.length then search edge 3 [1303, 3469] innerCandidates210_5 4772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_6

def innerTail210_4 : List Nat := 3109 :: innerTail210_5
def innerCandidates210_4 : List Nat := [4759]
set_option maxRecDepth 4096 in
theorem innerCandidates210_4_checked : candidates edge 4 1303 3109 innerTail210_5 best.bound = innerCandidates210_4 := by
  unfold candidates
  have row : edge 3109 = SearchData.row3109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_4 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_4 best = best := by
  rw [innerTail210_4, scan, if_neg (by decide), innerCandidates210_4_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_5
    (if 3 ≤ innerCandidates210_4.length then search edge 3 [1303, 3109] innerCandidates210_4 4412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_5

def innerTail210_3 : List Nat := 2953 :: innerTail210_4
def innerCandidates210_3 : List Nat := [7309]
set_option maxRecDepth 4096 in
theorem innerCandidates210_3_checked : candidates edge 4 1303 2953 innerTail210_4 best.bound = innerCandidates210_3 := by
  unfold candidates
  have row : edge 2953 = SearchData.row2953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_3 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_3 best = best := by
  rw [innerTail210_3, scan, if_neg (by decide), innerCandidates210_3_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_4
    (if 3 ≤ innerCandidates210_3.length then search edge 3 [1303, 2953] innerCandidates210_3 4256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_4

def innerTail210_2 : List Nat := 2857 :: innerTail210_3
def innerCandidates210_2 : List Nat := [3469, 4519, 4561]
set_option maxRecDepth 4096 in
theorem innerCandidates210_2_checked : candidates edge 4 1303 2857 innerTail210_3 best.bound = innerCandidates210_2 := by
  unfold candidates
  have row : edge 2857 = SearchData.row2857.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild210_2 : search edge 3 [1303, 2857] innerCandidates210_2 4160 best = best := by decide

theorem innerScan210_2 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_2 best = best := by
  rw [innerTail210_2, scan, if_neg (by decide), innerCandidates210_2_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_3
    (if 3 ≤ innerCandidates210_2.length then search edge 3 [1303, 2857] innerCandidates210_2 4160 best else best) = best
  rw [if_pos (by decide), innerChild210_2]
  exact innerScan210_3

def innerTail210_1 : List Nat := 2143 :: innerTail210_2
def innerCandidates210_1 : List Nat := [6871]
set_option maxRecDepth 4096 in
theorem innerCandidates210_1_checked : candidates edge 4 1303 2143 innerTail210_2 best.bound = innerCandidates210_1 := by
  unfold candidates
  have row : edge 2143 = PrimePairCache.smallRow2143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_1 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_1 best = best := by
  rw [innerTail210_1, scan, if_neg (by decide), innerCandidates210_1_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_2
    (if 3 ≤ innerCandidates210_1.length then search edge 3 [1303, 2143] innerCandidates210_1 3446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_2

def innerTail210_0 : List Nat := 1723 :: innerTail210_1
def innerCandidates210_0 : List Nat := [4363, 4993]
set_option maxRecDepth 4096 in
theorem innerCandidates210_0_checked : candidates edge 4 1303 1723 innerTail210_1 best.bound = innerCandidates210_0 := by
  unfold candidates
  have row : edge 1723 = PrimePairCache.smallRow1723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan210_0 : scan edge (search edge 3) 4 [1303] 1303 innerTail210_0 best = best := by
  rw [innerTail210_0, scan, if_neg (by decide), innerCandidates210_0_checked]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_1
    (if 3 ≤ innerCandidates210_0.length then search edge 3 [1303, 1723] innerCandidates210_0 3026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan210_1

theorem rootBranch210 : search edge 4 [1303] rootCandidates210 1303 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1303] 1303 innerTail210_0 best = best
  exact innerScan210_0

def rootCandidates209 : List Nat := [2141, 2393, 2579, 2927, 2969, 2999, 4583, 4973, 5003, 5279, 5531, 5669, 6011, 7127, 7283, 7481, 7643, 7757]
set_option maxRecDepth 4096 in
theorem rootCandidates209_checked : candidates edge 5 0 1301 tail210 best.bound = rootCandidates209 := by
  unfold candidates
  have row : edge 1301 = PrimePairCache.smallRow1301.contains := by funext q; rfl
  rw [row]
  decide

def innerTail209_18 : List Nat := []
theorem innerScan209_18 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_18 best = best := rfl

def innerTail209_17 : List Nat := 7757 :: innerTail209_18
def innerCandidates209_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates209_17_checked : candidates edge 4 1301 7757 innerTail209_18 best.bound = innerCandidates209_17 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_17 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_17 best = best := by
  rw [innerTail209_17, scan, if_neg (by decide), innerCandidates209_17_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_18
    (if 3 ≤ innerCandidates209_17.length then search edge 3 [1301, 7757] innerCandidates209_17 9058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_18

def innerTail209_16 : List Nat := 7643 :: innerTail209_17
def innerCandidates209_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates209_16_checked : candidates edge 4 1301 7643 innerTail209_17 best.bound = innerCandidates209_16 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_16 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_16 best = best := by
  rw [innerTail209_16, scan, if_neg (by decide), innerCandidates209_16_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_17
    (if 3 ≤ innerCandidates209_16.length then search edge 3 [1301, 7643] innerCandidates209_16 8944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_17

def innerTail209_15 : List Nat := 7481 :: innerTail209_16
def innerCandidates209_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates209_15_checked : candidates edge 4 1301 7481 innerTail209_16 best.bound = innerCandidates209_15 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_15 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_15 best = best := by
  rw [innerTail209_15, scan, if_neg (by decide), innerCandidates209_15_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_16
    (if 3 ≤ innerCandidates209_15.length then search edge 3 [1301, 7481] innerCandidates209_15 8782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_16

def innerTail209_14 : List Nat := 7283 :: innerTail209_15
def innerCandidates209_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates209_14_checked : candidates edge 4 1301 7283 innerTail209_15 best.bound = innerCandidates209_14 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_14 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_14 best = best := by
  rw [innerTail209_14, scan, if_neg (by decide), innerCandidates209_14_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_15
    (if 3 ≤ innerCandidates209_14.length then search edge 3 [1301, 7283] innerCandidates209_14 8584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_15

def innerTail209_13 : List Nat := 7127 :: innerTail209_14
def innerCandidates209_13 : List Nat := [7757]
set_option maxRecDepth 4096 in
theorem innerCandidates209_13_checked : candidates edge 4 1301 7127 innerTail209_14 best.bound = innerCandidates209_13 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_13 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_13 best = best := by
  rw [innerTail209_13, scan, if_neg (by decide), innerCandidates209_13_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_14
    (if 3 ≤ innerCandidates209_13.length then search edge 3 [1301, 7127] innerCandidates209_13 8428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_14

def innerTail209_12 : List Nat := 6011 :: innerTail209_13
def innerCandidates209_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates209_12_checked : candidates edge 4 1301 6011 innerTail209_13 best.bound = innerCandidates209_12 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_12 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_12 best = best := by
  rw [innerTail209_12, scan, if_neg (by decide), innerCandidates209_12_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_13
    (if 3 ≤ innerCandidates209_12.length then search edge 3 [1301, 6011] innerCandidates209_12 7312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_13

def innerTail209_11 : List Nat := 5669 :: innerTail209_12
def innerCandidates209_11 : List Nat := [6011]
set_option maxRecDepth 4096 in
theorem innerCandidates209_11_checked : candidates edge 4 1301 5669 innerTail209_12 best.bound = innerCandidates209_11 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_11 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_11 best = best := by
  rw [innerTail209_11, scan, if_neg (by decide), innerCandidates209_11_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_12
    (if 3 ≤ innerCandidates209_11.length then search edge 3 [1301, 5669] innerCandidates209_11 6970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_12

def innerTail209_10 : List Nat := 5531 :: innerTail209_11
def innerCandidates209_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates209_10_checked : candidates edge 4 1301 5531 innerTail209_11 best.bound = innerCandidates209_10 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_10 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_10 best = best := by
  rw [innerTail209_10, scan, if_neg (by decide), innerCandidates209_10_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_11
    (if 3 ≤ innerCandidates209_10.length then search edge 3 [1301, 5531] innerCandidates209_10 6832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_11

def innerTail209_9 : List Nat := 5279 :: innerTail209_10
def innerCandidates209_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates209_9_checked : candidates edge 4 1301 5279 innerTail209_10 best.bound = innerCandidates209_9 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_9 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_9 best = best := by
  rw [innerTail209_9, scan, if_neg (by decide), innerCandidates209_9_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_10
    (if 3 ≤ innerCandidates209_9.length then search edge 3 [1301, 5279] innerCandidates209_9 6580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_10

def innerTail209_8 : List Nat := 5003 :: innerTail209_9
def innerCandidates209_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates209_8_checked : candidates edge 4 1301 5003 innerTail209_9 best.bound = innerCandidates209_8 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_8 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_8 best = best := by
  rw [innerTail209_8, scan, if_neg (by decide), innerCandidates209_8_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_9
    (if 3 ≤ innerCandidates209_8.length then search edge 3 [1301, 5003] innerCandidates209_8 6304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_9

def innerTail209_7 : List Nat := 4973 :: innerTail209_8
def innerCandidates209_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates209_7_checked : candidates edge 4 1301 4973 innerTail209_8 best.bound = innerCandidates209_7 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_7 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_7 best = best := by
  rw [innerTail209_7, scan, if_neg (by decide), innerCandidates209_7_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_8
    (if 3 ≤ innerCandidates209_7.length then search edge 3 [1301, 4973] innerCandidates209_7 6274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_8

def innerTail209_6 : List Nat := 4583 :: innerTail209_7
def innerCandidates209_6 : List Nat := [5003]
set_option maxRecDepth 4096 in
theorem innerCandidates209_6_checked : candidates edge 4 1301 4583 innerTail209_7 best.bound = innerCandidates209_6 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_6 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_6 best = best := by
  rw [innerTail209_6, scan, if_neg (by decide), innerCandidates209_6_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_7
    (if 3 ≤ innerCandidates209_6.length then search edge 3 [1301, 4583] innerCandidates209_6 5884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_7

def innerTail209_5 : List Nat := 2999 :: innerTail209_6
def innerCandidates209_5 : List Nat := [5003, 7481]
set_option maxRecDepth 4096 in
theorem innerCandidates209_5_checked : candidates edge 4 1301 2999 innerTail209_6 best.bound = innerCandidates209_5 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_5 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_5 best = best := by
  rw [innerTail209_5, scan, if_neg (by decide), innerCandidates209_5_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_6
    (if 3 ≤ innerCandidates209_5.length then search edge 3 [1301, 2999] innerCandidates209_5 4300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_6

def innerTail209_4 : List Nat := 2969 :: innerTail209_5
def innerCandidates209_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates209_4_checked : candidates edge 4 1301 2969 innerTail209_5 best.bound = innerCandidates209_4 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_4 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_4 best = best := by
  rw [innerTail209_4, scan, if_neg (by decide), innerCandidates209_4_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_5
    (if 3 ≤ innerCandidates209_4.length then search edge 3 [1301, 2969] innerCandidates209_4 4270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_5

def innerTail209_3 : List Nat := 2927 :: innerTail209_4
def innerCandidates209_3 : List Nat := [5531]
set_option maxRecDepth 4096 in
theorem innerCandidates209_3_checked : candidates edge 4 1301 2927 innerTail209_4 best.bound = innerCandidates209_3 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_3 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_3 best = best := by
  rw [innerTail209_3, scan, if_neg (by decide), innerCandidates209_3_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_4
    (if 3 ≤ innerCandidates209_3.length then search edge 3 [1301, 2927] innerCandidates209_3 4228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_4

def innerTail209_2 : List Nat := 2579 :: innerTail209_3
def innerCandidates209_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates209_2_checked : candidates edge 4 1301 2579 innerTail209_3 best.bound = innerCandidates209_2 := by
  unfold candidates
  have row : edge 2579 = PrimePairCache.smallRow2579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_2 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_2 best = best := by
  rw [innerTail209_2, scan, if_neg (by decide), innerCandidates209_2_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_3
    (if 3 ≤ innerCandidates209_2.length then search edge 3 [1301, 2579] innerCandidates209_2 3880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_3

def innerTail209_1 : List Nat := 2393 :: innerTail209_2
def innerCandidates209_1 : List Nat := [2927, 7127]
set_option maxRecDepth 4096 in
theorem innerCandidates209_1_checked : candidates edge 4 1301 2393 innerTail209_2 best.bound = innerCandidates209_1 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_1 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_1 best = best := by
  rw [innerTail209_1, scan, if_neg (by decide), innerCandidates209_1_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_2
    (if 3 ≤ innerCandidates209_1.length then search edge 3 [1301, 2393] innerCandidates209_1 3694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_2

def innerTail209_0 : List Nat := 2141 :: innerTail209_1
def innerCandidates209_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates209_0_checked : candidates edge 4 1301 2141 innerTail209_1 best.bound = innerCandidates209_0 := by
  unfold candidates
  have row : edge 2141 = PrimePairCache.smallRow2141.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan209_0 : scan edge (search edge 3) 4 [1301] 1301 innerTail209_0 best = best := by
  rw [innerTail209_0, scan, if_neg (by decide), innerCandidates209_0_checked]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_1
    (if 3 ≤ innerCandidates209_0.length then search edge 3 [1301, 2141] innerCandidates209_0 3442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan209_1

theorem rootBranch209 : search edge 4 [1301] rootCandidates209 1301 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1301] 1301 innerTail209_0 best = best
  exact innerScan209_0

end ProjectEulerSolutions.PrimePairUpper.Search
