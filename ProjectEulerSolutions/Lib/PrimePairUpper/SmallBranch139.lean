import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates258 : List Nat := [2857, 3229, 3613, 4027, 4231, 4831, 4987, 5413, 5449, 5653, 6007, 6577, 7219, 7879, 7963, 8221]
set_option maxRecDepth 4096 in
theorem rootCandidates258_checked : candidates edge 5 0 1663 tail259 best.bound = rootCandidates258 := by
  unfold candidates
  have row : edge 1663 = PrimePairCache.smallRow1663.contains := by funext q; rfl
  rw [row]
  decide

def innerTail258_16 : List Nat := []
theorem innerScan258_16 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_16 best = best := rfl

def innerTail258_15 : List Nat := 8221 :: innerTail258_16
def innerCandidates258_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates258_15_checked : candidates edge 4 1663 8221 innerTail258_16 best.bound = innerCandidates258_15 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_15 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_15 best = best := by
  rw [innerTail258_15, scan, if_neg (by decide), innerCandidates258_15_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_16
    (if 3 ≤ innerCandidates258_15.length then search edge 3 [1663, 8221] innerCandidates258_15 9884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_16

def innerTail258_14 : List Nat := 7963 :: innerTail258_15
def innerCandidates258_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates258_14_checked : candidates edge 4 1663 7963 innerTail258_15 best.bound = innerCandidates258_14 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_14 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_14 best = best := by
  rw [innerTail258_14, scan, if_neg (by decide), innerCandidates258_14_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_15
    (if 3 ≤ innerCandidates258_14.length then search edge 3 [1663, 7963] innerCandidates258_14 9626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_15

def innerTail258_13 : List Nat := 7879 :: innerTail258_14
def innerCandidates258_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates258_13_checked : candidates edge 4 1663 7879 innerTail258_14 best.bound = innerCandidates258_13 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_13 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_13 best = best := by
  rw [innerTail258_13, scan, if_neg (by decide), innerCandidates258_13_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_14
    (if 3 ≤ innerCandidates258_13.length then search edge 3 [1663, 7879] innerCandidates258_13 9542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_14

def innerTail258_12 : List Nat := 7219 :: innerTail258_13
def innerCandidates258_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates258_12_checked : candidates edge 4 1663 7219 innerTail258_13 best.bound = innerCandidates258_12 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_12 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_12 best = best := by
  rw [innerTail258_12, scan, if_neg (by decide), innerCandidates258_12_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_13
    (if 3 ≤ innerCandidates258_12.length then search edge 3 [1663, 7219] innerCandidates258_12 8882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_13

def innerTail258_11 : List Nat := 6577 :: innerTail258_12
def innerCandidates258_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates258_11_checked : candidates edge 4 1663 6577 innerTail258_12 best.bound = innerCandidates258_11 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_11 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_11 best = best := by
  rw [innerTail258_11, scan, if_neg (by decide), innerCandidates258_11_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_12
    (if 3 ≤ innerCandidates258_11.length then search edge 3 [1663, 6577] innerCandidates258_11 8240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_12

def innerTail258_10 : List Nat := 6007 :: innerTail258_11
def innerCandidates258_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates258_10_checked : candidates edge 4 1663 6007 innerTail258_11 best.bound = innerCandidates258_10 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_10 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_10 best = best := by
  rw [innerTail258_10, scan, if_neg (by decide), innerCandidates258_10_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_11
    (if 3 ≤ innerCandidates258_10.length then search edge 3 [1663, 6007] innerCandidates258_10 7670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_11

def innerTail258_9 : List Nat := 5653 :: innerTail258_10
def innerCandidates258_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates258_9_checked : candidates edge 4 1663 5653 innerTail258_10 best.bound = innerCandidates258_9 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_9 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_9 best = best := by
  rw [innerTail258_9, scan, if_neg (by decide), innerCandidates258_9_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_10
    (if 3 ≤ innerCandidates258_9.length then search edge 3 [1663, 5653] innerCandidates258_9 7316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_10

def innerTail258_8 : List Nat := 5449 :: innerTail258_9
def innerCandidates258_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates258_8_checked : candidates edge 4 1663 5449 innerTail258_9 best.bound = innerCandidates258_8 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_8 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_8 best = best := by
  rw [innerTail258_8, scan, if_neg (by decide), innerCandidates258_8_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_9
    (if 3 ≤ innerCandidates258_8.length then search edge 3 [1663, 5449] innerCandidates258_8 7112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_9

def innerTail258_7 : List Nat := 5413 :: innerTail258_8
def innerCandidates258_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates258_7_checked : candidates edge 4 1663 5413 innerTail258_8 best.bound = innerCandidates258_7 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_7 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_7 best = best := by
  rw [innerTail258_7, scan, if_neg (by decide), innerCandidates258_7_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_8
    (if 3 ≤ innerCandidates258_7.length then search edge 3 [1663, 5413] innerCandidates258_7 7076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_8

def innerTail258_6 : List Nat := 4987 :: innerTail258_7
def innerCandidates258_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates258_6_checked : candidates edge 4 1663 4987 innerTail258_7 best.bound = innerCandidates258_6 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_6 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_6 best = best := by
  rw [innerTail258_6, scan, if_neg (by decide), innerCandidates258_6_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_7
    (if 3 ≤ innerCandidates258_6.length then search edge 3 [1663, 4987] innerCandidates258_6 6650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_7

def innerTail258_5 : List Nat := 4831 :: innerTail258_6
def innerCandidates258_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates258_5_checked : candidates edge 4 1663 4831 innerTail258_6 best.bound = innerCandidates258_5 := by
  unfold candidates
  have row : edge 4831 = SearchData.row4831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_5 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_5 best = best := by
  rw [innerTail258_5, scan, if_neg (by decide), innerCandidates258_5_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_6
    (if 3 ≤ innerCandidates258_5.length then search edge 3 [1663, 4831] innerCandidates258_5 6494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_6

def innerTail258_4 : List Nat := 4231 :: innerTail258_5
def innerCandidates258_4 : List Nat := [6577]
set_option maxRecDepth 4096 in
theorem innerCandidates258_4_checked : candidates edge 4 1663 4231 innerTail258_5 best.bound = innerCandidates258_4 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_4 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_4 best = best := by
  rw [innerTail258_4, scan, if_neg (by decide), innerCandidates258_4_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_5
    (if 3 ≤ innerCandidates258_4.length then search edge 3 [1663, 4231] innerCandidates258_4 5894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_5

def innerTail258_3 : List Nat := 4027 :: innerTail258_4
def innerCandidates258_3 : List Nat := [5413]
set_option maxRecDepth 4096 in
theorem innerCandidates258_3_checked : candidates edge 4 1663 4027 innerTail258_4 best.bound = innerCandidates258_3 := by
  unfold candidates
  have row : edge 4027 = SearchData.row4027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_3 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_3 best = best := by
  rw [innerTail258_3, scan, if_neg (by decide), innerCandidates258_3_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_4
    (if 3 ≤ innerCandidates258_3.length then search edge 3 [1663, 4027] innerCandidates258_3 5690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_4

def innerTail258_2 : List Nat := 3613 :: innerTail258_3
def innerCandidates258_2 : List Nat := [4831]
set_option maxRecDepth 4096 in
theorem innerCandidates258_2_checked : candidates edge 4 1663 3613 innerTail258_3 best.bound = innerCandidates258_2 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_2 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_2 best = best := by
  rw [innerTail258_2, scan, if_neg (by decide), innerCandidates258_2_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_3
    (if 3 ≤ innerCandidates258_2.length then search edge 3 [1663, 3613] innerCandidates258_2 5276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_3

def innerTail258_1 : List Nat := 3229 :: innerTail258_2
def innerCandidates258_1 : List Nat := [6007]
set_option maxRecDepth 4096 in
theorem innerCandidates258_1_checked : candidates edge 4 1663 3229 innerTail258_2 best.bound = innerCandidates258_1 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_1 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_1 best = best := by
  rw [innerTail258_1, scan, if_neg (by decide), innerCandidates258_1_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_2
    (if 3 ≤ innerCandidates258_1.length then search edge 3 [1663, 3229] innerCandidates258_1 4892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_2

def innerTail258_0 : List Nat := 2857 :: innerTail258_1
def innerCandidates258_0 : List Nat := [3229]
set_option maxRecDepth 4096 in
theorem innerCandidates258_0_checked : candidates edge 4 1663 2857 innerTail258_1 best.bound = innerCandidates258_0 := by
  unfold candidates
  have row : edge 2857 = SearchData.row2857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan258_0 : scan edge (search edge 3) 4 [1663] 1663 innerTail258_0 best = best := by
  rw [innerTail258_0, scan, if_neg (by decide), innerCandidates258_0_checked]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_1
    (if 3 ≤ innerCandidates258_0.length then search edge 3 [1663, 2857] innerCandidates258_0 4520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan258_1

theorem rootBranch258 : search edge 4 [1663] rootCandidates258 1663 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1663] 1663 innerTail258_0 best = best
  exact innerScan258_0

def rootCandidates257 : List Nat := [1699, 1879, 2203, 2683, 2791, 2887, 2917, 3463, 3967, 4111, 4273, 4549, 4969, 5059, 5101, 6397, 6733, 6907, 7237, 8017, 8167, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates257_checked : candidates edge 5 0 1657 tail258 best.bound = rootCandidates257 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext q; rfl
  rw [row]
  decide

def innerTail257_22 : List Nat := []
theorem innerScan257_22 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_22 best = best := rfl

def innerTail257_21 : List Nat := 8269 :: innerTail257_22
def innerCandidates257_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates257_21_checked : candidates edge 4 1657 8269 innerTail257_22 best.bound = innerCandidates257_21 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_21 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_21 best = best := by
  rw [innerTail257_21, scan, if_neg (by decide), innerCandidates257_21_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_22
    (if 3 ≤ innerCandidates257_21.length then search edge 3 [1657, 8269] innerCandidates257_21 9926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_22

def innerTail257_20 : List Nat := 8167 :: innerTail257_21
def innerCandidates257_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates257_20_checked : candidates edge 4 1657 8167 innerTail257_21 best.bound = innerCandidates257_20 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_20 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_20 best = best := by
  rw [innerTail257_20, scan, if_neg (by decide), innerCandidates257_20_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_21
    (if 3 ≤ innerCandidates257_20.length then search edge 3 [1657, 8167] innerCandidates257_20 9824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_21

def innerTail257_19 : List Nat := 8017 :: innerTail257_20
def innerCandidates257_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates257_19_checked : candidates edge 4 1657 8017 innerTail257_20 best.bound = innerCandidates257_19 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_19 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_19 best = best := by
  rw [innerTail257_19, scan, if_neg (by decide), innerCandidates257_19_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_20
    (if 3 ≤ innerCandidates257_19.length then search edge 3 [1657, 8017] innerCandidates257_19 9674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_20

def innerTail257_18 : List Nat := 7237 :: innerTail257_19
def innerCandidates257_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates257_18_checked : candidates edge 4 1657 7237 innerTail257_19 best.bound = innerCandidates257_18 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_18 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_18 best = best := by
  rw [innerTail257_18, scan, if_neg (by decide), innerCandidates257_18_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_19
    (if 3 ≤ innerCandidates257_18.length then search edge 3 [1657, 7237] innerCandidates257_18 8894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_19

def innerTail257_17 : List Nat := 6907 :: innerTail257_18
def innerCandidates257_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates257_17_checked : candidates edge 4 1657 6907 innerTail257_18 best.bound = innerCandidates257_17 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_17 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_17 best = best := by
  rw [innerTail257_17, scan, if_neg (by decide), innerCandidates257_17_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_18
    (if 3 ≤ innerCandidates257_17.length then search edge 3 [1657, 6907] innerCandidates257_17 8564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_18

def innerTail257_16 : List Nat := 6733 :: innerTail257_17
def innerCandidates257_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates257_16_checked : candidates edge 4 1657 6733 innerTail257_17 best.bound = innerCandidates257_16 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_16 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_16 best = best := by
  rw [innerTail257_16, scan, if_neg (by decide), innerCandidates257_16_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_17
    (if 3 ≤ innerCandidates257_16.length then search edge 3 [1657, 6733] innerCandidates257_16 8390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_17

def innerTail257_15 : List Nat := 6397 :: innerTail257_16
def innerCandidates257_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates257_15_checked : candidates edge 4 1657 6397 innerTail257_16 best.bound = innerCandidates257_15 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_15 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_15 best = best := by
  rw [innerTail257_15, scan, if_neg (by decide), innerCandidates257_15_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_16
    (if 3 ≤ innerCandidates257_15.length then search edge 3 [1657, 6397] innerCandidates257_15 8054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_16

def innerTail257_14 : List Nat := 5101 :: innerTail257_15
def innerCandidates257_14 : List Nat := [6397, 7237]
set_option maxRecDepth 4096 in
theorem innerCandidates257_14_checked : candidates edge 4 1657 5101 innerTail257_15 best.bound = innerCandidates257_14 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_14 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_14 best = best := by
  rw [innerTail257_14, scan, if_neg (by decide), innerCandidates257_14_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_15
    (if 3 ≤ innerCandidates257_14.length then search edge 3 [1657, 5101] innerCandidates257_14 6758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_15

def innerTail257_13 : List Nat := 5059 :: innerTail257_14
def innerCandidates257_13 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates257_13_checked : candidates edge 4 1657 5059 innerTail257_14 best.bound = innerCandidates257_13 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_13 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_13 best = best := by
  rw [innerTail257_13, scan, if_neg (by decide), innerCandidates257_13_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_14
    (if 3 ≤ innerCandidates257_13.length then search edge 3 [1657, 5059] innerCandidates257_13 6716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_14

def innerTail257_12 : List Nat := 4969 :: innerTail257_13
def innerCandidates257_12 : List Nat := [6397]
set_option maxRecDepth 4096 in
theorem innerCandidates257_12_checked : candidates edge 4 1657 4969 innerTail257_13 best.bound = innerCandidates257_12 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_12 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_12 best = best := by
  rw [innerTail257_12, scan, if_neg (by decide), innerCandidates257_12_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_13
    (if 3 ≤ innerCandidates257_12.length then search edge 3 [1657, 4969] innerCandidates257_12 6626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_13

def innerTail257_11 : List Nat := 4549 :: innerTail257_12
def innerCandidates257_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates257_11_checked : candidates edge 4 1657 4549 innerTail257_12 best.bound = innerCandidates257_11 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_11 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_11 best = best := by
  rw [innerTail257_11, scan, if_neg (by decide), innerCandidates257_11_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_12
    (if 3 ≤ innerCandidates257_11.length then search edge 3 [1657, 4549] innerCandidates257_11 6206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_12

def innerTail257_10 : List Nat := 4273 :: innerTail257_11
def innerCandidates257_10 : List Nat := [4969]
set_option maxRecDepth 4096 in
theorem innerCandidates257_10_checked : candidates edge 4 1657 4273 innerTail257_11 best.bound = innerCandidates257_10 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_10 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_10 best = best := by
  rw [innerTail257_10, scan, if_neg (by decide), innerCandidates257_10_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_11
    (if 3 ≤ innerCandidates257_10.length then search edge 3 [1657, 4273] innerCandidates257_10 5930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_11

def innerTail257_9 : List Nat := 4111 :: innerTail257_10
def innerCandidates257_9 : List Nat := [8167]
set_option maxRecDepth 4096 in
theorem innerCandidates257_9_checked : candidates edge 4 1657 4111 innerTail257_10 best.bound = innerCandidates257_9 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_9 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_9 best = best := by
  rw [innerTail257_9, scan, if_neg (by decide), innerCandidates257_9_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_10
    (if 3 ≤ innerCandidates257_9.length then search edge 3 [1657, 4111] innerCandidates257_9 5768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_10

def innerTail257_8 : List Nat := 3967 :: innerTail257_9
def innerCandidates257_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates257_8_checked : candidates edge 4 1657 3967 innerTail257_9 best.bound = innerCandidates257_8 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_8 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_8 best = best := by
  rw [innerTail257_8, scan, if_neg (by decide), innerCandidates257_8_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_9
    (if 3 ≤ innerCandidates257_8.length then search edge 3 [1657, 3967] innerCandidates257_8 5624 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_9

def innerTail257_7 : List Nat := 3463 :: innerTail257_8
def innerCandidates257_7 : List Nat := [4969]
set_option maxRecDepth 4096 in
theorem innerCandidates257_7_checked : candidates edge 4 1657 3463 innerTail257_8 best.bound = innerCandidates257_7 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_7 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_7 best = best := by
  rw [innerTail257_7, scan, if_neg (by decide), innerCandidates257_7_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_8
    (if 3 ≤ innerCandidates257_7.length then search edge 3 [1657, 3463] innerCandidates257_7 5120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_8

def innerTail257_6 : List Nat := 2917 :: innerTail257_7
def innerCandidates257_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates257_6_checked : candidates edge 4 1657 2917 innerTail257_7 best.bound = innerCandidates257_6 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_6 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_6 best = best := by
  rw [innerTail257_6, scan, if_neg (by decide), innerCandidates257_6_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_7
    (if 3 ≤ innerCandidates257_6.length then search edge 3 [1657, 2917] innerCandidates257_6 4574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_7

def innerTail257_5 : List Nat := 2887 :: innerTail257_6
def innerCandidates257_5 : List Nat := [2917, 4273, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates257_5_checked : candidates edge 4 1657 2887 innerTail257_6 best.bound = innerCandidates257_5 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild257_5 : search edge 3 [1657, 2887] innerCandidates257_5 4544 best = best := by decide

theorem innerScan257_5 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_5 best = best := by
  rw [innerTail257_5, scan, if_neg (by decide), innerCandidates257_5_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_6
    (if 3 ≤ innerCandidates257_5.length then search edge 3 [1657, 2887] innerCandidates257_5 4544 best else best) = best
  rw [if_pos (by decide), innerChild257_5]
  exact innerScan257_6

def innerTail257_4 : List Nat := 2791 :: innerTail257_5
def innerCandidates257_4 : List Nat := [3967]
set_option maxRecDepth 4096 in
theorem innerCandidates257_4_checked : candidates edge 4 1657 2791 innerTail257_5 best.bound = innerCandidates257_4 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_4 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_4 best = best := by
  rw [innerTail257_4, scan, if_neg (by decide), innerCandidates257_4_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_5
    (if 3 ≤ innerCandidates257_4.length then search edge 3 [1657, 2791] innerCandidates257_4 4448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_5

def innerTail257_3 : List Nat := 2683 :: innerTail257_4
def innerCandidates257_3 : List Nat := [2791, 8167]
set_option maxRecDepth 4096 in
theorem innerCandidates257_3_checked : candidates edge 4 1657 2683 innerTail257_4 best.bound = innerCandidates257_3 := by
  unfold candidates
  have row : edge 2683 = SearchData.row2683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_3 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_3 best = best := by
  rw [innerTail257_3, scan, if_neg (by decide), innerCandidates257_3_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_4
    (if 3 ≤ innerCandidates257_3.length then search edge 3 [1657, 2683] innerCandidates257_3 4340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_4

def innerTail257_2 : List Nat := 2203 :: innerTail257_3
def innerCandidates257_2 : List Nat := [2887]
set_option maxRecDepth 4096 in
theorem innerCandidates257_2_checked : candidates edge 4 1657 2203 innerTail257_3 best.bound = innerCandidates257_2 := by
  unfold candidates
  have row : edge 2203 = PrimePairCache.smallRow2203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_2 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_2 best = best := by
  rw [innerTail257_2, scan, if_neg (by decide), innerCandidates257_2_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_3
    (if 3 ≤ innerCandidates257_2.length then search edge 3 [1657, 2203] innerCandidates257_2 3860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_3

def innerTail257_1 : List Nat := 1879 :: innerTail257_2
def innerCandidates257_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates257_1_checked : candidates edge 4 1657 1879 innerTail257_2 best.bound = innerCandidates257_1 := by
  unfold candidates
  have row : edge 1879 = PrimePairCache.smallRow1879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_1 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_1 best = best := by
  rw [innerTail257_1, scan, if_neg (by decide), innerCandidates257_1_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_2
    (if 3 ≤ innerCandidates257_1.length then search edge 3 [1657, 1879] innerCandidates257_1 3536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_2

def innerTail257_0 : List Nat := 1699 :: innerTail257_1
def innerCandidates257_0 : List Nat := [2203]
set_option maxRecDepth 4096 in
theorem innerCandidates257_0_checked : candidates edge 4 1657 1699 innerTail257_1 best.bound = innerCandidates257_0 := by
  unfold candidates
  have row : edge 1699 = PrimePairCache.smallRow1699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan257_0 : scan edge (search edge 3) 4 [1657] 1657 innerTail257_0 best = best := by
  rw [innerTail257_0, scan, if_neg (by decide), innerCandidates257_0_checked]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_1
    (if 3 ≤ innerCandidates257_0.length then search edge 3 [1657, 1699] innerCandidates257_0 3356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan257_1

theorem rootBranch257 : search edge 4 [1657] rootCandidates257 1657 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1657] 1657 innerTail257_0 best = best
  exact innerScan257_0

def rootCandidates256 : List Nat := [1709, 1889, 2903, 3359, 3911, 4073, 4649, 4931, 4973, 5669, 5897, 6053, 7013, 7607, 7841, 8009, 8039, 8237, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates256_checked : candidates edge 5 0 1637 tail257 best.bound = rootCandidates256 := by
  unfold candidates
  have row : edge 1637 = PrimePairCache.smallRow1637.contains := by funext q; rfl
  rw [row]
  decide

def innerTail256_19 : List Nat := []
theorem innerScan256_19 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_19 best = best := rfl

def innerTail256_18 : List Nat := 8363 :: innerTail256_19
def innerCandidates256_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_18_checked : candidates edge 4 1637 8363 innerTail256_19 best.bound = innerCandidates256_18 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_18 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_18 best = best := by
  rw [innerTail256_18, scan, if_neg (by decide), innerCandidates256_18_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_19
    (if 3 ≤ innerCandidates256_18.length then search edge 3 [1637, 8363] innerCandidates256_18 10000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_19

def innerTail256_17 : List Nat := 8237 :: innerTail256_18
def innerCandidates256_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_17_checked : candidates edge 4 1637 8237 innerTail256_18 best.bound = innerCandidates256_17 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_17 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_17 best = best := by
  rw [innerTail256_17, scan, if_neg (by decide), innerCandidates256_17_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_18
    (if 3 ≤ innerCandidates256_17.length then search edge 3 [1637, 8237] innerCandidates256_17 9874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_18

def innerTail256_16 : List Nat := 8039 :: innerTail256_17
def innerCandidates256_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_16_checked : candidates edge 4 1637 8039 innerTail256_17 best.bound = innerCandidates256_16 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_16 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_16 best = best := by
  rw [innerTail256_16, scan, if_neg (by decide), innerCandidates256_16_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_17
    (if 3 ≤ innerCandidates256_16.length then search edge 3 [1637, 8039] innerCandidates256_16 9676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_17

def innerTail256_15 : List Nat := 8009 :: innerTail256_16
def innerCandidates256_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_15_checked : candidates edge 4 1637 8009 innerTail256_16 best.bound = innerCandidates256_15 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_15 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_15 best = best := by
  rw [innerTail256_15, scan, if_neg (by decide), innerCandidates256_15_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_16
    (if 3 ≤ innerCandidates256_15.length then search edge 3 [1637, 8009] innerCandidates256_15 9646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_16

def innerTail256_14 : List Nat := 7841 :: innerTail256_15
def innerCandidates256_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_14_checked : candidates edge 4 1637 7841 innerTail256_15 best.bound = innerCandidates256_14 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_14 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_14 best = best := by
  rw [innerTail256_14, scan, if_neg (by decide), innerCandidates256_14_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_15
    (if 3 ≤ innerCandidates256_14.length then search edge 3 [1637, 7841] innerCandidates256_14 9478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_15

def innerTail256_13 : List Nat := 7607 :: innerTail256_14
def innerCandidates256_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_13_checked : candidates edge 4 1637 7607 innerTail256_14 best.bound = innerCandidates256_13 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_13 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_13 best = best := by
  rw [innerTail256_13, scan, if_neg (by decide), innerCandidates256_13_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_14
    (if 3 ≤ innerCandidates256_13.length then search edge 3 [1637, 7607] innerCandidates256_13 9244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_14

def innerTail256_12 : List Nat := 7013 :: innerTail256_13
def innerCandidates256_12 : List Nat := [8363]
set_option maxRecDepth 4096 in
theorem innerCandidates256_12_checked : candidates edge 4 1637 7013 innerTail256_13 best.bound = innerCandidates256_12 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_12 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_12 best = best := by
  rw [innerTail256_12, scan, if_neg (by decide), innerCandidates256_12_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_13
    (if 3 ≤ innerCandidates256_12.length then search edge 3 [1637, 7013] innerCandidates256_12 8650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_13

def innerTail256_11 : List Nat := 6053 :: innerTail256_12
def innerCandidates256_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_11_checked : candidates edge 4 1637 6053 innerTail256_12 best.bound = innerCandidates256_11 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_11 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_11 best = best := by
  rw [innerTail256_11, scan, if_neg (by decide), innerCandidates256_11_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_12
    (if 3 ≤ innerCandidates256_11.length then search edge 3 [1637, 6053] innerCandidates256_11 7690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_12

def innerTail256_10 : List Nat := 5897 :: innerTail256_11
def innerCandidates256_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_10_checked : candidates edge 4 1637 5897 innerTail256_11 best.bound = innerCandidates256_10 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_10 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_10 best = best := by
  rw [innerTail256_10, scan, if_neg (by decide), innerCandidates256_10_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_11
    (if 3 ≤ innerCandidates256_10.length then search edge 3 [1637, 5897] innerCandidates256_10 7534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_11

def innerTail256_9 : List Nat := 5669 :: innerTail256_10
def innerCandidates256_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_9_checked : candidates edge 4 1637 5669 innerTail256_10 best.bound = innerCandidates256_9 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_9 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_9 best = best := by
  rw [innerTail256_9, scan, if_neg (by decide), innerCandidates256_9_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_10
    (if 3 ≤ innerCandidates256_9.length then search edge 3 [1637, 5669] innerCandidates256_9 7306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_10

def innerTail256_8 : List Nat := 4973 :: innerTail256_9
def innerCandidates256_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_8_checked : candidates edge 4 1637 4973 innerTail256_9 best.bound = innerCandidates256_8 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_8 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_8 best = best := by
  rw [innerTail256_8, scan, if_neg (by decide), innerCandidates256_8_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_9
    (if 3 ≤ innerCandidates256_8.length then search edge 3 [1637, 4973] innerCandidates256_8 6610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_9

def innerTail256_7 : List Nat := 4931 :: innerTail256_8
def innerCandidates256_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_7_checked : candidates edge 4 1637 4931 innerTail256_8 best.bound = innerCandidates256_7 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_7 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_7 best = best := by
  rw [innerTail256_7, scan, if_neg (by decide), innerCandidates256_7_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_8
    (if 3 ≤ innerCandidates256_7.length then search edge 3 [1637, 4931] innerCandidates256_7 6568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_8

def innerTail256_6 : List Nat := 4649 :: innerTail256_7
def innerCandidates256_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_6_checked : candidates edge 4 1637 4649 innerTail256_7 best.bound = innerCandidates256_6 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_6 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_6 best = best := by
  rw [innerTail256_6, scan, if_neg (by decide), innerCandidates256_6_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_7
    (if 3 ≤ innerCandidates256_6.length then search edge 3 [1637, 4649] innerCandidates256_6 6286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_7

def innerTail256_5 : List Nat := 4073 :: innerTail256_6
def innerCandidates256_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_5_checked : candidates edge 4 1637 4073 innerTail256_6 best.bound = innerCandidates256_5 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_5 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_5 best = best := by
  rw [innerTail256_5, scan, if_neg (by decide), innerCandidates256_5_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_6
    (if 3 ≤ innerCandidates256_5.length then search edge 3 [1637, 4073] innerCandidates256_5 5710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_6

def innerTail256_4 : List Nat := 3911 :: innerTail256_5
def innerCandidates256_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_4_checked : candidates edge 4 1637 3911 innerTail256_5 best.bound = innerCandidates256_4 := by
  unfold candidates
  have row : edge 3911 = SearchData.row3911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_4 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_4 best = best := by
  rw [innerTail256_4, scan, if_neg (by decide), innerCandidates256_4_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_5
    (if 3 ≤ innerCandidates256_4.length then search edge 3 [1637, 3911] innerCandidates256_4 5548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_5

def innerTail256_3 : List Nat := 3359 :: innerTail256_4
def innerCandidates256_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_3_checked : candidates edge 4 1637 3359 innerTail256_4 best.bound = innerCandidates256_3 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_3 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_3 best = best := by
  rw [innerTail256_3, scan, if_neg (by decide), innerCandidates256_3_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_4
    (if 3 ≤ innerCandidates256_3.length then search edge 3 [1637, 3359] innerCandidates256_3 4996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_4

def innerTail256_2 : List Nat := 2903 :: innerTail256_3
def innerCandidates256_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_2_checked : candidates edge 4 1637 2903 innerTail256_3 best.bound = innerCandidates256_2 := by
  unfold candidates
  have row : edge 2903 = SearchData.row2903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_2 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_2 best = best := by
  rw [innerTail256_2, scan, if_neg (by decide), innerCandidates256_2_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_3
    (if 3 ≤ innerCandidates256_2.length then search edge 3 [1637, 2903] innerCandidates256_2 4540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_3

def innerTail256_1 : List Nat := 1889 :: innerTail256_2
def innerCandidates256_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_1_checked : candidates edge 4 1637 1889 innerTail256_2 best.bound = innerCandidates256_1 := by
  unfold candidates
  have row : edge 1889 = PrimePairCache.smallRow1889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_1 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_1 best = best := by
  rw [innerTail256_1, scan, if_neg (by decide), innerCandidates256_1_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_2
    (if 3 ≤ innerCandidates256_1.length then search edge 3 [1637, 1889] innerCandidates256_1 3526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_2

def innerTail256_0 : List Nat := 1709 :: innerTail256_1
def innerCandidates256_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates256_0_checked : candidates edge 4 1637 1709 innerTail256_1 best.bound = innerCandidates256_0 := by
  unfold candidates
  have row : edge 1709 = PrimePairCache.smallRow1709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan256_0 : scan edge (search edge 3) 4 [1637] 1637 innerTail256_0 best = best := by
  rw [innerTail256_0, scan, if_neg (by decide), innerCandidates256_0_checked]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_1
    (if 3 ≤ innerCandidates256_0.length then search edge 3 [1637, 1709] innerCandidates256_0 3346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan256_1

theorem rootBranch256 : search edge 4 [1637] rootCandidates256 1637 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1637] 1637 innerTail256_0 best = best
  exact innerScan256_0

end ProjectEulerSolutions.PrimePairUpper.Search
