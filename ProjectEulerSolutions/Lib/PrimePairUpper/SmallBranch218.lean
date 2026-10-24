import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates66 : List Nat := [401, 443, 983, 1031, 1301, 1697, 1847, 2213, 3677, 3719, 3923, 4517, 5477, 5861, 5981, 6257, 6569, 6719, 6761, 7121, 7499, 7517, 7559, 7841, 8009]
set_option maxRecDepth 4096 in
theorem rootCandidates66_checked : candidates edge 5 0 347 tail67 best.bound = rootCandidates66 := by
  unfold candidates
  have row : edge 347 = PrimePairCache.smallRow347.contains := by funext q; rfl
  rw [row]
  decide

def innerTail66_25 : List Nat := []
theorem innerScan66_25 : scan edge (search edge 3) 4 [347] 347 innerTail66_25 best = best := rfl

def innerTail66_24 : List Nat := 8009 :: innerTail66_25
def innerCandidates66_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_24_checked : candidates edge 4 347 8009 innerTail66_25 best.bound = innerCandidates66_24 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_24 : scan edge (search edge 3) 4 [347] 347 innerTail66_24 best = best := by
  rw [innerTail66_24, scan, if_neg (by decide), innerCandidates66_24_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_25
    (if 3 ≤ innerCandidates66_24.length then search edge 3 [347, 8009] innerCandidates66_24 8356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_25

def innerTail66_23 : List Nat := 7841 :: innerTail66_24
def innerCandidates66_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_23_checked : candidates edge 4 347 7841 innerTail66_24 best.bound = innerCandidates66_23 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_23 : scan edge (search edge 3) 4 [347] 347 innerTail66_23 best = best := by
  rw [innerTail66_23, scan, if_neg (by decide), innerCandidates66_23_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_24
    (if 3 ≤ innerCandidates66_23.length then search edge 3 [347, 7841] innerCandidates66_23 8188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_24

def innerTail66_22 : List Nat := 7559 :: innerTail66_23
def innerCandidates66_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_22_checked : candidates edge 4 347 7559 innerTail66_23 best.bound = innerCandidates66_22 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_22 : scan edge (search edge 3) 4 [347] 347 innerTail66_22 best = best := by
  rw [innerTail66_22, scan, if_neg (by decide), innerCandidates66_22_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_23
    (if 3 ≤ innerCandidates66_22.length then search edge 3 [347, 7559] innerCandidates66_22 7906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_23

def innerTail66_21 : List Nat := 7517 :: innerTail66_22
def innerCandidates66_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_21_checked : candidates edge 4 347 7517 innerTail66_22 best.bound = innerCandidates66_21 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_21 : scan edge (search edge 3) 4 [347] 347 innerTail66_21 best = best := by
  rw [innerTail66_21, scan, if_neg (by decide), innerCandidates66_21_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_22
    (if 3 ≤ innerCandidates66_21.length then search edge 3 [347, 7517] innerCandidates66_21 7864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_22

def innerTail66_20 : List Nat := 7499 :: innerTail66_21
def innerCandidates66_20 : List Nat := [7841]
set_option maxRecDepth 4096 in
theorem innerCandidates66_20_checked : candidates edge 4 347 7499 innerTail66_21 best.bound = innerCandidates66_20 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_20 : scan edge (search edge 3) 4 [347] 347 innerTail66_20 best = best := by
  rw [innerTail66_20, scan, if_neg (by decide), innerCandidates66_20_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_21
    (if 3 ≤ innerCandidates66_20.length then search edge 3 [347, 7499] innerCandidates66_20 7846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_21

def innerTail66_19 : List Nat := 7121 :: innerTail66_20
def innerCandidates66_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_19_checked : candidates edge 4 347 7121 innerTail66_20 best.bound = innerCandidates66_19 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_19 : scan edge (search edge 3) 4 [347] 347 innerTail66_19 best = best := by
  rw [innerTail66_19, scan, if_neg (by decide), innerCandidates66_19_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_20
    (if 3 ≤ innerCandidates66_19.length then search edge 3 [347, 7121] innerCandidates66_19 7468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_20

def innerTail66_18 : List Nat := 6761 :: innerTail66_19
def innerCandidates66_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_18_checked : candidates edge 4 347 6761 innerTail66_19 best.bound = innerCandidates66_18 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_18 : scan edge (search edge 3) 4 [347] 347 innerTail66_18 best = best := by
  rw [innerTail66_18, scan, if_neg (by decide), innerCandidates66_18_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_19
    (if 3 ≤ innerCandidates66_18.length then search edge 3 [347, 6761] innerCandidates66_18 7108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_19

def innerTail66_17 : List Nat := 6719 :: innerTail66_18
def innerCandidates66_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_17_checked : candidates edge 4 347 6719 innerTail66_18 best.bound = innerCandidates66_17 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_17 : scan edge (search edge 3) 4 [347] 347 innerTail66_17 best = best := by
  rw [innerTail66_17, scan, if_neg (by decide), innerCandidates66_17_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_18
    (if 3 ≤ innerCandidates66_17.length then search edge 3 [347, 6719] innerCandidates66_17 7066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_18

def innerTail66_16 : List Nat := 6569 :: innerTail66_17
def innerCandidates66_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_16_checked : candidates edge 4 347 6569 innerTail66_17 best.bound = innerCandidates66_16 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_16 : scan edge (search edge 3) 4 [347] 347 innerTail66_16 best = best := by
  rw [innerTail66_16, scan, if_neg (by decide), innerCandidates66_16_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_17
    (if 3 ≤ innerCandidates66_16.length then search edge 3 [347, 6569] innerCandidates66_16 6916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_17

def innerTail66_15 : List Nat := 6257 :: innerTail66_16
def innerCandidates66_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_15_checked : candidates edge 4 347 6257 innerTail66_16 best.bound = innerCandidates66_15 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_15 : scan edge (search edge 3) 4 [347] 347 innerTail66_15 best = best := by
  rw [innerTail66_15, scan, if_neg (by decide), innerCandidates66_15_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_16
    (if 3 ≤ innerCandidates66_15.length then search edge 3 [347, 6257] innerCandidates66_15 6604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_16

def innerTail66_14 : List Nat := 5981 :: innerTail66_15
def innerCandidates66_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_14_checked : candidates edge 4 347 5981 innerTail66_15 best.bound = innerCandidates66_14 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_14 : scan edge (search edge 3) 4 [347] 347 innerTail66_14 best = best := by
  rw [innerTail66_14, scan, if_neg (by decide), innerCandidates66_14_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_15
    (if 3 ≤ innerCandidates66_14.length then search edge 3 [347, 5981] innerCandidates66_14 6328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_15

def innerTail66_13 : List Nat := 5861 :: innerTail66_14
def innerCandidates66_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_13_checked : candidates edge 4 347 5861 innerTail66_14 best.bound = innerCandidates66_13 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_13 : scan edge (search edge 3) 4 [347] 347 innerTail66_13 best = best := by
  rw [innerTail66_13, scan, if_neg (by decide), innerCandidates66_13_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_14
    (if 3 ≤ innerCandidates66_13.length then search edge 3 [347, 5861] innerCandidates66_13 6208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_14

def innerTail66_12 : List Nat := 5477 :: innerTail66_13
def innerCandidates66_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_12_checked : candidates edge 4 347 5477 innerTail66_13 best.bound = innerCandidates66_12 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_12 : scan edge (search edge 3) 4 [347] 347 innerTail66_12 best = best := by
  rw [innerTail66_12, scan, if_neg (by decide), innerCandidates66_12_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_13
    (if 3 ≤ innerCandidates66_12.length then search edge 3 [347, 5477] innerCandidates66_12 5824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_13

def innerTail66_11 : List Nat := 4517 :: innerTail66_12
def innerCandidates66_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_11_checked : candidates edge 4 347 4517 innerTail66_12 best.bound = innerCandidates66_11 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_11 : scan edge (search edge 3) 4 [347] 347 innerTail66_11 best = best := by
  rw [innerTail66_11, scan, if_neg (by decide), innerCandidates66_11_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_12
    (if 3 ≤ innerCandidates66_11.length then search edge 3 [347, 4517] innerCandidates66_11 4864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_12

def innerTail66_10 : List Nat := 3923 :: innerTail66_11
def innerCandidates66_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_10_checked : candidates edge 4 347 3923 innerTail66_11 best.bound = innerCandidates66_10 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_10 : scan edge (search edge 3) 4 [347] 347 innerTail66_10 best = best := by
  rw [innerTail66_10, scan, if_neg (by decide), innerCandidates66_10_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_11
    (if 3 ≤ innerCandidates66_10.length then search edge 3 [347, 3923] innerCandidates66_10 4270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_11

def innerTail66_9 : List Nat := 3719 :: innerTail66_10
def innerCandidates66_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_9_checked : candidates edge 4 347 3719 innerTail66_10 best.bound = innerCandidates66_9 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_9 : scan edge (search edge 3) 4 [347] 347 innerTail66_9 best = best := by
  rw [innerTail66_9, scan, if_neg (by decide), innerCandidates66_9_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_10
    (if 3 ≤ innerCandidates66_9.length then search edge 3 [347, 3719] innerCandidates66_9 4066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_10

def innerTail66_8 : List Nat := 3677 :: innerTail66_9
def innerCandidates66_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_8_checked : candidates edge 4 347 3677 innerTail66_9 best.bound = innerCandidates66_8 := by
  unfold candidates
  have row : edge 3677 = SearchData.row3677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_8 : scan edge (search edge 3) 4 [347] 347 innerTail66_8 best = best := by
  rw [innerTail66_8, scan, if_neg (by decide), innerCandidates66_8_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_9
    (if 3 ≤ innerCandidates66_8.length then search edge 3 [347, 3677] innerCandidates66_8 4024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_9

def innerTail66_7 : List Nat := 2213 :: innerTail66_8
def innerCandidates66_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_7_checked : candidates edge 4 347 2213 innerTail66_8 best.bound = innerCandidates66_7 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_7 : scan edge (search edge 3) 4 [347] 347 innerTail66_7 best = best := by
  rw [innerTail66_7, scan, if_neg (by decide), innerCandidates66_7_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_8
    (if 3 ≤ innerCandidates66_7.length then search edge 3 [347, 2213] innerCandidates66_7 2560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_8

def innerTail66_6 : List Nat := 1847 :: innerTail66_7
def innerCandidates66_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_6_checked : candidates edge 4 347 1847 innerTail66_7 best.bound = innerCandidates66_6 := by
  unfold candidates
  have row : edge 1847 = PrimePairCache.smallRow1847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_6 : scan edge (search edge 3) 4 [347] 347 innerTail66_6 best = best := by
  rw [innerTail66_6, scan, if_neg (by decide), innerCandidates66_6_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_7
    (if 3 ≤ innerCandidates66_6.length then search edge 3 [347, 1847] innerCandidates66_6 2194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_7

def innerTail66_5 : List Nat := 1697 :: innerTail66_6
def innerCandidates66_5 : List Nat := [5477]
set_option maxRecDepth 4096 in
theorem innerCandidates66_5_checked : candidates edge 4 347 1697 innerTail66_6 best.bound = innerCandidates66_5 := by
  unfold candidates
  have row : edge 1697 = PrimePairCache.smallRow1697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_5 : scan edge (search edge 3) 4 [347] 347 innerTail66_5 best = best := by
  rw [innerTail66_5, scan, if_neg (by decide), innerCandidates66_5_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_6
    (if 3 ≤ innerCandidates66_5.length then search edge 3 [347, 1697] innerCandidates66_5 2044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_6

def innerTail66_4 : List Nat := 1301 :: innerTail66_5
def innerCandidates66_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_4_checked : candidates edge 4 347 1301 innerTail66_5 best.bound = innerCandidates66_4 := by
  unfold candidates
  have row : edge 1301 = PrimePairCache.smallRow1301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_4 : scan edge (search edge 3) 4 [347] 347 innerTail66_4 best = best := by
  rw [innerTail66_4, scan, if_neg (by decide), innerCandidates66_4_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_5
    (if 3 ≤ innerCandidates66_4.length then search edge 3 [347, 1301] innerCandidates66_4 1648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_5

def innerTail66_3 : List Nat := 1031 :: innerTail66_4
def innerCandidates66_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates66_3_checked : candidates edge 4 347 1031 innerTail66_4 best.bound = innerCandidates66_3 := by
  unfold candidates
  have row : edge 1031 = PrimePairCache.smallRow1031.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_3 : scan edge (search edge 3) 4 [347] 347 innerTail66_3 best = best := by
  rw [innerTail66_3, scan, if_neg (by decide), innerCandidates66_3_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_4
    (if 3 ≤ innerCandidates66_3.length then search edge 3 [347, 1031] innerCandidates66_3 1378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_4

def innerTail66_2 : List Nat := 983 :: innerTail66_3
def innerCandidates66_2 : List Nat := [6719]
set_option maxRecDepth 4096 in
theorem innerCandidates66_2_checked : candidates edge 4 347 983 innerTail66_3 best.bound = innerCandidates66_2 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_2 : scan edge (search edge 3) 4 [347] 347 innerTail66_2 best = best := by
  rw [innerTail66_2, scan, if_neg (by decide), innerCandidates66_2_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_3
    (if 3 ≤ innerCandidates66_2.length then search edge 3 [347, 983] innerCandidates66_2 1330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_3

def innerTail66_1 : List Nat := 443 :: innerTail66_2
def innerCandidates66_1 : List Nat := [983]
set_option maxRecDepth 4096 in
theorem innerCandidates66_1_checked : candidates edge 4 347 443 innerTail66_2 best.bound = innerCandidates66_1 := by
  unfold candidates
  have row : edge 443 = PrimePairCache.smallRow443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_1 : scan edge (search edge 3) 4 [347] 347 innerTail66_1 best = best := by
  rw [innerTail66_1, scan, if_neg (by decide), innerCandidates66_1_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_2
    (if 3 ≤ innerCandidates66_1.length then search edge 3 [347, 443] innerCandidates66_1 790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_2

def innerTail66_0 : List Nat := 401 :: innerTail66_1
def innerCandidates66_0 : List Nat := [2213]
set_option maxRecDepth 4096 in
theorem innerCandidates66_0_checked : candidates edge 4 347 401 innerTail66_1 best.bound = innerCandidates66_0 := by
  unfold candidates
  have row : edge 401 = PrimePairCache.smallRow401.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan66_0 : scan edge (search edge 3) 4 [347] 347 innerTail66_0 best = best := by
  rw [innerTail66_0, scan, if_neg (by decide), innerCandidates66_0_checked]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_1
    (if 3 ≤ innerCandidates66_0.length then search edge 3 [347, 401] innerCandidates66_0 748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan66_1

theorem rootBranch66 : search edge 4 [347] rootCandidates66 347 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [347] 347 innerTail66_0 best = best
  exact innerScan66_0

def rootCandidates65 : List Nat := [349, 397, 607, 691, 919, 1153, 1297, 1867, 2371, 2797, 3169, 3217, 3343, 3499, 4507, 4729, 5431, 5857, 6781, 6949, 7723]
set_option maxRecDepth 4096 in
theorem rootCandidates65_checked : candidates edge 5 0 337 tail66 best.bound = rootCandidates65 := by
  unfold candidates
  have row : edge 337 = PrimePairCache.smallRow337.contains := by funext q; rfl
  rw [row]
  decide

def innerTail65_21 : List Nat := []
theorem innerScan65_21 : scan edge (search edge 3) 4 [337] 337 innerTail65_21 best = best := rfl

def innerTail65_20 : List Nat := 7723 :: innerTail65_21
def innerCandidates65_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_20_checked : candidates edge 4 337 7723 innerTail65_21 best.bound = innerCandidates65_20 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_20 : scan edge (search edge 3) 4 [337] 337 innerTail65_20 best = best := by
  rw [innerTail65_20, scan, if_neg (by decide), innerCandidates65_20_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_21
    (if 3 ≤ innerCandidates65_20.length then search edge 3 [337, 7723] innerCandidates65_20 8060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_21

def innerTail65_19 : List Nat := 6949 :: innerTail65_20
def innerCandidates65_19 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates65_19_checked : candidates edge 4 337 6949 innerTail65_20 best.bound = innerCandidates65_19 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_19 : scan edge (search edge 3) 4 [337] 337 innerTail65_19 best = best := by
  rw [innerTail65_19, scan, if_neg (by decide), innerCandidates65_19_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_20
    (if 3 ≤ innerCandidates65_19.length then search edge 3 [337, 6949] innerCandidates65_19 7286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_20

def innerTail65_18 : List Nat := 6781 :: innerTail65_19
def innerCandidates65_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_18_checked : candidates edge 4 337 6781 innerTail65_19 best.bound = innerCandidates65_18 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_18 : scan edge (search edge 3) 4 [337] 337 innerTail65_18 best = best := by
  rw [innerTail65_18, scan, if_neg (by decide), innerCandidates65_18_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_19
    (if 3 ≤ innerCandidates65_18.length then search edge 3 [337, 6781] innerCandidates65_18 7118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_19

def innerTail65_17 : List Nat := 5857 :: innerTail65_18
def innerCandidates65_17 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates65_17_checked : candidates edge 4 337 5857 innerTail65_18 best.bound = innerCandidates65_17 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_17 : scan edge (search edge 3) 4 [337] 337 innerTail65_17 best = best := by
  rw [innerTail65_17, scan, if_neg (by decide), innerCandidates65_17_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_18
    (if 3 ≤ innerCandidates65_17.length then search edge 3 [337, 5857] innerCandidates65_17 6194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_18

def innerTail65_16 : List Nat := 5431 :: innerTail65_17
def innerCandidates65_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_16_checked : candidates edge 4 337 5431 innerTail65_17 best.bound = innerCandidates65_16 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_16 : scan edge (search edge 3) 4 [337] 337 innerTail65_16 best = best := by
  rw [innerTail65_16, scan, if_neg (by decide), innerCandidates65_16_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_17
    (if 3 ≤ innerCandidates65_16.length then search edge 3 [337, 5431] innerCandidates65_16 5768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_17

def innerTail65_15 : List Nat := 4729 :: innerTail65_16
def innerCandidates65_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_15_checked : candidates edge 4 337 4729 innerTail65_16 best.bound = innerCandidates65_15 := by
  unfold candidates
  have row : edge 4729 = SearchData.row4729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_15 : scan edge (search edge 3) 4 [337] 337 innerTail65_15 best = best := by
  rw [innerTail65_15, scan, if_neg (by decide), innerCandidates65_15_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_16
    (if 3 ≤ innerCandidates65_15.length then search edge 3 [337, 4729] innerCandidates65_15 5066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_16

def innerTail65_14 : List Nat := 4507 :: innerTail65_15
def innerCandidates65_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_14_checked : candidates edge 4 337 4507 innerTail65_15 best.bound = innerCandidates65_14 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_14 : scan edge (search edge 3) 4 [337] 337 innerTail65_14 best = best := by
  rw [innerTail65_14, scan, if_neg (by decide), innerCandidates65_14_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_15
    (if 3 ≤ innerCandidates65_14.length then search edge 3 [337, 4507] innerCandidates65_14 4844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_15

def innerTail65_13 : List Nat := 3499 :: innerTail65_14
def innerCandidates65_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_13_checked : candidates edge 4 337 3499 innerTail65_14 best.bound = innerCandidates65_13 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_13 : scan edge (search edge 3) 4 [337] 337 innerTail65_13 best = best := by
  rw [innerTail65_13, scan, if_neg (by decide), innerCandidates65_13_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_14
    (if 3 ≤ innerCandidates65_13.length then search edge 3 [337, 3499] innerCandidates65_13 3836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_14

def innerTail65_12 : List Nat := 3343 :: innerTail65_13
def innerCandidates65_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_12_checked : candidates edge 4 337 3343 innerTail65_13 best.bound = innerCandidates65_12 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_12 : scan edge (search edge 3) 4 [337] 337 innerTail65_12 best = best := by
  rw [innerTail65_12, scan, if_neg (by decide), innerCandidates65_12_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_13
    (if 3 ≤ innerCandidates65_12.length then search edge 3 [337, 3343] innerCandidates65_12 3680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_13

def innerTail65_11 : List Nat := 3217 :: innerTail65_12
def innerCandidates65_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_11_checked : candidates edge 4 337 3217 innerTail65_12 best.bound = innerCandidates65_11 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_11 : scan edge (search edge 3) 4 [337] 337 innerTail65_11 best = best := by
  rw [innerTail65_11, scan, if_neg (by decide), innerCandidates65_11_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_12
    (if 3 ≤ innerCandidates65_11.length then search edge 3 [337, 3217] innerCandidates65_11 3554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_12

def innerTail65_10 : List Nat := 3169 :: innerTail65_11
def innerCandidates65_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_10_checked : candidates edge 4 337 3169 innerTail65_11 best.bound = innerCandidates65_10 := by
  unfold candidates
  have row : edge 3169 = SearchData.row3169.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_10 : scan edge (search edge 3) 4 [337] 337 innerTail65_10 best = best := by
  rw [innerTail65_10, scan, if_neg (by decide), innerCandidates65_10_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_11
    (if 3 ≤ innerCandidates65_10.length then search edge 3 [337, 3169] innerCandidates65_10 3506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_11

def innerTail65_9 : List Nat := 2797 :: innerTail65_10
def innerCandidates65_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_9_checked : candidates edge 4 337 2797 innerTail65_10 best.bound = innerCandidates65_9 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_9 : scan edge (search edge 3) 4 [337] 337 innerTail65_9 best = best := by
  rw [innerTail65_9, scan, if_neg (by decide), innerCandidates65_9_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_10
    (if 3 ≤ innerCandidates65_9.length then search edge 3 [337, 2797] innerCandidates65_9 3134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_10

def innerTail65_8 : List Nat := 2371 :: innerTail65_9
def innerCandidates65_8 : List Nat := [3169]
set_option maxRecDepth 4096 in
theorem innerCandidates65_8_checked : candidates edge 4 337 2371 innerTail65_9 best.bound = innerCandidates65_8 := by
  unfold candidates
  have row : edge 2371 = PrimePairCache.smallRow2371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_8 : scan edge (search edge 3) 4 [337] 337 innerTail65_8 best = best := by
  rw [innerTail65_8, scan, if_neg (by decide), innerCandidates65_8_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_9
    (if 3 ≤ innerCandidates65_8.length then search edge 3 [337, 2371] innerCandidates65_8 2708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_9

def innerTail65_7 : List Nat := 1867 :: innerTail65_8
def innerCandidates65_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_7_checked : candidates edge 4 337 1867 innerTail65_8 best.bound = innerCandidates65_7 := by
  unfold candidates
  have row : edge 1867 = PrimePairCache.smallRow1867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_7 : scan edge (search edge 3) 4 [337] 337 innerTail65_7 best = best := by
  rw [innerTail65_7, scan, if_neg (by decide), innerCandidates65_7_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_8
    (if 3 ≤ innerCandidates65_7.length then search edge 3 [337, 1867] innerCandidates65_7 2204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_8

def innerTail65_6 : List Nat := 1297 :: innerTail65_7
def innerCandidates65_6 : List Nat := [6781]
set_option maxRecDepth 4096 in
theorem innerCandidates65_6_checked : candidates edge 4 337 1297 innerTail65_7 best.bound = innerCandidates65_6 := by
  unfold candidates
  have row : edge 1297 = PrimePairCache.smallRow1297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_6 : scan edge (search edge 3) 4 [337] 337 innerTail65_6 best = best := by
  rw [innerTail65_6, scan, if_neg (by decide), innerCandidates65_6_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_7
    (if 3 ≤ innerCandidates65_6.length then search edge 3 [337, 1297] innerCandidates65_6 1634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_7

def innerTail65_5 : List Nat := 1153 :: innerTail65_6
def innerCandidates65_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_5_checked : candidates edge 4 337 1153 innerTail65_6 best.bound = innerCandidates65_5 := by
  unfold candidates
  have row : edge 1153 = PrimePairCache.smallRow1153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_5 : scan edge (search edge 3) 4 [337] 337 innerTail65_5 best = best := by
  rw [innerTail65_5, scan, if_neg (by decide), innerCandidates65_5_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_6
    (if 3 ≤ innerCandidates65_5.length then search edge 3 [337, 1153] innerCandidates65_5 1490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_6

def innerTail65_4 : List Nat := 919 :: innerTail65_5
def innerCandidates65_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_4_checked : candidates edge 4 337 919 innerTail65_5 best.bound = innerCandidates65_4 := by
  unfold candidates
  have row : edge 919 = PrimePairCache.smallRow919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_4 : scan edge (search edge 3) 4 [337] 337 innerTail65_4 best = best := by
  rw [innerTail65_4, scan, if_neg (by decide), innerCandidates65_4_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_5
    (if 3 ≤ innerCandidates65_4.length then search edge 3 [337, 919] innerCandidates65_4 1256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_5

def innerTail65_3 : List Nat := 691 :: innerTail65_4
def innerCandidates65_3 : List Nat := [919, 4729, 5431]
set_option maxRecDepth 4096 in
theorem innerCandidates65_3_checked : candidates edge 4 337 691 innerTail65_4 best.bound = innerCandidates65_3 := by
  unfold candidates
  have row : edge 691 = PrimePairCache.smallRow691.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild65_3 : search edge 3 [337, 691] innerCandidates65_3 1028 best = best := by decide

theorem innerScan65_3 : scan edge (search edge 3) 4 [337] 337 innerTail65_3 best = best := by
  rw [innerTail65_3, scan, if_neg (by decide), innerCandidates65_3_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_4
    (if 3 ≤ innerCandidates65_3.length then search edge 3 [337, 691] innerCandidates65_3 1028 best else best) = best
  rw [if_pos (by decide), innerChild65_3]
  exact innerScan65_4

def innerTail65_2 : List Nat := 607 :: innerTail65_3
def innerCandidates65_2 : List Nat := [1297]
set_option maxRecDepth 4096 in
theorem innerCandidates65_2_checked : candidates edge 4 337 607 innerTail65_3 best.bound = innerCandidates65_2 := by
  unfold candidates
  have row : edge 607 = PrimePairCache.smallRow607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_2 : scan edge (search edge 3) 4 [337] 337 innerTail65_2 best = best := by
  rw [innerTail65_2, scan, if_neg (by decide), innerCandidates65_2_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_3
    (if 3 ≤ innerCandidates65_2.length then search edge 3 [337, 607] innerCandidates65_2 944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_3

def innerTail65_1 : List Nat := 397 :: innerTail65_2
def innerCandidates65_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates65_1_checked : candidates edge 4 337 397 innerTail65_2 best.bound = innerCandidates65_1 := by
  unfold candidates
  have row : edge 397 = PrimePairCache.smallRow397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_1 : scan edge (search edge 3) 4 [337] 337 innerTail65_1 best = best := by
  rw [innerTail65_1, scan, if_neg (by decide), innerCandidates65_1_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_2
    (if 3 ≤ innerCandidates65_1.length then search edge 3 [337, 397] innerCandidates65_1 734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_2

def innerTail65_0 : List Nat := 349 :: innerTail65_1
def innerCandidates65_0 : List Nat := [919]
set_option maxRecDepth 4096 in
theorem innerCandidates65_0_checked : candidates edge 4 337 349 innerTail65_1 best.bound = innerCandidates65_0 := by
  unfold candidates
  have row : edge 349 = PrimePairCache.smallRow349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan65_0 : scan edge (search edge 3) 4 [337] 337 innerTail65_0 best = best := by
  rw [innerTail65_0, scan, if_neg (by decide), innerCandidates65_0_checked]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_1
    (if 3 ≤ innerCandidates65_0.length then search edge 3 [337, 349] innerCandidates65_0 686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan65_1

theorem rootBranch65 : search edge 4 [337] rootCandidates65 337 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [337] 337 innerTail65_0 best = best
  exact innerScan65_0

end ProjectEulerSolutions.PrimePairUpper.Search
