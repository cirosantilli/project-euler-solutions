import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates291 : List Nat := [2399, 2957, 4289, 4943, 5171, 5279, 5297, 5477, 6719, 7043]
set_option maxRecDepth 4096 in
theorem rootCandidates291_checked : candidates edge 5 0 1931 tail292 best.bound = rootCandidates291 := by
  unfold candidates
  have row : edge 1931 = PrimePairCache.smallRow1931.contains := by funext q; rfl
  rw [row]
  decide

def innerTail291_10 : List Nat := []
theorem innerScan291_10 : scan edge (search edge 3) 4 [1931] 1931 innerTail291_10 best = best := rfl

def innerTail291_9 : List Nat := 7043 :: innerTail291_10
def innerCandidates291_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates291_9_checked : candidates edge 4 1931 7043 innerTail291_10 best.bound = innerCandidates291_9 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan291_9 : scan edge (search edge 3) 4 [1931] 1931 innerTail291_9 best = best := by
  rw [innerTail291_9, scan, if_neg (by decide), innerCandidates291_9_checked]
  change scan edge (search edge 3) 4 [1931] 1931 innerTail291_10
    (if 3 ≤ innerCandidates291_9.length then search edge 3 [1931, 7043] innerCandidates291_9 8974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan291_10

def innerTail291_8 : List Nat := 6719 :: innerTail291_9
def innerCandidates291_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates291_8_checked : candidates edge 4 1931 6719 innerTail291_9 best.bound = innerCandidates291_8 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan291_8 : scan edge (search edge 3) 4 [1931] 1931 innerTail291_8 best = best := by
  rw [innerTail291_8, scan, if_neg (by decide), innerCandidates291_8_checked]
  change scan edge (search edge 3) 4 [1931] 1931 innerTail291_9
    (if 3 ≤ innerCandidates291_8.length then search edge 3 [1931, 6719] innerCandidates291_8 8650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan291_9

def innerTail291_7 : List Nat := 5477 :: innerTail291_8
def innerCandidates291_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates291_7_checked : candidates edge 4 1931 5477 innerTail291_8 best.bound = innerCandidates291_7 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan291_7 : scan edge (search edge 3) 4 [1931] 1931 innerTail291_7 best = best := by
  rw [innerTail291_7, scan, if_neg (by decide), innerCandidates291_7_checked]
  change scan edge (search edge 3) 4 [1931] 1931 innerTail291_8
    (if 3 ≤ innerCandidates291_7.length then search edge 3 [1931, 5477] innerCandidates291_7 7408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan291_8

def innerTail291_6 : List Nat := 5297 :: innerTail291_7
def innerCandidates291_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates291_6_checked : candidates edge 4 1931 5297 innerTail291_7 best.bound = innerCandidates291_6 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan291_6 : scan edge (search edge 3) 4 [1931] 1931 innerTail291_6 best = best := by
  rw [innerTail291_6, scan, if_neg (by decide), innerCandidates291_6_checked]
  change scan edge (search edge 3) 4 [1931] 1931 innerTail291_7
    (if 3 ≤ innerCandidates291_6.length then search edge 3 [1931, 5297] innerCandidates291_6 7228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan291_7

def innerTail291_5 : List Nat := 5279 :: innerTail291_6
def innerCandidates291_5 : List Nat := [6719]
set_option maxRecDepth 4096 in
theorem innerCandidates291_5_checked : candidates edge 4 1931 5279 innerTail291_6 best.bound = innerCandidates291_5 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan291_5 : scan edge (search edge 3) 4 [1931] 1931 innerTail291_5 best = best := by
  rw [innerTail291_5, scan, if_neg (by decide), innerCandidates291_5_checked]
  change scan edge (search edge 3) 4 [1931] 1931 innerTail291_6
    (if 3 ≤ innerCandidates291_5.length then search edge 3 [1931, 5279] innerCandidates291_5 7210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan291_6

def innerTail291_4 : List Nat := 5171 :: innerTail291_5
def innerCandidates291_4 : List Nat := [5297]
set_option maxRecDepth 4096 in
theorem innerCandidates291_4_checked : candidates edge 4 1931 5171 innerTail291_5 best.bound = innerCandidates291_4 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan291_4 : scan edge (search edge 3) 4 [1931] 1931 innerTail291_4 best = best := by
  rw [innerTail291_4, scan, if_neg (by decide), innerCandidates291_4_checked]
  change scan edge (search edge 3) 4 [1931] 1931 innerTail291_5
    (if 3 ≤ innerCandidates291_4.length then search edge 3 [1931, 5171] innerCandidates291_4 7102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan291_5

def innerTail291_3 : List Nat := 4943 :: innerTail291_4
def innerCandidates291_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates291_3_checked : candidates edge 4 1931 4943 innerTail291_4 best.bound = innerCandidates291_3 := by
  unfold candidates
  have row : edge 4943 = SearchData.row4943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan291_3 : scan edge (search edge 3) 4 [1931] 1931 innerTail291_3 best = best := by
  rw [innerTail291_3, scan, if_neg (by decide), innerCandidates291_3_checked]
  change scan edge (search edge 3) 4 [1931] 1931 innerTail291_4
    (if 3 ≤ innerCandidates291_3.length then search edge 3 [1931, 4943] innerCandidates291_3 6874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan291_4

def innerTail291_2 : List Nat := 4289 :: innerTail291_3
def innerCandidates291_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates291_2_checked : candidates edge 4 1931 4289 innerTail291_3 best.bound = innerCandidates291_2 := by
  unfold candidates
  have row : edge 4289 = SearchData.row4289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan291_2 : scan edge (search edge 3) 4 [1931] 1931 innerTail291_2 best = best := by
  rw [innerTail291_2, scan, if_neg (by decide), innerCandidates291_2_checked]
  change scan edge (search edge 3) 4 [1931] 1931 innerTail291_3
    (if 3 ≤ innerCandidates291_2.length then search edge 3 [1931, 4289] innerCandidates291_2 6220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan291_3

def innerTail291_1 : List Nat := 2957 :: innerTail291_2
def innerCandidates291_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates291_1_checked : candidates edge 4 1931 2957 innerTail291_2 best.bound = innerCandidates291_1 := by
  unfold candidates
  have row : edge 2957 = SearchData.row2957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan291_1 : scan edge (search edge 3) 4 [1931] 1931 innerTail291_1 best = best := by
  rw [innerTail291_1, scan, if_neg (by decide), innerCandidates291_1_checked]
  change scan edge (search edge 3) 4 [1931] 1931 innerTail291_2
    (if 3 ≤ innerCandidates291_1.length then search edge 3 [1931, 2957] innerCandidates291_1 4888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan291_2

def innerTail291_0 : List Nat := 2399 :: innerTail291_1
def innerCandidates291_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates291_0_checked : candidates edge 4 1931 2399 innerTail291_1 best.bound = innerCandidates291_0 := by
  unfold candidates
  have row : edge 2399 = PrimePairCache.smallRow2399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan291_0 : scan edge (search edge 3) 4 [1931] 1931 innerTail291_0 best = best := by
  rw [innerTail291_0, scan, if_neg (by decide), innerCandidates291_0_checked]
  change scan edge (search edge 3) 4 [1931] 1931 innerTail291_1
    (if 3 ≤ innerCandidates291_0.length then search edge 3 [1931, 2399] innerCandidates291_0 4330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan291_1

theorem rootBranch291 : search edge 4 [1931] rootCandidates291 1931 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1931] 1931 innerTail291_0 best = best
  exact innerScan291_0

def rootCandidates290 : List Nat := [2213, 2273, 2381, 2693, 2741, 2843, 3203, 4211, 5009, 5087, 5189, 5381, 5483, 5861, 6011, 6389, 7253, 7583, 7841, 7919, 8219, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates290_checked : candidates edge 5 0 1913 tail291 best.bound = rootCandidates290 := by
  unfold candidates
  have row : edge 1913 = PrimePairCache.smallRow1913.contains := by funext q; rfl
  rw [row]
  decide

def innerTail290_22 : List Nat := []
theorem innerScan290_22 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_22 best = best := rfl

def innerTail290_21 : List Nat := 8297 :: innerTail290_22
def innerCandidates290_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates290_21_checked : candidates edge 4 1913 8297 innerTail290_22 best.bound = innerCandidates290_21 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_21 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_21 best = best := by
  rw [innerTail290_21, scan, if_neg (by decide), innerCandidates290_21_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_22
    (if 3 ≤ innerCandidates290_21.length then search edge 3 [1913, 8297] innerCandidates290_21 10210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_22

def innerTail290_20 : List Nat := 8219 :: innerTail290_21
def innerCandidates290_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates290_20_checked : candidates edge 4 1913 8219 innerTail290_21 best.bound = innerCandidates290_20 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_20 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_20 best = best := by
  rw [innerTail290_20, scan, if_neg (by decide), innerCandidates290_20_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_21
    (if 3 ≤ innerCandidates290_20.length then search edge 3 [1913, 8219] innerCandidates290_20 10132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_21

def innerTail290_19 : List Nat := 7919 :: innerTail290_20
def innerCandidates290_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates290_19_checked : candidates edge 4 1913 7919 innerTail290_20 best.bound = innerCandidates290_19 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_19 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_19 best = best := by
  rw [innerTail290_19, scan, if_neg (by decide), innerCandidates290_19_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_20
    (if 3 ≤ innerCandidates290_19.length then search edge 3 [1913, 7919] innerCandidates290_19 9832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_20

def innerTail290_18 : List Nat := 7841 :: innerTail290_19
def innerCandidates290_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates290_18_checked : candidates edge 4 1913 7841 innerTail290_19 best.bound = innerCandidates290_18 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_18 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_18 best = best := by
  rw [innerTail290_18, scan, if_neg (by decide), innerCandidates290_18_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_19
    (if 3 ≤ innerCandidates290_18.length then search edge 3 [1913, 7841] innerCandidates290_18 9754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_19

def innerTail290_17 : List Nat := 7583 :: innerTail290_18
def innerCandidates290_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates290_17_checked : candidates edge 4 1913 7583 innerTail290_18 best.bound = innerCandidates290_17 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_17 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_17 best = best := by
  rw [innerTail290_17, scan, if_neg (by decide), innerCandidates290_17_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_18
    (if 3 ≤ innerCandidates290_17.length then search edge 3 [1913, 7583] innerCandidates290_17 9496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_18

def innerTail290_16 : List Nat := 7253 :: innerTail290_17
def innerCandidates290_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates290_16_checked : candidates edge 4 1913 7253 innerTail290_17 best.bound = innerCandidates290_16 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_16 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_16 best = best := by
  rw [innerTail290_16, scan, if_neg (by decide), innerCandidates290_16_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_17
    (if 3 ≤ innerCandidates290_16.length then search edge 3 [1913, 7253] innerCandidates290_16 9166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_17

def innerTail290_15 : List Nat := 6389 :: innerTail290_16
def innerCandidates290_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates290_15_checked : candidates edge 4 1913 6389 innerTail290_16 best.bound = innerCandidates290_15 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_15 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_15 best = best := by
  rw [innerTail290_15, scan, if_neg (by decide), innerCandidates290_15_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_16
    (if 3 ≤ innerCandidates290_15.length then search edge 3 [1913, 6389] innerCandidates290_15 8302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_16

def innerTail290_14 : List Nat := 6011 :: innerTail290_15
def innerCandidates290_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates290_14_checked : candidates edge 4 1913 6011 innerTail290_15 best.bound = innerCandidates290_14 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_14 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_14 best = best := by
  rw [innerTail290_14, scan, if_neg (by decide), innerCandidates290_14_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_15
    (if 3 ≤ innerCandidates290_14.length then search edge 3 [1913, 6011] innerCandidates290_14 7924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_15

def innerTail290_13 : List Nat := 5861 :: innerTail290_14
def innerCandidates290_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates290_13_checked : candidates edge 4 1913 5861 innerTail290_14 best.bound = innerCandidates290_13 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_13 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_13 best = best := by
  rw [innerTail290_13, scan, if_neg (by decide), innerCandidates290_13_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_14
    (if 3 ≤ innerCandidates290_13.length then search edge 3 [1913, 5861] innerCandidates290_13 7774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_14

def innerTail290_12 : List Nat := 5483 :: innerTail290_13
def innerCandidates290_12 : List Nat := [8297]
set_option maxRecDepth 4096 in
theorem innerCandidates290_12_checked : candidates edge 4 1913 5483 innerTail290_13 best.bound = innerCandidates290_12 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_12 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_12 best = best := by
  rw [innerTail290_12, scan, if_neg (by decide), innerCandidates290_12_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_13
    (if 3 ≤ innerCandidates290_12.length then search edge 3 [1913, 5483] innerCandidates290_12 7396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_13

def innerTail290_11 : List Nat := 5381 :: innerTail290_12
def innerCandidates290_11 : List Nat := [6011, 7919]
set_option maxRecDepth 4096 in
theorem innerCandidates290_11_checked : candidates edge 4 1913 5381 innerTail290_12 best.bound = innerCandidates290_11 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_11 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_11 best = best := by
  rw [innerTail290_11, scan, if_neg (by decide), innerCandidates290_11_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_12
    (if 3 ≤ innerCandidates290_11.length then search edge 3 [1913, 5381] innerCandidates290_11 7294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_12

def innerTail290_10 : List Nat := 5189 :: innerTail290_11
def innerCandidates290_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates290_10_checked : candidates edge 4 1913 5189 innerTail290_11 best.bound = innerCandidates290_10 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_10 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_10 best = best := by
  rw [innerTail290_10, scan, if_neg (by decide), innerCandidates290_10_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_11
    (if 3 ≤ innerCandidates290_10.length then search edge 3 [1913, 5189] innerCandidates290_10 7102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_11

def innerTail290_9 : List Nat := 5087 :: innerTail290_10
def innerCandidates290_9 : List Nat := [5861]
set_option maxRecDepth 4096 in
theorem innerCandidates290_9_checked : candidates edge 4 1913 5087 innerTail290_10 best.bound = innerCandidates290_9 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_9 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_9 best = best := by
  rw [innerTail290_9, scan, if_neg (by decide), innerCandidates290_9_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_10
    (if 3 ≤ innerCandidates290_9.length then search edge 3 [1913, 5087] innerCandidates290_9 7000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_10

def innerTail290_8 : List Nat := 5009 :: innerTail290_9
def innerCandidates290_8 : List Nat := [5861]
set_option maxRecDepth 4096 in
theorem innerCandidates290_8_checked : candidates edge 4 1913 5009 innerTail290_9 best.bound = innerCandidates290_8 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_8 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_8 best = best := by
  rw [innerTail290_8, scan, if_neg (by decide), innerCandidates290_8_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_9
    (if 3 ≤ innerCandidates290_8.length then search edge 3 [1913, 5009] innerCandidates290_8 6922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_9

def innerTail290_7 : List Nat := 4211 :: innerTail290_8
def innerCandidates290_7 : List Nat := [5483]
set_option maxRecDepth 4096 in
theorem innerCandidates290_7_checked : candidates edge 4 1913 4211 innerTail290_8 best.bound = innerCandidates290_7 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_7 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_7 best = best := by
  rw [innerTail290_7, scan, if_neg (by decide), innerCandidates290_7_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_8
    (if 3 ≤ innerCandidates290_7.length then search edge 3 [1913, 4211] innerCandidates290_7 6124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_8

def innerTail290_6 : List Nat := 3203 :: innerTail290_7
def innerCandidates290_6 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates290_6_checked : candidates edge 4 1913 3203 innerTail290_7 best.bound = innerCandidates290_6 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_6 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_6 best = best := by
  rw [innerTail290_6, scan, if_neg (by decide), innerCandidates290_6_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_7
    (if 3 ≤ innerCandidates290_6.length then search edge 3 [1913, 3203] innerCandidates290_6 5116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_7

def innerTail290_5 : List Nat := 2843 :: innerTail290_6
def innerCandidates290_5 : List Nat := [3203, 7841]
set_option maxRecDepth 4096 in
theorem innerCandidates290_5_checked : candidates edge 4 1913 2843 innerTail290_6 best.bound = innerCandidates290_5 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_5 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_5 best = best := by
  rw [innerTail290_5, scan, if_neg (by decide), innerCandidates290_5_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_6
    (if 3 ≤ innerCandidates290_5.length then search edge 3 [1913, 2843] innerCandidates290_5 4756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_6

def innerTail290_4 : List Nat := 2741 :: innerTail290_5
def innerCandidates290_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates290_4_checked : candidates edge 4 1913 2741 innerTail290_5 best.bound = innerCandidates290_4 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_4 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_4 best = best := by
  rw [innerTail290_4, scan, if_neg (by decide), innerCandidates290_4_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_5
    (if 3 ≤ innerCandidates290_4.length then search edge 3 [1913, 2741] innerCandidates290_4 4654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_5

def innerTail290_3 : List Nat := 2693 :: innerTail290_4
def innerCandidates290_3 : List Nat := [6011]
set_option maxRecDepth 4096 in
theorem innerCandidates290_3_checked : candidates edge 4 1913 2693 innerTail290_4 best.bound = innerCandidates290_3 := by
  unfold candidates
  have row : edge 2693 = SearchData.row2693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_3 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_3 best = best := by
  rw [innerTail290_3, scan, if_neg (by decide), innerCandidates290_3_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_4
    (if 3 ≤ innerCandidates290_3.length then search edge 3 [1913, 2693] innerCandidates290_3 4606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_4

def innerTail290_2 : List Nat := 2381 :: innerTail290_3
def innerCandidates290_2 : List Nat := [5483]
set_option maxRecDepth 4096 in
theorem innerCandidates290_2_checked : candidates edge 4 1913 2381 innerTail290_3 best.bound = innerCandidates290_2 := by
  unfold candidates
  have row : edge 2381 = PrimePairCache.smallRow2381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_2 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_2 best = best := by
  rw [innerTail290_2, scan, if_neg (by decide), innerCandidates290_2_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_3
    (if 3 ≤ innerCandidates290_2.length then search edge 3 [1913, 2381] innerCandidates290_2 4294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_3

def innerTail290_1 : List Nat := 2273 :: innerTail290_2
def innerCandidates290_1 : List Nat := [5861, 8297]
set_option maxRecDepth 4096 in
theorem innerCandidates290_1_checked : candidates edge 4 1913 2273 innerTail290_2 best.bound = innerCandidates290_1 := by
  unfold candidates
  have row : edge 2273 = PrimePairCache.smallRow2273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_1 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_1 best = best := by
  rw [innerTail290_1, scan, if_neg (by decide), innerCandidates290_1_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_2
    (if 3 ≤ innerCandidates290_1.length then search edge 3 [1913, 2273] innerCandidates290_1 4186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_2

def innerTail290_0 : List Nat := 2213 :: innerTail290_1
def innerCandidates290_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates290_0_checked : candidates edge 4 1913 2213 innerTail290_1 best.bound = innerCandidates290_0 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan290_0 : scan edge (search edge 3) 4 [1913] 1913 innerTail290_0 best = best := by
  rw [innerTail290_0, scan, if_neg (by decide), innerCandidates290_0_checked]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_1
    (if 3 ≤ innerCandidates290_0.length then search edge 3 [1913, 2213] innerCandidates290_0 4126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan290_1

theorem rootBranch290 : search edge 4 [1913] rootCandidates290 1913 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1913] 1913 innerTail290_0 best = best
  exact innerScan290_0

def rootCandidates289 : List Nat := [2063, 2243, 2621, 2777, 3083, 3581, 3797, 4127, 4637, 5099, 5669, 5897, 7481, 8231]
set_option maxRecDepth 4096 in
theorem rootCandidates289_checked : candidates edge 5 0 1907 tail290 best.bound = rootCandidates289 := by
  unfold candidates
  have row : edge 1907 = PrimePairCache.smallRow1907.contains := by funext q; rfl
  rw [row]
  decide

def innerTail289_14 : List Nat := []
theorem innerScan289_14 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_14 best = best := rfl

def innerTail289_13 : List Nat := 8231 :: innerTail289_14
def innerCandidates289_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates289_13_checked : candidates edge 4 1907 8231 innerTail289_14 best.bound = innerCandidates289_13 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_13 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_13 best = best := by
  rw [innerTail289_13, scan, if_neg (by decide), innerCandidates289_13_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_14
    (if 3 ≤ innerCandidates289_13.length then search edge 3 [1907, 8231] innerCandidates289_13 10138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_14

def innerTail289_12 : List Nat := 7481 :: innerTail289_13
def innerCandidates289_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates289_12_checked : candidates edge 4 1907 7481 innerTail289_13 best.bound = innerCandidates289_12 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_12 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_12 best = best := by
  rw [innerTail289_12, scan, if_neg (by decide), innerCandidates289_12_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_13
    (if 3 ≤ innerCandidates289_12.length then search edge 3 [1907, 7481] innerCandidates289_12 9388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_13

def innerTail289_11 : List Nat := 5897 :: innerTail289_12
def innerCandidates289_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates289_11_checked : candidates edge 4 1907 5897 innerTail289_12 best.bound = innerCandidates289_11 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_11 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_11 best = best := by
  rw [innerTail289_11, scan, if_neg (by decide), innerCandidates289_11_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_12
    (if 3 ≤ innerCandidates289_11.length then search edge 3 [1907, 5897] innerCandidates289_11 7804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_12

def innerTail289_10 : List Nat := 5669 :: innerTail289_11
def innerCandidates289_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates289_10_checked : candidates edge 4 1907 5669 innerTail289_11 best.bound = innerCandidates289_10 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_10 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_10 best = best := by
  rw [innerTail289_10, scan, if_neg (by decide), innerCandidates289_10_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_11
    (if 3 ≤ innerCandidates289_10.length then search edge 3 [1907, 5669] innerCandidates289_10 7576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_11

def innerTail289_9 : List Nat := 5099 :: innerTail289_10
def innerCandidates289_9 : List Nat := [8231]
set_option maxRecDepth 4096 in
theorem innerCandidates289_9_checked : candidates edge 4 1907 5099 innerTail289_10 best.bound = innerCandidates289_9 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_9 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_9 best = best := by
  rw [innerTail289_9, scan, if_neg (by decide), innerCandidates289_9_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_10
    (if 3 ≤ innerCandidates289_9.length then search edge 3 [1907, 5099] innerCandidates289_9 7006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_10

def innerTail289_8 : List Nat := 4637 :: innerTail289_9
def innerCandidates289_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates289_8_checked : candidates edge 4 1907 4637 innerTail289_9 best.bound = innerCandidates289_8 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_8 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_8 best = best := by
  rw [innerTail289_8, scan, if_neg (by decide), innerCandidates289_8_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_9
    (if 3 ≤ innerCandidates289_8.length then search edge 3 [1907, 4637] innerCandidates289_8 6544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_9

def innerTail289_7 : List Nat := 4127 :: innerTail289_8
def innerCandidates289_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates289_7_checked : candidates edge 4 1907 4127 innerTail289_8 best.bound = innerCandidates289_7 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_7 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_7 best = best := by
  rw [innerTail289_7, scan, if_neg (by decide), innerCandidates289_7_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_8
    (if 3 ≤ innerCandidates289_7.length then search edge 3 [1907, 4127] innerCandidates289_7 6034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_8

def innerTail289_6 : List Nat := 3797 :: innerTail289_7
def innerCandidates289_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates289_6_checked : candidates edge 4 1907 3797 innerTail289_7 best.bound = innerCandidates289_6 := by
  unfold candidates
  have row : edge 3797 = SearchData.row3797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_6 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_6 best = best := by
  rw [innerTail289_6, scan, if_neg (by decide), innerCandidates289_6_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_7
    (if 3 ≤ innerCandidates289_6.length then search edge 3 [1907, 3797] innerCandidates289_6 5704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_7

def innerTail289_5 : List Nat := 3581 :: innerTail289_6
def innerCandidates289_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates289_5_checked : candidates edge 4 1907 3581 innerTail289_6 best.bound = innerCandidates289_5 := by
  unfold candidates
  have row : edge 3581 = SearchData.row3581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_5 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_5 best = best := by
  rw [innerTail289_5, scan, if_neg (by decide), innerCandidates289_5_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_6
    (if 3 ≤ innerCandidates289_5.length then search edge 3 [1907, 3581] innerCandidates289_5 5488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_6

def innerTail289_4 : List Nat := 3083 :: innerTail289_5
def innerCandidates289_4 : List Nat := [3797]
set_option maxRecDepth 4096 in
theorem innerCandidates289_4_checked : candidates edge 4 1907 3083 innerTail289_5 best.bound = innerCandidates289_4 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_4 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_4 best = best := by
  rw [innerTail289_4, scan, if_neg (by decide), innerCandidates289_4_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_5
    (if 3 ≤ innerCandidates289_4.length then search edge 3 [1907, 3083] innerCandidates289_4 4990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_5

def innerTail289_3 : List Nat := 2777 :: innerTail289_4
def innerCandidates289_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates289_3_checked : candidates edge 4 1907 2777 innerTail289_4 best.bound = innerCandidates289_3 := by
  unfold candidates
  have row : edge 2777 = SearchData.row2777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_3 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_3 best = best := by
  rw [innerTail289_3, scan, if_neg (by decide), innerCandidates289_3_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_4
    (if 3 ≤ innerCandidates289_3.length then search edge 3 [1907, 2777] innerCandidates289_3 4684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_4

def innerTail289_2 : List Nat := 2621 :: innerTail289_3
def innerCandidates289_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates289_2_checked : candidates edge 4 1907 2621 innerTail289_3 best.bound = innerCandidates289_2 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_2 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_2 best = best := by
  rw [innerTail289_2, scan, if_neg (by decide), innerCandidates289_2_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_3
    (if 3 ≤ innerCandidates289_2.length then search edge 3 [1907, 2621] innerCandidates289_2 4528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_3

def innerTail289_1 : List Nat := 2243 :: innerTail289_2
def innerCandidates289_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates289_1_checked : candidates edge 4 1907 2243 innerTail289_2 best.bound = innerCandidates289_1 := by
  unfold candidates
  have row : edge 2243 = PrimePairCache.smallRow2243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_1 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_1 best = best := by
  rw [innerTail289_1, scan, if_neg (by decide), innerCandidates289_1_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_2
    (if 3 ≤ innerCandidates289_1.length then search edge 3 [1907, 2243] innerCandidates289_1 4150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_2

def innerTail289_0 : List Nat := 2063 :: innerTail289_1
def innerCandidates289_0 : List Nat := [2243]
set_option maxRecDepth 4096 in
theorem innerCandidates289_0_checked : candidates edge 4 1907 2063 innerTail289_1 best.bound = innerCandidates289_0 := by
  unfold candidates
  have row : edge 2063 = PrimePairCache.smallRow2063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan289_0 : scan edge (search edge 3) 4 [1907] 1907 innerTail289_0 best = best := by
  rw [innerTail289_0, scan, if_neg (by decide), innerCandidates289_0_checked]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_1
    (if 3 ≤ innerCandidates289_0.length then search edge 3 [1907, 2063] innerCandidates289_0 3970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan289_1

theorem rootBranch289 : search edge 4 [1907] rootCandidates289 1907 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1907] 1907 innerTail289_0 best = best
  exact innerScan289_0

def rootCandidates288 : List Nat := [2153, 2621, 2801, 2969, 3221, 4019, 4817, 5861, 7151, 7589, 7949, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates288_checked : candidates edge 5 0 1901 tail289 best.bound = rootCandidates288 := by
  unfold candidates
  have row : edge 1901 = PrimePairCache.smallRow1901.contains := by funext q; rfl
  rw [row]
  decide

def innerTail288_12 : List Nat := []
theorem innerScan288_12 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_12 best = best := rfl

def innerTail288_11 : List Nat := 8243 :: innerTail288_12
def innerCandidates288_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates288_11_checked : candidates edge 4 1901 8243 innerTail288_12 best.bound = innerCandidates288_11 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan288_11 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_11 best = best := by
  rw [innerTail288_11, scan, if_neg (by decide), innerCandidates288_11_checked]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_12
    (if 3 ≤ innerCandidates288_11.length then search edge 3 [1901, 8243] innerCandidates288_11 10144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan288_12

def innerTail288_10 : List Nat := 7949 :: innerTail288_11
def innerCandidates288_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates288_10_checked : candidates edge 4 1901 7949 innerTail288_11 best.bound = innerCandidates288_10 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan288_10 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_10 best = best := by
  rw [innerTail288_10, scan, if_neg (by decide), innerCandidates288_10_checked]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_11
    (if 3 ≤ innerCandidates288_10.length then search edge 3 [1901, 7949] innerCandidates288_10 9850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan288_11

def innerTail288_9 : List Nat := 7589 :: innerTail288_10
def innerCandidates288_9 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates288_9_checked : candidates edge 4 1901 7589 innerTail288_10 best.bound = innerCandidates288_9 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan288_9 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_9 best = best := by
  rw [innerTail288_9, scan, if_neg (by decide), innerCandidates288_9_checked]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_10
    (if 3 ≤ innerCandidates288_9.length then search edge 3 [1901, 7589] innerCandidates288_9 9490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan288_10

def innerTail288_8 : List Nat := 7151 :: innerTail288_9
def innerCandidates288_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates288_8_checked : candidates edge 4 1901 7151 innerTail288_9 best.bound = innerCandidates288_8 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan288_8 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_8 best = best := by
  rw [innerTail288_8, scan, if_neg (by decide), innerCandidates288_8_checked]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_9
    (if 3 ≤ innerCandidates288_8.length then search edge 3 [1901, 7151] innerCandidates288_8 9052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan288_9

def innerTail288_7 : List Nat := 5861 :: innerTail288_8
def innerCandidates288_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates288_7_checked : candidates edge 4 1901 5861 innerTail288_8 best.bound = innerCandidates288_7 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan288_7 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_7 best = best := by
  rw [innerTail288_7, scan, if_neg (by decide), innerCandidates288_7_checked]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_8
    (if 3 ≤ innerCandidates288_7.length then search edge 3 [1901, 5861] innerCandidates288_7 7762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan288_8

def innerTail288_6 : List Nat := 4817 :: innerTail288_7
def innerCandidates288_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates288_6_checked : candidates edge 4 1901 4817 innerTail288_7 best.bound = innerCandidates288_6 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan288_6 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_6 best = best := by
  rw [innerTail288_6, scan, if_neg (by decide), innerCandidates288_6_checked]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_7
    (if 3 ≤ innerCandidates288_6.length then search edge 3 [1901, 4817] innerCandidates288_6 6718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan288_7

def innerTail288_5 : List Nat := 4019 :: innerTail288_6
def innerCandidates288_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates288_5_checked : candidates edge 4 1901 4019 innerTail288_6 best.bound = innerCandidates288_5 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan288_5 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_5 best = best := by
  rw [innerTail288_5, scan, if_neg (by decide), innerCandidates288_5_checked]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_6
    (if 3 ≤ innerCandidates288_5.length then search edge 3 [1901, 4019] innerCandidates288_5 5920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan288_6

def innerTail288_4 : List Nat := 3221 :: innerTail288_5
def innerCandidates288_4 : List Nat := [4817]
set_option maxRecDepth 4096 in
theorem innerCandidates288_4_checked : candidates edge 4 1901 3221 innerTail288_5 best.bound = innerCandidates288_4 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan288_4 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_4 best = best := by
  rw [innerTail288_4, scan, if_neg (by decide), innerCandidates288_4_checked]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_5
    (if 3 ≤ innerCandidates288_4.length then search edge 3 [1901, 3221] innerCandidates288_4 5122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan288_5

def innerTail288_3 : List Nat := 2969 :: innerTail288_4
def innerCandidates288_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates288_3_checked : candidates edge 4 1901 2969 innerTail288_4 best.bound = innerCandidates288_3 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan288_3 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_3 best = best := by
  rw [innerTail288_3, scan, if_neg (by decide), innerCandidates288_3_checked]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_4
    (if 3 ≤ innerCandidates288_3.length then search edge 3 [1901, 2969] innerCandidates288_3 4870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan288_4

def innerTail288_2 : List Nat := 2801 :: innerTail288_3
def innerCandidates288_2 : List Nat := [3221]
set_option maxRecDepth 4096 in
theorem innerCandidates288_2_checked : candidates edge 4 1901 2801 innerTail288_3 best.bound = innerCandidates288_2 := by
  unfold candidates
  have row : edge 2801 = SearchData.row2801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan288_2 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_2 best = best := by
  rw [innerTail288_2, scan, if_neg (by decide), innerCandidates288_2_checked]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_3
    (if 3 ≤ innerCandidates288_2.length then search edge 3 [1901, 2801] innerCandidates288_2 4702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan288_3

def innerTail288_1 : List Nat := 2621 :: innerTail288_2
def innerCandidates288_1 : List Nat := [7151]
set_option maxRecDepth 4096 in
theorem innerCandidates288_1_checked : candidates edge 4 1901 2621 innerTail288_2 best.bound = innerCandidates288_1 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan288_1 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_1 best = best := by
  rw [innerTail288_1, scan, if_neg (by decide), innerCandidates288_1_checked]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_2
    (if 3 ≤ innerCandidates288_1.length then search edge 3 [1901, 2621] innerCandidates288_1 4522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan288_2

def innerTail288_0 : List Nat := 2153 :: innerTail288_1
def innerCandidates288_0 : List Nat := [2969, 7151]
set_option maxRecDepth 4096 in
theorem innerCandidates288_0_checked : candidates edge 4 1901 2153 innerTail288_1 best.bound = innerCandidates288_0 := by
  unfold candidates
  have row : edge 2153 = PrimePairCache.smallRow2153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan288_0 : scan edge (search edge 3) 4 [1901] 1901 innerTail288_0 best = best := by
  rw [innerTail288_0, scan, if_neg (by decide), innerCandidates288_0_checked]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_1
    (if 3 ≤ innerCandidates288_0.length then search edge 3 [1901, 2153] innerCandidates288_0 4054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan288_1

theorem rootBranch288 : search edge 4 [1901] rootCandidates288 1901 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1901] 1901 innerTail288_0 best = best
  exact innerScan288_0

end ProjectEulerSolutions.PrimePairUpper.Search
