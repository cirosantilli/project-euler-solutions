import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates314 : List Nat := [2129, 2381, 2393, 2693, 2843, 3083, 3527, 4241, 5279, 5717, 6863, 7211, 7517, 7841, 8069, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates314_checked : candidates edge 5 0 2099 tail315 best.bound = rootCandidates314 := by
  unfold candidates
  have row : edge 2099 = PrimePairCache.smallRow2099.contains := by funext q; rfl
  rw [row]
  decide

def innerTail314_16 : List Nat := []
theorem innerScan314_16 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_16 best = best := rfl

def innerTail314_15 : List Nat := 8093 :: innerTail314_16
def innerCandidates314_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_15_checked : candidates edge 4 2099 8093 innerTail314_16 best.bound = innerCandidates314_15 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_15 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_15 best = best := by
  rw [innerTail314_15, scan, if_neg (by decide), innerCandidates314_15_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_16
    (if 3 ≤ innerCandidates314_15.length then search edge 3 [2099, 8093] innerCandidates314_15 10192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_16

def innerTail314_14 : List Nat := 8069 :: innerTail314_15
def innerCandidates314_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_14_checked : candidates edge 4 2099 8069 innerTail314_15 best.bound = innerCandidates314_14 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_14 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_14 best = best := by
  rw [innerTail314_14, scan, if_neg (by decide), innerCandidates314_14_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_15
    (if 3 ≤ innerCandidates314_14.length then search edge 3 [2099, 8069] innerCandidates314_14 10168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_15

def innerTail314_13 : List Nat := 7841 :: innerTail314_14
def innerCandidates314_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_13_checked : candidates edge 4 2099 7841 innerTail314_14 best.bound = innerCandidates314_13 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_13 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_13 best = best := by
  rw [innerTail314_13, scan, if_neg (by decide), innerCandidates314_13_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_14
    (if 3 ≤ innerCandidates314_13.length then search edge 3 [2099, 7841] innerCandidates314_13 9940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_14

def innerTail314_12 : List Nat := 7517 :: innerTail314_13
def innerCandidates314_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_12_checked : candidates edge 4 2099 7517 innerTail314_13 best.bound = innerCandidates314_12 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_12 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_12 best = best := by
  rw [innerTail314_12, scan, if_neg (by decide), innerCandidates314_12_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_13
    (if 3 ≤ innerCandidates314_12.length then search edge 3 [2099, 7517] innerCandidates314_12 9616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_13

def innerTail314_11 : List Nat := 7211 :: innerTail314_12
def innerCandidates314_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_11_checked : candidates edge 4 2099 7211 innerTail314_12 best.bound = innerCandidates314_11 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_11 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_11 best = best := by
  rw [innerTail314_11, scan, if_neg (by decide), innerCandidates314_11_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_12
    (if 3 ≤ innerCandidates314_11.length then search edge 3 [2099, 7211] innerCandidates314_11 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_12

def innerTail314_10 : List Nat := 6863 :: innerTail314_11
def innerCandidates314_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_10_checked : candidates edge 4 2099 6863 innerTail314_11 best.bound = innerCandidates314_10 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_10 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_10 best = best := by
  rw [innerTail314_10, scan, if_neg (by decide), innerCandidates314_10_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_11
    (if 3 ≤ innerCandidates314_10.length then search edge 3 [2099, 6863] innerCandidates314_10 8962 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_11

def innerTail314_9 : List Nat := 5717 :: innerTail314_10
def innerCandidates314_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_9_checked : candidates edge 4 2099 5717 innerTail314_10 best.bound = innerCandidates314_9 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_9 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_9 best = best := by
  rw [innerTail314_9, scan, if_neg (by decide), innerCandidates314_9_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_10
    (if 3 ≤ innerCandidates314_9.length then search edge 3 [2099, 5717] innerCandidates314_9 7816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_10

def innerTail314_8 : List Nat := 5279 :: innerTail314_9
def innerCandidates314_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_8_checked : candidates edge 4 2099 5279 innerTail314_9 best.bound = innerCandidates314_8 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_8 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_8 best = best := by
  rw [innerTail314_8, scan, if_neg (by decide), innerCandidates314_8_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_9
    (if 3 ≤ innerCandidates314_8.length then search edge 3 [2099, 5279] innerCandidates314_8 7378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_9

def innerTail314_7 : List Nat := 4241 :: innerTail314_8
def innerCandidates314_7 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates314_7_checked : candidates edge 4 2099 4241 innerTail314_8 best.bound = innerCandidates314_7 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_7 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_7 best = best := by
  rw [innerTail314_7, scan, if_neg (by decide), innerCandidates314_7_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_8
    (if 3 ≤ innerCandidates314_7.length then search edge 3 [2099, 4241] innerCandidates314_7 6340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_8

def innerTail314_6 : List Nat := 3527 :: innerTail314_7
def innerCandidates314_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_6_checked : candidates edge 4 2099 3527 innerTail314_7 best.bound = innerCandidates314_6 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_6 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_6 best = best := by
  rw [innerTail314_6, scan, if_neg (by decide), innerCandidates314_6_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_7
    (if 3 ≤ innerCandidates314_6.length then search edge 3 [2099, 3527] innerCandidates314_6 5626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_7

def innerTail314_5 : List Nat := 3083 :: innerTail314_6
def innerCandidates314_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_5_checked : candidates edge 4 2099 3083 innerTail314_6 best.bound = innerCandidates314_5 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_5 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_5 best = best := by
  rw [innerTail314_5, scan, if_neg (by decide), innerCandidates314_5_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_6
    (if 3 ≤ innerCandidates314_5.length then search edge 3 [2099, 3083] innerCandidates314_5 5182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_6

def innerTail314_4 : List Nat := 2843 :: innerTail314_5
def innerCandidates314_4 : List Nat := [7841]
set_option maxRecDepth 4096 in
theorem innerCandidates314_4_checked : candidates edge 4 2099 2843 innerTail314_5 best.bound = innerCandidates314_4 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_4 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_4 best = best := by
  rw [innerTail314_4, scan, if_neg (by decide), innerCandidates314_4_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_5
    (if 3 ≤ innerCandidates314_4.length then search edge 3 [2099, 2843] innerCandidates314_4 4942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_5

def innerTail314_3 : List Nat := 2693 :: innerTail314_4
def innerCandidates314_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_3_checked : candidates edge 4 2099 2693 innerTail314_4 best.bound = innerCandidates314_3 := by
  unfold candidates
  have row : edge 2693 = SearchData.row2693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_3 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_3 best = best := by
  rw [innerTail314_3, scan, if_neg (by decide), innerCandidates314_3_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_4
    (if 3 ≤ innerCandidates314_3.length then search edge 3 [2099, 2693] innerCandidates314_3 4792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_4

def innerTail314_2 : List Nat := 2393 :: innerTail314_3
def innerCandidates314_2 : List Nat := [3083, 4241, 8069]
set_option maxRecDepth 4096 in
theorem innerCandidates314_2_checked : candidates edge 4 2099 2393 innerTail314_3 best.bound = innerCandidates314_2 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild314_2 : search edge 3 [2099, 2393] innerCandidates314_2 4492 best = best := by decide

theorem innerScan314_2 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_2 best = best := by
  rw [innerTail314_2, scan, if_neg (by decide), innerCandidates314_2_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_3
    (if 3 ≤ innerCandidates314_2.length then search edge 3 [2099, 2393] innerCandidates314_2 4492 best else best) = best
  rw [if_pos (by decide), innerChild314_2]
  exact innerScan314_3

def innerTail314_1 : List Nat := 2381 :: innerTail314_2
def innerCandidates314_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_1_checked : candidates edge 4 2099 2381 innerTail314_2 best.bound = innerCandidates314_1 := by
  unfold candidates
  have row : edge 2381 = PrimePairCache.smallRow2381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_1 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_1 best = best := by
  rw [innerTail314_1, scan, if_neg (by decide), innerCandidates314_1_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_2
    (if 3 ≤ innerCandidates314_1.length then search edge 3 [2099, 2381] innerCandidates314_1 4480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_2

def innerTail314_0 : List Nat := 2129 :: innerTail314_1
def innerCandidates314_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates314_0_checked : candidates edge 4 2099 2129 innerTail314_1 best.bound = innerCandidates314_0 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan314_0 : scan edge (search edge 3) 4 [2099] 2099 innerTail314_0 best = best := by
  rw [innerTail314_0, scan, if_neg (by decide), innerCandidates314_0_checked]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_1
    (if 3 ≤ innerCandidates314_0.length then search edge 3 [2099, 2129] innerCandidates314_0 4228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan314_1

theorem rootBranch314 : search edge 4 [2099] rootCandidates314 2099 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2099] 2099 innerTail314_0 best = best
  exact innerScan314_0

def rootCandidates313 : List Nat := [2341, 2953, 3181, 3643, 3931, 4099, 4219, 4597, 5443, 5827, 6277, 7039, 7237, 7297, 7411, 7873, 8209, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates313_checked : candidates edge 5 0 2089 tail314 best.bound = rootCandidates313 := by
  unfold candidates
  have row : edge 2089 = PrimePairCache.smallRow2089.contains := by funext q; rfl
  rw [row]
  decide

def innerTail313_18 : List Nat := []
theorem innerScan313_18 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_18 best = best := rfl

def innerTail313_17 : List Nat := 8263 :: innerTail313_18
def innerCandidates313_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_17_checked : candidates edge 4 2089 8263 innerTail313_18 best.bound = innerCandidates313_17 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_17 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_17 best = best := by
  rw [innerTail313_17, scan, if_neg (by decide), innerCandidates313_17_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_18
    (if 3 ≤ innerCandidates313_17.length then search edge 3 [2089, 8263] innerCandidates313_17 10352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_18

def innerTail313_16 : List Nat := 8209 :: innerTail313_17
def innerCandidates313_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_16_checked : candidates edge 4 2089 8209 innerTail313_17 best.bound = innerCandidates313_16 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_16 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_16 best = best := by
  rw [innerTail313_16, scan, if_neg (by decide), innerCandidates313_16_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_17
    (if 3 ≤ innerCandidates313_16.length then search edge 3 [2089, 8209] innerCandidates313_16 10298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_17

def innerTail313_15 : List Nat := 7873 :: innerTail313_16
def innerCandidates313_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_15_checked : candidates edge 4 2089 7873 innerTail313_16 best.bound = innerCandidates313_15 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_15 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_15 best = best := by
  rw [innerTail313_15, scan, if_neg (by decide), innerCandidates313_15_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_16
    (if 3 ≤ innerCandidates313_15.length then search edge 3 [2089, 7873] innerCandidates313_15 9962 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_16

def innerTail313_14 : List Nat := 7411 :: innerTail313_15
def innerCandidates313_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_14_checked : candidates edge 4 2089 7411 innerTail313_15 best.bound = innerCandidates313_14 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_14 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_14 best = best := by
  rw [innerTail313_14, scan, if_neg (by decide), innerCandidates313_14_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_15
    (if 3 ≤ innerCandidates313_14.length then search edge 3 [2089, 7411] innerCandidates313_14 9500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_15

def innerTail313_13 : List Nat := 7297 :: innerTail313_14
def innerCandidates313_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_13_checked : candidates edge 4 2089 7297 innerTail313_14 best.bound = innerCandidates313_13 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_13 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_13 best = best := by
  rw [innerTail313_13, scan, if_neg (by decide), innerCandidates313_13_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_14
    (if 3 ≤ innerCandidates313_13.length then search edge 3 [2089, 7297] innerCandidates313_13 9386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_14

def innerTail313_12 : List Nat := 7237 :: innerTail313_13
def innerCandidates313_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_12_checked : candidates edge 4 2089 7237 innerTail313_13 best.bound = innerCandidates313_12 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_12 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_12 best = best := by
  rw [innerTail313_12, scan, if_neg (by decide), innerCandidates313_12_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_13
    (if 3 ≤ innerCandidates313_12.length then search edge 3 [2089, 7237] innerCandidates313_12 9326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_13

def innerTail313_11 : List Nat := 7039 :: innerTail313_12
def innerCandidates313_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_11_checked : candidates edge 4 2089 7039 innerTail313_12 best.bound = innerCandidates313_11 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_11 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_11 best = best := by
  rw [innerTail313_11, scan, if_neg (by decide), innerCandidates313_11_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_12
    (if 3 ≤ innerCandidates313_11.length then search edge 3 [2089, 7039] innerCandidates313_11 9128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_12

def innerTail313_10 : List Nat := 6277 :: innerTail313_11
def innerCandidates313_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_10_checked : candidates edge 4 2089 6277 innerTail313_11 best.bound = innerCandidates313_10 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_10 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_10 best = best := by
  rw [innerTail313_10, scan, if_neg (by decide), innerCandidates313_10_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_11
    (if 3 ≤ innerCandidates313_10.length then search edge 3 [2089, 6277] innerCandidates313_10 8366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_11

def innerTail313_9 : List Nat := 5827 :: innerTail313_10
def innerCandidates313_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_9_checked : candidates edge 4 2089 5827 innerTail313_10 best.bound = innerCandidates313_9 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_9 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_9 best = best := by
  rw [innerTail313_9, scan, if_neg (by decide), innerCandidates313_9_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_10
    (if 3 ≤ innerCandidates313_9.length then search edge 3 [2089, 5827] innerCandidates313_9 7916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_10

def innerTail313_8 : List Nat := 5443 :: innerTail313_9
def innerCandidates313_8 : List Nat := [7039]
set_option maxRecDepth 4096 in
theorem innerCandidates313_8_checked : candidates edge 4 2089 5443 innerTail313_9 best.bound = innerCandidates313_8 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_8 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_8 best = best := by
  rw [innerTail313_8, scan, if_neg (by decide), innerCandidates313_8_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_9
    (if 3 ≤ innerCandidates313_8.length then search edge 3 [2089, 5443] innerCandidates313_8 7532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_9

def innerTail313_7 : List Nat := 4597 :: innerTail313_8
def innerCandidates313_7 : List Nat := [8263]
set_option maxRecDepth 4096 in
theorem innerCandidates313_7_checked : candidates edge 4 2089 4597 innerTail313_8 best.bound = innerCandidates313_7 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_7 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_7 best = best := by
  rw [innerTail313_7, scan, if_neg (by decide), innerCandidates313_7_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_8
    (if 3 ≤ innerCandidates313_7.length then search edge 3 [2089, 4597] innerCandidates313_7 6686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_8

def innerTail313_6 : List Nat := 4219 :: innerTail313_7
def innerCandidates313_6 : List Nat := [4597, 8209]
set_option maxRecDepth 4096 in
theorem innerCandidates313_6_checked : candidates edge 4 2089 4219 innerTail313_7 best.bound = innerCandidates313_6 := by
  unfold candidates
  have row : edge 4219 = SearchData.row4219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_6 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_6 best = best := by
  rw [innerTail313_6, scan, if_neg (by decide), innerCandidates313_6_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_7
    (if 3 ≤ innerCandidates313_6.length then search edge 3 [2089, 4219] innerCandidates313_6 6308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_7

def innerTail313_5 : List Nat := 4099 :: innerTail313_6
def innerCandidates313_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_5_checked : candidates edge 4 2089 4099 innerTail313_6 best.bound = innerCandidates313_5 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_5 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_5 best = best := by
  rw [innerTail313_5, scan, if_neg (by decide), innerCandidates313_5_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_6
    (if 3 ≤ innerCandidates313_5.length then search edge 3 [2089, 4099] innerCandidates313_5 6188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_6

def innerTail313_4 : List Nat := 3931 :: innerTail313_5
def innerCandidates313_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_4_checked : candidates edge 4 2089 3931 innerTail313_5 best.bound = innerCandidates313_4 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_4 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_4 best = best := by
  rw [innerTail313_4, scan, if_neg (by decide), innerCandidates313_4_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_5
    (if 3 ≤ innerCandidates313_4.length then search edge 3 [2089, 3931] innerCandidates313_4 6020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_5

def innerTail313_3 : List Nat := 3643 :: innerTail313_4
def innerCandidates313_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_3_checked : candidates edge 4 2089 3643 innerTail313_4 best.bound = innerCandidates313_3 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_3 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_3 best = best := by
  rw [innerTail313_3, scan, if_neg (by decide), innerCandidates313_3_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_4
    (if 3 ≤ innerCandidates313_3.length then search edge 3 [2089, 3643] innerCandidates313_3 5732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_4

def innerTail313_2 : List Nat := 3181 :: innerTail313_3
def innerCandidates313_2 : List Nat := [4099, 4219]
set_option maxRecDepth 4096 in
theorem innerCandidates313_2_checked : candidates edge 4 2089 3181 innerTail313_3 best.bound = innerCandidates313_2 := by
  unfold candidates
  have row : edge 3181 = SearchData.row3181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_2 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_2 best = best := by
  rw [innerTail313_2, scan, if_neg (by decide), innerCandidates313_2_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_3
    (if 3 ≤ innerCandidates313_2.length then search edge 3 [2089, 3181] innerCandidates313_2 5270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_3

def innerTail313_1 : List Nat := 2953 :: innerTail313_2
def innerCandidates313_1 : List Nat := [3181, 3643]
set_option maxRecDepth 4096 in
theorem innerCandidates313_1_checked : candidates edge 4 2089 2953 innerTail313_2 best.bound = innerCandidates313_1 := by
  unfold candidates
  have row : edge 2953 = SearchData.row2953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_1 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_1 best = best := by
  rw [innerTail313_1, scan, if_neg (by decide), innerCandidates313_1_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_2
    (if 3 ≤ innerCandidates313_1.length then search edge 3 [2089, 2953] innerCandidates313_1 5042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_2

def innerTail313_0 : List Nat := 2341 :: innerTail313_1
def innerCandidates313_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates313_0_checked : candidates edge 4 2089 2341 innerTail313_1 best.bound = innerCandidates313_0 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan313_0 : scan edge (search edge 3) 4 [2089] 2089 innerTail313_0 best = best := by
  rw [innerTail313_0, scan, if_neg (by decide), innerCandidates313_0_checked]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_1
    (if 3 ≤ innerCandidates313_0.length then search edge 3 [2089, 2341] innerCandidates313_0 4430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan313_1

theorem rootBranch313 : search edge 4 [2089] rootCandidates313 2089 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2089] 2089 innerTail313_0 best = best
  exact innerScan313_0

def rootCandidates312 : List Nat := [2153, 2417, 3821, 4463, 4937, 5009, 5333, 6689, 7079, 7559, 7907]
set_option maxRecDepth 4096 in
theorem rootCandidates312_checked : candidates edge 5 0 2087 tail313 best.bound = rootCandidates312 := by
  unfold candidates
  have row : edge 2087 = PrimePairCache.smallRow2087.contains := by funext q; rfl
  rw [row]
  decide

def innerTail312_11 : List Nat := []
theorem innerScan312_11 : scan edge (search edge 3) 4 [2087] 2087 innerTail312_11 best = best := rfl

def innerTail312_10 : List Nat := 7907 :: innerTail312_11
def innerCandidates312_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates312_10_checked : candidates edge 4 2087 7907 innerTail312_11 best.bound = innerCandidates312_10 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan312_10 : scan edge (search edge 3) 4 [2087] 2087 innerTail312_10 best = best := by
  rw [innerTail312_10, scan, if_neg (by decide), innerCandidates312_10_checked]
  change scan edge (search edge 3) 4 [2087] 2087 innerTail312_11
    (if 3 ≤ innerCandidates312_10.length then search edge 3 [2087, 7907] innerCandidates312_10 9994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan312_11

def innerTail312_9 : List Nat := 7559 :: innerTail312_10
def innerCandidates312_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates312_9_checked : candidates edge 4 2087 7559 innerTail312_10 best.bound = innerCandidates312_9 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan312_9 : scan edge (search edge 3) 4 [2087] 2087 innerTail312_9 best = best := by
  rw [innerTail312_9, scan, if_neg (by decide), innerCandidates312_9_checked]
  change scan edge (search edge 3) 4 [2087] 2087 innerTail312_10
    (if 3 ≤ innerCandidates312_9.length then search edge 3 [2087, 7559] innerCandidates312_9 9646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan312_10

def innerTail312_8 : List Nat := 7079 :: innerTail312_9
def innerCandidates312_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates312_8_checked : candidates edge 4 2087 7079 innerTail312_9 best.bound = innerCandidates312_8 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan312_8 : scan edge (search edge 3) 4 [2087] 2087 innerTail312_8 best = best := by
  rw [innerTail312_8, scan, if_neg (by decide), innerCandidates312_8_checked]
  change scan edge (search edge 3) 4 [2087] 2087 innerTail312_9
    (if 3 ≤ innerCandidates312_8.length then search edge 3 [2087, 7079] innerCandidates312_8 9166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan312_9

def innerTail312_7 : List Nat := 6689 :: innerTail312_8
def innerCandidates312_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates312_7_checked : candidates edge 4 2087 6689 innerTail312_8 best.bound = innerCandidates312_7 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan312_7 : scan edge (search edge 3) 4 [2087] 2087 innerTail312_7 best = best := by
  rw [innerTail312_7, scan, if_neg (by decide), innerCandidates312_7_checked]
  change scan edge (search edge 3) 4 [2087] 2087 innerTail312_8
    (if 3 ≤ innerCandidates312_7.length then search edge 3 [2087, 6689] innerCandidates312_7 8776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan312_8

def innerTail312_6 : List Nat := 5333 :: innerTail312_7
def innerCandidates312_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates312_6_checked : candidates edge 4 2087 5333 innerTail312_7 best.bound = innerCandidates312_6 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan312_6 : scan edge (search edge 3) 4 [2087] 2087 innerTail312_6 best = best := by
  rw [innerTail312_6, scan, if_neg (by decide), innerCandidates312_6_checked]
  change scan edge (search edge 3) 4 [2087] 2087 innerTail312_7
    (if 3 ≤ innerCandidates312_6.length then search edge 3 [2087, 5333] innerCandidates312_6 7420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan312_7

def innerTail312_5 : List Nat := 5009 :: innerTail312_6
def innerCandidates312_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates312_5_checked : candidates edge 4 2087 5009 innerTail312_6 best.bound = innerCandidates312_5 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan312_5 : scan edge (search edge 3) 4 [2087] 2087 innerTail312_5 best = best := by
  rw [innerTail312_5, scan, if_neg (by decide), innerCandidates312_5_checked]
  change scan edge (search edge 3) 4 [2087] 2087 innerTail312_6
    (if 3 ≤ innerCandidates312_5.length then search edge 3 [2087, 5009] innerCandidates312_5 7096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan312_6

def innerTail312_4 : List Nat := 4937 :: innerTail312_5
def innerCandidates312_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates312_4_checked : candidates edge 4 2087 4937 innerTail312_5 best.bound = innerCandidates312_4 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan312_4 : scan edge (search edge 3) 4 [2087] 2087 innerTail312_4 best = best := by
  rw [innerTail312_4, scan, if_neg (by decide), innerCandidates312_4_checked]
  change scan edge (search edge 3) 4 [2087] 2087 innerTail312_5
    (if 3 ≤ innerCandidates312_4.length then search edge 3 [2087, 4937] innerCandidates312_4 7024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan312_5

def innerTail312_3 : List Nat := 4463 :: innerTail312_4
def innerCandidates312_3 : List Nat := [7079]
set_option maxRecDepth 4096 in
theorem innerCandidates312_3_checked : candidates edge 4 2087 4463 innerTail312_4 best.bound = innerCandidates312_3 := by
  unfold candidates
  have row : edge 4463 = SearchData.row4463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan312_3 : scan edge (search edge 3) 4 [2087] 2087 innerTail312_3 best = best := by
  rw [innerTail312_3, scan, if_neg (by decide), innerCandidates312_3_checked]
  change scan edge (search edge 3) 4 [2087] 2087 innerTail312_4
    (if 3 ≤ innerCandidates312_3.length then search edge 3 [2087, 4463] innerCandidates312_3 6550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan312_4

def innerTail312_2 : List Nat := 3821 :: innerTail312_3
def innerCandidates312_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates312_2_checked : candidates edge 4 2087 3821 innerTail312_3 best.bound = innerCandidates312_2 := by
  unfold candidates
  have row : edge 3821 = SearchData.row3821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan312_2 : scan edge (search edge 3) 4 [2087] 2087 innerTail312_2 best = best := by
  rw [innerTail312_2, scan, if_neg (by decide), innerCandidates312_2_checked]
  change scan edge (search edge 3) 4 [2087] 2087 innerTail312_3
    (if 3 ≤ innerCandidates312_2.length then search edge 3 [2087, 3821] innerCandidates312_2 5908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan312_3

def innerTail312_1 : List Nat := 2417 :: innerTail312_2
def innerCandidates312_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates312_1_checked : candidates edge 4 2087 2417 innerTail312_2 best.bound = innerCandidates312_1 := by
  unfold candidates
  have row : edge 2417 = PrimePairCache.smallRow2417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan312_1 : scan edge (search edge 3) 4 [2087] 2087 innerTail312_1 best = best := by
  rw [innerTail312_1, scan, if_neg (by decide), innerCandidates312_1_checked]
  change scan edge (search edge 3) 4 [2087] 2087 innerTail312_2
    (if 3 ≤ innerCandidates312_1.length then search edge 3 [2087, 2417] innerCandidates312_1 4504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan312_2

def innerTail312_0 : List Nat := 2153 :: innerTail312_1
def innerCandidates312_0 : List Nat := [2417]
set_option maxRecDepth 4096 in
theorem innerCandidates312_0_checked : candidates edge 4 2087 2153 innerTail312_1 best.bound = innerCandidates312_0 := by
  unfold candidates
  have row : edge 2153 = PrimePairCache.smallRow2153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan312_0 : scan edge (search edge 3) 4 [2087] 2087 innerTail312_0 best = best := by
  rw [innerTail312_0, scan, if_neg (by decide), innerCandidates312_0_checked]
  change scan edge (search edge 3) 4 [2087] 2087 innerTail312_1
    (if 3 ≤ innerCandidates312_0.length then search edge 3 [2087, 2153] innerCandidates312_0 4240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan312_1

theorem rootBranch312 : search edge 4 [2087] rootCandidates312 2087 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2087] 2087 innerTail312_0 best = best
  exact innerScan312_0

def rootCandidates311 : List Nat := [2971, 3343, 3511, 4111, 4243, 4651, 5443, 5827, 5923, 6121, 6997, 7459, 8221]
set_option maxRecDepth 4096 in
theorem rootCandidates311_checked : candidates edge 5 0 2083 tail312 best.bound = rootCandidates311 := by
  unfold candidates
  have row : edge 2083 = PrimePairCache.smallRow2083.contains := by funext q; rfl
  rw [row]
  decide

def innerTail311_13 : List Nat := []
theorem innerScan311_13 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_13 best = best := rfl

def innerTail311_12 : List Nat := 8221 :: innerTail311_13
def innerCandidates311_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates311_12_checked : candidates edge 4 2083 8221 innerTail311_13 best.bound = innerCandidates311_12 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_12 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_12 best = best := by
  rw [innerTail311_12, scan, if_neg (by decide), innerCandidates311_12_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_13
    (if 3 ≤ innerCandidates311_12.length then search edge 3 [2083, 8221] innerCandidates311_12 10304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_13

def innerTail311_11 : List Nat := 7459 :: innerTail311_12
def innerCandidates311_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates311_11_checked : candidates edge 4 2083 7459 innerTail311_12 best.bound = innerCandidates311_11 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_11 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_11 best = best := by
  rw [innerTail311_11, scan, if_neg (by decide), innerCandidates311_11_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_12
    (if 3 ≤ innerCandidates311_11.length then search edge 3 [2083, 7459] innerCandidates311_11 9542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_12

def innerTail311_10 : List Nat := 6997 :: innerTail311_11
def innerCandidates311_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates311_10_checked : candidates edge 4 2083 6997 innerTail311_11 best.bound = innerCandidates311_10 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_10 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_10 best = best := by
  rw [innerTail311_10, scan, if_neg (by decide), innerCandidates311_10_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_11
    (if 3 ≤ innerCandidates311_10.length then search edge 3 [2083, 6997] innerCandidates311_10 9080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_11

def innerTail311_9 : List Nat := 6121 :: innerTail311_10
def innerCandidates311_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates311_9_checked : candidates edge 4 2083 6121 innerTail311_10 best.bound = innerCandidates311_9 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_9 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_9 best = best := by
  rw [innerTail311_9, scan, if_neg (by decide), innerCandidates311_9_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_10
    (if 3 ≤ innerCandidates311_9.length then search edge 3 [2083, 6121] innerCandidates311_9 8204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_10

def innerTail311_8 : List Nat := 5923 :: innerTail311_9
def innerCandidates311_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates311_8_checked : candidates edge 4 2083 5923 innerTail311_9 best.bound = innerCandidates311_8 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_8 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_8 best = best := by
  rw [innerTail311_8, scan, if_neg (by decide), innerCandidates311_8_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_9
    (if 3 ≤ innerCandidates311_8.length then search edge 3 [2083, 5923] innerCandidates311_8 8006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_9

def innerTail311_7 : List Nat := 5827 :: innerTail311_8
def innerCandidates311_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates311_7_checked : candidates edge 4 2083 5827 innerTail311_8 best.bound = innerCandidates311_7 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_7 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_7 best = best := by
  rw [innerTail311_7, scan, if_neg (by decide), innerCandidates311_7_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_8
    (if 3 ≤ innerCandidates311_7.length then search edge 3 [2083, 5827] innerCandidates311_7 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_8

def innerTail311_6 : List Nat := 5443 :: innerTail311_7
def innerCandidates311_6 : List Nat := [5923]
set_option maxRecDepth 4096 in
theorem innerCandidates311_6_checked : candidates edge 4 2083 5443 innerTail311_7 best.bound = innerCandidates311_6 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_6 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_6 best = best := by
  rw [innerTail311_6, scan, if_neg (by decide), innerCandidates311_6_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_7
    (if 3 ≤ innerCandidates311_6.length then search edge 3 [2083, 5443] innerCandidates311_6 7526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_7

def innerTail311_5 : List Nat := 4651 :: innerTail311_6
def innerCandidates311_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates311_5_checked : candidates edge 4 2083 4651 innerTail311_6 best.bound = innerCandidates311_5 := by
  unfold candidates
  have row : edge 4651 = SearchData.row4651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_5 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_5 best = best := by
  rw [innerTail311_5, scan, if_neg (by decide), innerCandidates311_5_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_6
    (if 3 ≤ innerCandidates311_5.length then search edge 3 [2083, 4651] innerCandidates311_5 6734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_6

def innerTail311_4 : List Nat := 4243 :: innerTail311_5
def innerCandidates311_4 : List Nat := [6997]
set_option maxRecDepth 4096 in
theorem innerCandidates311_4_checked : candidates edge 4 2083 4243 innerTail311_5 best.bound = innerCandidates311_4 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_4 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_4 best = best := by
  rw [innerTail311_4, scan, if_neg (by decide), innerCandidates311_4_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_5
    (if 3 ≤ innerCandidates311_4.length then search edge 3 [2083, 4243] innerCandidates311_4 6326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_5

def innerTail311_3 : List Nat := 4111 :: innerTail311_4
def innerCandidates311_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates311_3_checked : candidates edge 4 2083 4111 innerTail311_4 best.bound = innerCandidates311_3 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_3 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_3 best = best := by
  rw [innerTail311_3, scan, if_neg (by decide), innerCandidates311_3_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_4
    (if 3 ≤ innerCandidates311_3.length then search edge 3 [2083, 4111] innerCandidates311_3 6194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_4

def innerTail311_2 : List Nat := 3511 :: innerTail311_3
def innerCandidates311_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates311_2_checked : candidates edge 4 2083 3511 innerTail311_3 best.bound = innerCandidates311_2 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_2 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_2 best = best := by
  rw [innerTail311_2, scan, if_neg (by decide), innerCandidates311_2_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_3
    (if 3 ≤ innerCandidates311_2.length then search edge 3 [2083, 3511] innerCandidates311_2 5594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_3

def innerTail311_1 : List Nat := 3343 :: innerTail311_2
def innerCandidates311_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates311_1_checked : candidates edge 4 2083 3343 innerTail311_2 best.bound = innerCandidates311_1 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_1 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_1 best = best := by
  rw [innerTail311_1, scan, if_neg (by decide), innerCandidates311_1_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_2
    (if 3 ≤ innerCandidates311_1.length then search edge 3 [2083, 3343] innerCandidates311_1 5426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_2

def innerTail311_0 : List Nat := 2971 :: innerTail311_1
def innerCandidates311_0 : List Nat := [7459]
set_option maxRecDepth 4096 in
theorem innerCandidates311_0_checked : candidates edge 4 2083 2971 innerTail311_1 best.bound = innerCandidates311_0 := by
  unfold candidates
  have row : edge 2971 = SearchData.row2971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan311_0 : scan edge (search edge 3) 4 [2083] 2083 innerTail311_0 best = best := by
  rw [innerTail311_0, scan, if_neg (by decide), innerCandidates311_0_checked]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_1
    (if 3 ≤ innerCandidates311_0.length then search edge 3 [2083, 2971] innerCandidates311_0 5054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan311_1

theorem rootBranch311 : search edge 4 [2083] rootCandidates311 2083 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2083] 2083 innerTail311_0 best = best
  exact innerScan311_0

end ProjectEulerSolutions.PrimePairUpper.Search
