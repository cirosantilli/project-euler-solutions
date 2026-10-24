import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates135 : List Nat := [823, 1483, 2017, 2671, 2731, 3301, 3457, 3631, 4129, 4177, 4507, 5347, 5653, 6007, 6367, 6733, 7159, 7873, 7993, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates135_checked : candidates edge 5 0 787 tail136 best.bound = rootCandidates135 := by
  unfold candidates
  have row : edge 787 = PrimePairCache.smallRow787.contains := by funext q; rfl
  rw [row]
  decide

def innerTail135_20 : List Nat := []
theorem innerScan135_20 : scan edge (search edge 3) 4 [787] 787 innerTail135_20 best = best := rfl

def innerTail135_19 : List Nat := 8317 :: innerTail135_20
def innerCandidates135_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates135_19_checked : candidates edge 4 787 8317 innerTail135_20 best.bound = innerCandidates135_19 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_19 : scan edge (search edge 3) 4 [787] 787 innerTail135_19 best = best := by
  rw [innerTail135_19, scan, if_neg (by decide), innerCandidates135_19_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_20
    (if 3 ≤ innerCandidates135_19.length then search edge 3 [787, 8317] innerCandidates135_19 9104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_20

def innerTail135_18 : List Nat := 7993 :: innerTail135_19
def innerCandidates135_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates135_18_checked : candidates edge 4 787 7993 innerTail135_19 best.bound = innerCandidates135_18 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_18 : scan edge (search edge 3) 4 [787] 787 innerTail135_18 best = best := by
  rw [innerTail135_18, scan, if_neg (by decide), innerCandidates135_18_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_19
    (if 3 ≤ innerCandidates135_18.length then search edge 3 [787, 7993] innerCandidates135_18 8780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_19

def innerTail135_17 : List Nat := 7873 :: innerTail135_18
def innerCandidates135_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates135_17_checked : candidates edge 4 787 7873 innerTail135_18 best.bound = innerCandidates135_17 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_17 : scan edge (search edge 3) 4 [787] 787 innerTail135_17 best = best := by
  rw [innerTail135_17, scan, if_neg (by decide), innerCandidates135_17_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_18
    (if 3 ≤ innerCandidates135_17.length then search edge 3 [787, 7873] innerCandidates135_17 8660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_18

def innerTail135_16 : List Nat := 7159 :: innerTail135_17
def innerCandidates135_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates135_16_checked : candidates edge 4 787 7159 innerTail135_17 best.bound = innerCandidates135_16 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_16 : scan edge (search edge 3) 4 [787] 787 innerTail135_16 best = best := by
  rw [innerTail135_16, scan, if_neg (by decide), innerCandidates135_16_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_17
    (if 3 ≤ innerCandidates135_16.length then search edge 3 [787, 7159] innerCandidates135_16 7946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_17

def innerTail135_15 : List Nat := 6733 :: innerTail135_16
def innerCandidates135_15 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates135_15_checked : candidates edge 4 787 6733 innerTail135_16 best.bound = innerCandidates135_15 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_15 : scan edge (search edge 3) 4 [787] 787 innerTail135_15 best = best := by
  rw [innerTail135_15, scan, if_neg (by decide), innerCandidates135_15_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_16
    (if 3 ≤ innerCandidates135_15.length then search edge 3 [787, 6733] innerCandidates135_15 7520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_16

def innerTail135_14 : List Nat := 6367 :: innerTail135_15
def innerCandidates135_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates135_14_checked : candidates edge 4 787 6367 innerTail135_15 best.bound = innerCandidates135_14 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_14 : scan edge (search edge 3) 4 [787] 787 innerTail135_14 best = best := by
  rw [innerTail135_14, scan, if_neg (by decide), innerCandidates135_14_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_15
    (if 3 ≤ innerCandidates135_14.length then search edge 3 [787, 6367] innerCandidates135_14 7154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_15

def innerTail135_13 : List Nat := 6007 :: innerTail135_14
def innerCandidates135_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates135_13_checked : candidates edge 4 787 6007 innerTail135_14 best.bound = innerCandidates135_13 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_13 : scan edge (search edge 3) 4 [787] 787 innerTail135_13 best = best := by
  rw [innerTail135_13, scan, if_neg (by decide), innerCandidates135_13_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_14
    (if 3 ≤ innerCandidates135_13.length then search edge 3 [787, 6007] innerCandidates135_13 6794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_14

def innerTail135_12 : List Nat := 5653 :: innerTail135_13
def innerCandidates135_12 : List Nat := [6367]
set_option maxRecDepth 4096 in
theorem innerCandidates135_12_checked : candidates edge 4 787 5653 innerTail135_13 best.bound = innerCandidates135_12 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_12 : scan edge (search edge 3) 4 [787] 787 innerTail135_12 best = best := by
  rw [innerTail135_12, scan, if_neg (by decide), innerCandidates135_12_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_13
    (if 3 ≤ innerCandidates135_12.length then search edge 3 [787, 5653] innerCandidates135_12 6440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_13

def innerTail135_11 : List Nat := 5347 :: innerTail135_12
def innerCandidates135_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates135_11_checked : candidates edge 4 787 5347 innerTail135_12 best.bound = innerCandidates135_11 := by
  unfold candidates
  have row : edge 5347 = SearchData.row5347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_11 : scan edge (search edge 3) 4 [787] 787 innerTail135_11 best = best := by
  rw [innerTail135_11, scan, if_neg (by decide), innerCandidates135_11_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_12
    (if 3 ≤ innerCandidates135_11.length then search edge 3 [787, 5347] innerCandidates135_11 6134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_12

def innerTail135_10 : List Nat := 4507 :: innerTail135_11
def innerCandidates135_10 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates135_10_checked : candidates edge 4 787 4507 innerTail135_11 best.bound = innerCandidates135_10 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_10 : scan edge (search edge 3) 4 [787] 787 innerTail135_10 best = best := by
  rw [innerTail135_10, scan, if_neg (by decide), innerCandidates135_10_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_11
    (if 3 ≤ innerCandidates135_10.length then search edge 3 [787, 4507] innerCandidates135_10 5294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_11

def innerTail135_9 : List Nat := 4177 :: innerTail135_10
def innerCandidates135_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates135_9_checked : candidates edge 4 787 4177 innerTail135_10 best.bound = innerCandidates135_9 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_9 : scan edge (search edge 3) 4 [787] 787 innerTail135_9 best = best := by
  rw [innerTail135_9, scan, if_neg (by decide), innerCandidates135_9_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_10
    (if 3 ≤ innerCandidates135_9.length then search edge 3 [787, 4177] innerCandidates135_9 4964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_10

def innerTail135_8 : List Nat := 4129 :: innerTail135_9
def innerCandidates135_8 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates135_8_checked : candidates edge 4 787 4129 innerTail135_9 best.bound = innerCandidates135_8 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_8 : scan edge (search edge 3) 4 [787] 787 innerTail135_8 best = best := by
  rw [innerTail135_8, scan, if_neg (by decide), innerCandidates135_8_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_9
    (if 3 ≤ innerCandidates135_8.length then search edge 3 [787, 4129] innerCandidates135_8 4916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_9

def innerTail135_7 : List Nat := 3631 :: innerTail135_8
def innerCandidates135_7 : List Nat := [5347]
set_option maxRecDepth 4096 in
theorem innerCandidates135_7_checked : candidates edge 4 787 3631 innerTail135_8 best.bound = innerCandidates135_7 := by
  unfold candidates
  have row : edge 3631 = SearchData.row3631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_7 : scan edge (search edge 3) 4 [787] 787 innerTail135_7 best = best := by
  rw [innerTail135_7, scan, if_neg (by decide), innerCandidates135_7_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_8
    (if 3 ≤ innerCandidates135_7.length then search edge 3 [787, 3631] innerCandidates135_7 4418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_8

def innerTail135_6 : List Nat := 3457 :: innerTail135_7
def innerCandidates135_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates135_6_checked : candidates edge 4 787 3457 innerTail135_7 best.bound = innerCandidates135_6 := by
  unfold candidates
  have row : edge 3457 = SearchData.row3457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_6 : scan edge (search edge 3) 4 [787] 787 innerTail135_6 best = best := by
  rw [innerTail135_6, scan, if_neg (by decide), innerCandidates135_6_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_7
    (if 3 ≤ innerCandidates135_6.length then search edge 3 [787, 3457] innerCandidates135_6 4244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_7

def innerTail135_5 : List Nat := 3301 :: innerTail135_6
def innerCandidates135_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates135_5_checked : candidates edge 4 787 3301 innerTail135_6 best.bound = innerCandidates135_5 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_5 : scan edge (search edge 3) 4 [787] 787 innerTail135_5 best = best := by
  rw [innerTail135_5, scan, if_neg (by decide), innerCandidates135_5_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_6
    (if 3 ≤ innerCandidates135_5.length then search edge 3 [787, 3301] innerCandidates135_5 4088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_6

def innerTail135_4 : List Nat := 2731 :: innerTail135_5
def innerCandidates135_4 : List Nat := [3301, 4129, 7993]
set_option maxRecDepth 4096 in
theorem innerCandidates135_4_checked : candidates edge 4 787 2731 innerTail135_5 best.bound = innerCandidates135_4 := by
  unfold candidates
  have row : edge 2731 = SearchData.row2731.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild135_4 : search edge 3 [787, 2731] innerCandidates135_4 3518 best = best := by decide

theorem innerScan135_4 : scan edge (search edge 3) 4 [787] 787 innerTail135_4 best = best := by
  rw [innerTail135_4, scan, if_neg (by decide), innerCandidates135_4_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_5
    (if 3 ≤ innerCandidates135_4.length then search edge 3 [787, 2731] innerCandidates135_4 3518 best else best) = best
  rw [if_pos (by decide), innerChild135_4]
  exact innerScan135_5

def innerTail135_3 : List Nat := 2671 :: innerTail135_4
def innerCandidates135_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates135_3_checked : candidates edge 4 787 2671 innerTail135_4 best.bound = innerCandidates135_3 := by
  unfold candidates
  have row : edge 2671 = SearchData.row2671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_3 : scan edge (search edge 3) 4 [787] 787 innerTail135_3 best = best := by
  rw [innerTail135_3, scan, if_neg (by decide), innerCandidates135_3_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_4
    (if 3 ≤ innerCandidates135_3.length then search edge 3 [787, 2671] innerCandidates135_3 3458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_4

def innerTail135_2 : List Nat := 2017 :: innerTail135_3
def innerCandidates135_2 : List Nat := [5653, 8317]
set_option maxRecDepth 4096 in
theorem innerCandidates135_2_checked : candidates edge 4 787 2017 innerTail135_3 best.bound = innerCandidates135_2 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_2 : scan edge (search edge 3) 4 [787] 787 innerTail135_2 best = best := by
  rw [innerTail135_2, scan, if_neg (by decide), innerCandidates135_2_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_3
    (if 3 ≤ innerCandidates135_2.length then search edge 3 [787, 2017] innerCandidates135_2 2804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_3

def innerTail135_1 : List Nat := 1483 :: innerTail135_2
def innerCandidates135_1 : List Nat := [3457, 7159, 7873]
set_option maxRecDepth 4096 in
theorem innerCandidates135_1_checked : candidates edge 4 787 1483 innerTail135_2 best.bound = innerCandidates135_1 := by
  unfold candidates
  have row : edge 1483 = PrimePairCache.smallRow1483.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild135_1 : search edge 3 [787, 1483] innerCandidates135_1 2270 best = best := by decide

theorem innerScan135_1 : scan edge (search edge 3) 4 [787] 787 innerTail135_1 best = best := by
  rw [innerTail135_1, scan, if_neg (by decide), innerCandidates135_1_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_2
    (if 3 ≤ innerCandidates135_1.length then search edge 3 [787, 1483] innerCandidates135_1 2270 best else best) = best
  rw [if_pos (by decide), innerChild135_1]
  exact innerScan135_2

def innerTail135_0 : List Nat := 823 :: innerTail135_1
def innerCandidates135_0 : List Nat := [2017, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates135_0_checked : candidates edge 4 787 823 innerTail135_1 best.bound = innerCandidates135_0 := by
  unfold candidates
  have row : edge 823 = PrimePairCache.smallRow823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan135_0 : scan edge (search edge 3) 4 [787] 787 innerTail135_0 best = best := by
  rw [innerTail135_0, scan, if_neg (by decide), innerCandidates135_0_checked]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_1
    (if 3 ≤ innerCandidates135_0.length then search edge 3 [787, 823] innerCandidates135_0 1610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan135_1

theorem rootBranch135 : search edge 4 [787] rootCandidates135 787 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [787] 787 innerTail135_0 best = best
  exact innerScan135_0

def rootCandidates134 : List Nat := [953, 1439, 1667, 1721, 1877, 2063, 2129, 2339, 2591, 3701, 4241, 4409, 5081, 5381, 5669, 5711, 5927, 6101, 6263, 6269, 7547]
set_option maxRecDepth 4096 in
theorem rootCandidates134_checked : candidates edge 5 0 773 tail135 best.bound = rootCandidates134 := by
  unfold candidates
  have row : edge 773 = PrimePairCache.smallRow773.contains := by funext q; rfl
  rw [row]
  decide

def innerTail134_21 : List Nat := []
theorem innerScan134_21 : scan edge (search edge 3) 4 [773] 773 innerTail134_21 best = best := rfl

def innerTail134_20 : List Nat := 7547 :: innerTail134_21
def innerCandidates134_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_20_checked : candidates edge 4 773 7547 innerTail134_21 best.bound = innerCandidates134_20 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_20 : scan edge (search edge 3) 4 [773] 773 innerTail134_20 best = best := by
  rw [innerTail134_20, scan, if_neg (by decide), innerCandidates134_20_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_21
    (if 3 ≤ innerCandidates134_20.length then search edge 3 [773, 7547] innerCandidates134_20 8320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_21

def innerTail134_19 : List Nat := 6269 :: innerTail134_20
def innerCandidates134_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_19_checked : candidates edge 4 773 6269 innerTail134_20 best.bound = innerCandidates134_19 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_19 : scan edge (search edge 3) 4 [773] 773 innerTail134_19 best = best := by
  rw [innerTail134_19, scan, if_neg (by decide), innerCandidates134_19_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_20
    (if 3 ≤ innerCandidates134_19.length then search edge 3 [773, 6269] innerCandidates134_19 7042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_20

def innerTail134_18 : List Nat := 6263 :: innerTail134_19
def innerCandidates134_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_18_checked : candidates edge 4 773 6263 innerTail134_19 best.bound = innerCandidates134_18 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_18 : scan edge (search edge 3) 4 [773] 773 innerTail134_18 best = best := by
  rw [innerTail134_18, scan, if_neg (by decide), innerCandidates134_18_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_19
    (if 3 ≤ innerCandidates134_18.length then search edge 3 [773, 6263] innerCandidates134_18 7036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_19

def innerTail134_17 : List Nat := 6101 :: innerTail134_18
def innerCandidates134_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_17_checked : candidates edge 4 773 6101 innerTail134_18 best.bound = innerCandidates134_17 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_17 : scan edge (search edge 3) 4 [773] 773 innerTail134_17 best = best := by
  rw [innerTail134_17, scan, if_neg (by decide), innerCandidates134_17_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_18
    (if 3 ≤ innerCandidates134_17.length then search edge 3 [773, 6101] innerCandidates134_17 6874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_18

def innerTail134_16 : List Nat := 5927 :: innerTail134_17
def innerCandidates134_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_16_checked : candidates edge 4 773 5927 innerTail134_17 best.bound = innerCandidates134_16 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_16 : scan edge (search edge 3) 4 [773] 773 innerTail134_16 best = best := by
  rw [innerTail134_16, scan, if_neg (by decide), innerCandidates134_16_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_17
    (if 3 ≤ innerCandidates134_16.length then search edge 3 [773, 5927] innerCandidates134_16 6700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_17

def innerTail134_15 : List Nat := 5711 :: innerTail134_16
def innerCandidates134_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_15_checked : candidates edge 4 773 5711 innerTail134_16 best.bound = innerCandidates134_15 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_15 : scan edge (search edge 3) 4 [773] 773 innerTail134_15 best = best := by
  rw [innerTail134_15, scan, if_neg (by decide), innerCandidates134_15_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_16
    (if 3 ≤ innerCandidates134_15.length then search edge 3 [773, 5711] innerCandidates134_15 6484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_16

def innerTail134_14 : List Nat := 5669 :: innerTail134_15
def innerCandidates134_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_14_checked : candidates edge 4 773 5669 innerTail134_15 best.bound = innerCandidates134_14 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_14 : scan edge (search edge 3) 4 [773] 773 innerTail134_14 best = best := by
  rw [innerTail134_14, scan, if_neg (by decide), innerCandidates134_14_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_15
    (if 3 ≤ innerCandidates134_14.length then search edge 3 [773, 5669] innerCandidates134_14 6442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_15

def innerTail134_13 : List Nat := 5381 :: innerTail134_14
def innerCandidates134_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_13_checked : candidates edge 4 773 5381 innerTail134_14 best.bound = innerCandidates134_13 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_13 : scan edge (search edge 3) 4 [773] 773 innerTail134_13 best = best := by
  rw [innerTail134_13, scan, if_neg (by decide), innerCandidates134_13_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_14
    (if 3 ≤ innerCandidates134_13.length then search edge 3 [773, 5381] innerCandidates134_13 6154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_14

def innerTail134_12 : List Nat := 5081 :: innerTail134_13
def innerCandidates134_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_12_checked : candidates edge 4 773 5081 innerTail134_13 best.bound = innerCandidates134_12 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_12 : scan edge (search edge 3) 4 [773] 773 innerTail134_12 best = best := by
  rw [innerTail134_12, scan, if_neg (by decide), innerCandidates134_12_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_13
    (if 3 ≤ innerCandidates134_12.length then search edge 3 [773, 5081] innerCandidates134_12 5854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_13

def innerTail134_11 : List Nat := 4409 :: innerTail134_12
def innerCandidates134_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_11_checked : candidates edge 4 773 4409 innerTail134_12 best.bound = innerCandidates134_11 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_11 : scan edge (search edge 3) 4 [773] 773 innerTail134_11 best = best := by
  rw [innerTail134_11, scan, if_neg (by decide), innerCandidates134_11_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_12
    (if 3 ≤ innerCandidates134_11.length then search edge 3 [773, 4409] innerCandidates134_11 5182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_12

def innerTail134_10 : List Nat := 4241 :: innerTail134_11
def innerCandidates134_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_10_checked : candidates edge 4 773 4241 innerTail134_11 best.bound = innerCandidates134_10 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_10 : scan edge (search edge 3) 4 [773] 773 innerTail134_10 best = best := by
  rw [innerTail134_10, scan, if_neg (by decide), innerCandidates134_10_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_11
    (if 3 ≤ innerCandidates134_10.length then search edge 3 [773, 4241] innerCandidates134_10 5014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_11

def innerTail134_9 : List Nat := 3701 :: innerTail134_10
def innerCandidates134_9 : List Nat := [5081]
set_option maxRecDepth 4096 in
theorem innerCandidates134_9_checked : candidates edge 4 773 3701 innerTail134_10 best.bound = innerCandidates134_9 := by
  unfold candidates
  have row : edge 3701 = SearchData.row3701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_9 : scan edge (search edge 3) 4 [773] 773 innerTail134_9 best = best := by
  rw [innerTail134_9, scan, if_neg (by decide), innerCandidates134_9_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_10
    (if 3 ≤ innerCandidates134_9.length then search edge 3 [773, 3701] innerCandidates134_9 4474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_10

def innerTail134_8 : List Nat := 2591 :: innerTail134_9
def innerCandidates134_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_8_checked : candidates edge 4 773 2591 innerTail134_9 best.bound = innerCandidates134_8 := by
  unfold candidates
  have row : edge 2591 = PrimePairCache.smallRow2591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_8 : scan edge (search edge 3) 4 [773] 773 innerTail134_8 best = best := by
  rw [innerTail134_8, scan, if_neg (by decide), innerCandidates134_8_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_9
    (if 3 ≤ innerCandidates134_8.length then search edge 3 [773, 2591] innerCandidates134_8 3364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_9

def innerTail134_7 : List Nat := 2339 :: innerTail134_8
def innerCandidates134_7 : List Nat := [5081, 6101]
set_option maxRecDepth 4096 in
theorem innerCandidates134_7_checked : candidates edge 4 773 2339 innerTail134_8 best.bound = innerCandidates134_7 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_7 : scan edge (search edge 3) 4 [773] 773 innerTail134_7 best = best := by
  rw [innerTail134_7, scan, if_neg (by decide), innerCandidates134_7_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_8
    (if 3 ≤ innerCandidates134_7.length then search edge 3 [773, 2339] innerCandidates134_7 3112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_8

def innerTail134_6 : List Nat := 2129 :: innerTail134_7
def innerCandidates134_6 : List Nat := [6269]
set_option maxRecDepth 4096 in
theorem innerCandidates134_6_checked : candidates edge 4 773 2129 innerTail134_7 best.bound = innerCandidates134_6 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_6 : scan edge (search edge 3) 4 [773] 773 innerTail134_6 best = best := by
  rw [innerTail134_6, scan, if_neg (by decide), innerCandidates134_6_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_7
    (if 3 ≤ innerCandidates134_6.length then search edge 3 [773, 2129] innerCandidates134_6 2902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_7

def innerTail134_5 : List Nat := 2063 :: innerTail134_6
def innerCandidates134_5 : List Nat := [5711]
set_option maxRecDepth 4096 in
theorem innerCandidates134_5_checked : candidates edge 4 773 2063 innerTail134_6 best.bound = innerCandidates134_5 := by
  unfold candidates
  have row : edge 2063 = PrimePairCache.smallRow2063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_5 : scan edge (search edge 3) 4 [773] 773 innerTail134_5 best = best := by
  rw [innerTail134_5, scan, if_neg (by decide), innerCandidates134_5_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_6
    (if 3 ≤ innerCandidates134_5.length then search edge 3 [773, 2063] innerCandidates134_5 2836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_6

def innerTail134_4 : List Nat := 1877 :: innerTail134_5
def innerCandidates134_4 : List Nat := [5711]
set_option maxRecDepth 4096 in
theorem innerCandidates134_4_checked : candidates edge 4 773 1877 innerTail134_5 best.bound = innerCandidates134_4 := by
  unfold candidates
  have row : edge 1877 = PrimePairCache.smallRow1877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_4 : scan edge (search edge 3) 4 [773] 773 innerTail134_4 best = best := by
  rw [innerTail134_4, scan, if_neg (by decide), innerCandidates134_4_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_5
    (if 3 ≤ innerCandidates134_4.length then search edge 3 [773, 1877] innerCandidates134_4 2650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_5

def innerTail134_3 : List Nat := 1721 :: innerTail134_4
def innerCandidates134_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_3_checked : candidates edge 4 773 1721 innerTail134_4 best.bound = innerCandidates134_3 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_3 : scan edge (search edge 3) 4 [773] 773 innerTail134_3 best = best := by
  rw [innerTail134_3, scan, if_neg (by decide), innerCandidates134_3_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_4
    (if 3 ≤ innerCandidates134_3.length then search edge 3 [773, 1721] innerCandidates134_3 2494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_4

def innerTail134_2 : List Nat := 1667 :: innerTail134_3
def innerCandidates134_2 : List Nat := [5669]
set_option maxRecDepth 4096 in
theorem innerCandidates134_2_checked : candidates edge 4 773 1667 innerTail134_3 best.bound = innerCandidates134_2 := by
  unfold candidates
  have row : edge 1667 = PrimePairCache.smallRow1667.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_2 : scan edge (search edge 3) 4 [773] 773 innerTail134_2 best = best := by
  rw [innerTail134_2, scan, if_neg (by decide), innerCandidates134_2_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_3
    (if 3 ≤ innerCandidates134_2.length then search edge 3 [773, 1667] innerCandidates134_2 2440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_3

def innerTail134_1 : List Nat := 1439 :: innerTail134_2
def innerCandidates134_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_1_checked : candidates edge 4 773 1439 innerTail134_2 best.bound = innerCandidates134_1 := by
  unfold candidates
  have row : edge 1439 = PrimePairCache.smallRow1439.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_1 : scan edge (search edge 3) 4 [773] 773 innerTail134_1 best = best := by
  rw [innerTail134_1, scan, if_neg (by decide), innerCandidates134_1_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_2
    (if 3 ≤ innerCandidates134_1.length then search edge 3 [773, 1439] innerCandidates134_1 2212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_2

def innerTail134_0 : List Nat := 953 :: innerTail134_1
def innerCandidates134_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates134_0_checked : candidates edge 4 773 953 innerTail134_1 best.bound = innerCandidates134_0 := by
  unfold candidates
  have row : edge 953 = PrimePairCache.smallRow953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan134_0 : scan edge (search edge 3) 4 [773] 773 innerTail134_0 best = best := by
  rw [innerTail134_0, scan, if_neg (by decide), innerCandidates134_0_checked]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_1
    (if 3 ≤ innerCandidates134_0.length then search edge 3 [773, 953] innerCandidates134_0 1726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan134_1

theorem rootBranch134 : search edge 4 [773] rootCandidates134 773 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [773] 773 innerTail134_0 best = best
  exact innerScan134_0

def rootCandidates133 : List Nat := [919, 997, 1381, 2053, 2647, 3313, 4021, 4651, 4987, 5113, 5197, 5227, 5503, 5791, 6133, 6211, 6343, 6763, 6967, 7057, 7507]
set_option maxRecDepth 4096 in
theorem rootCandidates133_checked : candidates edge 5 0 769 tail134 best.bound = rootCandidates133 := by
  unfold candidates
  have row : edge 769 = PrimePairCache.smallRow769.contains := by funext q; rfl
  rw [row]
  decide

def innerTail133_21 : List Nat := []
theorem innerScan133_21 : scan edge (search edge 3) 4 [769] 769 innerTail133_21 best = best := rfl

def innerTail133_20 : List Nat := 7507 :: innerTail133_21
def innerCandidates133_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_20_checked : candidates edge 4 769 7507 innerTail133_21 best.bound = innerCandidates133_20 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_20 : scan edge (search edge 3) 4 [769] 769 innerTail133_20 best = best := by
  rw [innerTail133_20, scan, if_neg (by decide), innerCandidates133_20_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_21
    (if 3 ≤ innerCandidates133_20.length then search edge 3 [769, 7507] innerCandidates133_20 8276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_21

def innerTail133_19 : List Nat := 7057 :: innerTail133_20
def innerCandidates133_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_19_checked : candidates edge 4 769 7057 innerTail133_20 best.bound = innerCandidates133_19 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_19 : scan edge (search edge 3) 4 [769] 769 innerTail133_19 best = best := by
  rw [innerTail133_19, scan, if_neg (by decide), innerCandidates133_19_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_20
    (if 3 ≤ innerCandidates133_19.length then search edge 3 [769, 7057] innerCandidates133_19 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_20

def innerTail133_18 : List Nat := 6967 :: innerTail133_19
def innerCandidates133_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_18_checked : candidates edge 4 769 6967 innerTail133_19 best.bound = innerCandidates133_18 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_18 : scan edge (search edge 3) 4 [769] 769 innerTail133_18 best = best := by
  rw [innerTail133_18, scan, if_neg (by decide), innerCandidates133_18_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_19
    (if 3 ≤ innerCandidates133_18.length then search edge 3 [769, 6967] innerCandidates133_18 7736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_19

def innerTail133_17 : List Nat := 6763 :: innerTail133_18
def innerCandidates133_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_17_checked : candidates edge 4 769 6763 innerTail133_18 best.bound = innerCandidates133_17 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_17 : scan edge (search edge 3) 4 [769] 769 innerTail133_17 best = best := by
  rw [innerTail133_17, scan, if_neg (by decide), innerCandidates133_17_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_18
    (if 3 ≤ innerCandidates133_17.length then search edge 3 [769, 6763] innerCandidates133_17 7532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_18

def innerTail133_16 : List Nat := 6343 :: innerTail133_17
def innerCandidates133_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_16_checked : candidates edge 4 769 6343 innerTail133_17 best.bound = innerCandidates133_16 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_16 : scan edge (search edge 3) 4 [769] 769 innerTail133_16 best = best := by
  rw [innerTail133_16, scan, if_neg (by decide), innerCandidates133_16_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_17
    (if 3 ≤ innerCandidates133_16.length then search edge 3 [769, 6343] innerCandidates133_16 7112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_17

def innerTail133_15 : List Nat := 6211 :: innerTail133_16
def innerCandidates133_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_15_checked : candidates edge 4 769 6211 innerTail133_16 best.bound = innerCandidates133_15 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_15 : scan edge (search edge 3) 4 [769] 769 innerTail133_15 best = best := by
  rw [innerTail133_15, scan, if_neg (by decide), innerCandidates133_15_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_16
    (if 3 ≤ innerCandidates133_15.length then search edge 3 [769, 6211] innerCandidates133_15 6980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_16

def innerTail133_14 : List Nat := 6133 :: innerTail133_15
def innerCandidates133_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_14_checked : candidates edge 4 769 6133 innerTail133_15 best.bound = innerCandidates133_14 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_14 : scan edge (search edge 3) 4 [769] 769 innerTail133_14 best = best := by
  rw [innerTail133_14, scan, if_neg (by decide), innerCandidates133_14_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_15
    (if 3 ≤ innerCandidates133_14.length then search edge 3 [769, 6133] innerCandidates133_14 6902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_15

def innerTail133_13 : List Nat := 5791 :: innerTail133_14
def innerCandidates133_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_13_checked : candidates edge 4 769 5791 innerTail133_14 best.bound = innerCandidates133_13 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_13 : scan edge (search edge 3) 4 [769] 769 innerTail133_13 best = best := by
  rw [innerTail133_13, scan, if_neg (by decide), innerCandidates133_13_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_14
    (if 3 ≤ innerCandidates133_13.length then search edge 3 [769, 5791] innerCandidates133_13 6560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_14

def innerTail133_12 : List Nat := 5503 :: innerTail133_13
def innerCandidates133_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_12_checked : candidates edge 4 769 5503 innerTail133_13 best.bound = innerCandidates133_12 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_12 : scan edge (search edge 3) 4 [769] 769 innerTail133_12 best = best := by
  rw [innerTail133_12, scan, if_neg (by decide), innerCandidates133_12_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_13
    (if 3 ≤ innerCandidates133_12.length then search edge 3 [769, 5503] innerCandidates133_12 6272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_13

def innerTail133_11 : List Nat := 5227 :: innerTail133_12
def innerCandidates133_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_11_checked : candidates edge 4 769 5227 innerTail133_12 best.bound = innerCandidates133_11 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_11 : scan edge (search edge 3) 4 [769] 769 innerTail133_11 best = best := by
  rw [innerTail133_11, scan, if_neg (by decide), innerCandidates133_11_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_12
    (if 3 ≤ innerCandidates133_11.length then search edge 3 [769, 5227] innerCandidates133_11 5996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_12

def innerTail133_10 : List Nat := 5197 :: innerTail133_11
def innerCandidates133_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_10_checked : candidates edge 4 769 5197 innerTail133_11 best.bound = innerCandidates133_10 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_10 : scan edge (search edge 3) 4 [769] 769 innerTail133_10 best = best := by
  rw [innerTail133_10, scan, if_neg (by decide), innerCandidates133_10_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_11
    (if 3 ≤ innerCandidates133_10.length then search edge 3 [769, 5197] innerCandidates133_10 5966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_11

def innerTail133_9 : List Nat := 5113 :: innerTail133_10
def innerCandidates133_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_9_checked : candidates edge 4 769 5113 innerTail133_10 best.bound = innerCandidates133_9 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_9 : scan edge (search edge 3) 4 [769] 769 innerTail133_9 best = best := by
  rw [innerTail133_9, scan, if_neg (by decide), innerCandidates133_9_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_10
    (if 3 ≤ innerCandidates133_9.length then search edge 3 [769, 5113] innerCandidates133_9 5882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_10

def innerTail133_8 : List Nat := 4987 :: innerTail133_9
def innerCandidates133_8 : List Nat := [5197]
set_option maxRecDepth 4096 in
theorem innerCandidates133_8_checked : candidates edge 4 769 4987 innerTail133_9 best.bound = innerCandidates133_8 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_8 : scan edge (search edge 3) 4 [769] 769 innerTail133_8 best = best := by
  rw [innerTail133_8, scan, if_neg (by decide), innerCandidates133_8_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_9
    (if 3 ≤ innerCandidates133_8.length then search edge 3 [769, 4987] innerCandidates133_8 5756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_9

def innerTail133_7 : List Nat := 4651 :: innerTail133_8
def innerCandidates133_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_7_checked : candidates edge 4 769 4651 innerTail133_8 best.bound = innerCandidates133_7 := by
  unfold candidates
  have row : edge 4651 = SearchData.row4651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_7 : scan edge (search edge 3) 4 [769] 769 innerTail133_7 best = best := by
  rw [innerTail133_7, scan, if_neg (by decide), innerCandidates133_7_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_8
    (if 3 ≤ innerCandidates133_7.length then search edge 3 [769, 4651] innerCandidates133_7 5420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_8

def innerTail133_6 : List Nat := 4021 :: innerTail133_7
def innerCandidates133_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_6_checked : candidates edge 4 769 4021 innerTail133_7 best.bound = innerCandidates133_6 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_6 : scan edge (search edge 3) 4 [769] 769 innerTail133_6 best = best := by
  rw [innerTail133_6, scan, if_neg (by decide), innerCandidates133_6_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_7
    (if 3 ≤ innerCandidates133_6.length then search edge 3 [769, 4021] innerCandidates133_6 4790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_7

def innerTail133_5 : List Nat := 3313 :: innerTail133_6
def innerCandidates133_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_5_checked : candidates edge 4 769 3313 innerTail133_6 best.bound = innerCandidates133_5 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_5 : scan edge (search edge 3) 4 [769] 769 innerTail133_5 best = best := by
  rw [innerTail133_5, scan, if_neg (by decide), innerCandidates133_5_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_6
    (if 3 ≤ innerCandidates133_5.length then search edge 3 [769, 3313] innerCandidates133_5 4082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_6

def innerTail133_4 : List Nat := 2647 :: innerTail133_5
def innerCandidates133_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_4_checked : candidates edge 4 769 2647 innerTail133_5 best.bound = innerCandidates133_4 := by
  unfold candidates
  have row : edge 2647 = PrimePairCache.smallRow2647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_4 : scan edge (search edge 3) 4 [769] 769 innerTail133_4 best = best := by
  rw [innerTail133_4, scan, if_neg (by decide), innerCandidates133_4_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_5
    (if 3 ≤ innerCandidates133_4.length then search edge 3 [769, 2647] innerCandidates133_4 3416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_5

def innerTail133_3 : List Nat := 2053 :: innerTail133_4
def innerCandidates133_3 : List Nat := [6343]
set_option maxRecDepth 4096 in
theorem innerCandidates133_3_checked : candidates edge 4 769 2053 innerTail133_4 best.bound = innerCandidates133_3 := by
  unfold candidates
  have row : edge 2053 = PrimePairCache.smallRow2053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_3 : scan edge (search edge 3) 4 [769] 769 innerTail133_3 best = best := by
  rw [innerTail133_3, scan, if_neg (by decide), innerCandidates133_3_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_4
    (if 3 ≤ innerCandidates133_3.length then search edge 3 [769, 2053] innerCandidates133_3 2822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_4

def innerTail133_2 : List Nat := 1381 :: innerTail133_3
def innerCandidates133_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_2_checked : candidates edge 4 769 1381 innerTail133_3 best.bound = innerCandidates133_2 := by
  unfold candidates
  have row : edge 1381 = PrimePairCache.smallRow1381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_2 : scan edge (search edge 3) 4 [769] 769 innerTail133_2 best = best := by
  rw [innerTail133_2, scan, if_neg (by decide), innerCandidates133_2_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_3
    (if 3 ≤ innerCandidates133_2.length then search edge 3 [769, 1381] innerCandidates133_2 2150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_3

def innerTail133_1 : List Nat := 997 :: innerTail133_2
def innerCandidates133_1 : List Nat := [6763, 7057]
set_option maxRecDepth 4096 in
theorem innerCandidates133_1_checked : candidates edge 4 769 997 innerTail133_2 best.bound = innerCandidates133_1 := by
  unfold candidates
  have row : edge 997 = PrimePairCache.smallRow997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_1 : scan edge (search edge 3) 4 [769] 769 innerTail133_1 best = best := by
  rw [innerTail133_1, scan, if_neg (by decide), innerCandidates133_1_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_2
    (if 3 ≤ innerCandidates133_1.length then search edge 3 [769, 997] innerCandidates133_1 1766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_2

def innerTail133_0 : List Nat := 919 :: innerTail133_1
def innerCandidates133_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates133_0_checked : candidates edge 4 769 919 innerTail133_1 best.bound = innerCandidates133_0 := by
  unfold candidates
  have row : edge 919 = PrimePairCache.smallRow919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan133_0 : scan edge (search edge 3) 4 [769] 769 innerTail133_0 best = best := by
  rw [innerTail133_0, scan, if_neg (by decide), innerCandidates133_0_checked]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_1
    (if 3 ≤ innerCandidates133_0.length then search edge 3 [769, 919] innerCandidates133_0 1688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan133_1

theorem rootBranch133 : search edge 4 [769] rootCandidates133 769 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [769] 769 innerTail133_0 best = best
  exact innerScan133_0

end ProjectEulerSolutions.PrimePairUpper.Search
