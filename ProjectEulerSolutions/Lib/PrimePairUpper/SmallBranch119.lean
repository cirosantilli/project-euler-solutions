import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates324 : List Nat := [2503, 2917, 3169, 3253, 3469, 3631, 4261, 5119, 5227, 5647, 5857, 6343, 6529]
set_option maxRecDepth 4096 in
theorem rootCandidates324_checked : candidates edge 5 0 2179 tail325 best.bound = rootCandidates324 := by
  unfold candidates
  have row : edge 2179 = PrimePairCache.smallRow2179.contains := by funext q; rfl
  rw [row]
  decide

def innerTail324_13 : List Nat := []
theorem innerScan324_13 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_13 best = best := rfl

def innerTail324_12 : List Nat := 6529 :: innerTail324_13
def innerCandidates324_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates324_12_checked : candidates edge 4 2179 6529 innerTail324_13 best.bound = innerCandidates324_12 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan324_12 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_12 best = best := by
  rw [innerTail324_12, scan, if_neg (by decide), innerCandidates324_12_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_13
    (if 3 ≤ innerCandidates324_12.length then search edge 3 [2179, 6529] innerCandidates324_12 8708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan324_13

def innerTail324_11 : List Nat := 6343 :: innerTail324_12
def innerCandidates324_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates324_11_checked : candidates edge 4 2179 6343 innerTail324_12 best.bound = innerCandidates324_11 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan324_11 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_11 best = best := by
  rw [innerTail324_11, scan, if_neg (by decide), innerCandidates324_11_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_12
    (if 3 ≤ innerCandidates324_11.length then search edge 3 [2179, 6343] innerCandidates324_11 8522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan324_12

def innerTail324_10 : List Nat := 5857 :: innerTail324_11
def innerCandidates324_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates324_10_checked : candidates edge 4 2179 5857 innerTail324_11 best.bound = innerCandidates324_10 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan324_10 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_10 best = best := by
  rw [innerTail324_10, scan, if_neg (by decide), innerCandidates324_10_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_11
    (if 3 ≤ innerCandidates324_10.length then search edge 3 [2179, 5857] innerCandidates324_10 8036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan324_11

def innerTail324_9 : List Nat := 5647 :: innerTail324_10
def innerCandidates324_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates324_9_checked : candidates edge 4 2179 5647 innerTail324_10 best.bound = innerCandidates324_9 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan324_9 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_9 best = best := by
  rw [innerTail324_9, scan, if_neg (by decide), innerCandidates324_9_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_10
    (if 3 ≤ innerCandidates324_9.length then search edge 3 [2179, 5647] innerCandidates324_9 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan324_10

def innerTail324_8 : List Nat := 5227 :: innerTail324_9
def innerCandidates324_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates324_8_checked : candidates edge 4 2179 5227 innerTail324_9 best.bound = innerCandidates324_8 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan324_8 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_8 best = best := by
  rw [innerTail324_8, scan, if_neg (by decide), innerCandidates324_8_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_9
    (if 3 ≤ innerCandidates324_8.length then search edge 3 [2179, 5227] innerCandidates324_8 7406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan324_9

def innerTail324_7 : List Nat := 5119 :: innerTail324_8
def innerCandidates324_7 : List Nat := [6529]
set_option maxRecDepth 4096 in
theorem innerCandidates324_7_checked : candidates edge 4 2179 5119 innerTail324_8 best.bound = innerCandidates324_7 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan324_7 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_7 best = best := by
  rw [innerTail324_7, scan, if_neg (by decide), innerCandidates324_7_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_8
    (if 3 ≤ innerCandidates324_7.length then search edge 3 [2179, 5119] innerCandidates324_7 7298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan324_8

def innerTail324_6 : List Nat := 4261 :: innerTail324_7
def innerCandidates324_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates324_6_checked : candidates edge 4 2179 4261 innerTail324_7 best.bound = innerCandidates324_6 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan324_6 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_6 best = best := by
  rw [innerTail324_6, scan, if_neg (by decide), innerCandidates324_6_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_7
    (if 3 ≤ innerCandidates324_6.length then search edge 3 [2179, 4261] innerCandidates324_6 6440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan324_7

def innerTail324_5 : List Nat := 3631 :: innerTail324_6
def innerCandidates324_5 : List Nat := [5119]
set_option maxRecDepth 4096 in
theorem innerCandidates324_5_checked : candidates edge 4 2179 3631 innerTail324_6 best.bound = innerCandidates324_5 := by
  unfold candidates
  have row : edge 3631 = SearchData.row3631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan324_5 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_5 best = best := by
  rw [innerTail324_5, scan, if_neg (by decide), innerCandidates324_5_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_6
    (if 3 ≤ innerCandidates324_5.length then search edge 3 [2179, 3631] innerCandidates324_5 5810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan324_6

def innerTail324_4 : List Nat := 3469 :: innerTail324_5
def innerCandidates324_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates324_4_checked : candidates edge 4 2179 3469 innerTail324_5 best.bound = innerCandidates324_4 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan324_4 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_4 best = best := by
  rw [innerTail324_4, scan, if_neg (by decide), innerCandidates324_4_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_5
    (if 3 ≤ innerCandidates324_4.length then search edge 3 [2179, 3469] innerCandidates324_4 5648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan324_5

def innerTail324_3 : List Nat := 3253 :: innerTail324_4
def innerCandidates324_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates324_3_checked : candidates edge 4 2179 3253 innerTail324_4 best.bound = innerCandidates324_3 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan324_3 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_3 best = best := by
  rw [innerTail324_3, scan, if_neg (by decide), innerCandidates324_3_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_4
    (if 3 ≤ innerCandidates324_3.length then search edge 3 [2179, 3253] innerCandidates324_3 5432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan324_4

def innerTail324_2 : List Nat := 3169 :: innerTail324_3
def innerCandidates324_2 : List Nat := [5119]
set_option maxRecDepth 4096 in
theorem innerCandidates324_2_checked : candidates edge 4 2179 3169 innerTail324_3 best.bound = innerCandidates324_2 := by
  unfold candidates
  have row : edge 3169 = SearchData.row3169.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan324_2 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_2 best = best := by
  rw [innerTail324_2, scan, if_neg (by decide), innerCandidates324_2_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_3
    (if 3 ≤ innerCandidates324_2.length then search edge 3 [2179, 3169] innerCandidates324_2 5348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan324_3

def innerTail324_1 : List Nat := 2917 :: innerTail324_2
def innerCandidates324_1 : List Nat := [3169, 3631, 5227]
set_option maxRecDepth 4096 in
theorem innerCandidates324_1_checked : candidates edge 4 2179 2917 innerTail324_2 best.bound = innerCandidates324_1 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild324_1 : search edge 3 [2179, 2917] innerCandidates324_1 5096 best = best := by decide

theorem innerScan324_1 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_1 best = best := by
  rw [innerTail324_1, scan, if_neg (by decide), innerCandidates324_1_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_2
    (if 3 ≤ innerCandidates324_1.length then search edge 3 [2179, 2917] innerCandidates324_1 5096 best else best) = best
  rw [if_pos (by decide), innerChild324_1]
  exact innerScan324_2

def innerTail324_0 : List Nat := 2503 :: innerTail324_1
def innerCandidates324_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates324_0_checked : candidates edge 4 2179 2503 innerTail324_1 best.bound = innerCandidates324_0 := by
  unfold candidates
  have row : edge 2503 = PrimePairCache.smallRow2503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan324_0 : scan edge (search edge 3) 4 [2179] 2179 innerTail324_0 best = best := by
  rw [innerTail324_0, scan, if_neg (by decide), innerCandidates324_0_checked]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_1
    (if 3 ≤ innerCandidates324_0.length then search edge 3 [2179, 2503] innerCandidates324_0 4682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan324_1

theorem rootBranch324 : search edge 4 [2179] rootCandidates324 2179 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2179] 2179 innerTail324_0 best = best
  exact innerScan324_0

def rootCandidates323 : List Nat := [2887, 4357, 4597, 5479, 6079, 6211, 6337, 6373, 6841, 6949, 7027, 7237, 7411, 7621, 7681, 7759]
set_option maxRecDepth 4096 in
theorem rootCandidates323_checked : candidates edge 5 0 2161 tail324 best.bound = rootCandidates323 := by
  unfold candidates
  have row : edge 2161 = PrimePairCache.smallRow2161.contains := by funext q; rfl
  rw [row]
  decide

def innerTail323_16 : List Nat := []
theorem innerScan323_16 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_16 best = best := rfl

def innerTail323_15 : List Nat := 7759 :: innerTail323_16
def innerCandidates323_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates323_15_checked : candidates edge 4 2161 7759 innerTail323_16 best.bound = innerCandidates323_15 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_15 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_15 best = best := by
  rw [innerTail323_15, scan, if_neg (by decide), innerCandidates323_15_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_16
    (if 3 ≤ innerCandidates323_15.length then search edge 3 [2161, 7759] innerCandidates323_15 9920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_16

def innerTail323_14 : List Nat := 7681 :: innerTail323_15
def innerCandidates323_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates323_14_checked : candidates edge 4 2161 7681 innerTail323_15 best.bound = innerCandidates323_14 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_14 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_14 best = best := by
  rw [innerTail323_14, scan, if_neg (by decide), innerCandidates323_14_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_15
    (if 3 ≤ innerCandidates323_14.length then search edge 3 [2161, 7681] innerCandidates323_14 9842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_15

def innerTail323_13 : List Nat := 7621 :: innerTail323_14
def innerCandidates323_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates323_13_checked : candidates edge 4 2161 7621 innerTail323_14 best.bound = innerCandidates323_13 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_13 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_13 best = best := by
  rw [innerTail323_13, scan, if_neg (by decide), innerCandidates323_13_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_14
    (if 3 ≤ innerCandidates323_13.length then search edge 3 [2161, 7621] innerCandidates323_13 9782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_14

def innerTail323_12 : List Nat := 7411 :: innerTail323_13
def innerCandidates323_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates323_12_checked : candidates edge 4 2161 7411 innerTail323_13 best.bound = innerCandidates323_12 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_12 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_12 best = best := by
  rw [innerTail323_12, scan, if_neg (by decide), innerCandidates323_12_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_13
    (if 3 ≤ innerCandidates323_12.length then search edge 3 [2161, 7411] innerCandidates323_12 9572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_13

def innerTail323_11 : List Nat := 7237 :: innerTail323_12
def innerCandidates323_11 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates323_11_checked : candidates edge 4 2161 7237 innerTail323_12 best.bound = innerCandidates323_11 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_11 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_11 best = best := by
  rw [innerTail323_11, scan, if_neg (by decide), innerCandidates323_11_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_12
    (if 3 ≤ innerCandidates323_11.length then search edge 3 [2161, 7237] innerCandidates323_11 9398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_12

def innerTail323_10 : List Nat := 7027 :: innerTail323_11
def innerCandidates323_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates323_10_checked : candidates edge 4 2161 7027 innerTail323_11 best.bound = innerCandidates323_10 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_10 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_10 best = best := by
  rw [innerTail323_10, scan, if_neg (by decide), innerCandidates323_10_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_11
    (if 3 ≤ innerCandidates323_10.length then search edge 3 [2161, 7027] innerCandidates323_10 9188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_11

def innerTail323_9 : List Nat := 6949 :: innerTail323_10
def innerCandidates323_9 : List Nat := [7621]
set_option maxRecDepth 4096 in
theorem innerCandidates323_9_checked : candidates edge 4 2161 6949 innerTail323_10 best.bound = innerCandidates323_9 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_9 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_9 best = best := by
  rw [innerTail323_9, scan, if_neg (by decide), innerCandidates323_9_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_10
    (if 3 ≤ innerCandidates323_9.length then search edge 3 [2161, 6949] innerCandidates323_9 9110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_10

def innerTail323_8 : List Nat := 6841 :: innerTail323_9
def innerCandidates323_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates323_8_checked : candidates edge 4 2161 6841 innerTail323_9 best.bound = innerCandidates323_8 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_8 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_8 best = best := by
  rw [innerTail323_8, scan, if_neg (by decide), innerCandidates323_8_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_9
    (if 3 ≤ innerCandidates323_8.length then search edge 3 [2161, 6841] innerCandidates323_8 9002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_9

def innerTail323_7 : List Nat := 6373 :: innerTail323_8
def innerCandidates323_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates323_7_checked : candidates edge 4 2161 6373 innerTail323_8 best.bound = innerCandidates323_7 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_7 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_7 best = best := by
  rw [innerTail323_7, scan, if_neg (by decide), innerCandidates323_7_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_8
    (if 3 ≤ innerCandidates323_7.length then search edge 3 [2161, 6373] innerCandidates323_7 8534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_8

def innerTail323_6 : List Nat := 6337 :: innerTail323_7
def innerCandidates323_6 : List Nat := [6949]
set_option maxRecDepth 4096 in
theorem innerCandidates323_6_checked : candidates edge 4 2161 6337 innerTail323_7 best.bound = innerCandidates323_6 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_6 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_6 best = best := by
  rw [innerTail323_6, scan, if_neg (by decide), innerCandidates323_6_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_7
    (if 3 ≤ innerCandidates323_6.length then search edge 3 [2161, 6337] innerCandidates323_6 8498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_7

def innerTail323_5 : List Nat := 6211 :: innerTail323_6
def innerCandidates323_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates323_5_checked : candidates edge 4 2161 6211 innerTail323_6 best.bound = innerCandidates323_5 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_5 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_5 best = best := by
  rw [innerTail323_5, scan, if_neg (by decide), innerCandidates323_5_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_6
    (if 3 ≤ innerCandidates323_5.length then search edge 3 [2161, 6211] innerCandidates323_5 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_6

def innerTail323_4 : List Nat := 6079 :: innerTail323_5
def innerCandidates323_4 : List Nat := [7621]
set_option maxRecDepth 4096 in
theorem innerCandidates323_4_checked : candidates edge 4 2161 6079 innerTail323_5 best.bound = innerCandidates323_4 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_4 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_4 best = best := by
  rw [innerTail323_4, scan, if_neg (by decide), innerCandidates323_4_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_5
    (if 3 ≤ innerCandidates323_4.length then search edge 3 [2161, 6079] innerCandidates323_4 8240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_5

def innerTail323_3 : List Nat := 5479 :: innerTail323_4
def innerCandidates323_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates323_3_checked : candidates edge 4 2161 5479 innerTail323_4 best.bound = innerCandidates323_3 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_3 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_3 best = best := by
  rw [innerTail323_3, scan, if_neg (by decide), innerCandidates323_3_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_4
    (if 3 ≤ innerCandidates323_3.length then search edge 3 [2161, 5479] innerCandidates323_3 7640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_4

def innerTail323_2 : List Nat := 4597 :: innerTail323_3
def innerCandidates323_2 : List Nat := [6079]
set_option maxRecDepth 4096 in
theorem innerCandidates323_2_checked : candidates edge 4 2161 4597 innerTail323_3 best.bound = innerCandidates323_2 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_2 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_2 best = best := by
  rw [innerTail323_2, scan, if_neg (by decide), innerCandidates323_2_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_3
    (if 3 ≤ innerCandidates323_2.length then search edge 3 [2161, 4597] innerCandidates323_2 6758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_3

def innerTail323_1 : List Nat := 4357 :: innerTail323_2
def innerCandidates323_1 : List Nat := [5479, 7411]
set_option maxRecDepth 4096 in
theorem innerCandidates323_1_checked : candidates edge 4 2161 4357 innerTail323_2 best.bound = innerCandidates323_1 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_1 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_1 best = best := by
  rw [innerTail323_1, scan, if_neg (by decide), innerCandidates323_1_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_2
    (if 3 ≤ innerCandidates323_1.length then search edge 3 [2161, 4357] innerCandidates323_1 6518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_2

def innerTail323_0 : List Nat := 2887 :: innerTail323_1
def innerCandidates323_0 : List Nat := [4597]
set_option maxRecDepth 4096 in
theorem innerCandidates323_0_checked : candidates edge 4 2161 2887 innerTail323_1 best.bound = innerCandidates323_0 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan323_0 : scan edge (search edge 3) 4 [2161] 2161 innerTail323_0 best = best := by
  rw [innerTail323_0, scan, if_neg (by decide), innerCandidates323_0_checked]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_1
    (if 3 ≤ innerCandidates323_0.length then search edge 3 [2161, 2887] innerCandidates323_0 5048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan323_1

theorem rootBranch323 : search edge 4 [2161] rootCandidates323 2161 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2161] 2161 innerTail323_0 best = best
  exact innerScan323_0

def rootCandidates322 : List Nat := [2417, 2663, 2969, 4397, 4967, 5237, 5273, 5351, 6143, 6863, 6869, 7121, 7151]
set_option maxRecDepth 4096 in
theorem rootCandidates322_checked : candidates edge 5 0 2153 tail323 best.bound = rootCandidates322 := by
  unfold candidates
  have row : edge 2153 = PrimePairCache.smallRow2153.contains := by funext q; rfl
  rw [row]
  decide

def innerTail322_13 : List Nat := []
theorem innerScan322_13 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_13 best = best := rfl

def innerTail322_12 : List Nat := 7151 :: innerTail322_13
def innerCandidates322_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates322_12_checked : candidates edge 4 2153 7151 innerTail322_13 best.bound = innerCandidates322_12 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_12 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_12 best = best := by
  rw [innerTail322_12, scan, if_neg (by decide), innerCandidates322_12_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_13
    (if 3 ≤ innerCandidates322_12.length then search edge 3 [2153, 7151] innerCandidates322_12 9304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_13

def innerTail322_11 : List Nat := 7121 :: innerTail322_12
def innerCandidates322_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates322_11_checked : candidates edge 4 2153 7121 innerTail322_12 best.bound = innerCandidates322_11 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_11 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_11 best = best := by
  rw [innerTail322_11, scan, if_neg (by decide), innerCandidates322_11_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_12
    (if 3 ≤ innerCandidates322_11.length then search edge 3 [2153, 7121] innerCandidates322_11 9274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_12

def innerTail322_10 : List Nat := 6869 :: innerTail322_11
def innerCandidates322_10 : List Nat := [7121]
set_option maxRecDepth 4096 in
theorem innerCandidates322_10_checked : candidates edge 4 2153 6869 innerTail322_11 best.bound = innerCandidates322_10 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_10 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_10 best = best := by
  rw [innerTail322_10, scan, if_neg (by decide), innerCandidates322_10_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_11
    (if 3 ≤ innerCandidates322_10.length then search edge 3 [2153, 6869] innerCandidates322_10 9022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_11

def innerTail322_9 : List Nat := 6863 :: innerTail322_10
def innerCandidates322_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates322_9_checked : candidates edge 4 2153 6863 innerTail322_10 best.bound = innerCandidates322_9 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_9 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_9 best = best := by
  rw [innerTail322_9, scan, if_neg (by decide), innerCandidates322_9_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_10
    (if 3 ≤ innerCandidates322_9.length then search edge 3 [2153, 6863] innerCandidates322_9 9016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_10

def innerTail322_8 : List Nat := 6143 :: innerTail322_9
def innerCandidates322_8 : List Nat := [7151]
set_option maxRecDepth 4096 in
theorem innerCandidates322_8_checked : candidates edge 4 2153 6143 innerTail322_9 best.bound = innerCandidates322_8 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_8 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_8 best = best := by
  rw [innerTail322_8, scan, if_neg (by decide), innerCandidates322_8_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_9
    (if 3 ≤ innerCandidates322_8.length then search edge 3 [2153, 6143] innerCandidates322_8 8296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_9

def innerTail322_7 : List Nat := 5351 :: innerTail322_8
def innerCandidates322_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates322_7_checked : candidates edge 4 2153 5351 innerTail322_8 best.bound = innerCandidates322_7 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_7 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_7 best = best := by
  rw [innerTail322_7, scan, if_neg (by decide), innerCandidates322_7_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_8
    (if 3 ≤ innerCandidates322_7.length then search edge 3 [2153, 5351] innerCandidates322_7 7504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_8

def innerTail322_6 : List Nat := 5273 :: innerTail322_7
def innerCandidates322_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates322_6_checked : candidates edge 4 2153 5273 innerTail322_7 best.bound = innerCandidates322_6 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_6 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_6 best = best := by
  rw [innerTail322_6, scan, if_neg (by decide), innerCandidates322_6_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_7
    (if 3 ≤ innerCandidates322_6.length then search edge 3 [2153, 5273] innerCandidates322_6 7426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_7

def innerTail322_5 : List Nat := 5237 :: innerTail322_6
def innerCandidates322_5 : List Nat := [7151]
set_option maxRecDepth 4096 in
theorem innerCandidates322_5_checked : candidates edge 4 2153 5237 innerTail322_6 best.bound = innerCandidates322_5 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_5 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_5 best = best := by
  rw [innerTail322_5, scan, if_neg (by decide), innerCandidates322_5_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_6
    (if 3 ≤ innerCandidates322_5.length then search edge 3 [2153, 5237] innerCandidates322_5 7390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_6

def innerTail322_4 : List Nat := 4967 :: innerTail322_5
def innerCandidates322_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates322_4_checked : candidates edge 4 2153 4967 innerTail322_5 best.bound = innerCandidates322_4 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_4 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_4 best = best := by
  rw [innerTail322_4, scan, if_neg (by decide), innerCandidates322_4_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_5
    (if 3 ≤ innerCandidates322_4.length then search edge 3 [2153, 4967] innerCandidates322_4 7120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_5

def innerTail322_3 : List Nat := 4397 :: innerTail322_4
def innerCandidates322_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates322_3_checked : candidates edge 4 2153 4397 innerTail322_4 best.bound = innerCandidates322_3 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_3 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_3 best = best := by
  rw [innerTail322_3, scan, if_neg (by decide), innerCandidates322_3_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_4
    (if 3 ≤ innerCandidates322_3.length then search edge 3 [2153, 4397] innerCandidates322_3 6550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_4

def innerTail322_2 : List Nat := 2969 :: innerTail322_3
def innerCandidates322_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates322_2_checked : candidates edge 4 2153 2969 innerTail322_3 best.bound = innerCandidates322_2 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_2 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_2 best = best := by
  rw [innerTail322_2, scan, if_neg (by decide), innerCandidates322_2_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_3
    (if 3 ≤ innerCandidates322_2.length then search edge 3 [2153, 2969] innerCandidates322_2 5122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_3

def innerTail322_1 : List Nat := 2663 :: innerTail322_2
def innerCandidates322_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates322_1_checked : candidates edge 4 2153 2663 innerTail322_2 best.bound = innerCandidates322_1 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_1 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_1 best = best := by
  rw [innerTail322_1, scan, if_neg (by decide), innerCandidates322_1_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_2
    (if 3 ≤ innerCandidates322_1.length then search edge 3 [2153, 2663] innerCandidates322_1 4816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_2

def innerTail322_0 : List Nat := 2417 :: innerTail322_1
def innerCandidates322_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates322_0_checked : candidates edge 4 2153 2417 innerTail322_1 best.bound = innerCandidates322_0 := by
  unfold candidates
  have row : edge 2417 = PrimePairCache.smallRow2417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan322_0 : scan edge (search edge 3) 4 [2153] 2153 innerTail322_0 best = best := by
  rw [innerTail322_0, scan, if_neg (by decide), innerCandidates322_0_checked]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_1
    (if 3 ≤ innerCandidates322_0.length then search edge 3 [2153, 2417] innerCandidates322_0 4570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan322_1

theorem rootBranch322 : search edge 4 [2153] rootCandidates322 2153 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2153] 2153 innerTail322_0 best = best
  exact innerScan322_0

def rootCandidates321 : List Nat := [2239, 2347, 4657, 4801, 6229, 6301, 6691, 6871, 8017, 8089]
set_option maxRecDepth 4096 in
theorem rootCandidates321_checked : candidates edge 5 0 2143 tail322 best.bound = rootCandidates321 := by
  unfold candidates
  have row : edge 2143 = PrimePairCache.smallRow2143.contains := by funext q; rfl
  rw [row]
  decide

def innerTail321_10 : List Nat := []
theorem innerScan321_10 : scan edge (search edge 3) 4 [2143] 2143 innerTail321_10 best = best := rfl

def innerTail321_9 : List Nat := 8089 :: innerTail321_10
def innerCandidates321_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates321_9_checked : candidates edge 4 2143 8089 innerTail321_10 best.bound = innerCandidates321_9 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan321_9 : scan edge (search edge 3) 4 [2143] 2143 innerTail321_9 best = best := by
  rw [innerTail321_9, scan, if_neg (by decide), innerCandidates321_9_checked]
  change scan edge (search edge 3) 4 [2143] 2143 innerTail321_10
    (if 3 ≤ innerCandidates321_9.length then search edge 3 [2143, 8089] innerCandidates321_9 10232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan321_10

def innerTail321_8 : List Nat := 8017 :: innerTail321_9
def innerCandidates321_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates321_8_checked : candidates edge 4 2143 8017 innerTail321_9 best.bound = innerCandidates321_8 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan321_8 : scan edge (search edge 3) 4 [2143] 2143 innerTail321_8 best = best := by
  rw [innerTail321_8, scan, if_neg (by decide), innerCandidates321_8_checked]
  change scan edge (search edge 3) 4 [2143] 2143 innerTail321_9
    (if 3 ≤ innerCandidates321_8.length then search edge 3 [2143, 8017] innerCandidates321_8 10160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan321_9

def innerTail321_7 : List Nat := 6871 :: innerTail321_8
def innerCandidates321_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates321_7_checked : candidates edge 4 2143 6871 innerTail321_8 best.bound = innerCandidates321_7 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan321_7 : scan edge (search edge 3) 4 [2143] 2143 innerTail321_7 best = best := by
  rw [innerTail321_7, scan, if_neg (by decide), innerCandidates321_7_checked]
  change scan edge (search edge 3) 4 [2143] 2143 innerTail321_8
    (if 3 ≤ innerCandidates321_7.length then search edge 3 [2143, 6871] innerCandidates321_7 9014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan321_8

def innerTail321_6 : List Nat := 6691 :: innerTail321_7
def innerCandidates321_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates321_6_checked : candidates edge 4 2143 6691 innerTail321_7 best.bound = innerCandidates321_6 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan321_6 : scan edge (search edge 3) 4 [2143] 2143 innerTail321_6 best = best := by
  rw [innerTail321_6, scan, if_neg (by decide), innerCandidates321_6_checked]
  change scan edge (search edge 3) 4 [2143] 2143 innerTail321_7
    (if 3 ≤ innerCandidates321_6.length then search edge 3 [2143, 6691] innerCandidates321_6 8834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan321_7

def innerTail321_5 : List Nat := 6301 :: innerTail321_6
def innerCandidates321_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates321_5_checked : candidates edge 4 2143 6301 innerTail321_6 best.bound = innerCandidates321_5 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan321_5 : scan edge (search edge 3) 4 [2143] 2143 innerTail321_5 best = best := by
  rw [innerTail321_5, scan, if_neg (by decide), innerCandidates321_5_checked]
  change scan edge (search edge 3) 4 [2143] 2143 innerTail321_6
    (if 3 ≤ innerCandidates321_5.length then search edge 3 [2143, 6301] innerCandidates321_5 8444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan321_6

def innerTail321_4 : List Nat := 6229 :: innerTail321_5
def innerCandidates321_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates321_4_checked : candidates edge 4 2143 6229 innerTail321_5 best.bound = innerCandidates321_4 := by
  unfold candidates
  have row : edge 6229 = SearchData.row6229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan321_4 : scan edge (search edge 3) 4 [2143] 2143 innerTail321_4 best = best := by
  rw [innerTail321_4, scan, if_neg (by decide), innerCandidates321_4_checked]
  change scan edge (search edge 3) 4 [2143] 2143 innerTail321_5
    (if 3 ≤ innerCandidates321_4.length then search edge 3 [2143, 6229] innerCandidates321_4 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan321_5

def innerTail321_3 : List Nat := 4801 :: innerTail321_4
def innerCandidates321_3 : List Nat := [6229]
set_option maxRecDepth 4096 in
theorem innerCandidates321_3_checked : candidates edge 4 2143 4801 innerTail321_4 best.bound = innerCandidates321_3 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan321_3 : scan edge (search edge 3) 4 [2143] 2143 innerTail321_3 best = best := by
  rw [innerTail321_3, scan, if_neg (by decide), innerCandidates321_3_checked]
  change scan edge (search edge 3) 4 [2143] 2143 innerTail321_4
    (if 3 ≤ innerCandidates321_3.length then search edge 3 [2143, 4801] innerCandidates321_3 6944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan321_4

def innerTail321_2 : List Nat := 4657 :: innerTail321_3
def innerCandidates321_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates321_2_checked : candidates edge 4 2143 4657 innerTail321_3 best.bound = innerCandidates321_2 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan321_2 : scan edge (search edge 3) 4 [2143] 2143 innerTail321_2 best = best := by
  rw [innerTail321_2, scan, if_neg (by decide), innerCandidates321_2_checked]
  change scan edge (search edge 3) 4 [2143] 2143 innerTail321_3
    (if 3 ≤ innerCandidates321_2.length then search edge 3 [2143, 4657] innerCandidates321_2 6800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan321_3

def innerTail321_1 : List Nat := 2347 :: innerTail321_2
def innerCandidates321_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates321_1_checked : candidates edge 4 2143 2347 innerTail321_2 best.bound = innerCandidates321_1 := by
  unfold candidates
  have row : edge 2347 = PrimePairCache.smallRow2347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan321_1 : scan edge (search edge 3) 4 [2143] 2143 innerTail321_1 best = best := by
  rw [innerTail321_1, scan, if_neg (by decide), innerCandidates321_1_checked]
  change scan edge (search edge 3) 4 [2143] 2143 innerTail321_2
    (if 3 ≤ innerCandidates321_1.length then search edge 3 [2143, 2347] innerCandidates321_1 4490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan321_2

def innerTail321_0 : List Nat := 2239 :: innerTail321_1
def innerCandidates321_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates321_0_checked : candidates edge 4 2143 2239 innerTail321_1 best.bound = innerCandidates321_0 := by
  unfold candidates
  have row : edge 2239 = PrimePairCache.smallRow2239.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan321_0 : scan edge (search edge 3) 4 [2143] 2143 innerTail321_0 best = best := by
  rw [innerTail321_0, scan, if_neg (by decide), innerCandidates321_0_checked]
  change scan edge (search edge 3) 4 [2143] 2143 innerTail321_1
    (if 3 ≤ innerCandidates321_0.length then search edge 3 [2143, 2239] innerCandidates321_0 4382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan321_1

theorem rootBranch321 : search edge 4 [2143] rootCandidates321 2143 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2143] 2143 innerTail321_0 best = best
  exact innerScan321_0

end ProjectEulerSolutions.PrimePairUpper.Search
