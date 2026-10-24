import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates298 : List Nat := [2221, 3271, 3457, 4093, 4483, 5011, 5581, 5743, 5869, 6151, 6211, 6337, 6481, 6781, 7507, 8209, 8287, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates298_checked : candidates edge 5 0 1993 tail299 best.bound = rootCandidates298 := by
  unfold candidates
  have row : edge 1993 = PrimePairCache.smallRow1993.contains := by funext q; rfl
  rw [row]
  decide

def innerTail298_18 : List Nat := []
theorem innerScan298_18 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_18 best = best := rfl

def innerTail298_17 : List Nat := 8377 :: innerTail298_18
def innerCandidates298_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates298_17_checked : candidates edge 4 1993 8377 innerTail298_18 best.bound = innerCandidates298_17 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_17 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_17 best = best := by
  rw [innerTail298_17, scan, if_neg (by decide), innerCandidates298_17_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_18
    (if 3 ≤ innerCandidates298_17.length then search edge 3 [1993, 8377] innerCandidates298_17 10370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_18

def innerTail298_16 : List Nat := 8287 :: innerTail298_17
def innerCandidates298_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates298_16_checked : candidates edge 4 1993 8287 innerTail298_17 best.bound = innerCandidates298_16 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_16 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_16 best = best := by
  rw [innerTail298_16, scan, if_neg (by decide), innerCandidates298_16_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_17
    (if 3 ≤ innerCandidates298_16.length then search edge 3 [1993, 8287] innerCandidates298_16 10280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_17

def innerTail298_15 : List Nat := 8209 :: innerTail298_16
def innerCandidates298_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates298_15_checked : candidates edge 4 1993 8209 innerTail298_16 best.bound = innerCandidates298_15 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_15 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_15 best = best := by
  rw [innerTail298_15, scan, if_neg (by decide), innerCandidates298_15_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_16
    (if 3 ≤ innerCandidates298_15.length then search edge 3 [1993, 8209] innerCandidates298_15 10202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_16

def innerTail298_14 : List Nat := 7507 :: innerTail298_15
def innerCandidates298_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates298_14_checked : candidates edge 4 1993 7507 innerTail298_15 best.bound = innerCandidates298_14 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_14 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_14 best = best := by
  rw [innerTail298_14, scan, if_neg (by decide), innerCandidates298_14_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_15
    (if 3 ≤ innerCandidates298_14.length then search edge 3 [1993, 7507] innerCandidates298_14 9500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_15

def innerTail298_13 : List Nat := 6781 :: innerTail298_14
def innerCandidates298_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates298_13_checked : candidates edge 4 1993 6781 innerTail298_14 best.bound = innerCandidates298_13 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_13 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_13 best = best := by
  rw [innerTail298_13, scan, if_neg (by decide), innerCandidates298_13_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_14
    (if 3 ≤ innerCandidates298_13.length then search edge 3 [1993, 6781] innerCandidates298_13 8774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_14

def innerTail298_12 : List Nat := 6481 :: innerTail298_13
def innerCandidates298_12 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates298_12_checked : candidates edge 4 1993 6481 innerTail298_13 best.bound = innerCandidates298_12 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_12 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_12 best = best := by
  rw [innerTail298_12, scan, if_neg (by decide), innerCandidates298_12_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_13
    (if 3 ≤ innerCandidates298_12.length then search edge 3 [1993, 6481] innerCandidates298_12 8474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_13

def innerTail298_11 : List Nat := 6337 :: innerTail298_12
def innerCandidates298_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates298_11_checked : candidates edge 4 1993 6337 innerTail298_12 best.bound = innerCandidates298_11 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_11 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_11 best = best := by
  rw [innerTail298_11, scan, if_neg (by decide), innerCandidates298_11_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_12
    (if 3 ≤ innerCandidates298_11.length then search edge 3 [1993, 6337] innerCandidates298_11 8330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_12

def innerTail298_10 : List Nat := 6211 :: innerTail298_11
def innerCandidates298_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates298_10_checked : candidates edge 4 1993 6211 innerTail298_11 best.bound = innerCandidates298_10 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_10 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_10 best = best := by
  rw [innerTail298_10, scan, if_neg (by decide), innerCandidates298_10_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_11
    (if 3 ≤ innerCandidates298_10.length then search edge 3 [1993, 6211] innerCandidates298_10 8204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_11

def innerTail298_9 : List Nat := 6151 :: innerTail298_10
def innerCandidates298_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates298_9_checked : candidates edge 4 1993 6151 innerTail298_10 best.bound = innerCandidates298_9 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_9 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_9 best = best := by
  rw [innerTail298_9, scan, if_neg (by decide), innerCandidates298_9_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_10
    (if 3 ≤ innerCandidates298_9.length then search edge 3 [1993, 6151] innerCandidates298_9 8144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_10

def innerTail298_8 : List Nat := 5869 :: innerTail298_9
def innerCandidates298_8 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem innerCandidates298_8_checked : candidates edge 4 1993 5869 innerTail298_9 best.bound = innerCandidates298_8 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_8 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_8 best = best := by
  rw [innerTail298_8, scan, if_neg (by decide), innerCandidates298_8_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_9
    (if 3 ≤ innerCandidates298_8.length then search edge 3 [1993, 5869] innerCandidates298_8 7862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_9

def innerTail298_7 : List Nat := 5743 :: innerTail298_8
def innerCandidates298_7 : List Nat := [7507, 8287, 8377]
set_option maxRecDepth 4096 in
theorem innerCandidates298_7_checked : candidates edge 4 1993 5743 innerTail298_8 best.bound = innerCandidates298_7 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild298_7 : search edge 3 [1993, 5743] innerCandidates298_7 7736 best = best := by decide

theorem innerScan298_7 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_7 best = best := by
  rw [innerTail298_7, scan, if_neg (by decide), innerCandidates298_7_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_8
    (if 3 ≤ innerCandidates298_7.length then search edge 3 [1993, 5743] innerCandidates298_7 7736 best else best) = best
  rw [if_pos (by decide), innerChild298_7]
  exact innerScan298_8

def innerTail298_6 : List Nat := 5581 :: innerTail298_7
def innerCandidates298_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates298_6_checked : candidates edge 4 1993 5581 innerTail298_7 best.bound = innerCandidates298_6 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_6 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_6 best = best := by
  rw [innerTail298_6, scan, if_neg (by decide), innerCandidates298_6_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_7
    (if 3 ≤ innerCandidates298_6.length then search edge 3 [1993, 5581] innerCandidates298_6 7574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_7

def innerTail298_5 : List Nat := 5011 :: innerTail298_6
def innerCandidates298_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates298_5_checked : candidates edge 4 1993 5011 innerTail298_6 best.bound = innerCandidates298_5 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_5 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_5 best = best := by
  rw [innerTail298_5, scan, if_neg (by decide), innerCandidates298_5_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_6
    (if 3 ≤ innerCandidates298_5.length then search edge 3 [1993, 5011] innerCandidates298_5 7004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_6

def innerTail298_4 : List Nat := 4483 :: innerTail298_5
def innerCandidates298_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates298_4_checked : candidates edge 4 1993 4483 innerTail298_5 best.bound = innerCandidates298_4 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_4 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_4 best = best := by
  rw [innerTail298_4, scan, if_neg (by decide), innerCandidates298_4_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_5
    (if 3 ≤ innerCandidates298_4.length then search edge 3 [1993, 4483] innerCandidates298_4 6476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_5

def innerTail298_3 : List Nat := 4093 :: innerTail298_4
def innerCandidates298_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates298_3_checked : candidates edge 4 1993 4093 innerTail298_4 best.bound = innerCandidates298_3 := by
  unfold candidates
  have row : edge 4093 = SearchData.row4093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_3 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_3 best = best := by
  rw [innerTail298_3, scan, if_neg (by decide), innerCandidates298_3_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_4
    (if 3 ≤ innerCandidates298_3.length then search edge 3 [1993, 4093] innerCandidates298_3 6086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_4

def innerTail298_2 : List Nat := 3457 :: innerTail298_3
def innerCandidates298_2 : List Nat := [5869, 8209, 8377]
set_option maxRecDepth 4096 in
theorem innerCandidates298_2_checked : candidates edge 4 1993 3457 innerTail298_3 best.bound = innerCandidates298_2 := by
  unfold candidates
  have row : edge 3457 = SearchData.row3457.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild298_2 : search edge 3 [1993, 3457] innerCandidates298_2 5450 best = best := by decide

theorem innerScan298_2 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_2 best = best := by
  rw [innerTail298_2, scan, if_neg (by decide), innerCandidates298_2_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_3
    (if 3 ≤ innerCandidates298_2.length then search edge 3 [1993, 3457] innerCandidates298_2 5450 best else best) = best
  rw [if_pos (by decide), innerChild298_2]
  exact innerScan298_3

def innerTail298_1 : List Nat := 3271 :: innerTail298_2
def innerCandidates298_1 : List Nat := [8377]
set_option maxRecDepth 4096 in
theorem innerCandidates298_1_checked : candidates edge 4 1993 3271 innerTail298_2 best.bound = innerCandidates298_1 := by
  unfold candidates
  have row : edge 3271 = SearchData.row3271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_1 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_1 best = best := by
  rw [innerTail298_1, scan, if_neg (by decide), innerCandidates298_1_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_2
    (if 3 ≤ innerCandidates298_1.length then search edge 3 [1993, 3271] innerCandidates298_1 5264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_2

def innerTail298_0 : List Nat := 2221 :: innerTail298_1
def innerCandidates298_0 : List Nat := [5581]
set_option maxRecDepth 4096 in
theorem innerCandidates298_0_checked : candidates edge 4 1993 2221 innerTail298_1 best.bound = innerCandidates298_0 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan298_0 : scan edge (search edge 3) 4 [1993] 1993 innerTail298_0 best = best := by
  rw [innerTail298_0, scan, if_neg (by decide), innerCandidates298_0_checked]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_1
    (if 3 ≤ innerCandidates298_0.length then search edge 3 [1993, 2221] innerCandidates298_0 4214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan298_1

theorem rootBranch298 : search edge 4 [1993] rootCandidates298 1993 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1993] 1993 innerTail298_0 best = best
  exact innerScan298_0

def rootCandidates297 : List Nat := [2131, 2383, 2467, 2551, 2707, 2719, 3079, 3331, 3361, 3499, 3517, 3709, 3877, 4423, 4639, 4801, 5521, 5563, 5659, 6079, 6133, 6481, 7027, 7237, 7549, 8191, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates297_checked : candidates edge 5 0 1987 tail298 best.bound = rootCandidates297 := by
  unfold candidates
  have row : edge 1987 = PrimePairCache.smallRow1987.contains := by funext q; rfl
  rw [row]
  decide

def innerTail297_27 : List Nat := []
theorem innerScan297_27 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_27 best = best := rfl

def innerTail297_26 : List Nat := 8263 :: innerTail297_27
def innerCandidates297_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_26_checked : candidates edge 4 1987 8263 innerTail297_27 best.bound = innerCandidates297_26 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_26 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_26 best = best := by
  rw [innerTail297_26, scan, if_neg (by decide), innerCandidates297_26_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_27
    (if 3 ≤ innerCandidates297_26.length then search edge 3 [1987, 8263] innerCandidates297_26 10250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_27

def innerTail297_25 : List Nat := 8191 :: innerTail297_26
def innerCandidates297_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_25_checked : candidates edge 4 1987 8191 innerTail297_26 best.bound = innerCandidates297_25 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_25 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_25 best = best := by
  rw [innerTail297_25, scan, if_neg (by decide), innerCandidates297_25_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_26
    (if 3 ≤ innerCandidates297_25.length then search edge 3 [1987, 8191] innerCandidates297_25 10178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_26

def innerTail297_24 : List Nat := 7549 :: innerTail297_25
def innerCandidates297_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_24_checked : candidates edge 4 1987 7549 innerTail297_25 best.bound = innerCandidates297_24 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_24 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_24 best = best := by
  rw [innerTail297_24, scan, if_neg (by decide), innerCandidates297_24_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_25
    (if 3 ≤ innerCandidates297_24.length then search edge 3 [1987, 7549] innerCandidates297_24 9536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_25

def innerTail297_23 : List Nat := 7237 :: innerTail297_24
def innerCandidates297_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_23_checked : candidates edge 4 1987 7237 innerTail297_24 best.bound = innerCandidates297_23 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_23 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_23 best = best := by
  rw [innerTail297_23, scan, if_neg (by decide), innerCandidates297_23_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_24
    (if 3 ≤ innerCandidates297_23.length then search edge 3 [1987, 7237] innerCandidates297_23 9224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_24

def innerTail297_22 : List Nat := 7027 :: innerTail297_23
def innerCandidates297_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_22_checked : candidates edge 4 1987 7027 innerTail297_23 best.bound = innerCandidates297_22 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_22 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_22 best = best := by
  rw [innerTail297_22, scan, if_neg (by decide), innerCandidates297_22_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_23
    (if 3 ≤ innerCandidates297_22.length then search edge 3 [1987, 7027] innerCandidates297_22 9014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_23

def innerTail297_21 : List Nat := 6481 :: innerTail297_22
def innerCandidates297_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_21_checked : candidates edge 4 1987 6481 innerTail297_22 best.bound = innerCandidates297_21 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_21 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_21 best = best := by
  rw [innerTail297_21, scan, if_neg (by decide), innerCandidates297_21_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_22
    (if 3 ≤ innerCandidates297_21.length then search edge 3 [1987, 6481] innerCandidates297_21 8468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_22

def innerTail297_20 : List Nat := 6133 :: innerTail297_21
def innerCandidates297_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_20_checked : candidates edge 4 1987 6133 innerTail297_21 best.bound = innerCandidates297_20 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_20 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_20 best = best := by
  rw [innerTail297_20, scan, if_neg (by decide), innerCandidates297_20_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_21
    (if 3 ≤ innerCandidates297_20.length then search edge 3 [1987, 6133] innerCandidates297_20 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_21

def innerTail297_19 : List Nat := 6079 :: innerTail297_20
def innerCandidates297_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_19_checked : candidates edge 4 1987 6079 innerTail297_20 best.bound = innerCandidates297_19 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_19 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_19 best = best := by
  rw [innerTail297_19, scan, if_neg (by decide), innerCandidates297_19_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_20
    (if 3 ≤ innerCandidates297_19.length then search edge 3 [1987, 6079] innerCandidates297_19 8066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_20

def innerTail297_18 : List Nat := 5659 :: innerTail297_19
def innerCandidates297_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_18_checked : candidates edge 4 1987 5659 innerTail297_19 best.bound = innerCandidates297_18 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_18 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_18 best = best := by
  rw [innerTail297_18, scan, if_neg (by decide), innerCandidates297_18_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_19
    (if 3 ≤ innerCandidates297_18.length then search edge 3 [1987, 5659] innerCandidates297_18 7646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_19

def innerTail297_17 : List Nat := 5563 :: innerTail297_18
def innerCandidates297_17 : List Nat := [6079]
set_option maxRecDepth 4096 in
theorem innerCandidates297_17_checked : candidates edge 4 1987 5563 innerTail297_18 best.bound = innerCandidates297_17 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_17 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_17 best = best := by
  rw [innerTail297_17, scan, if_neg (by decide), innerCandidates297_17_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_18
    (if 3 ≤ innerCandidates297_17.length then search edge 3 [1987, 5563] innerCandidates297_17 7550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_18

def innerTail297_16 : List Nat := 5521 :: innerTail297_17
def innerCandidates297_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_16_checked : candidates edge 4 1987 5521 innerTail297_17 best.bound = innerCandidates297_16 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_16 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_16 best = best := by
  rw [innerTail297_16, scan, if_neg (by decide), innerCandidates297_16_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_17
    (if 3 ≤ innerCandidates297_16.length then search edge 3 [1987, 5521] innerCandidates297_16 7508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_17

def innerTail297_15 : List Nat := 4801 :: innerTail297_16
def innerCandidates297_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_15_checked : candidates edge 4 1987 4801 innerTail297_16 best.bound = innerCandidates297_15 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_15 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_15 best = best := by
  rw [innerTail297_15, scan, if_neg (by decide), innerCandidates297_15_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_16
    (if 3 ≤ innerCandidates297_15.length then search edge 3 [1987, 4801] innerCandidates297_15 6788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_16

def innerTail297_14 : List Nat := 4639 :: innerTail297_15
def innerCandidates297_14 : List Nat := [7237]
set_option maxRecDepth 4096 in
theorem innerCandidates297_14_checked : candidates edge 4 1987 4639 innerTail297_15 best.bound = innerCandidates297_14 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_14 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_14 best = best := by
  rw [innerTail297_14, scan, if_neg (by decide), innerCandidates297_14_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_15
    (if 3 ≤ innerCandidates297_14.length then search edge 3 [1987, 4639] innerCandidates297_14 6626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_15

def innerTail297_13 : List Nat := 4423 :: innerTail297_14
def innerCandidates297_13 : List Nat := [6133]
set_option maxRecDepth 4096 in
theorem innerCandidates297_13_checked : candidates edge 4 1987 4423 innerTail297_14 best.bound = innerCandidates297_13 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_13 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_13 best = best := by
  rw [innerTail297_13, scan, if_neg (by decide), innerCandidates297_13_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_14
    (if 3 ≤ innerCandidates297_13.length then search edge 3 [1987, 4423] innerCandidates297_13 6410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_14

def innerTail297_12 : List Nat := 3877 :: innerTail297_13
def innerCandidates297_12 : List Nat := [8263]
set_option maxRecDepth 4096 in
theorem innerCandidates297_12_checked : candidates edge 4 1987 3877 innerTail297_13 best.bound = innerCandidates297_12 := by
  unfold candidates
  have row : edge 3877 = SearchData.row3877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_12 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_12 best = best := by
  rw [innerTail297_12, scan, if_neg (by decide), innerCandidates297_12_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_13
    (if 3 ≤ innerCandidates297_12.length then search edge 3 [1987, 3877] innerCandidates297_12 5864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_13

def innerTail297_11 : List Nat := 3709 :: innerTail297_12
def innerCandidates297_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_11_checked : candidates edge 4 1987 3709 innerTail297_12 best.bound = innerCandidates297_11 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_11 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_11 best = best := by
  rw [innerTail297_11, scan, if_neg (by decide), innerCandidates297_11_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_12
    (if 3 ≤ innerCandidates297_11.length then search edge 3 [1987, 3709] innerCandidates297_11 5696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_12

def innerTail297_10 : List Nat := 3517 :: innerTail297_11
def innerCandidates297_10 : List Nat := [5659, 8263]
set_option maxRecDepth 4096 in
theorem innerCandidates297_10_checked : candidates edge 4 1987 3517 innerTail297_11 best.bound = innerCandidates297_10 := by
  unfold candidates
  have row : edge 3517 = SearchData.row3517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_10 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_10 best = best := by
  rw [innerTail297_10, scan, if_neg (by decide), innerCandidates297_10_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_11
    (if 3 ≤ innerCandidates297_10.length then search edge 3 [1987, 3517] innerCandidates297_10 5504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_11

def innerTail297_9 : List Nat := 3499 :: innerTail297_10
def innerCandidates297_9 : List Nat := [8191]
set_option maxRecDepth 4096 in
theorem innerCandidates297_9_checked : candidates edge 4 1987 3499 innerTail297_10 best.bound = innerCandidates297_9 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_9 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_9 best = best := by
  rw [innerTail297_9, scan, if_neg (by decide), innerCandidates297_9_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_10
    (if 3 ≤ innerCandidates297_9.length then search edge 3 [1987, 3499] innerCandidates297_9 5486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_10

def innerTail297_8 : List Nat := 3361 :: innerTail297_9
def innerCandidates297_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_8_checked : candidates edge 4 1987 3361 innerTail297_9 best.bound = innerCandidates297_8 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_8 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_8 best = best := by
  rw [innerTail297_8, scan, if_neg (by decide), innerCandidates297_8_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_9
    (if 3 ≤ innerCandidates297_8.length then search edge 3 [1987, 3361] innerCandidates297_8 5348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_9

def innerTail297_7 : List Nat := 3331 :: innerTail297_8
def innerCandidates297_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_7_checked : candidates edge 4 1987 3331 innerTail297_8 best.bound = innerCandidates297_7 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_7 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_7 best = best := by
  rw [innerTail297_7, scan, if_neg (by decide), innerCandidates297_7_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_8
    (if 3 ≤ innerCandidates297_7.length then search edge 3 [1987, 3331] innerCandidates297_7 5318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_8

def innerTail297_6 : List Nat := 3079 :: innerTail297_7
def innerCandidates297_6 : List Nat := [3709, 3877]
set_option maxRecDepth 4096 in
theorem innerCandidates297_6_checked : candidates edge 4 1987 3079 innerTail297_7 best.bound = innerCandidates297_6 := by
  unfold candidates
  have row : edge 3079 = SearchData.row3079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_6 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_6 best = best := by
  rw [innerTail297_6, scan, if_neg (by decide), innerCandidates297_6_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_7
    (if 3 ≤ innerCandidates297_6.length then search edge 3 [1987, 3079] innerCandidates297_6 5066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_7

def innerTail297_5 : List Nat := 2719 :: innerTail297_6
def innerCandidates297_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates297_5_checked : candidates edge 4 1987 2719 innerTail297_6 best.bound = innerCandidates297_5 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_5 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_5 best = best := by
  rw [innerTail297_5, scan, if_neg (by decide), innerCandidates297_5_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_6
    (if 3 ≤ innerCandidates297_5.length then search edge 3 [1987, 2719] innerCandidates297_5 4706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_6

def innerTail297_4 : List Nat := 2707 :: innerTail297_5
def innerCandidates297_4 : List Nat := [3517, 6079]
set_option maxRecDepth 4096 in
theorem innerCandidates297_4_checked : candidates edge 4 1987 2707 innerTail297_5 best.bound = innerCandidates297_4 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_4 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_4 best = best := by
  rw [innerTail297_4, scan, if_neg (by decide), innerCandidates297_4_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_5
    (if 3 ≤ innerCandidates297_4.length then search edge 3 [1987, 2707] innerCandidates297_4 4694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_5

def innerTail297_3 : List Nat := 2551 :: innerTail297_4
def innerCandidates297_3 : List Nat := [3331]
set_option maxRecDepth 4096 in
theorem innerCandidates297_3_checked : candidates edge 4 1987 2551 innerTail297_4 best.bound = innerCandidates297_3 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_3 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_3 best = best := by
  rw [innerTail297_3, scan, if_neg (by decide), innerCandidates297_3_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_4
    (if 3 ≤ innerCandidates297_3.length then search edge 3 [1987, 2551] innerCandidates297_3 4538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_4

def innerTail297_2 : List Nat := 2467 :: innerTail297_3
def innerCandidates297_2 : List Nat := [2551]
set_option maxRecDepth 4096 in
theorem innerCandidates297_2_checked : candidates edge 4 1987 2467 innerTail297_3 best.bound = innerCandidates297_2 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_2 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_2 best = best := by
  rw [innerTail297_2, scan, if_neg (by decide), innerCandidates297_2_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_3
    (if 3 ≤ innerCandidates297_2.length then search edge 3 [1987, 2467] innerCandidates297_2 4454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_3

def innerTail297_1 : List Nat := 2383 :: innerTail297_2
def innerCandidates297_1 : List Nat := [4639, 8263]
set_option maxRecDepth 4096 in
theorem innerCandidates297_1_checked : candidates edge 4 1987 2383 innerTail297_2 best.bound = innerCandidates297_1 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_1 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_1 best = best := by
  rw [innerTail297_1, scan, if_neg (by decide), innerCandidates297_1_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_2
    (if 3 ≤ innerCandidates297_1.length then search edge 3 [1987, 2383] innerCandidates297_1 4370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_2

def innerTail297_0 : List Nat := 2131 :: innerTail297_1
def innerCandidates297_0 : List Nat := [5563, 7027]
set_option maxRecDepth 4096 in
theorem innerCandidates297_0_checked : candidates edge 4 1987 2131 innerTail297_1 best.bound = innerCandidates297_0 := by
  unfold candidates
  have row : edge 2131 = PrimePairCache.smallRow2131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan297_0 : scan edge (search edge 3) 4 [1987] 1987 innerTail297_0 best = best := by
  rw [innerTail297_0, scan, if_neg (by decide), innerCandidates297_0_checked]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_1
    (if 3 ≤ innerCandidates297_0.length then search edge 3 [1987, 2131] innerCandidates297_0 4118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan297_1

theorem rootBranch297 : search edge 4 [1987] rootCandidates297 1987 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1987] 1987 innerTail297_0 best = best
  exact innerScan297_0

def rootCandidates296 : List Nat := [2039, 2207, 2693, 2999, 3911, 4241, 4937, 6869, 7187, 7433, 8291, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates296_checked : candidates edge 5 0 1979 tail297 best.bound = rootCandidates296 := by
  unfold candidates
  have row : edge 1979 = PrimePairCache.smallRow1979.contains := by funext q; rfl
  rw [row]
  decide

def innerTail296_12 : List Nat := []
theorem innerScan296_12 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_12 best = best := rfl

def innerTail296_11 : List Nat := 8297 :: innerTail296_12
def innerCandidates296_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates296_11_checked : candidates edge 4 1979 8297 innerTail296_12 best.bound = innerCandidates296_11 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan296_11 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_11 best = best := by
  rw [innerTail296_11, scan, if_neg (by decide), innerCandidates296_11_checked]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_12
    (if 3 ≤ innerCandidates296_11.length then search edge 3 [1979, 8297] innerCandidates296_11 10276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan296_12

def innerTail296_10 : List Nat := 8291 :: innerTail296_11
def innerCandidates296_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates296_10_checked : candidates edge 4 1979 8291 innerTail296_11 best.bound = innerCandidates296_10 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan296_10 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_10 best = best := by
  rw [innerTail296_10, scan, if_neg (by decide), innerCandidates296_10_checked]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_11
    (if 3 ≤ innerCandidates296_10.length then search edge 3 [1979, 8291] innerCandidates296_10 10270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan296_11

def innerTail296_9 : List Nat := 7433 :: innerTail296_10
def innerCandidates296_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates296_9_checked : candidates edge 4 1979 7433 innerTail296_10 best.bound = innerCandidates296_9 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan296_9 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_9 best = best := by
  rw [innerTail296_9, scan, if_neg (by decide), innerCandidates296_9_checked]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_10
    (if 3 ≤ innerCandidates296_9.length then search edge 3 [1979, 7433] innerCandidates296_9 9412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan296_10

def innerTail296_8 : List Nat := 7187 :: innerTail296_9
def innerCandidates296_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates296_8_checked : candidates edge 4 1979 7187 innerTail296_9 best.bound = innerCandidates296_8 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan296_8 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_8 best = best := by
  rw [innerTail296_8, scan, if_neg (by decide), innerCandidates296_8_checked]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_9
    (if 3 ≤ innerCandidates296_8.length then search edge 3 [1979, 7187] innerCandidates296_8 9166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan296_9

def innerTail296_7 : List Nat := 6869 :: innerTail296_8
def innerCandidates296_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates296_7_checked : candidates edge 4 1979 6869 innerTail296_8 best.bound = innerCandidates296_7 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan296_7 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_7 best = best := by
  rw [innerTail296_7, scan, if_neg (by decide), innerCandidates296_7_checked]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_8
    (if 3 ≤ innerCandidates296_7.length then search edge 3 [1979, 6869] innerCandidates296_7 8848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan296_8

def innerTail296_6 : List Nat := 4937 :: innerTail296_7
def innerCandidates296_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates296_6_checked : candidates edge 4 1979 4937 innerTail296_7 best.bound = innerCandidates296_6 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan296_6 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_6 best = best := by
  rw [innerTail296_6, scan, if_neg (by decide), innerCandidates296_6_checked]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_7
    (if 3 ≤ innerCandidates296_6.length then search edge 3 [1979, 4937] innerCandidates296_6 6916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan296_7

def innerTail296_5 : List Nat := 4241 :: innerTail296_6
def innerCandidates296_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates296_5_checked : candidates edge 4 1979 4241 innerTail296_6 best.bound = innerCandidates296_5 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan296_5 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_5 best = best := by
  rw [innerTail296_5, scan, if_neg (by decide), innerCandidates296_5_checked]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_6
    (if 3 ≤ innerCandidates296_5.length then search edge 3 [1979, 4241] innerCandidates296_5 6220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan296_6

def innerTail296_4 : List Nat := 3911 :: innerTail296_5
def innerCandidates296_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates296_4_checked : candidates edge 4 1979 3911 innerTail296_5 best.bound = innerCandidates296_4 := by
  unfold candidates
  have row : edge 3911 = SearchData.row3911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan296_4 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_4 best = best := by
  rw [innerTail296_4, scan, if_neg (by decide), innerCandidates296_4_checked]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_5
    (if 3 ≤ innerCandidates296_4.length then search edge 3 [1979, 3911] innerCandidates296_4 5890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan296_5

def innerTail296_3 : List Nat := 2999 :: innerTail296_4
def innerCandidates296_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates296_3_checked : candidates edge 4 1979 2999 innerTail296_4 best.bound = innerCandidates296_3 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan296_3 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_3 best = best := by
  rw [innerTail296_3, scan, if_neg (by decide), innerCandidates296_3_checked]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_4
    (if 3 ≤ innerCandidates296_3.length then search edge 3 [1979, 2999] innerCandidates296_3 4978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan296_4

def innerTail296_2 : List Nat := 2693 :: innerTail296_3
def innerCandidates296_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates296_2_checked : candidates edge 4 1979 2693 innerTail296_3 best.bound = innerCandidates296_2 := by
  unfold candidates
  have row : edge 2693 = SearchData.row2693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan296_2 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_2 best = best := by
  rw [innerTail296_2, scan, if_neg (by decide), innerCandidates296_2_checked]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_3
    (if 3 ≤ innerCandidates296_2.length then search edge 3 [1979, 2693] innerCandidates296_2 4672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan296_3

def innerTail296_1 : List Nat := 2207 :: innerTail296_2
def innerCandidates296_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates296_1_checked : candidates edge 4 1979 2207 innerTail296_2 best.bound = innerCandidates296_1 := by
  unfold candidates
  have row : edge 2207 = PrimePairCache.smallRow2207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan296_1 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_1 best = best := by
  rw [innerTail296_1, scan, if_neg (by decide), innerCandidates296_1_checked]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_2
    (if 3 ≤ innerCandidates296_1.length then search edge 3 [1979, 2207] innerCandidates296_1 4186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan296_2

def innerTail296_0 : List Nat := 2039 :: innerTail296_1
def innerCandidates296_0 : List Nat := [4937]
set_option maxRecDepth 4096 in
theorem innerCandidates296_0_checked : candidates edge 4 1979 2039 innerTail296_1 best.bound = innerCandidates296_0 := by
  unfold candidates
  have row : edge 2039 = PrimePairCache.smallRow2039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan296_0 : scan edge (search edge 3) 4 [1979] 1979 innerTail296_0 best = best := by
  rw [innerTail296_0, scan, if_neg (by decide), innerCandidates296_0_checked]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_1
    (if 3 ≤ innerCandidates296_0.length then search edge 3 [1979, 2039] innerCandidates296_0 4018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan296_1

theorem rootBranch296 : search edge 4 [1979] rootCandidates296 1979 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1979] 1979 innerTail296_0 best = best
  exact innerScan296_0

end ProjectEulerSolutions.PrimePairUpper.Search
