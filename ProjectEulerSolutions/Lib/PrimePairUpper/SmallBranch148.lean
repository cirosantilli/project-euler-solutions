import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates232 : List Nat := [1867, 2269, 2281, 2341, 2593, 2791, 3457, 4357, 5683, 5869, 6163, 6961, 7027, 7159, 7393, 7591, 7789, 7873, 8191, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates232_checked : candidates edge 5 0 1483 tail233 best.bound = rootCandidates232 := by
  unfold candidates
  have row : edge 1483 = PrimePairCache.smallRow1483.contains := by funext q; rfl
  rw [row]
  decide

def innerTail232_20 : List Nat := []
theorem innerScan232_20 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_20 best = best := rfl

def innerTail232_19 : List Nat := 8209 :: innerTail232_20
def innerCandidates232_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates232_19_checked : candidates edge 4 1483 8209 innerTail232_20 best.bound = innerCandidates232_19 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_19 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_19 best = best := by
  rw [innerTail232_19, scan, if_neg (by decide), innerCandidates232_19_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_20
    (if 3 ≤ innerCandidates232_19.length then search edge 3 [1483, 8209] innerCandidates232_19 9692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_20

def innerTail232_18 : List Nat := 8191 :: innerTail232_19
def innerCandidates232_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates232_18_checked : candidates edge 4 1483 8191 innerTail232_19 best.bound = innerCandidates232_18 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_18 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_18 best = best := by
  rw [innerTail232_18, scan, if_neg (by decide), innerCandidates232_18_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_19
    (if 3 ≤ innerCandidates232_18.length then search edge 3 [1483, 8191] innerCandidates232_18 9674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_19

def innerTail232_17 : List Nat := 7873 :: innerTail232_18
def innerCandidates232_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates232_17_checked : candidates edge 4 1483 7873 innerTail232_18 best.bound = innerCandidates232_17 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_17 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_17 best = best := by
  rw [innerTail232_17, scan, if_neg (by decide), innerCandidates232_17_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_18
    (if 3 ≤ innerCandidates232_17.length then search edge 3 [1483, 7873] innerCandidates232_17 9356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_18

def innerTail232_16 : List Nat := 7789 :: innerTail232_17
def innerCandidates232_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates232_16_checked : candidates edge 4 1483 7789 innerTail232_17 best.bound = innerCandidates232_16 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_16 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_16 best = best := by
  rw [innerTail232_16, scan, if_neg (by decide), innerCandidates232_16_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_17
    (if 3 ≤ innerCandidates232_16.length then search edge 3 [1483, 7789] innerCandidates232_16 9272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_17

def innerTail232_15 : List Nat := 7591 :: innerTail232_16
def innerCandidates232_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates232_15_checked : candidates edge 4 1483 7591 innerTail232_16 best.bound = innerCandidates232_15 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_15 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_15 best = best := by
  rw [innerTail232_15, scan, if_neg (by decide), innerCandidates232_15_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_16
    (if 3 ≤ innerCandidates232_15.length then search edge 3 [1483, 7591] innerCandidates232_15 9074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_16

def innerTail232_14 : List Nat := 7393 :: innerTail232_15
def innerCandidates232_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates232_14_checked : candidates edge 4 1483 7393 innerTail232_15 best.bound = innerCandidates232_14 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_14 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_14 best = best := by
  rw [innerTail232_14, scan, if_neg (by decide), innerCandidates232_14_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_15
    (if 3 ≤ innerCandidates232_14.length then search edge 3 [1483, 7393] innerCandidates232_14 8876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_15

def innerTail232_13 : List Nat := 7159 :: innerTail232_14
def innerCandidates232_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates232_13_checked : candidates edge 4 1483 7159 innerTail232_14 best.bound = innerCandidates232_13 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_13 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_13 best = best := by
  rw [innerTail232_13, scan, if_neg (by decide), innerCandidates232_13_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_14
    (if 3 ≤ innerCandidates232_13.length then search edge 3 [1483, 7159] innerCandidates232_13 8642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_14

def innerTail232_12 : List Nat := 7027 :: innerTail232_13
def innerCandidates232_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates232_12_checked : candidates edge 4 1483 7027 innerTail232_13 best.bound = innerCandidates232_12 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_12 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_12 best = best := by
  rw [innerTail232_12, scan, if_neg (by decide), innerCandidates232_12_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_13
    (if 3 ≤ innerCandidates232_12.length then search edge 3 [1483, 7027] innerCandidates232_12 8510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_13

def innerTail232_11 : List Nat := 6961 :: innerTail232_12
def innerCandidates232_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates232_11_checked : candidates edge 4 1483 6961 innerTail232_12 best.bound = innerCandidates232_11 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_11 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_11 best = best := by
  rw [innerTail232_11, scan, if_neg (by decide), innerCandidates232_11_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_12
    (if 3 ≤ innerCandidates232_11.length then search edge 3 [1483, 6961] innerCandidates232_11 8444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_12

def innerTail232_10 : List Nat := 6163 :: innerTail232_11
def innerCandidates232_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates232_10_checked : candidates edge 4 1483 6163 innerTail232_11 best.bound = innerCandidates232_10 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_10 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_10 best = best := by
  rw [innerTail232_10, scan, if_neg (by decide), innerCandidates232_10_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_11
    (if 3 ≤ innerCandidates232_10.length then search edge 3 [1483, 6163] innerCandidates232_10 7646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_11

def innerTail232_9 : List Nat := 5869 :: innerTail232_10
def innerCandidates232_9 : List Nat := [6163, 6961, 8209]
set_option maxRecDepth 4096 in
theorem innerCandidates232_9_checked : candidates edge 4 1483 5869 innerTail232_10 best.bound = innerCandidates232_9 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild232_9 : search edge 3 [1483, 5869] innerCandidates232_9 7352 best = best := by decide

theorem innerScan232_9 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_9 best = best := by
  rw [innerTail232_9, scan, if_neg (by decide), innerCandidates232_9_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_10
    (if 3 ≤ innerCandidates232_9.length then search edge 3 [1483, 5869] innerCandidates232_9 7352 best else best) = best
  rw [if_pos (by decide), innerChild232_9]
  exact innerScan232_10

def innerTail232_8 : List Nat := 5683 :: innerTail232_9
def innerCandidates232_8 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates232_8_checked : candidates edge 4 1483 5683 innerTail232_9 best.bound = innerCandidates232_8 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_8 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_8 best = best := by
  rw [innerTail232_8, scan, if_neg (by decide), innerCandidates232_8_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_9
    (if 3 ≤ innerCandidates232_8.length then search edge 3 [1483, 5683] innerCandidates232_8 7166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_9

def innerTail232_7 : List Nat := 4357 :: innerTail232_8
def innerCandidates232_7 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem innerCandidates232_7_checked : candidates edge 4 1483 4357 innerTail232_8 best.bound = innerCandidates232_7 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_7 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_7 best = best := by
  rw [innerTail232_7, scan, if_neg (by decide), innerCandidates232_7_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_8
    (if 3 ≤ innerCandidates232_7.length then search edge 3 [1483, 4357] innerCandidates232_7 5840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_8

def innerTail232_6 : List Nat := 3457 :: innerTail232_7
def innerCandidates232_6 : List Nat := [5869, 8209]
set_option maxRecDepth 4096 in
theorem innerCandidates232_6_checked : candidates edge 4 1483 3457 innerTail232_7 best.bound = innerCandidates232_6 := by
  unfold candidates
  have row : edge 3457 = SearchData.row3457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_6 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_6 best = best := by
  rw [innerTail232_6, scan, if_neg (by decide), innerCandidates232_6_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_7
    (if 3 ≤ innerCandidates232_6.length then search edge 3 [1483, 3457] innerCandidates232_6 4940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_7

def innerTail232_5 : List Nat := 2791 :: innerTail232_6
def innerCandidates232_5 : List Nat := [6961]
set_option maxRecDepth 4096 in
theorem innerCandidates232_5_checked : candidates edge 4 1483 2791 innerTail232_6 best.bound = innerCandidates232_5 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_5 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_5 best = best := by
  rw [innerTail232_5, scan, if_neg (by decide), innerCandidates232_5_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_6
    (if 3 ≤ innerCandidates232_5.length then search edge 3 [1483, 2791] innerCandidates232_5 4274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_6

def innerTail232_4 : List Nat := 2593 :: innerTail232_5
def innerCandidates232_4 : List Nat := [5683, 7027]
set_option maxRecDepth 4096 in
theorem innerCandidates232_4_checked : candidates edge 4 1483 2593 innerTail232_5 best.bound = innerCandidates232_4 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_4 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_4 best = best := by
  rw [innerTail232_4, scan, if_neg (by decide), innerCandidates232_4_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_5
    (if 3 ≤ innerCandidates232_4.length then search edge 3 [1483, 2593] innerCandidates232_4 4076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_5

def innerTail232_3 : List Nat := 2341 :: innerTail232_4
def innerCandidates232_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates232_3_checked : candidates edge 4 1483 2341 innerTail232_4 best.bound = innerCandidates232_3 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_3 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_3 best = best := by
  rw [innerTail232_3, scan, if_neg (by decide), innerCandidates232_3_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_4
    (if 3 ≤ innerCandidates232_3.length then search edge 3 [1483, 2341] innerCandidates232_3 3824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_4

def innerTail232_2 : List Nat := 2281 :: innerTail232_3
def innerCandidates232_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates232_2_checked : candidates edge 4 1483 2281 innerTail232_3 best.bound = innerCandidates232_2 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_2 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_2 best = best := by
  rw [innerTail232_2, scan, if_neg (by decide), innerCandidates232_2_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_3
    (if 3 ≤ innerCandidates232_2.length then search edge 3 [1483, 2281] innerCandidates232_2 3764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_3

def innerTail232_1 : List Nat := 2269 :: innerTail232_2
def innerCandidates232_1 : List Nat := [2281, 7027]
set_option maxRecDepth 4096 in
theorem innerCandidates232_1_checked : candidates edge 4 1483 2269 innerTail232_2 best.bound = innerCandidates232_1 := by
  unfold candidates
  have row : edge 2269 = PrimePairCache.smallRow2269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_1 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_1 best = best := by
  rw [innerTail232_1, scan, if_neg (by decide), innerCandidates232_1_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_2
    (if 3 ≤ innerCandidates232_1.length then search edge 3 [1483, 2269] innerCandidates232_1 3752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_2

def innerTail232_0 : List Nat := 1867 :: innerTail232_1
def innerCandidates232_0 : List Nat := [7873]
set_option maxRecDepth 4096 in
theorem innerCandidates232_0_checked : candidates edge 4 1483 1867 innerTail232_1 best.bound = innerCandidates232_0 := by
  unfold candidates
  have row : edge 1867 = PrimePairCache.smallRow1867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan232_0 : scan edge (search edge 3) 4 [1483] 1483 innerTail232_0 best = best := by
  rw [innerTail232_0, scan, if_neg (by decide), innerCandidates232_0_checked]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_1
    (if 3 ≤ innerCandidates232_0.length then search edge 3 [1483, 1867] innerCandidates232_0 3350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan232_1

theorem rootBranch232 : search edge 4 [1483] rootCandidates232 1483 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1483] 1483 innerTail232_0 best = best
  exact innerScan232_0

def rootCandidates231 : List Nat := [1733, 2663, 2879, 3089, 3167, 3803, 3881, 4127, 4211, 5189, 5639, 5981, 6569, 6869, 7529, 7703, 7823]
set_option maxRecDepth 4096 in
theorem rootCandidates231_checked : candidates edge 5 0 1481 tail232 best.bound = rootCandidates231 := by
  unfold candidates
  have row : edge 1481 = PrimePairCache.smallRow1481.contains := by funext q; rfl
  rw [row]
  decide

def innerTail231_17 : List Nat := []
theorem innerScan231_17 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_17 best = best := rfl

def innerTail231_16 : List Nat := 7823 :: innerTail231_17
def innerCandidates231_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_16_checked : candidates edge 4 1481 7823 innerTail231_17 best.bound = innerCandidates231_16 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_16 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_16 best = best := by
  rw [innerTail231_16, scan, if_neg (by decide), innerCandidates231_16_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_17
    (if 3 ≤ innerCandidates231_16.length then search edge 3 [1481, 7823] innerCandidates231_16 9304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_17

def innerTail231_15 : List Nat := 7703 :: innerTail231_16
def innerCandidates231_15 : List Nat := [7823]
set_option maxRecDepth 4096 in
theorem innerCandidates231_15_checked : candidates edge 4 1481 7703 innerTail231_16 best.bound = innerCandidates231_15 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_15 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_15 best = best := by
  rw [innerTail231_15, scan, if_neg (by decide), innerCandidates231_15_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_16
    (if 3 ≤ innerCandidates231_15.length then search edge 3 [1481, 7703] innerCandidates231_15 9184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_16

def innerTail231_14 : List Nat := 7529 :: innerTail231_15
def innerCandidates231_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_14_checked : candidates edge 4 1481 7529 innerTail231_15 best.bound = innerCandidates231_14 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_14 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_14 best = best := by
  rw [innerTail231_14, scan, if_neg (by decide), innerCandidates231_14_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_15
    (if 3 ≤ innerCandidates231_14.length then search edge 3 [1481, 7529] innerCandidates231_14 9010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_15

def innerTail231_13 : List Nat := 6869 :: innerTail231_14
def innerCandidates231_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_13_checked : candidates edge 4 1481 6869 innerTail231_14 best.bound = innerCandidates231_13 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_13 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_13 best = best := by
  rw [innerTail231_13, scan, if_neg (by decide), innerCandidates231_13_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_14
    (if 3 ≤ innerCandidates231_13.length then search edge 3 [1481, 6869] innerCandidates231_13 8350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_14

def innerTail231_12 : List Nat := 6569 :: innerTail231_13
def innerCandidates231_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_12_checked : candidates edge 4 1481 6569 innerTail231_13 best.bound = innerCandidates231_12 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_12 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_12 best = best := by
  rw [innerTail231_12, scan, if_neg (by decide), innerCandidates231_12_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_13
    (if 3 ≤ innerCandidates231_12.length then search edge 3 [1481, 6569] innerCandidates231_12 8050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_13

def innerTail231_11 : List Nat := 5981 :: innerTail231_12
def innerCandidates231_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_11_checked : candidates edge 4 1481 5981 innerTail231_12 best.bound = innerCandidates231_11 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_11 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_11 best = best := by
  rw [innerTail231_11, scan, if_neg (by decide), innerCandidates231_11_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_12
    (if 3 ≤ innerCandidates231_11.length then search edge 3 [1481, 5981] innerCandidates231_11 7462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_12

def innerTail231_10 : List Nat := 5639 :: innerTail231_11
def innerCandidates231_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_10_checked : candidates edge 4 1481 5639 innerTail231_11 best.bound = innerCandidates231_10 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_10 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_10 best = best := by
  rw [innerTail231_10, scan, if_neg (by decide), innerCandidates231_10_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_11
    (if 3 ≤ innerCandidates231_10.length then search edge 3 [1481, 5639] innerCandidates231_10 7120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_11

def innerTail231_9 : List Nat := 5189 :: innerTail231_10
def innerCandidates231_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_9_checked : candidates edge 4 1481 5189 innerTail231_10 best.bound = innerCandidates231_9 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_9 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_9 best = best := by
  rw [innerTail231_9, scan, if_neg (by decide), innerCandidates231_9_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_10
    (if 3 ≤ innerCandidates231_9.length then search edge 3 [1481, 5189] innerCandidates231_9 6670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_10

def innerTail231_8 : List Nat := 4211 :: innerTail231_9
def innerCandidates231_8 : List Nat := [7703]
set_option maxRecDepth 4096 in
theorem innerCandidates231_8_checked : candidates edge 4 1481 4211 innerTail231_9 best.bound = innerCandidates231_8 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_8 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_8 best = best := by
  rw [innerTail231_8, scan, if_neg (by decide), innerCandidates231_8_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_9
    (if 3 ≤ innerCandidates231_8.length then search edge 3 [1481, 4211] innerCandidates231_8 5692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_9

def innerTail231_7 : List Nat := 4127 :: innerTail231_8
def innerCandidates231_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_7_checked : candidates edge 4 1481 4127 innerTail231_8 best.bound = innerCandidates231_7 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_7 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_7 best = best := by
  rw [innerTail231_7, scan, if_neg (by decide), innerCandidates231_7_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_8
    (if 3 ≤ innerCandidates231_7.length then search edge 3 [1481, 4127] innerCandidates231_7 5608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_8

def innerTail231_6 : List Nat := 3881 :: innerTail231_7
def innerCandidates231_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_6_checked : candidates edge 4 1481 3881 innerTail231_7 best.bound = innerCandidates231_6 := by
  unfold candidates
  have row : edge 3881 = SearchData.row3881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_6 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_6 best = best := by
  rw [innerTail231_6, scan, if_neg (by decide), innerCandidates231_6_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_7
    (if 3 ≤ innerCandidates231_6.length then search edge 3 [1481, 3881] innerCandidates231_6 5362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_7

def innerTail231_5 : List Nat := 3803 :: innerTail231_6
def innerCandidates231_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_5_checked : candidates edge 4 1481 3803 innerTail231_6 best.bound = innerCandidates231_5 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_5 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_5 best = best := by
  rw [innerTail231_5, scan, if_neg (by decide), innerCandidates231_5_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_6
    (if 3 ≤ innerCandidates231_5.length then search edge 3 [1481, 3803] innerCandidates231_5 5284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_6

def innerTail231_4 : List Nat := 3167 :: innerTail231_5
def innerCandidates231_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_4_checked : candidates edge 4 1481 3167 innerTail231_5 best.bound = innerCandidates231_4 := by
  unfold candidates
  have row : edge 3167 = SearchData.row3167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_4 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_4 best = best := by
  rw [innerTail231_4, scan, if_neg (by decide), innerCandidates231_4_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_5
    (if 3 ≤ innerCandidates231_4.length then search edge 3 [1481, 3167] innerCandidates231_4 4648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_5

def innerTail231_3 : List Nat := 3089 :: innerTail231_4
def innerCandidates231_3 : List Nat := [5639]
set_option maxRecDepth 4096 in
theorem innerCandidates231_3_checked : candidates edge 4 1481 3089 innerTail231_4 best.bound = innerCandidates231_3 := by
  unfold candidates
  have row : edge 3089 = SearchData.row3089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_3 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_3 best = best := by
  rw [innerTail231_3, scan, if_neg (by decide), innerCandidates231_3_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_4
    (if 3 ≤ innerCandidates231_3.length then search edge 3 [1481, 3089] innerCandidates231_3 4570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_4

def innerTail231_2 : List Nat := 2879 :: innerTail231_3
def innerCandidates231_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_2_checked : candidates edge 4 1481 2879 innerTail231_3 best.bound = innerCandidates231_2 := by
  unfold candidates
  have row : edge 2879 = SearchData.row2879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_2 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_2 best = best := by
  rw [innerTail231_2, scan, if_neg (by decide), innerCandidates231_2_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_3
    (if 3 ≤ innerCandidates231_2.length then search edge 3 [1481, 2879] innerCandidates231_2 4360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_3

def innerTail231_1 : List Nat := 2663 :: innerTail231_2
def innerCandidates231_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates231_1_checked : candidates edge 4 1481 2663 innerTail231_2 best.bound = innerCandidates231_1 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_1 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_1 best = best := by
  rw [innerTail231_1, scan, if_neg (by decide), innerCandidates231_1_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_2
    (if 3 ≤ innerCandidates231_1.length then search edge 3 [1481, 2663] innerCandidates231_1 4144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_2

def innerTail231_0 : List Nat := 1733 :: innerTail231_1
def innerCandidates231_0 : List Nat := [2663]
set_option maxRecDepth 4096 in
theorem innerCandidates231_0_checked : candidates edge 4 1481 1733 innerTail231_1 best.bound = innerCandidates231_0 := by
  unfold candidates
  have row : edge 1733 = PrimePairCache.smallRow1733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan231_0 : scan edge (search edge 3) 4 [1481] 1481 innerTail231_0 best = best := by
  rw [innerTail231_0, scan, if_neg (by decide), innerCandidates231_0_checked]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_1
    (if 3 ≤ innerCandidates231_0.length then search edge 3 [1481, 1733] innerCandidates231_0 3214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan231_1

theorem rootBranch231 : search edge 4 [1481] rootCandidates231 1481 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1481] 1481 innerTail231_0 best = best
  exact innerScan231_0

def rootCandidates230 : List Nat := [1597, 2221, 2311, 2389, 2521, 3541, 3571, 3613, 3931, 4159, 4561, 4951, 5179, 5419, 6073, 6133, 6301, 7069, 7207, 7321, 7933]
set_option maxRecDepth 4096 in
theorem rootCandidates230_checked : candidates edge 5 0 1471 tail231 best.bound = rootCandidates230 := by
  unfold candidates
  have row : edge 1471 = PrimePairCache.smallRow1471.contains := by funext q; rfl
  rw [row]
  decide

def innerTail230_21 : List Nat := []
theorem innerScan230_21 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_21 best = best := rfl

def innerTail230_20 : List Nat := 7933 :: innerTail230_21
def innerCandidates230_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_20_checked : candidates edge 4 1471 7933 innerTail230_21 best.bound = innerCandidates230_20 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_20 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_20 best = best := by
  rw [innerTail230_20, scan, if_neg (by decide), innerCandidates230_20_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_21
    (if 3 ≤ innerCandidates230_20.length then search edge 3 [1471, 7933] innerCandidates230_20 9404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_21

def innerTail230_19 : List Nat := 7321 :: innerTail230_20
def innerCandidates230_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_19_checked : candidates edge 4 1471 7321 innerTail230_20 best.bound = innerCandidates230_19 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_19 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_19 best = best := by
  rw [innerTail230_19, scan, if_neg (by decide), innerCandidates230_19_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_20
    (if 3 ≤ innerCandidates230_19.length then search edge 3 [1471, 7321] innerCandidates230_19 8792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_20

def innerTail230_18 : List Nat := 7207 :: innerTail230_19
def innerCandidates230_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_18_checked : candidates edge 4 1471 7207 innerTail230_19 best.bound = innerCandidates230_18 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_18 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_18 best = best := by
  rw [innerTail230_18, scan, if_neg (by decide), innerCandidates230_18_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_19
    (if 3 ≤ innerCandidates230_18.length then search edge 3 [1471, 7207] innerCandidates230_18 8678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_19

def innerTail230_17 : List Nat := 7069 :: innerTail230_18
def innerCandidates230_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_17_checked : candidates edge 4 1471 7069 innerTail230_18 best.bound = innerCandidates230_17 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_17 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_17 best = best := by
  rw [innerTail230_17, scan, if_neg (by decide), innerCandidates230_17_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_18
    (if 3 ≤ innerCandidates230_17.length then search edge 3 [1471, 7069] innerCandidates230_17 8540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_18

def innerTail230_16 : List Nat := 6301 :: innerTail230_17
def innerCandidates230_16 : List Nat := [7321]
set_option maxRecDepth 4096 in
theorem innerCandidates230_16_checked : candidates edge 4 1471 6301 innerTail230_17 best.bound = innerCandidates230_16 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_16 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_16 best = best := by
  rw [innerTail230_16, scan, if_neg (by decide), innerCandidates230_16_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_17
    (if 3 ≤ innerCandidates230_16.length then search edge 3 [1471, 6301] innerCandidates230_16 7772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_17

def innerTail230_15 : List Nat := 6133 :: innerTail230_16
def innerCandidates230_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_15_checked : candidates edge 4 1471 6133 innerTail230_16 best.bound = innerCandidates230_15 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_15 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_15 best = best := by
  rw [innerTail230_15, scan, if_neg (by decide), innerCandidates230_15_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_16
    (if 3 ≤ innerCandidates230_15.length then search edge 3 [1471, 6133] innerCandidates230_15 7604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_16

def innerTail230_14 : List Nat := 6073 :: innerTail230_15
def innerCandidates230_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_14_checked : candidates edge 4 1471 6073 innerTail230_15 best.bound = innerCandidates230_14 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_14 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_14 best = best := by
  rw [innerTail230_14, scan, if_neg (by decide), innerCandidates230_14_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_15
    (if 3 ≤ innerCandidates230_14.length then search edge 3 [1471, 6073] innerCandidates230_14 7544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_15

def innerTail230_13 : List Nat := 5419 :: innerTail230_14
def innerCandidates230_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_13_checked : candidates edge 4 1471 5419 innerTail230_14 best.bound = innerCandidates230_13 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_13 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_13 best = best := by
  rw [innerTail230_13, scan, if_neg (by decide), innerCandidates230_13_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_14
    (if 3 ≤ innerCandidates230_13.length then search edge 3 [1471, 5419] innerCandidates230_13 6890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_14

def innerTail230_12 : List Nat := 5179 :: innerTail230_13
def innerCandidates230_12 : List Nat := [6301]
set_option maxRecDepth 4096 in
theorem innerCandidates230_12_checked : candidates edge 4 1471 5179 innerTail230_13 best.bound = innerCandidates230_12 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_12 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_12 best = best := by
  rw [innerTail230_12, scan, if_neg (by decide), innerCandidates230_12_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_13
    (if 3 ≤ innerCandidates230_12.length then search edge 3 [1471, 5179] innerCandidates230_12 6650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_13

def innerTail230_11 : List Nat := 4951 :: innerTail230_12
def innerCandidates230_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_11_checked : candidates edge 4 1471 4951 innerTail230_12 best.bound = innerCandidates230_11 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_11 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_11 best = best := by
  rw [innerTail230_11, scan, if_neg (by decide), innerCandidates230_11_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_12
    (if 3 ≤ innerCandidates230_11.length then search edge 3 [1471, 4951] innerCandidates230_11 6422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_12

def innerTail230_10 : List Nat := 4561 :: innerTail230_11
def innerCandidates230_10 : List Nat := [5419, 6073]
set_option maxRecDepth 4096 in
theorem innerCandidates230_10_checked : candidates edge 4 1471 4561 innerTail230_11 best.bound = innerCandidates230_10 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_10 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_10 best = best := by
  rw [innerTail230_10, scan, if_neg (by decide), innerCandidates230_10_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_11
    (if 3 ≤ innerCandidates230_10.length then search edge 3 [1471, 4561] innerCandidates230_10 6032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_11

def innerTail230_9 : List Nat := 4159 :: innerTail230_10
def innerCandidates230_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_9_checked : candidates edge 4 1471 4159 innerTail230_10 best.bound = innerCandidates230_9 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_9 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_9 best = best := by
  rw [innerTail230_9, scan, if_neg (by decide), innerCandidates230_9_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_10
    (if 3 ≤ innerCandidates230_9.length then search edge 3 [1471, 4159] innerCandidates230_9 5630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_10

def innerTail230_8 : List Nat := 3931 :: innerTail230_9
def innerCandidates230_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_8_checked : candidates edge 4 1471 3931 innerTail230_9 best.bound = innerCandidates230_8 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_8 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_8 best = best := by
  rw [innerTail230_8, scan, if_neg (by decide), innerCandidates230_8_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_9
    (if 3 ≤ innerCandidates230_8.length then search edge 3 [1471, 3931] innerCandidates230_8 5402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_9

def innerTail230_7 : List Nat := 3613 :: innerTail230_8
def innerCandidates230_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_7_checked : candidates edge 4 1471 3613 innerTail230_8 best.bound = innerCandidates230_7 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_7 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_7 best = best := by
  rw [innerTail230_7, scan, if_neg (by decide), innerCandidates230_7_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_8
    (if 3 ≤ innerCandidates230_7.length then search edge 3 [1471, 3613] innerCandidates230_7 5084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_8

def innerTail230_6 : List Nat := 3571 :: innerTail230_7
def innerCandidates230_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_6_checked : candidates edge 4 1471 3571 innerTail230_7 best.bound = innerCandidates230_6 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_6 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_6 best = best := by
  rw [innerTail230_6, scan, if_neg (by decide), innerCandidates230_6_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_7
    (if 3 ≤ innerCandidates230_6.length then search edge 3 [1471, 3571] innerCandidates230_6 5042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_7

def innerTail230_5 : List Nat := 3541 :: innerTail230_6
def innerCandidates230_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_5_checked : candidates edge 4 1471 3541 innerTail230_6 best.bound = innerCandidates230_5 := by
  unfold candidates
  have row : edge 3541 = SearchData.row3541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_5 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_5 best = best := by
  rw [innerTail230_5, scan, if_neg (by decide), innerCandidates230_5_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_6
    (if 3 ≤ innerCandidates230_5.length then search edge 3 [1471, 3541] innerCandidates230_5 5012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_6

def innerTail230_4 : List Nat := 2521 :: innerTail230_5
def innerCandidates230_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_4_checked : candidates edge 4 1471 2521 innerTail230_5 best.bound = innerCandidates230_4 := by
  unfold candidates
  have row : edge 2521 = PrimePairCache.smallRow2521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_4 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_4 best = best := by
  rw [innerTail230_4, scan, if_neg (by decide), innerCandidates230_4_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_5
    (if 3 ≤ innerCandidates230_4.length then search edge 3 [1471, 2521] innerCandidates230_4 3992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_5

def innerTail230_3 : List Nat := 2389 :: innerTail230_4
def innerCandidates230_3 : List Nat := [2521]
set_option maxRecDepth 4096 in
theorem innerCandidates230_3_checked : candidates edge 4 1471 2389 innerTail230_4 best.bound = innerCandidates230_3 := by
  unfold candidates
  have row : edge 2389 = PrimePairCache.smallRow2389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_3 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_3 best = best := by
  rw [innerTail230_3, scan, if_neg (by decide), innerCandidates230_3_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_4
    (if 3 ≤ innerCandidates230_3.length then search edge 3 [1471, 2389] innerCandidates230_3 3860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_4

def innerTail230_2 : List Nat := 2311 :: innerTail230_3
def innerCandidates230_2 : List Nat := [5179]
set_option maxRecDepth 4096 in
theorem innerCandidates230_2_checked : candidates edge 4 1471 2311 innerTail230_3 best.bound = innerCandidates230_2 := by
  unfold candidates
  have row : edge 2311 = PrimePairCache.smallRow2311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_2 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_2 best = best := by
  rw [innerTail230_2, scan, if_neg (by decide), innerCandidates230_2_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_3
    (if 3 ≤ innerCandidates230_2.length then search edge 3 [1471, 2311] innerCandidates230_2 3782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_3

def innerTail230_1 : List Nat := 2221 :: innerTail230_2
def innerCandidates230_1 : List Nat := [2389]
set_option maxRecDepth 4096 in
theorem innerCandidates230_1_checked : candidates edge 4 1471 2221 innerTail230_2 best.bound = innerCandidates230_1 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_1 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_1 best = best := by
  rw [innerTail230_1, scan, if_neg (by decide), innerCandidates230_1_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_2
    (if 3 ≤ innerCandidates230_1.length then search edge 3 [1471, 2221] innerCandidates230_1 3692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_2

def innerTail230_0 : List Nat := 1597 :: innerTail230_1
def innerCandidates230_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates230_0_checked : candidates edge 4 1471 1597 innerTail230_1 best.bound = innerCandidates230_0 := by
  unfold candidates
  have row : edge 1597 = PrimePairCache.smallRow1597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan230_0 : scan edge (search edge 3) 4 [1471] 1471 innerTail230_0 best = best := by
  rw [innerTail230_0, scan, if_neg (by decide), innerCandidates230_0_checked]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_1
    (if 3 ≤ innerCandidates230_0.length then search edge 3 [1471, 1597] innerCandidates230_0 3068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan230_1

theorem rootBranch230 : search edge 4 [1471] rootCandidates230 1471 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1471] 1471 innerTail230_0 best = best
  exact innerScan230_0

end ProjectEulerSolutions.PrimePairUpper.Search
