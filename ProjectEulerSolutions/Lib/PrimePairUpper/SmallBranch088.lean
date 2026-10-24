import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates442 : List Nat := [3433, 3673, 4201, 4219, 4357, 4507, 4597, 4639, 4903, 5851, 7393, 7507, 7759, 8221]
set_option maxRecDepth 4096 in
theorem rootCandidates442_checked : candidates edge 5 0 3121 tail443 best.bound = rootCandidates442 := by
  unfold candidates
  have row : edge 3121 = SearchData.row3121.contains := by funext q; rfl
  rw [row]
  decide

def innerTail442_14 : List Nat := []
theorem innerScan442_14 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_14 best = best := rfl

def innerTail442_13 : List Nat := 8221 :: innerTail442_14
def innerCandidates442_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates442_13_checked : candidates edge 4 3121 8221 innerTail442_14 best.bound = innerCandidates442_13 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_13 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_13 best = best := by
  rw [innerTail442_13, scan, if_neg (by decide), innerCandidates442_13_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_14
    (if 3 ≤ innerCandidates442_13.length then search edge 3 [3121, 8221] innerCandidates442_13 11342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_14

def innerTail442_12 : List Nat := 7759 :: innerTail442_13
def innerCandidates442_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates442_12_checked : candidates edge 4 3121 7759 innerTail442_13 best.bound = innerCandidates442_12 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_12 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_12 best = best := by
  rw [innerTail442_12, scan, if_neg (by decide), innerCandidates442_12_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_13
    (if 3 ≤ innerCandidates442_12.length then search edge 3 [3121, 7759] innerCandidates442_12 10880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_13

def innerTail442_11 : List Nat := 7507 :: innerTail442_12
def innerCandidates442_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates442_11_checked : candidates edge 4 3121 7507 innerTail442_12 best.bound = innerCandidates442_11 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_11 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_11 best = best := by
  rw [innerTail442_11, scan, if_neg (by decide), innerCandidates442_11_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_12
    (if 3 ≤ innerCandidates442_11.length then search edge 3 [3121, 7507] innerCandidates442_11 10628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_12

def innerTail442_10 : List Nat := 7393 :: innerTail442_11
def innerCandidates442_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates442_10_checked : candidates edge 4 3121 7393 innerTail442_11 best.bound = innerCandidates442_10 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_10 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_10 best = best := by
  rw [innerTail442_10, scan, if_neg (by decide), innerCandidates442_10_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_11
    (if 3 ≤ innerCandidates442_10.length then search edge 3 [3121, 7393] innerCandidates442_10 10514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_11

def innerTail442_9 : List Nat := 5851 :: innerTail442_10
def innerCandidates442_9 : List Nat := [7507]
set_option maxRecDepth 4096 in
theorem innerCandidates442_9_checked : candidates edge 4 3121 5851 innerTail442_10 best.bound = innerCandidates442_9 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_9 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_9 best = best := by
  rw [innerTail442_9, scan, if_neg (by decide), innerCandidates442_9_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_10
    (if 3 ≤ innerCandidates442_9.length then search edge 3 [3121, 5851] innerCandidates442_9 8972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_10

def innerTail442_8 : List Nat := 4903 :: innerTail442_9
def innerCandidates442_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates442_8_checked : candidates edge 4 3121 4903 innerTail442_9 best.bound = innerCandidates442_8 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_8 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_8 best = best := by
  rw [innerTail442_8, scan, if_neg (by decide), innerCandidates442_8_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_9
    (if 3 ≤ innerCandidates442_8.length then search edge 3 [3121, 4903] innerCandidates442_8 8024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_9

def innerTail442_7 : List Nat := 4639 :: innerTail442_8
def innerCandidates442_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates442_7_checked : candidates edge 4 3121 4639 innerTail442_8 best.bound = innerCandidates442_7 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_7 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_7 best = best := by
  rw [innerTail442_7, scan, if_neg (by decide), innerCandidates442_7_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_8
    (if 3 ≤ innerCandidates442_7.length then search edge 3 [3121, 4639] innerCandidates442_7 7760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_8

def innerTail442_6 : List Nat := 4597 :: innerTail442_7
def innerCandidates442_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates442_6_checked : candidates edge 4 3121 4597 innerTail442_7 best.bound = innerCandidates442_6 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_6 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_6 best = best := by
  rw [innerTail442_6, scan, if_neg (by decide), innerCandidates442_6_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_7
    (if 3 ≤ innerCandidates442_6.length then search edge 3 [3121, 4597] innerCandidates442_6 7718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_7

def innerTail442_5 : List Nat := 4507 :: innerTail442_6
def innerCandidates442_5 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates442_5_checked : candidates edge 4 3121 4507 innerTail442_6 best.bound = innerCandidates442_5 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_5 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_5 best = best := by
  rw [innerTail442_5, scan, if_neg (by decide), innerCandidates442_5_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_6
    (if 3 ≤ innerCandidates442_5.length then search edge 3 [3121, 4507] innerCandidates442_5 7628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_6

def innerTail442_4 : List Nat := 4357 :: innerTail442_5
def innerCandidates442_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates442_4_checked : candidates edge 4 3121 4357 innerTail442_5 best.bound = innerCandidates442_4 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_4 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_4 best = best := by
  rw [innerTail442_4, scan, if_neg (by decide), innerCandidates442_4_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_5
    (if 3 ≤ innerCandidates442_4.length then search edge 3 [3121, 4357] innerCandidates442_4 7478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_5

def innerTail442_3 : List Nat := 4219 :: innerTail442_4
def innerCandidates442_3 : List Nat := [4597]
set_option maxRecDepth 4096 in
theorem innerCandidates442_3_checked : candidates edge 4 3121 4219 innerTail442_4 best.bound = innerCandidates442_3 := by
  unfold candidates
  have row : edge 4219 = SearchData.row4219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_3 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_3 best = best := by
  rw [innerTail442_3, scan, if_neg (by decide), innerCandidates442_3_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_4
    (if 3 ≤ innerCandidates442_3.length then search edge 3 [3121, 4219] innerCandidates442_3 7340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_4

def innerTail442_2 : List Nat := 4201 :: innerTail442_3
def innerCandidates442_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates442_2_checked : candidates edge 4 3121 4201 innerTail442_3 best.bound = innerCandidates442_2 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_2 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_2 best = best := by
  rw [innerTail442_2, scan, if_neg (by decide), innerCandidates442_2_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_3
    (if 3 ≤ innerCandidates442_2.length then search edge 3 [3121, 4201] innerCandidates442_2 7322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_3

def innerTail442_1 : List Nat := 3673 :: innerTail442_2
def innerCandidates442_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates442_1_checked : candidates edge 4 3121 3673 innerTail442_2 best.bound = innerCandidates442_1 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_1 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_1 best = best := by
  rw [innerTail442_1, scan, if_neg (by decide), innerCandidates442_1_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_2
    (if 3 ≤ innerCandidates442_1.length then search edge 3 [3121, 3673] innerCandidates442_1 6794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_2

def innerTail442_0 : List Nat := 3433 :: innerTail442_1
def innerCandidates442_0 : List Nat := [8221]
set_option maxRecDepth 4096 in
theorem innerCandidates442_0_checked : candidates edge 4 3121 3433 innerTail442_1 best.bound = innerCandidates442_0 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan442_0 : scan edge (search edge 3) 4 [3121] 3121 innerTail442_0 best = best := by
  rw [innerTail442_0, scan, if_neg (by decide), innerCandidates442_0_checked]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_1
    (if 3 ≤ innerCandidates442_0.length then search edge 3 [3121, 3433] innerCandidates442_0 6554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan442_1

theorem rootBranch442 : search edge 4 [3121] rootCandidates442 3121 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3121] 3121 innerTail442_0 best = best
  exact innerScan442_0

def rootCandidates441 : List Nat := [3677, 4391, 4679, 5231, 5237, 5741, 5783, 6029, 6311, 6359, 6491, 6857, 7109]
set_option maxRecDepth 4096 in
theorem rootCandidates441_checked : candidates edge 5 0 3119 tail442 best.bound = rootCandidates441 := by
  unfold candidates
  have row : edge 3119 = SearchData.row3119.contains := by funext q; rfl
  rw [row]
  decide

def innerTail441_13 : List Nat := []
theorem innerScan441_13 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_13 best = best := rfl

def innerTail441_12 : List Nat := 7109 :: innerTail441_13
def innerCandidates441_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates441_12_checked : candidates edge 4 3119 7109 innerTail441_13 best.bound = innerCandidates441_12 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_12 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_12 best = best := by
  rw [innerTail441_12, scan, if_neg (by decide), innerCandidates441_12_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_13
    (if 3 ≤ innerCandidates441_12.length then search edge 3 [3119, 7109] innerCandidates441_12 10228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_13

def innerTail441_11 : List Nat := 6857 :: innerTail441_12
def innerCandidates441_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates441_11_checked : candidates edge 4 3119 6857 innerTail441_12 best.bound = innerCandidates441_11 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_11 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_11 best = best := by
  rw [innerTail441_11, scan, if_neg (by decide), innerCandidates441_11_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_12
    (if 3 ≤ innerCandidates441_11.length then search edge 3 [3119, 6857] innerCandidates441_11 9976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_12

def innerTail441_10 : List Nat := 6491 :: innerTail441_11
def innerCandidates441_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates441_10_checked : candidates edge 4 3119 6491 innerTail441_11 best.bound = innerCandidates441_10 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_10 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_10 best = best := by
  rw [innerTail441_10, scan, if_neg (by decide), innerCandidates441_10_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_11
    (if 3 ≤ innerCandidates441_10.length then search edge 3 [3119, 6491] innerCandidates441_10 9610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_11

def innerTail441_9 : List Nat := 6359 :: innerTail441_10
def innerCandidates441_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates441_9_checked : candidates edge 4 3119 6359 innerTail441_10 best.bound = innerCandidates441_9 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_9 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_9 best = best := by
  rw [innerTail441_9, scan, if_neg (by decide), innerCandidates441_9_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_10
    (if 3 ≤ innerCandidates441_9.length then search edge 3 [3119, 6359] innerCandidates441_9 9478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_10

def innerTail441_8 : List Nat := 6311 :: innerTail441_9
def innerCandidates441_8 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates441_8_checked : candidates edge 4 3119 6311 innerTail441_9 best.bound = innerCandidates441_8 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_8 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_8 best = best := by
  rw [innerTail441_8, scan, if_neg (by decide), innerCandidates441_8_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_9
    (if 3 ≤ innerCandidates441_8.length then search edge 3 [3119, 6311] innerCandidates441_8 9430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_9

def innerTail441_7 : List Nat := 6029 :: innerTail441_8
def innerCandidates441_7 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates441_7_checked : candidates edge 4 3119 6029 innerTail441_8 best.bound = innerCandidates441_7 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_7 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_7 best = best := by
  rw [innerTail441_7, scan, if_neg (by decide), innerCandidates441_7_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_8
    (if 3 ≤ innerCandidates441_7.length then search edge 3 [3119, 6029] innerCandidates441_7 9148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_8

def innerTail441_6 : List Nat := 5783 :: innerTail441_7
def innerCandidates441_6 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates441_6_checked : candidates edge 4 3119 5783 innerTail441_7 best.bound = innerCandidates441_6 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_6 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_6 best = best := by
  rw [innerTail441_6, scan, if_neg (by decide), innerCandidates441_6_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_7
    (if 3 ≤ innerCandidates441_6.length then search edge 3 [3119, 5783] innerCandidates441_6 8902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_7

def innerTail441_5 : List Nat := 5741 :: innerTail441_6
def innerCandidates441_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates441_5_checked : candidates edge 4 3119 5741 innerTail441_6 best.bound = innerCandidates441_5 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_5 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_5 best = best := by
  rw [innerTail441_5, scan, if_neg (by decide), innerCandidates441_5_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_6
    (if 3 ≤ innerCandidates441_5.length then search edge 3 [3119, 5741] innerCandidates441_5 8860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_6

def innerTail441_4 : List Nat := 5237 :: innerTail441_5
def innerCandidates441_4 : List Nat := [5783]
set_option maxRecDepth 4096 in
theorem innerCandidates441_4_checked : candidates edge 4 3119 5237 innerTail441_5 best.bound = innerCandidates441_4 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_4 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_4 best = best := by
  rw [innerTail441_4, scan, if_neg (by decide), innerCandidates441_4_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_5
    (if 3 ≤ innerCandidates441_4.length then search edge 3 [3119, 5237] innerCandidates441_4 8356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_5

def innerTail441_3 : List Nat := 5231 :: innerTail441_4
def innerCandidates441_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates441_3_checked : candidates edge 4 3119 5231 innerTail441_4 best.bound = innerCandidates441_3 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_3 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_3 best = best := by
  rw [innerTail441_3, scan, if_neg (by decide), innerCandidates441_3_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_4
    (if 3 ≤ innerCandidates441_3.length then search edge 3 [3119, 5231] innerCandidates441_3 8350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_4

def innerTail441_2 : List Nat := 4679 :: innerTail441_3
def innerCandidates441_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates441_2_checked : candidates edge 4 3119 4679 innerTail441_3 best.bound = innerCandidates441_2 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_2 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_2 best = best := by
  rw [innerTail441_2, scan, if_neg (by decide), innerCandidates441_2_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_3
    (if 3 ≤ innerCandidates441_2.length then search edge 3 [3119, 4679] innerCandidates441_2 7798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_3

def innerTail441_1 : List Nat := 4391 :: innerTail441_2
def innerCandidates441_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates441_1_checked : candidates edge 4 3119 4391 innerTail441_2 best.bound = innerCandidates441_1 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_1 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_1 best = best := by
  rw [innerTail441_1, scan, if_neg (by decide), innerCandidates441_1_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_2
    (if 3 ≤ innerCandidates441_1.length then search edge 3 [3119, 4391] innerCandidates441_1 7510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_2

def innerTail441_0 : List Nat := 3677 :: innerTail441_1
def innerCandidates441_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates441_0_checked : candidates edge 4 3119 3677 innerTail441_1 best.bound = innerCandidates441_0 := by
  unfold candidates
  have row : edge 3677 = SearchData.row3677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan441_0 : scan edge (search edge 3) 4 [3119] 3119 innerTail441_0 best = best := by
  rw [innerTail441_0, scan, if_neg (by decide), innerCandidates441_0_checked]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_1
    (if 3 ≤ innerCandidates441_0.length then search edge 3 [3119, 3677] innerCandidates441_0 6796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan441_1

theorem rootBranch441 : search edge 4 [3119] rootCandidates441 3119 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3119] 3119 innerTail441_0 best = best
  exact innerScan441_0

def rootCandidates440 : List Nat := [3541, 3877, 4027, 4057, 4099, 4339, 4447, 4759, 4801, 5791, 6073, 6607, 6733, 6883, 6997, 7069, 7177, 7603, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates440_checked : candidates edge 5 0 3109 tail441 best.bound = rootCandidates440 := by
  unfold candidates
  have row : edge 3109 = SearchData.row3109.contains := by funext q; rfl
  rw [row]
  decide

def innerTail440_19 : List Nat := []
theorem innerScan440_19 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_19 best = best := rfl

def innerTail440_18 : List Nat := 7993 :: innerTail440_19
def innerCandidates440_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_18_checked : candidates edge 4 3109 7993 innerTail440_19 best.bound = innerCandidates440_18 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_18 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_18 best = best := by
  rw [innerTail440_18, scan, if_neg (by decide), innerCandidates440_18_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_19
    (if 3 ≤ innerCandidates440_18.length then search edge 3 [3109, 7993] innerCandidates440_18 11102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_19

def innerTail440_17 : List Nat := 7603 :: innerTail440_18
def innerCandidates440_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_17_checked : candidates edge 4 3109 7603 innerTail440_18 best.bound = innerCandidates440_17 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_17 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_17 best = best := by
  rw [innerTail440_17, scan, if_neg (by decide), innerCandidates440_17_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_18
    (if 3 ≤ innerCandidates440_17.length then search edge 3 [3109, 7603] innerCandidates440_17 10712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_18

def innerTail440_16 : List Nat := 7177 :: innerTail440_17
def innerCandidates440_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_16_checked : candidates edge 4 3109 7177 innerTail440_17 best.bound = innerCandidates440_16 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_16 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_16 best = best := by
  rw [innerTail440_16, scan, if_neg (by decide), innerCandidates440_16_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_17
    (if 3 ≤ innerCandidates440_16.length then search edge 3 [3109, 7177] innerCandidates440_16 10286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_17

def innerTail440_15 : List Nat := 7069 :: innerTail440_16
def innerCandidates440_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_15_checked : candidates edge 4 3109 7069 innerTail440_16 best.bound = innerCandidates440_15 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_15 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_15 best = best := by
  rw [innerTail440_15, scan, if_neg (by decide), innerCandidates440_15_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_16
    (if 3 ≤ innerCandidates440_15.length then search edge 3 [3109, 7069] innerCandidates440_15 10178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_16

def innerTail440_14 : List Nat := 6997 :: innerTail440_15
def innerCandidates440_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_14_checked : candidates edge 4 3109 6997 innerTail440_15 best.bound = innerCandidates440_14 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_14 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_14 best = best := by
  rw [innerTail440_14, scan, if_neg (by decide), innerCandidates440_14_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_15
    (if 3 ≤ innerCandidates440_14.length then search edge 3 [3109, 6997] innerCandidates440_14 10106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_15

def innerTail440_13 : List Nat := 6883 :: innerTail440_14
def innerCandidates440_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_13_checked : candidates edge 4 3109 6883 innerTail440_14 best.bound = innerCandidates440_13 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_13 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_13 best = best := by
  rw [innerTail440_13, scan, if_neg (by decide), innerCandidates440_13_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_14
    (if 3 ≤ innerCandidates440_13.length then search edge 3 [3109, 6883] innerCandidates440_13 9992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_14

def innerTail440_12 : List Nat := 6733 :: innerTail440_13
def innerCandidates440_12 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates440_12_checked : candidates edge 4 3109 6733 innerTail440_13 best.bound = innerCandidates440_12 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_12 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_12 best = best := by
  rw [innerTail440_12, scan, if_neg (by decide), innerCandidates440_12_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_13
    (if 3 ≤ innerCandidates440_12.length then search edge 3 [3109, 6733] innerCandidates440_12 9842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_13

def innerTail440_11 : List Nat := 6607 :: innerTail440_12
def innerCandidates440_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_11_checked : candidates edge 4 3109 6607 innerTail440_12 best.bound = innerCandidates440_11 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_11 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_11 best = best := by
  rw [innerTail440_11, scan, if_neg (by decide), innerCandidates440_11_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_12
    (if 3 ≤ innerCandidates440_11.length then search edge 3 [3109, 6607] innerCandidates440_11 9716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_12

def innerTail440_10 : List Nat := 6073 :: innerTail440_11
def innerCandidates440_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_10_checked : candidates edge 4 3109 6073 innerTail440_11 best.bound = innerCandidates440_10 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_10 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_10 best = best := by
  rw [innerTail440_10, scan, if_neg (by decide), innerCandidates440_10_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_11
    (if 3 ≤ innerCandidates440_10.length then search edge 3 [3109, 6073] innerCandidates440_10 9182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_11

def innerTail440_9 : List Nat := 5791 :: innerTail440_10
def innerCandidates440_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_9_checked : candidates edge 4 3109 5791 innerTail440_10 best.bound = innerCandidates440_9 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_9 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_9 best = best := by
  rw [innerTail440_9, scan, if_neg (by decide), innerCandidates440_9_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_10
    (if 3 ≤ innerCandidates440_9.length then search edge 3 [3109, 5791] innerCandidates440_9 8900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_10

def innerTail440_8 : List Nat := 4801 :: innerTail440_9
def innerCandidates440_8 : List Nat := [6733, 7993]
set_option maxRecDepth 4096 in
theorem innerCandidates440_8_checked : candidates edge 4 3109 4801 innerTail440_9 best.bound = innerCandidates440_8 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_8 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_8 best = best := by
  rw [innerTail440_8, scan, if_neg (by decide), innerCandidates440_8_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_9
    (if 3 ≤ innerCandidates440_8.length then search edge 3 [3109, 4801] innerCandidates440_8 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_9

def innerTail440_7 : List Nat := 4759 :: innerTail440_8
def innerCandidates440_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_7_checked : candidates edge 4 3109 4759 innerTail440_8 best.bound = innerCandidates440_7 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_7 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_7 best = best := by
  rw [innerTail440_7, scan, if_neg (by decide), innerCandidates440_7_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_8
    (if 3 ≤ innerCandidates440_7.length then search edge 3 [3109, 4759] innerCandidates440_7 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_8

def innerTail440_6 : List Nat := 4447 :: innerTail440_7
def innerCandidates440_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_6_checked : candidates edge 4 3109 4447 innerTail440_7 best.bound = innerCandidates440_6 := by
  unfold candidates
  have row : edge 4447 = SearchData.row4447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_6 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_6 best = best := by
  rw [innerTail440_6, scan, if_neg (by decide), innerCandidates440_6_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_7
    (if 3 ≤ innerCandidates440_6.length then search edge 3 [3109, 4447] innerCandidates440_6 7556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_7

def innerTail440_5 : List Nat := 4339 :: innerTail440_6
def innerCandidates440_5 : List Nat := [7603]
set_option maxRecDepth 4096 in
theorem innerCandidates440_5_checked : candidates edge 4 3109 4339 innerTail440_6 best.bound = innerCandidates440_5 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_5 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_5 best = best := by
  rw [innerTail440_5, scan, if_neg (by decide), innerCandidates440_5_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_6
    (if 3 ≤ innerCandidates440_5.length then search edge 3 [3109, 4339] innerCandidates440_5 7448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_6

def innerTail440_4 : List Nat := 4099 :: innerTail440_5
def innerCandidates440_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_4_checked : candidates edge 4 3109 4099 innerTail440_5 best.bound = innerCandidates440_4 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_4 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_4 best = best := by
  rw [innerTail440_4, scan, if_neg (by decide), innerCandidates440_4_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_5
    (if 3 ≤ innerCandidates440_4.length then search edge 3 [3109, 4099] innerCandidates440_4 7208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_5

def innerTail440_3 : List Nat := 4057 :: innerTail440_4
def innerCandidates440_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_3_checked : candidates edge 4 3109 4057 innerTail440_4 best.bound = innerCandidates440_3 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_3 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_3 best = best := by
  rw [innerTail440_3, scan, if_neg (by decide), innerCandidates440_3_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_4
    (if 3 ≤ innerCandidates440_3.length then search edge 3 [3109, 4057] innerCandidates440_3 7166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_4

def innerTail440_2 : List Nat := 4027 :: innerTail440_3
def innerCandidates440_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_2_checked : candidates edge 4 3109 4027 innerTail440_3 best.bound = innerCandidates440_2 := by
  unfold candidates
  have row : edge 4027 = SearchData.row4027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_2 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_2 best = best := by
  rw [innerTail440_2, scan, if_neg (by decide), innerCandidates440_2_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_3
    (if 3 ≤ innerCandidates440_2.length then search edge 3 [3109, 4027] innerCandidates440_2 7136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_3

def innerTail440_1 : List Nat := 3877 :: innerTail440_2
def innerCandidates440_1 : List Nat := [6607]
set_option maxRecDepth 4096 in
theorem innerCandidates440_1_checked : candidates edge 4 3109 3877 innerTail440_2 best.bound = innerCandidates440_1 := by
  unfold candidates
  have row : edge 3877 = SearchData.row3877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_1 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_1 best = best := by
  rw [innerTail440_1, scan, if_neg (by decide), innerCandidates440_1_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_2
    (if 3 ≤ innerCandidates440_1.length then search edge 3 [3109, 3877] innerCandidates440_1 6986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_2

def innerTail440_0 : List Nat := 3541 :: innerTail440_1
def innerCandidates440_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates440_0_checked : candidates edge 4 3109 3541 innerTail440_1 best.bound = innerCandidates440_0 := by
  unfold candidates
  have row : edge 3541 = SearchData.row3541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan440_0 : scan edge (search edge 3) 4 [3109] 3109 innerTail440_0 best = best := by
  rw [innerTail440_0, scan, if_neg (by decide), innerCandidates440_0_checked]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_1
    (if 3 ≤ innerCandidates440_0.length then search edge 3 [3109, 3541] innerCandidates440_0 6650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan440_1

theorem rootBranch440 : search edge 4 [3109] rootCandidates440 3109 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3109] 3109 innerTail440_0 best = best
  exact innerScan440_0

def rootCandidates439 : List Nat := [3299, 3851, 3917, 5147, 5303, 5639, 5867, 6689, 7457, 7541, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates439_checked : candidates edge 5 0 3089 tail440 best.bound = rootCandidates439 := by
  unfold candidates
  have row : edge 3089 = SearchData.row3089.contains := by funext q; rfl
  rw [row]
  decide

def innerTail439_11 : List Nat := []
theorem innerScan439_11 : scan edge (search edge 3) 4 [3089] 3089 innerTail439_11 best = best := rfl

def innerTail439_10 : List Nat := 8093 :: innerTail439_11
def innerCandidates439_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates439_10_checked : candidates edge 4 3089 8093 innerTail439_11 best.bound = innerCandidates439_10 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan439_10 : scan edge (search edge 3) 4 [3089] 3089 innerTail439_10 best = best := by
  rw [innerTail439_10, scan, if_neg (by decide), innerCandidates439_10_checked]
  change scan edge (search edge 3) 4 [3089] 3089 innerTail439_11
    (if 3 ≤ innerCandidates439_10.length then search edge 3 [3089, 8093] innerCandidates439_10 11182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan439_11

def innerTail439_9 : List Nat := 7541 :: innerTail439_10
def innerCandidates439_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates439_9_checked : candidates edge 4 3089 7541 innerTail439_10 best.bound = innerCandidates439_9 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan439_9 : scan edge (search edge 3) 4 [3089] 3089 innerTail439_9 best = best := by
  rw [innerTail439_9, scan, if_neg (by decide), innerCandidates439_9_checked]
  change scan edge (search edge 3) 4 [3089] 3089 innerTail439_10
    (if 3 ≤ innerCandidates439_9.length then search edge 3 [3089, 7541] innerCandidates439_9 10630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan439_10

def innerTail439_8 : List Nat := 7457 :: innerTail439_9
def innerCandidates439_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates439_8_checked : candidates edge 4 3089 7457 innerTail439_9 best.bound = innerCandidates439_8 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan439_8 : scan edge (search edge 3) 4 [3089] 3089 innerTail439_8 best = best := by
  rw [innerTail439_8, scan, if_neg (by decide), innerCandidates439_8_checked]
  change scan edge (search edge 3) 4 [3089] 3089 innerTail439_9
    (if 3 ≤ innerCandidates439_8.length then search edge 3 [3089, 7457] innerCandidates439_8 10546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan439_9

def innerTail439_7 : List Nat := 6689 :: innerTail439_8
def innerCandidates439_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates439_7_checked : candidates edge 4 3089 6689 innerTail439_8 best.bound = innerCandidates439_7 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan439_7 : scan edge (search edge 3) 4 [3089] 3089 innerTail439_7 best = best := by
  rw [innerTail439_7, scan, if_neg (by decide), innerCandidates439_7_checked]
  change scan edge (search edge 3) 4 [3089] 3089 innerTail439_8
    (if 3 ≤ innerCandidates439_7.length then search edge 3 [3089, 6689] innerCandidates439_7 9778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan439_8

def innerTail439_6 : List Nat := 5867 :: innerTail439_7
def innerCandidates439_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates439_6_checked : candidates edge 4 3089 5867 innerTail439_7 best.bound = innerCandidates439_6 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan439_6 : scan edge (search edge 3) 4 [3089] 3089 innerTail439_6 best = best := by
  rw [innerTail439_6, scan, if_neg (by decide), innerCandidates439_6_checked]
  change scan edge (search edge 3) 4 [3089] 3089 innerTail439_7
    (if 3 ≤ innerCandidates439_6.length then search edge 3 [3089, 5867] innerCandidates439_6 8956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan439_7

def innerTail439_5 : List Nat := 5639 :: innerTail439_6
def innerCandidates439_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates439_5_checked : candidates edge 4 3089 5639 innerTail439_6 best.bound = innerCandidates439_5 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan439_5 : scan edge (search edge 3) 4 [3089] 3089 innerTail439_5 best = best := by
  rw [innerTail439_5, scan, if_neg (by decide), innerCandidates439_5_checked]
  change scan edge (search edge 3) 4 [3089] 3089 innerTail439_6
    (if 3 ≤ innerCandidates439_5.length then search edge 3 [3089, 5639] innerCandidates439_5 8728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan439_6

def innerTail439_4 : List Nat := 5303 :: innerTail439_5
def innerCandidates439_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates439_4_checked : candidates edge 4 3089 5303 innerTail439_5 best.bound = innerCandidates439_4 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan439_4 : scan edge (search edge 3) 4 [3089] 3089 innerTail439_4 best = best := by
  rw [innerTail439_4, scan, if_neg (by decide), innerCandidates439_4_checked]
  change scan edge (search edge 3) 4 [3089] 3089 innerTail439_5
    (if 3 ≤ innerCandidates439_4.length then search edge 3 [3089, 5303] innerCandidates439_4 8392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan439_5

def innerTail439_3 : List Nat := 5147 :: innerTail439_4
def innerCandidates439_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates439_3_checked : candidates edge 4 3089 5147 innerTail439_4 best.bound = innerCandidates439_3 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan439_3 : scan edge (search edge 3) 4 [3089] 3089 innerTail439_3 best = best := by
  rw [innerTail439_3, scan, if_neg (by decide), innerCandidates439_3_checked]
  change scan edge (search edge 3) 4 [3089] 3089 innerTail439_4
    (if 3 ≤ innerCandidates439_3.length then search edge 3 [3089, 5147] innerCandidates439_3 8236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan439_4

def innerTail439_2 : List Nat := 3917 :: innerTail439_3
def innerCandidates439_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates439_2_checked : candidates edge 4 3089 3917 innerTail439_3 best.bound = innerCandidates439_2 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan439_2 : scan edge (search edge 3) 4 [3089] 3089 innerTail439_2 best = best := by
  rw [innerTail439_2, scan, if_neg (by decide), innerCandidates439_2_checked]
  change scan edge (search edge 3) 4 [3089] 3089 innerTail439_3
    (if 3 ≤ innerCandidates439_2.length then search edge 3 [3089, 3917] innerCandidates439_2 7006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan439_3

def innerTail439_1 : List Nat := 3851 :: innerTail439_2
def innerCandidates439_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates439_1_checked : candidates edge 4 3089 3851 innerTail439_2 best.bound = innerCandidates439_1 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan439_1 : scan edge (search edge 3) 4 [3089] 3089 innerTail439_1 best = best := by
  rw [innerTail439_1, scan, if_neg (by decide), innerCandidates439_1_checked]
  change scan edge (search edge 3) 4 [3089] 3089 innerTail439_2
    (if 3 ≤ innerCandidates439_1.length then search edge 3 [3089, 3851] innerCandidates439_1 6940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan439_2

def innerTail439_0 : List Nat := 3299 :: innerTail439_1
def innerCandidates439_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates439_0_checked : candidates edge 4 3089 3299 innerTail439_1 best.bound = innerCandidates439_0 := by
  unfold candidates
  have row : edge 3299 = SearchData.row3299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan439_0 : scan edge (search edge 3) 4 [3089] 3089 innerTail439_0 best = best := by
  rw [innerTail439_0, scan, if_neg (by decide), innerCandidates439_0_checked]
  change scan edge (search edge 3) 4 [3089] 3089 innerTail439_1
    (if 3 ≤ innerCandidates439_0.length then search edge 3 [3089, 3299] innerCandidates439_0 6388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan439_1

theorem rootBranch439 : search edge 4 [3089] rootCandidates439 3089 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3089] 3089 innerTail439_0 best = best
  exact innerScan439_0

end ProjectEulerSolutions.PrimePairUpper.Search
