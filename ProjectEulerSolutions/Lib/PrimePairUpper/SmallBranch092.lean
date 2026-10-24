import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates425 : List Nat := [3539, 3557, 4547, 4703, 5009, 5153, 6701, 6803, 7673, 7823, 7841]
set_option maxRecDepth 4096 in
theorem rootCandidates425_checked : candidates edge 5 0 2969 tail426 best.bound = rootCandidates425 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext q; rfl
  rw [row]
  decide

def innerTail425_11 : List Nat := []
theorem innerScan425_11 : scan edge (search edge 3) 4 [2969] 2969 innerTail425_11 best = best := rfl

def innerTail425_10 : List Nat := 7841 :: innerTail425_11
def innerCandidates425_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates425_10_checked : candidates edge 4 2969 7841 innerTail425_11 best.bound = innerCandidates425_10 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan425_10 : scan edge (search edge 3) 4 [2969] 2969 innerTail425_10 best = best := by
  rw [innerTail425_10, scan, if_neg (by decide), innerCandidates425_10_checked]
  change scan edge (search edge 3) 4 [2969] 2969 innerTail425_11
    (if 3 ≤ innerCandidates425_10.length then search edge 3 [2969, 7841] innerCandidates425_10 10810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan425_11

def innerTail425_9 : List Nat := 7823 :: innerTail425_10
def innerCandidates425_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates425_9_checked : candidates edge 4 2969 7823 innerTail425_10 best.bound = innerCandidates425_9 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan425_9 : scan edge (search edge 3) 4 [2969] 2969 innerTail425_9 best = best := by
  rw [innerTail425_9, scan, if_neg (by decide), innerCandidates425_9_checked]
  change scan edge (search edge 3) 4 [2969] 2969 innerTail425_10
    (if 3 ≤ innerCandidates425_9.length then search edge 3 [2969, 7823] innerCandidates425_9 10792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan425_10

def innerTail425_8 : List Nat := 7673 :: innerTail425_9
def innerCandidates425_8 : List Nat := [7823]
set_option maxRecDepth 4096 in
theorem innerCandidates425_8_checked : candidates edge 4 2969 7673 innerTail425_9 best.bound = innerCandidates425_8 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan425_8 : scan edge (search edge 3) 4 [2969] 2969 innerTail425_8 best = best := by
  rw [innerTail425_8, scan, if_neg (by decide), innerCandidates425_8_checked]
  change scan edge (search edge 3) 4 [2969] 2969 innerTail425_9
    (if 3 ≤ innerCandidates425_8.length then search edge 3 [2969, 7673] innerCandidates425_8 10642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan425_9

def innerTail425_7 : List Nat := 6803 :: innerTail425_8
def innerCandidates425_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates425_7_checked : candidates edge 4 2969 6803 innerTail425_8 best.bound = innerCandidates425_7 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan425_7 : scan edge (search edge 3) 4 [2969] 2969 innerTail425_7 best = best := by
  rw [innerTail425_7, scan, if_neg (by decide), innerCandidates425_7_checked]
  change scan edge (search edge 3) 4 [2969] 2969 innerTail425_8
    (if 3 ≤ innerCandidates425_7.length then search edge 3 [2969, 6803] innerCandidates425_7 9772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan425_8

def innerTail425_6 : List Nat := 6701 :: innerTail425_7
def innerCandidates425_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates425_6_checked : candidates edge 4 2969 6701 innerTail425_7 best.bound = innerCandidates425_6 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan425_6 : scan edge (search edge 3) 4 [2969] 2969 innerTail425_6 best = best := by
  rw [innerTail425_6, scan, if_neg (by decide), innerCandidates425_6_checked]
  change scan edge (search edge 3) 4 [2969] 2969 innerTail425_7
    (if 3 ≤ innerCandidates425_6.length then search edge 3 [2969, 6701] innerCandidates425_6 9670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan425_7

def innerTail425_5 : List Nat := 5153 :: innerTail425_6
def innerCandidates425_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates425_5_checked : candidates edge 4 2969 5153 innerTail425_6 best.bound = innerCandidates425_5 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan425_5 : scan edge (search edge 3) 4 [2969] 2969 innerTail425_5 best = best := by
  rw [innerTail425_5, scan, if_neg (by decide), innerCandidates425_5_checked]
  change scan edge (search edge 3) 4 [2969] 2969 innerTail425_6
    (if 3 ≤ innerCandidates425_5.length then search edge 3 [2969, 5153] innerCandidates425_5 8122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan425_6

def innerTail425_4 : List Nat := 5009 :: innerTail425_5
def innerCandidates425_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates425_4_checked : candidates edge 4 2969 5009 innerTail425_5 best.bound = innerCandidates425_4 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan425_4 : scan edge (search edge 3) 4 [2969] 2969 innerTail425_4 best = best := by
  rw [innerTail425_4, scan, if_neg (by decide), innerCandidates425_4_checked]
  change scan edge (search edge 3) 4 [2969] 2969 innerTail425_5
    (if 3 ≤ innerCandidates425_4.length then search edge 3 [2969, 5009] innerCandidates425_4 7978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan425_5

def innerTail425_3 : List Nat := 4703 :: innerTail425_4
def innerCandidates425_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates425_3_checked : candidates edge 4 2969 4703 innerTail425_4 best.bound = innerCandidates425_3 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan425_3 : scan edge (search edge 3) 4 [2969] 2969 innerTail425_3 best = best := by
  rw [innerTail425_3, scan, if_neg (by decide), innerCandidates425_3_checked]
  change scan edge (search edge 3) 4 [2969] 2969 innerTail425_4
    (if 3 ≤ innerCandidates425_3.length then search edge 3 [2969, 4703] innerCandidates425_3 7672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan425_4

def innerTail425_2 : List Nat := 4547 :: innerTail425_3
def innerCandidates425_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates425_2_checked : candidates edge 4 2969 4547 innerTail425_3 best.bound = innerCandidates425_2 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan425_2 : scan edge (search edge 3) 4 [2969] 2969 innerTail425_2 best = best := by
  rw [innerTail425_2, scan, if_neg (by decide), innerCandidates425_2_checked]
  change scan edge (search edge 3) 4 [2969] 2969 innerTail425_3
    (if 3 ≤ innerCandidates425_2.length then search edge 3 [2969, 4547] innerCandidates425_2 7516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan425_3

def innerTail425_1 : List Nat := 3557 :: innerTail425_2
def innerCandidates425_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates425_1_checked : candidates edge 4 2969 3557 innerTail425_2 best.bound = innerCandidates425_1 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan425_1 : scan edge (search edge 3) 4 [2969] 2969 innerTail425_1 best = best := by
  rw [innerTail425_1, scan, if_neg (by decide), innerCandidates425_1_checked]
  change scan edge (search edge 3) 4 [2969] 2969 innerTail425_2
    (if 3 ≤ innerCandidates425_1.length then search edge 3 [2969, 3557] innerCandidates425_1 6526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan425_2

def innerTail425_0 : List Nat := 3539 :: innerTail425_1
def innerCandidates425_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates425_0_checked : candidates edge 4 2969 3539 innerTail425_1 best.bound = innerCandidates425_0 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan425_0 : scan edge (search edge 3) 4 [2969] 2969 innerTail425_0 best = best := by
  rw [innerTail425_0, scan, if_neg (by decide), innerCandidates425_0_checked]
  change scan edge (search edge 3) 4 [2969] 2969 innerTail425_1
    (if 3 ≤ innerCandidates425_0.length then search edge 3 [2969, 3539] innerCandidates425_0 6508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan425_1

theorem rootBranch425 : search edge 4 [2969] rootCandidates425 2969 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2969] 2969 innerTail425_0 best = best
  exact innerScan425_0

def rootCandidates424 : List Nat := [3557, 4019, 4079, 4391, 4421, 4691, 4919, 5381, 5693, 5861, 6449, 6689, 7151, 7187, 7607, 7691, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates424_checked : candidates edge 5 0 2963 tail425 best.bound = rootCandidates424 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext q; rfl
  rw [row]
  decide

def innerTail424_17 : List Nat := []
theorem innerScan424_17 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_17 best = best := rfl

def innerTail424_16 : List Nat := 8111 :: innerTail424_17
def innerCandidates424_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates424_16_checked : candidates edge 4 2963 8111 innerTail424_17 best.bound = innerCandidates424_16 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_16 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_16 best = best := by
  rw [innerTail424_16, scan, if_neg (by decide), innerCandidates424_16_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_17
    (if 3 ≤ innerCandidates424_16.length then search edge 3 [2963, 8111] innerCandidates424_16 11074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_17

def innerTail424_15 : List Nat := 7691 :: innerTail424_16
def innerCandidates424_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates424_15_checked : candidates edge 4 2963 7691 innerTail424_16 best.bound = innerCandidates424_15 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_15 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_15 best = best := by
  rw [innerTail424_15, scan, if_neg (by decide), innerCandidates424_15_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_16
    (if 3 ≤ innerCandidates424_15.length then search edge 3 [2963, 7691] innerCandidates424_15 10654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_16

def innerTail424_14 : List Nat := 7607 :: innerTail424_15
def innerCandidates424_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates424_14_checked : candidates edge 4 2963 7607 innerTail424_15 best.bound = innerCandidates424_14 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_14 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_14 best = best := by
  rw [innerTail424_14, scan, if_neg (by decide), innerCandidates424_14_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_15
    (if 3 ≤ innerCandidates424_14.length then search edge 3 [2963, 7607] innerCandidates424_14 10570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_15

def innerTail424_13 : List Nat := 7187 :: innerTail424_14
def innerCandidates424_13 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates424_13_checked : candidates edge 4 2963 7187 innerTail424_14 best.bound = innerCandidates424_13 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_13 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_13 best = best := by
  rw [innerTail424_13, scan, if_neg (by decide), innerCandidates424_13_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_14
    (if 3 ≤ innerCandidates424_13.length then search edge 3 [2963, 7187] innerCandidates424_13 10150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_14

def innerTail424_12 : List Nat := 7151 :: innerTail424_13
def innerCandidates424_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates424_12_checked : candidates edge 4 2963 7151 innerTail424_13 best.bound = innerCandidates424_12 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_12 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_12 best = best := by
  rw [innerTail424_12, scan, if_neg (by decide), innerCandidates424_12_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_13
    (if 3 ≤ innerCandidates424_12.length then search edge 3 [2963, 7151] innerCandidates424_12 10114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_13

def innerTail424_11 : List Nat := 6689 :: innerTail424_12
def innerCandidates424_11 : List Nat := [7151]
set_option maxRecDepth 4096 in
theorem innerCandidates424_11_checked : candidates edge 4 2963 6689 innerTail424_12 best.bound = innerCandidates424_11 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_11 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_11 best = best := by
  rw [innerTail424_11, scan, if_neg (by decide), innerCandidates424_11_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_12
    (if 3 ≤ innerCandidates424_11.length then search edge 3 [2963, 6689] innerCandidates424_11 9652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_12

def innerTail424_10 : List Nat := 6449 :: innerTail424_11
def innerCandidates424_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates424_10_checked : candidates edge 4 2963 6449 innerTail424_11 best.bound = innerCandidates424_10 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_10 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_10 best = best := by
  rw [innerTail424_10, scan, if_neg (by decide), innerCandidates424_10_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_11
    (if 3 ≤ innerCandidates424_10.length then search edge 3 [2963, 6449] innerCandidates424_10 9412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_11

def innerTail424_9 : List Nat := 5861 :: innerTail424_10
def innerCandidates424_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates424_9_checked : candidates edge 4 2963 5861 innerTail424_10 best.bound = innerCandidates424_9 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_9 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_9 best = best := by
  rw [innerTail424_9, scan, if_neg (by decide), innerCandidates424_9_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_10
    (if 3 ≤ innerCandidates424_9.length then search edge 3 [2963, 5861] innerCandidates424_9 8824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_10

def innerTail424_8 : List Nat := 5693 :: innerTail424_9
def innerCandidates424_8 : List Nat := [7187]
set_option maxRecDepth 4096 in
theorem innerCandidates424_8_checked : candidates edge 4 2963 5693 innerTail424_9 best.bound = innerCandidates424_8 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_8 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_8 best = best := by
  rw [innerTail424_8, scan, if_neg (by decide), innerCandidates424_8_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_9
    (if 3 ≤ innerCandidates424_8.length then search edge 3 [2963, 5693] innerCandidates424_8 8656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_9

def innerTail424_7 : List Nat := 5381 :: innerTail424_8
def innerCandidates424_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates424_7_checked : candidates edge 4 2963 5381 innerTail424_8 best.bound = innerCandidates424_7 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_7 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_7 best = best := by
  rw [innerTail424_7, scan, if_neg (by decide), innerCandidates424_7_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_8
    (if 3 ≤ innerCandidates424_7.length then search edge 3 [2963, 5381] innerCandidates424_7 8344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_8

def innerTail424_6 : List Nat := 4919 :: innerTail424_7
def innerCandidates424_6 : List Nat := [7607]
set_option maxRecDepth 4096 in
theorem innerCandidates424_6_checked : candidates edge 4 2963 4919 innerTail424_7 best.bound = innerCandidates424_6 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_6 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_6 best = best := by
  rw [innerTail424_6, scan, if_neg (by decide), innerCandidates424_6_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_7
    (if 3 ≤ innerCandidates424_6.length then search edge 3 [2963, 4919] innerCandidates424_6 7882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_7

def innerTail424_5 : List Nat := 4691 :: innerTail424_6
def innerCandidates424_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates424_5_checked : candidates edge 4 2963 4691 innerTail424_6 best.bound = innerCandidates424_5 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_5 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_5 best = best := by
  rw [innerTail424_5, scan, if_neg (by decide), innerCandidates424_5_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_6
    (if 3 ≤ innerCandidates424_5.length then search edge 3 [2963, 4691] innerCandidates424_5 7654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_6

def innerTail424_4 : List Nat := 4421 :: innerTail424_5
def innerCandidates424_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates424_4_checked : candidates edge 4 2963 4421 innerTail424_5 best.bound = innerCandidates424_4 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_4 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_4 best = best := by
  rw [innerTail424_4, scan, if_neg (by decide), innerCandidates424_4_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_5
    (if 3 ≤ innerCandidates424_4.length then search edge 3 [2963, 4421] innerCandidates424_4 7384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_5

def innerTail424_3 : List Nat := 4391 :: innerTail424_4
def innerCandidates424_3 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates424_3_checked : candidates edge 4 2963 4391 innerTail424_4 best.bound = innerCandidates424_3 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_3 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_3 best = best := by
  rw [innerTail424_3, scan, if_neg (by decide), innerCandidates424_3_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_4
    (if 3 ≤ innerCandidates424_3.length then search edge 3 [2963, 4391] innerCandidates424_3 7354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_4

def innerTail424_2 : List Nat := 4079 :: innerTail424_3
def innerCandidates424_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates424_2_checked : candidates edge 4 2963 4079 innerTail424_3 best.bound = innerCandidates424_2 := by
  unfold candidates
  have row : edge 4079 = SearchData.row4079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_2 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_2 best = best := by
  rw [innerTail424_2, scan, if_neg (by decide), innerCandidates424_2_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_3
    (if 3 ≤ innerCandidates424_2.length then search edge 3 [2963, 4079] innerCandidates424_2 7042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_3

def innerTail424_1 : List Nat := 4019 :: innerTail424_2
def innerCandidates424_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates424_1_checked : candidates edge 4 2963 4019 innerTail424_2 best.bound = innerCandidates424_1 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_1 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_1 best = best := by
  rw [innerTail424_1, scan, if_neg (by decide), innerCandidates424_1_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_2
    (if 3 ≤ innerCandidates424_1.length then search edge 3 [2963, 4019] innerCandidates424_1 6982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_2

def innerTail424_0 : List Nat := 3557 :: innerTail424_1
def innerCandidates424_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates424_0_checked : candidates edge 4 2963 3557 innerTail424_1 best.bound = innerCandidates424_0 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan424_0 : scan edge (search edge 3) 4 [2963] 2963 innerTail424_0 best = best := by
  rw [innerTail424_0, scan, if_neg (by decide), innerCandidates424_0_checked]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_1
    (if 3 ≤ innerCandidates424_0.length then search edge 3 [2963, 3557] innerCandidates424_0 6520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan424_1

theorem rootBranch424 : search edge 4 [2963] rootCandidates424 2963 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2963] 2963 innerTail424_0 best = best
  exact innerScan424_0

def rootCandidates423 : List Nat := [3041, 3449, 4451, 4973, 6101, 6221, 6803, 7307, 7451, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates423_checked : candidates edge 5 0 2957 tail424 best.bound = rootCandidates423 := by
  unfold candidates
  have row : edge 2957 = SearchData.row2957.contains := by funext q; rfl
  rw [row]
  decide

def innerTail423_10 : List Nat := []
theorem innerScan423_10 : scan edge (search edge 3) 4 [2957] 2957 innerTail423_10 best = best := rfl

def innerTail423_9 : List Nat := 8291 :: innerTail423_10
def innerCandidates423_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates423_9_checked : candidates edge 4 2957 8291 innerTail423_10 best.bound = innerCandidates423_9 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan423_9 : scan edge (search edge 3) 4 [2957] 2957 innerTail423_9 best = best := by
  rw [innerTail423_9, scan, if_neg (by decide), innerCandidates423_9_checked]
  change scan edge (search edge 3) 4 [2957] 2957 innerTail423_10
    (if 3 ≤ innerCandidates423_9.length then search edge 3 [2957, 8291] innerCandidates423_9 11248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan423_10

def innerTail423_8 : List Nat := 7451 :: innerTail423_9
def innerCandidates423_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates423_8_checked : candidates edge 4 2957 7451 innerTail423_9 best.bound = innerCandidates423_8 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan423_8 : scan edge (search edge 3) 4 [2957] 2957 innerTail423_8 best = best := by
  rw [innerTail423_8, scan, if_neg (by decide), innerCandidates423_8_checked]
  change scan edge (search edge 3) 4 [2957] 2957 innerTail423_9
    (if 3 ≤ innerCandidates423_8.length then search edge 3 [2957, 7451] innerCandidates423_8 10408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan423_9

def innerTail423_7 : List Nat := 7307 :: innerTail423_8
def innerCandidates423_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates423_7_checked : candidates edge 4 2957 7307 innerTail423_8 best.bound = innerCandidates423_7 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan423_7 : scan edge (search edge 3) 4 [2957] 2957 innerTail423_7 best = best := by
  rw [innerTail423_7, scan, if_neg (by decide), innerCandidates423_7_checked]
  change scan edge (search edge 3) 4 [2957] 2957 innerTail423_8
    (if 3 ≤ innerCandidates423_7.length then search edge 3 [2957, 7307] innerCandidates423_7 10264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan423_8

def innerTail423_6 : List Nat := 6803 :: innerTail423_7
def innerCandidates423_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates423_6_checked : candidates edge 4 2957 6803 innerTail423_7 best.bound = innerCandidates423_6 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan423_6 : scan edge (search edge 3) 4 [2957] 2957 innerTail423_6 best = best := by
  rw [innerTail423_6, scan, if_neg (by decide), innerCandidates423_6_checked]
  change scan edge (search edge 3) 4 [2957] 2957 innerTail423_7
    (if 3 ≤ innerCandidates423_6.length then search edge 3 [2957, 6803] innerCandidates423_6 9760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan423_7

def innerTail423_5 : List Nat := 6221 :: innerTail423_6
def innerCandidates423_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates423_5_checked : candidates edge 4 2957 6221 innerTail423_6 best.bound = innerCandidates423_5 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan423_5 : scan edge (search edge 3) 4 [2957] 2957 innerTail423_5 best = best := by
  rw [innerTail423_5, scan, if_neg (by decide), innerCandidates423_5_checked]
  change scan edge (search edge 3) 4 [2957] 2957 innerTail423_6
    (if 3 ≤ innerCandidates423_5.length then search edge 3 [2957, 6221] innerCandidates423_5 9178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan423_6

def innerTail423_4 : List Nat := 6101 :: innerTail423_5
def innerCandidates423_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates423_4_checked : candidates edge 4 2957 6101 innerTail423_5 best.bound = innerCandidates423_4 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan423_4 : scan edge (search edge 3) 4 [2957] 2957 innerTail423_4 best = best := by
  rw [innerTail423_4, scan, if_neg (by decide), innerCandidates423_4_checked]
  change scan edge (search edge 3) 4 [2957] 2957 innerTail423_5
    (if 3 ≤ innerCandidates423_4.length then search edge 3 [2957, 6101] innerCandidates423_4 9058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan423_5

def innerTail423_3 : List Nat := 4973 :: innerTail423_4
def innerCandidates423_3 : List Nat := [6221]
set_option maxRecDepth 4096 in
theorem innerCandidates423_3_checked : candidates edge 4 2957 4973 innerTail423_4 best.bound = innerCandidates423_3 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan423_3 : scan edge (search edge 3) 4 [2957] 2957 innerTail423_3 best = best := by
  rw [innerTail423_3, scan, if_neg (by decide), innerCandidates423_3_checked]
  change scan edge (search edge 3) 4 [2957] 2957 innerTail423_4
    (if 3 ≤ innerCandidates423_3.length then search edge 3 [2957, 4973] innerCandidates423_3 7930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan423_4

def innerTail423_2 : List Nat := 4451 :: innerTail423_3
def innerCandidates423_2 : List Nat := [6803, 7307]
set_option maxRecDepth 4096 in
theorem innerCandidates423_2_checked : candidates edge 4 2957 4451 innerTail423_3 best.bound = innerCandidates423_2 := by
  unfold candidates
  have row : edge 4451 = SearchData.row4451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan423_2 : scan edge (search edge 3) 4 [2957] 2957 innerTail423_2 best = best := by
  rw [innerTail423_2, scan, if_neg (by decide), innerCandidates423_2_checked]
  change scan edge (search edge 3) 4 [2957] 2957 innerTail423_3
    (if 3 ≤ innerCandidates423_2.length then search edge 3 [2957, 4451] innerCandidates423_2 7408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan423_3

def innerTail423_1 : List Nat := 3449 :: innerTail423_2
def innerCandidates423_1 : List Nat := [4451]
set_option maxRecDepth 4096 in
theorem innerCandidates423_1_checked : candidates edge 4 2957 3449 innerTail423_2 best.bound = innerCandidates423_1 := by
  unfold candidates
  have row : edge 3449 = SearchData.row3449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan423_1 : scan edge (search edge 3) 4 [2957] 2957 innerTail423_1 best = best := by
  rw [innerTail423_1, scan, if_neg (by decide), innerCandidates423_1_checked]
  change scan edge (search edge 3) 4 [2957] 2957 innerTail423_2
    (if 3 ≤ innerCandidates423_1.length then search edge 3 [2957, 3449] innerCandidates423_1 6406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan423_2

def innerTail423_0 : List Nat := 3041 :: innerTail423_1
def innerCandidates423_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates423_0_checked : candidates edge 4 2957 3041 innerTail423_1 best.bound = innerCandidates423_0 := by
  unfold candidates
  have row : edge 3041 = SearchData.row3041.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan423_0 : scan edge (search edge 3) 4 [2957] 2957 innerTail423_0 best = best := by
  rw [innerTail423_0, scan, if_neg (by decide), innerCandidates423_0_checked]
  change scan edge (search edge 3) 4 [2957] 2957 innerTail423_1
    (if 3 ≤ innerCandidates423_0.length then search edge 3 [2957, 3041] innerCandidates423_0 5998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan423_1

theorem rootBranch423 : search edge 4 [2957] rootCandidates423 2957 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2957] 2957 innerTail423_0 best = best
  exact innerScan423_0

def rootCandidates422 : List Nat := [3181, 3433, 3643, 4273, 4357, 4957, 6217, 6271, 6679, 6709, 6733, 6907, 7213, 7309, 7369, 7537, 7591, 7759, 7867]
set_option maxRecDepth 4096 in
theorem rootCandidates422_checked : candidates edge 5 0 2953 tail423 best.bound = rootCandidates422 := by
  unfold candidates
  have row : edge 2953 = SearchData.row2953.contains := by funext q; rfl
  rw [row]
  decide

def innerTail422_19 : List Nat := []
theorem innerScan422_19 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_19 best = best := rfl

def innerTail422_18 : List Nat := 7867 :: innerTail422_19
def innerCandidates422_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates422_18_checked : candidates edge 4 2953 7867 innerTail422_19 best.bound = innerCandidates422_18 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_18 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_18 best = best := by
  rw [innerTail422_18, scan, if_neg (by decide), innerCandidates422_18_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_19
    (if 3 ≤ innerCandidates422_18.length then search edge 3 [2953, 7867] innerCandidates422_18 10820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_19

def innerTail422_17 : List Nat := 7759 :: innerTail422_18
def innerCandidates422_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates422_17_checked : candidates edge 4 2953 7759 innerTail422_18 best.bound = innerCandidates422_17 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_17 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_17 best = best := by
  rw [innerTail422_17, scan, if_neg (by decide), innerCandidates422_17_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_18
    (if 3 ≤ innerCandidates422_17.length then search edge 3 [2953, 7759] innerCandidates422_17 10712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_18

def innerTail422_16 : List Nat := 7591 :: innerTail422_17
def innerCandidates422_16 : List Nat := [7867]
set_option maxRecDepth 4096 in
theorem innerCandidates422_16_checked : candidates edge 4 2953 7591 innerTail422_17 best.bound = innerCandidates422_16 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_16 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_16 best = best := by
  rw [innerTail422_16, scan, if_neg (by decide), innerCandidates422_16_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_17
    (if 3 ≤ innerCandidates422_16.length then search edge 3 [2953, 7591] innerCandidates422_16 10544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_17

def innerTail422_15 : List Nat := 7537 :: innerTail422_16
def innerCandidates422_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates422_15_checked : candidates edge 4 2953 7537 innerTail422_16 best.bound = innerCandidates422_15 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_15 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_15 best = best := by
  rw [innerTail422_15, scan, if_neg (by decide), innerCandidates422_15_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_16
    (if 3 ≤ innerCandidates422_15.length then search edge 3 [2953, 7537] innerCandidates422_15 10490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_16

def innerTail422_14 : List Nat := 7369 :: innerTail422_15
def innerCandidates422_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates422_14_checked : candidates edge 4 2953 7369 innerTail422_15 best.bound = innerCandidates422_14 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_14 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_14 best = best := by
  rw [innerTail422_14, scan, if_neg (by decide), innerCandidates422_14_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_15
    (if 3 ≤ innerCandidates422_14.length then search edge 3 [2953, 7369] innerCandidates422_14 10322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_15

def innerTail422_13 : List Nat := 7309 :: innerTail422_14
def innerCandidates422_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates422_13_checked : candidates edge 4 2953 7309 innerTail422_14 best.bound = innerCandidates422_13 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_13 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_13 best = best := by
  rw [innerTail422_13, scan, if_neg (by decide), innerCandidates422_13_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_14
    (if 3 ≤ innerCandidates422_13.length then search edge 3 [2953, 7309] innerCandidates422_13 10262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_14

def innerTail422_12 : List Nat := 7213 :: innerTail422_13
def innerCandidates422_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates422_12_checked : candidates edge 4 2953 7213 innerTail422_13 best.bound = innerCandidates422_12 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_12 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_12 best = best := by
  rw [innerTail422_12, scan, if_neg (by decide), innerCandidates422_12_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_13
    (if 3 ≤ innerCandidates422_12.length then search edge 3 [2953, 7213] innerCandidates422_12 10166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_13

def innerTail422_11 : List Nat := 6907 :: innerTail422_12
def innerCandidates422_11 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates422_11_checked : candidates edge 4 2953 6907 innerTail422_12 best.bound = innerCandidates422_11 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_11 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_11 best = best := by
  rw [innerTail422_11, scan, if_neg (by decide), innerCandidates422_11_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_12
    (if 3 ≤ innerCandidates422_11.length then search edge 3 [2953, 6907] innerCandidates422_11 9860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_12

def innerTail422_10 : List Nat := 6733 :: innerTail422_11
def innerCandidates422_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates422_10_checked : candidates edge 4 2953 6733 innerTail422_11 best.bound = innerCandidates422_10 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_10 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_10 best = best := by
  rw [innerTail422_10, scan, if_neg (by decide), innerCandidates422_10_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_11
    (if 3 ≤ innerCandidates422_10.length then search edge 3 [2953, 6733] innerCandidates422_10 9686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_11

def innerTail422_9 : List Nat := 6709 :: innerTail422_10
def innerCandidates422_9 : List Nat := [7369]
set_option maxRecDepth 4096 in
theorem innerCandidates422_9_checked : candidates edge 4 2953 6709 innerTail422_10 best.bound = innerCandidates422_9 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_9 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_9 best = best := by
  rw [innerTail422_9, scan, if_neg (by decide), innerCandidates422_9_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_10
    (if 3 ≤ innerCandidates422_9.length then search edge 3 [2953, 6709] innerCandidates422_9 9662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_10

def innerTail422_8 : List Nat := 6679 :: innerTail422_9
def innerCandidates422_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates422_8_checked : candidates edge 4 2953 6679 innerTail422_9 best.bound = innerCandidates422_8 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_8 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_8 best = best := by
  rw [innerTail422_8, scan, if_neg (by decide), innerCandidates422_8_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_9
    (if 3 ≤ innerCandidates422_8.length then search edge 3 [2953, 6679] innerCandidates422_8 9632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_9

def innerTail422_7 : List Nat := 6271 :: innerTail422_8
def innerCandidates422_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates422_7_checked : candidates edge 4 2953 6271 innerTail422_8 best.bound = innerCandidates422_7 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_7 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_7 best = best := by
  rw [innerTail422_7, scan, if_neg (by decide), innerCandidates422_7_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_8
    (if 3 ≤ innerCandidates422_7.length then search edge 3 [2953, 6271] innerCandidates422_7 9224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_8

def innerTail422_6 : List Nat := 6217 :: innerTail422_7
def innerCandidates422_6 : List Nat := [6679]
set_option maxRecDepth 4096 in
theorem innerCandidates422_6_checked : candidates edge 4 2953 6217 innerTail422_7 best.bound = innerCandidates422_6 := by
  unfold candidates
  have row : edge 6217 = SearchData.row6217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_6 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_6 best = best := by
  rw [innerTail422_6, scan, if_neg (by decide), innerCandidates422_6_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_7
    (if 3 ≤ innerCandidates422_6.length then search edge 3 [2953, 6217] innerCandidates422_6 9170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_7

def innerTail422_5 : List Nat := 4957 :: innerTail422_6
def innerCandidates422_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates422_5_checked : candidates edge 4 2953 4957 innerTail422_6 best.bound = innerCandidates422_5 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_5 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_5 best = best := by
  rw [innerTail422_5, scan, if_neg (by decide), innerCandidates422_5_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_6
    (if 3 ≤ innerCandidates422_5.length then search edge 3 [2953, 4957] innerCandidates422_5 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_6

def innerTail422_4 : List Nat := 4357 :: innerTail422_5
def innerCandidates422_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates422_4_checked : candidates edge 4 2953 4357 innerTail422_5 best.bound = innerCandidates422_4 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_4 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_4 best = best := by
  rw [innerTail422_4, scan, if_neg (by decide), innerCandidates422_4_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_5
    (if 3 ≤ innerCandidates422_4.length then search edge 3 [2953, 4357] innerCandidates422_4 7310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_5

def innerTail422_3 : List Nat := 4273 :: innerTail422_4
def innerCandidates422_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates422_3_checked : candidates edge 4 2953 4273 innerTail422_4 best.bound = innerCandidates422_3 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_3 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_3 best = best := by
  rw [innerTail422_3, scan, if_neg (by decide), innerCandidates422_3_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_4
    (if 3 ≤ innerCandidates422_3.length then search edge 3 [2953, 4273] innerCandidates422_3 7226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_4

def innerTail422_2 : List Nat := 3643 :: innerTail422_3
def innerCandidates422_2 : List Nat := [4273]
set_option maxRecDepth 4096 in
theorem innerCandidates422_2_checked : candidates edge 4 2953 3643 innerTail422_3 best.bound = innerCandidates422_2 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_2 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_2 best = best := by
  rw [innerTail422_2, scan, if_neg (by decide), innerCandidates422_2_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_3
    (if 3 ≤ innerCandidates422_2.length then search edge 3 [2953, 3643] innerCandidates422_2 6596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_3

def innerTail422_1 : List Nat := 3433 :: innerTail422_2
def innerCandidates422_1 : List Nat := [3643, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates422_1_checked : candidates edge 4 2953 3433 innerTail422_2 best.bound = innerCandidates422_1 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_1 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_1 best = best := by
  rw [innerTail422_1, scan, if_neg (by decide), innerCandidates422_1_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_2
    (if 3 ≤ innerCandidates422_1.length then search edge 3 [2953, 3433] innerCandidates422_1 6386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_2

def innerTail422_0 : List Nat := 3181 :: innerTail422_1
def innerCandidates422_0 : List Nat := [4273, 6709]
set_option maxRecDepth 4096 in
theorem innerCandidates422_0_checked : candidates edge 4 2953 3181 innerTail422_1 best.bound = innerCandidates422_0 := by
  unfold candidates
  have row : edge 3181 = SearchData.row3181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan422_0 : scan edge (search edge 3) 4 [2953] 2953 innerTail422_0 best = best := by
  rw [innerTail422_0, scan, if_neg (by decide), innerCandidates422_0_checked]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_1
    (if 3 ≤ innerCandidates422_0.length then search edge 3 [2953, 3181] innerCandidates422_0 6134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan422_1

theorem rootBranch422 : search edge 4 [2953] rootCandidates422 2953 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2953] 2953 innerTail422_0 best = best
  exact innerScan422_0

end ProjectEulerSolutions.PrimePairUpper.Search
