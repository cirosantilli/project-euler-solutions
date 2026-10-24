import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates386 : List Nat := [2791, 2797, 3253, 3823, 3931, 4447, 5023, 5209, 5563, 6451, 6619, 8059, 8167, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates386_checked : candidates edge 5 0 2683 tail387 best.bound = rootCandidates386 := by
  unfold candidates
  have row : edge 2683 = SearchData.row2683.contains := by funext q; rfl
  rw [row]
  decide

def innerTail386_14 : List Nat := []
theorem innerScan386_14 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_14 best = best := rfl

def innerTail386_13 : List Nat := 8293 :: innerTail386_14
def innerCandidates386_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates386_13_checked : candidates edge 4 2683 8293 innerTail386_14 best.bound = innerCandidates386_13 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_13 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_13 best = best := by
  rw [innerTail386_13, scan, if_neg (by decide), innerCandidates386_13_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_14
    (if 3 ≤ innerCandidates386_13.length then search edge 3 [2683, 8293] innerCandidates386_13 10976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_14

def innerTail386_12 : List Nat := 8167 :: innerTail386_13
def innerCandidates386_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates386_12_checked : candidates edge 4 2683 8167 innerTail386_13 best.bound = innerCandidates386_12 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_12 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_12 best = best := by
  rw [innerTail386_12, scan, if_neg (by decide), innerCandidates386_12_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_13
    (if 3 ≤ innerCandidates386_12.length then search edge 3 [2683, 8167] innerCandidates386_12 10850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_13

def innerTail386_11 : List Nat := 8059 :: innerTail386_12
def innerCandidates386_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates386_11_checked : candidates edge 4 2683 8059 innerTail386_12 best.bound = innerCandidates386_11 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_11 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_11 best = best := by
  rw [innerTail386_11, scan, if_neg (by decide), innerCandidates386_11_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_12
    (if 3 ≤ innerCandidates386_11.length then search edge 3 [2683, 8059] innerCandidates386_11 10742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_12

def innerTail386_10 : List Nat := 6619 :: innerTail386_11
def innerCandidates386_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates386_10_checked : candidates edge 4 2683 6619 innerTail386_11 best.bound = innerCandidates386_10 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_10 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_10 best = best := by
  rw [innerTail386_10, scan, if_neg (by decide), innerCandidates386_10_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_11
    (if 3 ≤ innerCandidates386_10.length then search edge 3 [2683, 6619] innerCandidates386_10 9302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_11

def innerTail386_9 : List Nat := 6451 :: innerTail386_10
def innerCandidates386_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates386_9_checked : candidates edge 4 2683 6451 innerTail386_10 best.bound = innerCandidates386_9 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_9 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_9 best = best := by
  rw [innerTail386_9, scan, if_neg (by decide), innerCandidates386_9_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_10
    (if 3 ≤ innerCandidates386_9.length then search edge 3 [2683, 6451] innerCandidates386_9 9134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_10

def innerTail386_8 : List Nat := 5563 :: innerTail386_9
def innerCandidates386_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates386_8_checked : candidates edge 4 2683 5563 innerTail386_9 best.bound = innerCandidates386_8 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_8 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_8 best = best := by
  rw [innerTail386_8, scan, if_neg (by decide), innerCandidates386_8_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_9
    (if 3 ≤ innerCandidates386_8.length then search edge 3 [2683, 5563] innerCandidates386_8 8246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_9

def innerTail386_7 : List Nat := 5209 :: innerTail386_8
def innerCandidates386_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates386_7_checked : candidates edge 4 2683 5209 innerTail386_8 best.bound = innerCandidates386_7 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_7 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_7 best = best := by
  rw [innerTail386_7, scan, if_neg (by decide), innerCandidates386_7_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_8
    (if 3 ≤ innerCandidates386_7.length then search edge 3 [2683, 5209] innerCandidates386_7 7892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_8

def innerTail386_6 : List Nat := 5023 :: innerTail386_7
def innerCandidates386_6 : List Nat := [8059]
set_option maxRecDepth 4096 in
theorem innerCandidates386_6_checked : candidates edge 4 2683 5023 innerTail386_7 best.bound = innerCandidates386_6 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_6 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_6 best = best := by
  rw [innerTail386_6, scan, if_neg (by decide), innerCandidates386_6_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_7
    (if 3 ≤ innerCandidates386_6.length then search edge 3 [2683, 5023] innerCandidates386_6 7706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_7

def innerTail386_5 : List Nat := 4447 :: innerTail386_6
def innerCandidates386_5 : List Nat := [5023, 8293]
set_option maxRecDepth 4096 in
theorem innerCandidates386_5_checked : candidates edge 4 2683 4447 innerTail386_6 best.bound = innerCandidates386_5 := by
  unfold candidates
  have row : edge 4447 = SearchData.row4447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_5 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_5 best = best := by
  rw [innerTail386_5, scan, if_neg (by decide), innerCandidates386_5_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_6
    (if 3 ≤ innerCandidates386_5.length then search edge 3 [2683, 4447] innerCandidates386_5 7130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_6

def innerTail386_4 : List Nat := 3931 :: innerTail386_5
def innerCandidates386_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates386_4_checked : candidates edge 4 2683 3931 innerTail386_5 best.bound = innerCandidates386_4 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_4 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_4 best = best := by
  rw [innerTail386_4, scan, if_neg (by decide), innerCandidates386_4_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_5
    (if 3 ≤ innerCandidates386_4.length then search edge 3 [2683, 3931] innerCandidates386_4 6614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_5

def innerTail386_3 : List Nat := 3823 :: innerTail386_4
def innerCandidates386_3 : List Nat := [5209]
set_option maxRecDepth 4096 in
theorem innerCandidates386_3_checked : candidates edge 4 2683 3823 innerTail386_4 best.bound = innerCandidates386_3 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_3 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_3 best = best := by
  rw [innerTail386_3, scan, if_neg (by decide), innerCandidates386_3_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_4
    (if 3 ≤ innerCandidates386_3.length then search edge 3 [2683, 3823] innerCandidates386_3 6506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_4

def innerTail386_2 : List Nat := 3253 :: innerTail386_3
def innerCandidates386_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates386_2_checked : candidates edge 4 2683 3253 innerTail386_3 best.bound = innerCandidates386_2 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_2 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_2 best = best := by
  rw [innerTail386_2, scan, if_neg (by decide), innerCandidates386_2_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_3
    (if 3 ≤ innerCandidates386_2.length then search edge 3 [2683, 3253] innerCandidates386_2 5936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_3

def innerTail386_1 : List Nat := 2797 :: innerTail386_2
def innerCandidates386_1 : List Nat := [5023]
set_option maxRecDepth 4096 in
theorem innerCandidates386_1_checked : candidates edge 4 2683 2797 innerTail386_2 best.bound = innerCandidates386_1 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_1 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_1 best = best := by
  rw [innerTail386_1, scan, if_neg (by decide), innerCandidates386_1_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_2
    (if 3 ≤ innerCandidates386_1.length then search edge 3 [2683, 2797] innerCandidates386_1 5480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_2

def innerTail386_0 : List Nat := 2791 :: innerTail386_1
def innerCandidates386_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates386_0_checked : candidates edge 4 2683 2791 innerTail386_1 best.bound = innerCandidates386_0 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan386_0 : scan edge (search edge 3) 4 [2683] 2683 innerTail386_0 best = best := by
  rw [innerTail386_0, scan, if_neg (by decide), innerCandidates386_0_checked]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_1
    (if 3 ≤ innerCandidates386_0.length then search edge 3 [2683, 2791] innerCandidates386_0 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan386_1

theorem rootBranch386 : search edge 4 [2683] rootCandidates386 2683 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2683] 2683 innerTail386_0 best = best
  exact innerScan386_0

def rootCandidates385 : List Nat := [2833, 2971, 3049, 3469, 4261, 5563, 5569, 5737, 5851, 6247, 6961, 7369, 7741]
set_option maxRecDepth 4096 in
theorem rootCandidates385_checked : candidates edge 5 0 2677 tail386 best.bound = rootCandidates385 := by
  unfold candidates
  have row : edge 2677 = SearchData.row2677.contains := by funext q; rfl
  rw [row]
  decide

def innerTail385_13 : List Nat := []
theorem innerScan385_13 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_13 best = best := rfl

def innerTail385_12 : List Nat := 7741 :: innerTail385_13
def innerCandidates385_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates385_12_checked : candidates edge 4 2677 7741 innerTail385_13 best.bound = innerCandidates385_12 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_12 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_12 best = best := by
  rw [innerTail385_12, scan, if_neg (by decide), innerCandidates385_12_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_13
    (if 3 ≤ innerCandidates385_12.length then search edge 3 [2677, 7741] innerCandidates385_12 10418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_13

def innerTail385_11 : List Nat := 7369 :: innerTail385_12
def innerCandidates385_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates385_11_checked : candidates edge 4 2677 7369 innerTail385_12 best.bound = innerCandidates385_11 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_11 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_11 best = best := by
  rw [innerTail385_11, scan, if_neg (by decide), innerCandidates385_11_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_12
    (if 3 ≤ innerCandidates385_11.length then search edge 3 [2677, 7369] innerCandidates385_11 10046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_12

def innerTail385_10 : List Nat := 6961 :: innerTail385_11
def innerCandidates385_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates385_10_checked : candidates edge 4 2677 6961 innerTail385_11 best.bound = innerCandidates385_10 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_10 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_10 best = best := by
  rw [innerTail385_10, scan, if_neg (by decide), innerCandidates385_10_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_11
    (if 3 ≤ innerCandidates385_10.length then search edge 3 [2677, 6961] innerCandidates385_10 9638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_11

def innerTail385_9 : List Nat := 6247 :: innerTail385_10
def innerCandidates385_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates385_9_checked : candidates edge 4 2677 6247 innerTail385_10 best.bound = innerCandidates385_9 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_9 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_9 best = best := by
  rw [innerTail385_9, scan, if_neg (by decide), innerCandidates385_9_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_10
    (if 3 ≤ innerCandidates385_9.length then search edge 3 [2677, 6247] innerCandidates385_9 8924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_10

def innerTail385_8 : List Nat := 5851 :: innerTail385_9
def innerCandidates385_8 : List Nat := [6961]
set_option maxRecDepth 4096 in
theorem innerCandidates385_8_checked : candidates edge 4 2677 5851 innerTail385_9 best.bound = innerCandidates385_8 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_8 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_8 best = best := by
  rw [innerTail385_8, scan, if_neg (by decide), innerCandidates385_8_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_9
    (if 3 ≤ innerCandidates385_8.length then search edge 3 [2677, 5851] innerCandidates385_8 8528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_9

def innerTail385_7 : List Nat := 5737 :: innerTail385_8
def innerCandidates385_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates385_7_checked : candidates edge 4 2677 5737 innerTail385_8 best.bound = innerCandidates385_7 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_7 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_7 best = best := by
  rw [innerTail385_7, scan, if_neg (by decide), innerCandidates385_7_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_8
    (if 3 ≤ innerCandidates385_7.length then search edge 3 [2677, 5737] innerCandidates385_7 8414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_8

def innerTail385_6 : List Nat := 5569 :: innerTail385_7
def innerCandidates385_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates385_6_checked : candidates edge 4 2677 5569 innerTail385_7 best.bound = innerCandidates385_6 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_6 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_6 best = best := by
  rw [innerTail385_6, scan, if_neg (by decide), innerCandidates385_6_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_7
    (if 3 ≤ innerCandidates385_6.length then search edge 3 [2677, 5569] innerCandidates385_6 8246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_7

def innerTail385_5 : List Nat := 5563 :: innerTail385_6
def innerCandidates385_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates385_5_checked : candidates edge 4 2677 5563 innerTail385_6 best.bound = innerCandidates385_5 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_5 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_5 best = best := by
  rw [innerTail385_5, scan, if_neg (by decide), innerCandidates385_5_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_6
    (if 3 ≤ innerCandidates385_5.length then search edge 3 [2677, 5563] innerCandidates385_5 8240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_6

def innerTail385_4 : List Nat := 4261 :: innerTail385_5
def innerCandidates385_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates385_4_checked : candidates edge 4 2677 4261 innerTail385_5 best.bound = innerCandidates385_4 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_4 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_4 best = best := by
  rw [innerTail385_4, scan, if_neg (by decide), innerCandidates385_4_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_5
    (if 3 ≤ innerCandidates385_4.length then search edge 3 [2677, 4261] innerCandidates385_4 6938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_5

def innerTail385_3 : List Nat := 3469 :: innerTail385_4
def innerCandidates385_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates385_3_checked : candidates edge 4 2677 3469 innerTail385_4 best.bound = innerCandidates385_3 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_3 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_3 best = best := by
  rw [innerTail385_3, scan, if_neg (by decide), innerCandidates385_3_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_4
    (if 3 ≤ innerCandidates385_3.length then search edge 3 [2677, 3469] innerCandidates385_3 6146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_4

def innerTail385_2 : List Nat := 3049 :: innerTail385_3
def innerCandidates385_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates385_2_checked : candidates edge 4 2677 3049 innerTail385_3 best.bound = innerCandidates385_2 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_2 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_2 best = best := by
  rw [innerTail385_2, scan, if_neg (by decide), innerCandidates385_2_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_3
    (if 3 ≤ innerCandidates385_2.length then search edge 3 [2677, 3049] innerCandidates385_2 5726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_3

def innerTail385_1 : List Nat := 2971 :: innerTail385_2
def innerCandidates385_1 : List Nat := [6961]
set_option maxRecDepth 4096 in
theorem innerCandidates385_1_checked : candidates edge 4 2677 2971 innerTail385_2 best.bound = innerCandidates385_1 := by
  unfold candidates
  have row : edge 2971 = SearchData.row2971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_1 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_1 best = best := by
  rw [innerTail385_1, scan, if_neg (by decide), innerCandidates385_1_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_2
    (if 3 ≤ innerCandidates385_1.length then search edge 3 [2677, 2971] innerCandidates385_1 5648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_2

def innerTail385_0 : List Nat := 2833 :: innerTail385_1
def innerCandidates385_0 : List Nat := [5851]
set_option maxRecDepth 4096 in
theorem innerCandidates385_0_checked : candidates edge 4 2677 2833 innerTail385_1 best.bound = innerCandidates385_0 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan385_0 : scan edge (search edge 3) 4 [2677] 2677 innerTail385_0 best = best := by
  rw [innerTail385_0, scan, if_neg (by decide), innerCandidates385_0_checked]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_1
    (if 3 ≤ innerCandidates385_0.length then search edge 3 [2677, 2833] innerCandidates385_0 5510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan385_1

theorem rootBranch385 : search edge 4 [2677] rootCandidates385 2677 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2677] 2677 innerTail385_0 best = best
  exact innerScan385_0

def rootCandidates384 : List Nat := [3361, 3571, 3769, 3853, 4231, 4603, 5077, 6211, 6343, 6709, 6829, 7129, 7507, 7723, 8053]
set_option maxRecDepth 4096 in
theorem rootCandidates384_checked : candidates edge 5 0 2671 tail385 best.bound = rootCandidates384 := by
  unfold candidates
  have row : edge 2671 = SearchData.row2671.contains := by funext q; rfl
  rw [row]
  decide

def innerTail384_15 : List Nat := []
theorem innerScan384_15 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_15 best = best := rfl

def innerTail384_14 : List Nat := 8053 :: innerTail384_15
def innerCandidates384_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates384_14_checked : candidates edge 4 2671 8053 innerTail384_15 best.bound = innerCandidates384_14 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_14 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_14 best = best := by
  rw [innerTail384_14, scan, if_neg (by decide), innerCandidates384_14_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_15
    (if 3 ≤ innerCandidates384_14.length then search edge 3 [2671, 8053] innerCandidates384_14 10724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_15

def innerTail384_13 : List Nat := 7723 :: innerTail384_14
def innerCandidates384_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates384_13_checked : candidates edge 4 2671 7723 innerTail384_14 best.bound = innerCandidates384_13 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_13 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_13 best = best := by
  rw [innerTail384_13, scan, if_neg (by decide), innerCandidates384_13_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_14
    (if 3 ≤ innerCandidates384_13.length then search edge 3 [2671, 7723] innerCandidates384_13 10394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_14

def innerTail384_12 : List Nat := 7507 :: innerTail384_13
def innerCandidates384_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates384_12_checked : candidates edge 4 2671 7507 innerTail384_13 best.bound = innerCandidates384_12 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_12 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_12 best = best := by
  rw [innerTail384_12, scan, if_neg (by decide), innerCandidates384_12_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_13
    (if 3 ≤ innerCandidates384_12.length then search edge 3 [2671, 7507] innerCandidates384_12 10178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_13

def innerTail384_11 : List Nat := 7129 :: innerTail384_12
def innerCandidates384_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates384_11_checked : candidates edge 4 2671 7129 innerTail384_12 best.bound = innerCandidates384_11 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_11 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_11 best = best := by
  rw [innerTail384_11, scan, if_neg (by decide), innerCandidates384_11_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_12
    (if 3 ≤ innerCandidates384_11.length then search edge 3 [2671, 7129] innerCandidates384_11 9800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_12

def innerTail384_10 : List Nat := 6829 :: innerTail384_11
def innerCandidates384_10 : List Nat := [7507]
set_option maxRecDepth 4096 in
theorem innerCandidates384_10_checked : candidates edge 4 2671 6829 innerTail384_11 best.bound = innerCandidates384_10 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_10 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_10 best = best := by
  rw [innerTail384_10, scan, if_neg (by decide), innerCandidates384_10_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_11
    (if 3 ≤ innerCandidates384_10.length then search edge 3 [2671, 6829] innerCandidates384_10 9500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_11

def innerTail384_9 : List Nat := 6709 :: innerTail384_10
def innerCandidates384_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates384_9_checked : candidates edge 4 2671 6709 innerTail384_10 best.bound = innerCandidates384_9 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_9 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_9 best = best := by
  rw [innerTail384_9, scan, if_neg (by decide), innerCandidates384_9_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_10
    (if 3 ≤ innerCandidates384_9.length then search edge 3 [2671, 6709] innerCandidates384_9 9380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_10

def innerTail384_8 : List Nat := 6343 :: innerTail384_9
def innerCandidates384_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates384_8_checked : candidates edge 4 2671 6343 innerTail384_9 best.bound = innerCandidates384_8 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_8 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_8 best = best := by
  rw [innerTail384_8, scan, if_neg (by decide), innerCandidates384_8_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_9
    (if 3 ≤ innerCandidates384_8.length then search edge 3 [2671, 6343] innerCandidates384_8 9014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_9

def innerTail384_7 : List Nat := 6211 :: innerTail384_8
def innerCandidates384_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates384_7_checked : candidates edge 4 2671 6211 innerTail384_8 best.bound = innerCandidates384_7 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_7 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_7 best = best := by
  rw [innerTail384_7, scan, if_neg (by decide), innerCandidates384_7_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_8
    (if 3 ≤ innerCandidates384_7.length then search edge 3 [2671, 6211] innerCandidates384_7 8882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_8

def innerTail384_6 : List Nat := 5077 :: innerTail384_7
def innerCandidates384_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates384_6_checked : candidates edge 4 2671 5077 innerTail384_7 best.bound = innerCandidates384_6 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_6 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_6 best = best := by
  rw [innerTail384_6, scan, if_neg (by decide), innerCandidates384_6_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_7
    (if 3 ≤ innerCandidates384_6.length then search edge 3 [2671, 5077] innerCandidates384_6 7748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_7

def innerTail384_5 : List Nat := 4603 :: innerTail384_6
def innerCandidates384_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates384_5_checked : candidates edge 4 2671 4603 innerTail384_6 best.bound = innerCandidates384_5 := by
  unfold candidates
  have row : edge 4603 = SearchData.row4603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_5 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_5 best = best := by
  rw [innerTail384_5, scan, if_neg (by decide), innerCandidates384_5_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_6
    (if 3 ≤ innerCandidates384_5.length then search edge 3 [2671, 4603] innerCandidates384_5 7274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_6

def innerTail384_4 : List Nat := 4231 :: innerTail384_5
def innerCandidates384_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates384_4_checked : candidates edge 4 2671 4231 innerTail384_5 best.bound = innerCandidates384_4 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_4 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_4 best = best := by
  rw [innerTail384_4, scan, if_neg (by decide), innerCandidates384_4_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_5
    (if 3 ≤ innerCandidates384_4.length then search edge 3 [2671, 4231] innerCandidates384_4 6902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_5

def innerTail384_3 : List Nat := 3853 :: innerTail384_4
def innerCandidates384_3 : List Nat := [6709]
set_option maxRecDepth 4096 in
theorem innerCandidates384_3_checked : candidates edge 4 2671 3853 innerTail384_4 best.bound = innerCandidates384_3 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_3 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_3 best = best := by
  rw [innerTail384_3, scan, if_neg (by decide), innerCandidates384_3_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_4
    (if 3 ≤ innerCandidates384_3.length then search edge 3 [2671, 3853] innerCandidates384_3 6524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_4

def innerTail384_2 : List Nat := 3769 :: innerTail384_3
def innerCandidates384_2 : List Nat := [4603]
set_option maxRecDepth 4096 in
theorem innerCandidates384_2_checked : candidates edge 4 2671 3769 innerTail384_3 best.bound = innerCandidates384_2 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_2 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_2 best = best := by
  rw [innerTail384_2, scan, if_neg (by decide), innerCandidates384_2_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_3
    (if 3 ≤ innerCandidates384_2.length then search edge 3 [2671, 3769] innerCandidates384_2 6440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_3

def innerTail384_1 : List Nat := 3571 :: innerTail384_2
def innerCandidates384_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates384_1_checked : candidates edge 4 2671 3571 innerTail384_2 best.bound = innerCandidates384_1 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_1 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_1 best = best := by
  rw [innerTail384_1, scan, if_neg (by decide), innerCandidates384_1_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_2
    (if 3 ≤ innerCandidates384_1.length then search edge 3 [2671, 3571] innerCandidates384_1 6242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_2

def innerTail384_0 : List Nat := 3361 :: innerTail384_1
def innerCandidates384_0 : List Nat := [6343]
set_option maxRecDepth 4096 in
theorem innerCandidates384_0_checked : candidates edge 4 2671 3361 innerTail384_1 best.bound = innerCandidates384_0 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan384_0 : scan edge (search edge 3) 4 [2671] 2671 innerTail384_0 best = best := by
  rw [innerTail384_0, scan, if_neg (by decide), innerCandidates384_0_checked]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_1
    (if 3 ≤ innerCandidates384_0.length then search edge 3 [2671, 3361] innerCandidates384_0 6032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan384_1

theorem rootBranch384 : search edge 4 [2671] rootCandidates384 2671 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2671] 2671 innerTail384_0 best = best
  exact innerScan384_0

def rootCandidates383 : List Nat := [3119, 3251, 3533, 3779, 4457, 5471, 5669, 5813, 5849, 6359, 7607, 7901, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates383_checked : candidates edge 5 0 2663 tail384 best.bound = rootCandidates383 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext q; rfl
  rw [row]
  decide

def innerTail383_13 : List Nat := []
theorem innerScan383_13 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_13 best = best := rfl

def innerTail383_12 : List Nat := 8291 :: innerTail383_13
def innerCandidates383_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates383_12_checked : candidates edge 4 2663 8291 innerTail383_13 best.bound = innerCandidates383_12 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_12 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_12 best = best := by
  rw [innerTail383_12, scan, if_neg (by decide), innerCandidates383_12_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_13
    (if 3 ≤ innerCandidates383_12.length then search edge 3 [2663, 8291] innerCandidates383_12 10954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_13

def innerTail383_11 : List Nat := 7901 :: innerTail383_12
def innerCandidates383_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates383_11_checked : candidates edge 4 2663 7901 innerTail383_12 best.bound = innerCandidates383_11 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_11 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_11 best = best := by
  rw [innerTail383_11, scan, if_neg (by decide), innerCandidates383_11_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_12
    (if 3 ≤ innerCandidates383_11.length then search edge 3 [2663, 7901] innerCandidates383_11 10564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_12

def innerTail383_10 : List Nat := 7607 :: innerTail383_11
def innerCandidates383_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates383_10_checked : candidates edge 4 2663 7607 innerTail383_11 best.bound = innerCandidates383_10 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_10 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_10 best = best := by
  rw [innerTail383_10, scan, if_neg (by decide), innerCandidates383_10_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_11
    (if 3 ≤ innerCandidates383_10.length then search edge 3 [2663, 7607] innerCandidates383_10 10270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_11

def innerTail383_9 : List Nat := 6359 :: innerTail383_10
def innerCandidates383_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates383_9_checked : candidates edge 4 2663 6359 innerTail383_10 best.bound = innerCandidates383_9 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_9 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_9 best = best := by
  rw [innerTail383_9, scan, if_neg (by decide), innerCandidates383_9_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_10
    (if 3 ≤ innerCandidates383_9.length then search edge 3 [2663, 6359] innerCandidates383_9 9022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_10

def innerTail383_8 : List Nat := 5849 :: innerTail383_9
def innerCandidates383_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates383_8_checked : candidates edge 4 2663 5849 innerTail383_9 best.bound = innerCandidates383_8 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_8 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_8 best = best := by
  rw [innerTail383_8, scan, if_neg (by decide), innerCandidates383_8_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_9
    (if 3 ≤ innerCandidates383_8.length then search edge 3 [2663, 5849] innerCandidates383_8 8512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_9

def innerTail383_7 : List Nat := 5813 :: innerTail383_8
def innerCandidates383_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates383_7_checked : candidates edge 4 2663 5813 innerTail383_8 best.bound = innerCandidates383_7 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_7 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_7 best = best := by
  rw [innerTail383_7, scan, if_neg (by decide), innerCandidates383_7_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_8
    (if 3 ≤ innerCandidates383_7.length then search edge 3 [2663, 5813] innerCandidates383_7 8476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_8

def innerTail383_6 : List Nat := 5669 :: innerTail383_7
def innerCandidates383_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates383_6_checked : candidates edge 4 2663 5669 innerTail383_7 best.bound = innerCandidates383_6 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_6 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_6 best = best := by
  rw [innerTail383_6, scan, if_neg (by decide), innerCandidates383_6_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_7
    (if 3 ≤ innerCandidates383_6.length then search edge 3 [2663, 5669] innerCandidates383_6 8332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_7

def innerTail383_5 : List Nat := 5471 :: innerTail383_6
def innerCandidates383_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates383_5_checked : candidates edge 4 2663 5471 innerTail383_6 best.bound = innerCandidates383_5 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_5 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_5 best = best := by
  rw [innerTail383_5, scan, if_neg (by decide), innerCandidates383_5_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_6
    (if 3 ≤ innerCandidates383_5.length then search edge 3 [2663, 5471] innerCandidates383_5 8134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_6

def innerTail383_4 : List Nat := 4457 :: innerTail383_5
def innerCandidates383_4 : List Nat := [7607]
set_option maxRecDepth 4096 in
theorem innerCandidates383_4_checked : candidates edge 4 2663 4457 innerTail383_5 best.bound = innerCandidates383_4 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_4 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_4 best = best := by
  rw [innerTail383_4, scan, if_neg (by decide), innerCandidates383_4_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_5
    (if 3 ≤ innerCandidates383_4.length then search edge 3 [2663, 4457] innerCandidates383_4 7120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_5

def innerTail383_3 : List Nat := 3779 :: innerTail383_4
def innerCandidates383_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates383_3_checked : candidates edge 4 2663 3779 innerTail383_4 best.bound = innerCandidates383_3 := by
  unfold candidates
  have row : edge 3779 = SearchData.row3779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_3 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_3 best = best := by
  rw [innerTail383_3, scan, if_neg (by decide), innerCandidates383_3_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_4
    (if 3 ≤ innerCandidates383_3.length then search edge 3 [2663, 3779] innerCandidates383_3 6442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_4

def innerTail383_2 : List Nat := 3533 :: innerTail383_3
def innerCandidates383_2 : List Nat := [7607]
set_option maxRecDepth 4096 in
theorem innerCandidates383_2_checked : candidates edge 4 2663 3533 innerTail383_3 best.bound = innerCandidates383_2 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_2 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_2 best = best := by
  rw [innerTail383_2, scan, if_neg (by decide), innerCandidates383_2_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_3
    (if 3 ≤ innerCandidates383_2.length then search edge 3 [2663, 3533] innerCandidates383_2 6196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_3

def innerTail383_1 : List Nat := 3251 :: innerTail383_2
def innerCandidates383_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates383_1_checked : candidates edge 4 2663 3251 innerTail383_2 best.bound = innerCandidates383_1 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_1 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_1 best = best := by
  rw [innerTail383_1, scan, if_neg (by decide), innerCandidates383_1_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_2
    (if 3 ≤ innerCandidates383_1.length then search edge 3 [2663, 3251] innerCandidates383_1 5914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_2

def innerTail383_0 : List Nat := 3119 :: innerTail383_1
def innerCandidates383_0 : List Nat := [6359]
set_option maxRecDepth 4096 in
theorem innerCandidates383_0_checked : candidates edge 4 2663 3119 innerTail383_1 best.bound = innerCandidates383_0 := by
  unfold candidates
  have row : edge 3119 = SearchData.row3119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan383_0 : scan edge (search edge 3) 4 [2663] 2663 innerTail383_0 best = best := by
  rw [innerTail383_0, scan, if_neg (by decide), innerCandidates383_0_checked]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_1
    (if 3 ≤ innerCandidates383_0.length then search edge 3 [2663, 3119] innerCandidates383_0 5782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan383_1

theorem rootBranch383 : search edge 4 [2663] rootCandidates383 2663 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2663] 2663 innerTail383_0 best = best
  exact innerScan383_0

end ProjectEulerSolutions.PrimePairUpper.Search
