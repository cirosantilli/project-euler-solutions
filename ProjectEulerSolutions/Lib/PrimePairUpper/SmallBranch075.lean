import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates501 : List Nat := [4099, 5023, 5077, 6367, 6379, 6571, 6577, 7351, 7669, 8089, 8101]
set_option maxRecDepth 4096 in
theorem rootCandidates501_checked : candidates edge 5 0 3607 tail502 best.bound = rootCandidates501 := by
  unfold candidates
  have row : edge 3607 = SearchData.row3607.contains := by funext q; rfl
  rw [row]
  decide

def innerTail501_11 : List Nat := []
theorem innerScan501_11 : scan edge (search edge 3) 4 [3607] 3607 innerTail501_11 best = best := rfl

def innerTail501_10 : List Nat := 8101 :: innerTail501_11
def innerCandidates501_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates501_10_checked : candidates edge 4 3607 8101 innerTail501_11 best.bound = innerCandidates501_10 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan501_10 : scan edge (search edge 3) 4 [3607] 3607 innerTail501_10 best = best := by
  rw [innerTail501_10, scan, if_neg (by decide), innerCandidates501_10_checked]
  change scan edge (search edge 3) 4 [3607] 3607 innerTail501_11
    (if 3 ≤ innerCandidates501_10.length then search edge 3 [3607, 8101] innerCandidates501_10 11708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan501_11

def innerTail501_9 : List Nat := 8089 :: innerTail501_10
def innerCandidates501_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates501_9_checked : candidates edge 4 3607 8089 innerTail501_10 best.bound = innerCandidates501_9 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan501_9 : scan edge (search edge 3) 4 [3607] 3607 innerTail501_9 best = best := by
  rw [innerTail501_9, scan, if_neg (by decide), innerCandidates501_9_checked]
  change scan edge (search edge 3) 4 [3607] 3607 innerTail501_10
    (if 3 ≤ innerCandidates501_9.length then search edge 3 [3607, 8089] innerCandidates501_9 11696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan501_10

def innerTail501_8 : List Nat := 7669 :: innerTail501_9
def innerCandidates501_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates501_8_checked : candidates edge 4 3607 7669 innerTail501_9 best.bound = innerCandidates501_8 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan501_8 : scan edge (search edge 3) 4 [3607] 3607 innerTail501_8 best = best := by
  rw [innerTail501_8, scan, if_neg (by decide), innerCandidates501_8_checked]
  change scan edge (search edge 3) 4 [3607] 3607 innerTail501_9
    (if 3 ≤ innerCandidates501_8.length then search edge 3 [3607, 7669] innerCandidates501_8 11276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan501_9

def innerTail501_7 : List Nat := 7351 :: innerTail501_8
def innerCandidates501_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates501_7_checked : candidates edge 4 3607 7351 innerTail501_8 best.bound = innerCandidates501_7 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan501_7 : scan edge (search edge 3) 4 [3607] 3607 innerTail501_7 best = best := by
  rw [innerTail501_7, scan, if_neg (by decide), innerCandidates501_7_checked]
  change scan edge (search edge 3) 4 [3607] 3607 innerTail501_8
    (if 3 ≤ innerCandidates501_7.length then search edge 3 [3607, 7351] innerCandidates501_7 10958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan501_8

def innerTail501_6 : List Nat := 6577 :: innerTail501_7
def innerCandidates501_6 : List Nat := [7669]
set_option maxRecDepth 4096 in
theorem innerCandidates501_6_checked : candidates edge 4 3607 6577 innerTail501_7 best.bound = innerCandidates501_6 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan501_6 : scan edge (search edge 3) 4 [3607] 3607 innerTail501_6 best = best := by
  rw [innerTail501_6, scan, if_neg (by decide), innerCandidates501_6_checked]
  change scan edge (search edge 3) 4 [3607] 3607 innerTail501_7
    (if 3 ≤ innerCandidates501_6.length then search edge 3 [3607, 6577] innerCandidates501_6 10184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan501_7

def innerTail501_5 : List Nat := 6571 :: innerTail501_6
def innerCandidates501_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates501_5_checked : candidates edge 4 3607 6571 innerTail501_6 best.bound = innerCandidates501_5 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan501_5 : scan edge (search edge 3) 4 [3607] 3607 innerTail501_5 best = best := by
  rw [innerTail501_5, scan, if_neg (by decide), innerCandidates501_5_checked]
  change scan edge (search edge 3) 4 [3607] 3607 innerTail501_6
    (if 3 ≤ innerCandidates501_5.length then search edge 3 [3607, 6571] innerCandidates501_5 10178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan501_6

def innerTail501_4 : List Nat := 6379 :: innerTail501_5
def innerCandidates501_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates501_4_checked : candidates edge 4 3607 6379 innerTail501_5 best.bound = innerCandidates501_4 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan501_4 : scan edge (search edge 3) 4 [3607] 3607 innerTail501_4 best = best := by
  rw [innerTail501_4, scan, if_neg (by decide), innerCandidates501_4_checked]
  change scan edge (search edge 3) 4 [3607] 3607 innerTail501_5
    (if 3 ≤ innerCandidates501_4.length then search edge 3 [3607, 6379] innerCandidates501_4 9986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan501_5

def innerTail501_3 : List Nat := 6367 :: innerTail501_4
def innerCandidates501_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates501_3_checked : candidates edge 4 3607 6367 innerTail501_4 best.bound = innerCandidates501_3 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan501_3 : scan edge (search edge 3) 4 [3607] 3607 innerTail501_3 best = best := by
  rw [innerTail501_3, scan, if_neg (by decide), innerCandidates501_3_checked]
  change scan edge (search edge 3) 4 [3607] 3607 innerTail501_4
    (if 3 ≤ innerCandidates501_3.length then search edge 3 [3607, 6367] innerCandidates501_3 9974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan501_4

def innerTail501_2 : List Nat := 5077 :: innerTail501_3
def innerCandidates501_2 : List Nat := [6367]
set_option maxRecDepth 4096 in
theorem innerCandidates501_2_checked : candidates edge 4 3607 5077 innerTail501_3 best.bound = innerCandidates501_2 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan501_2 : scan edge (search edge 3) 4 [3607] 3607 innerTail501_2 best = best := by
  rw [innerTail501_2, scan, if_neg (by decide), innerCandidates501_2_checked]
  change scan edge (search edge 3) 4 [3607] 3607 innerTail501_3
    (if 3 ≤ innerCandidates501_2.length then search edge 3 [3607, 5077] innerCandidates501_2 8684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan501_3

def innerTail501_1 : List Nat := 5023 :: innerTail501_2
def innerCandidates501_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates501_1_checked : candidates edge 4 3607 5023 innerTail501_2 best.bound = innerCandidates501_1 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan501_1 : scan edge (search edge 3) 4 [3607] 3607 innerTail501_1 best = best := by
  rw [innerTail501_1, scan, if_neg (by decide), innerCandidates501_1_checked]
  change scan edge (search edge 3) 4 [3607] 3607 innerTail501_2
    (if 3 ≤ innerCandidates501_1.length then search edge 3 [3607, 5023] innerCandidates501_1 8630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan501_2

def innerTail501_0 : List Nat := 4099 :: innerTail501_1
def innerCandidates501_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates501_0_checked : candidates edge 4 3607 4099 innerTail501_1 best.bound = innerCandidates501_0 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan501_0 : scan edge (search edge 3) 4 [3607] 3607 innerTail501_0 best = best := by
  rw [innerTail501_0, scan, if_neg (by decide), innerCandidates501_0_checked]
  change scan edge (search edge 3) 4 [3607] 3607 innerTail501_1
    (if 3 ≤ innerCandidates501_0.length then search edge 3 [3607, 4099] innerCandidates501_0 7706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan501_1

theorem rootBranch501 : search edge 4 [3607] rootCandidates501 3607 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3607] 3607 innerTail501_0 best = best
  exact innerScan501_0

def rootCandidates500 : List Nat := [3719, 4001, 4517, 5021, 5087, 5279, 5693, 5903, 6311, 7529, 7877, 7907, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates500_checked : candidates edge 5 0 3593 tail501 best.bound = rootCandidates500 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext q; rfl
  rw [row]
  decide

def innerTail500_13 : List Nat := []
theorem innerScan500_13 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_13 best = best := rfl

def innerTail500_12 : List Nat := 8297 :: innerTail500_13
def innerCandidates500_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates500_12_checked : candidates edge 4 3593 8297 innerTail500_13 best.bound = innerCandidates500_12 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_12 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_12 best = best := by
  rw [innerTail500_12, scan, if_neg (by decide), innerCandidates500_12_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_13
    (if 3 ≤ innerCandidates500_12.length then search edge 3 [3593, 8297] innerCandidates500_12 11890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_13

def innerTail500_11 : List Nat := 7907 :: innerTail500_12
def innerCandidates500_11 : List Nat := [8297]
set_option maxRecDepth 4096 in
theorem innerCandidates500_11_checked : candidates edge 4 3593 7907 innerTail500_12 best.bound = innerCandidates500_11 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_11 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_11 best = best := by
  rw [innerTail500_11, scan, if_neg (by decide), innerCandidates500_11_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_12
    (if 3 ≤ innerCandidates500_11.length then search edge 3 [3593, 7907] innerCandidates500_11 11500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_12

def innerTail500_10 : List Nat := 7877 :: innerTail500_11
def innerCandidates500_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates500_10_checked : candidates edge 4 3593 7877 innerTail500_11 best.bound = innerCandidates500_10 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_10 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_10 best = best := by
  rw [innerTail500_10, scan, if_neg (by decide), innerCandidates500_10_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_11
    (if 3 ≤ innerCandidates500_10.length then search edge 3 [3593, 7877] innerCandidates500_10 11470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_11

def innerTail500_9 : List Nat := 7529 :: innerTail500_10
def innerCandidates500_9 : List Nat := [7877]
set_option maxRecDepth 4096 in
theorem innerCandidates500_9_checked : candidates edge 4 3593 7529 innerTail500_10 best.bound = innerCandidates500_9 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_9 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_9 best = best := by
  rw [innerTail500_9, scan, if_neg (by decide), innerCandidates500_9_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_10
    (if 3 ≤ innerCandidates500_9.length then search edge 3 [3593, 7529] innerCandidates500_9 11122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_10

def innerTail500_8 : List Nat := 6311 :: innerTail500_9
def innerCandidates500_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates500_8_checked : candidates edge 4 3593 6311 innerTail500_9 best.bound = innerCandidates500_8 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_8 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_8 best = best := by
  rw [innerTail500_8, scan, if_neg (by decide), innerCandidates500_8_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_9
    (if 3 ≤ innerCandidates500_8.length then search edge 3 [3593, 6311] innerCandidates500_8 9904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_9

def innerTail500_7 : List Nat := 5903 :: innerTail500_8
def innerCandidates500_7 : List Nat := [6311, 7877]
set_option maxRecDepth 4096 in
theorem innerCandidates500_7_checked : candidates edge 4 3593 5903 innerTail500_8 best.bound = innerCandidates500_7 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_7 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_7 best = best := by
  rw [innerTail500_7, scan, if_neg (by decide), innerCandidates500_7_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_8
    (if 3 ≤ innerCandidates500_7.length then search edge 3 [3593, 5903] innerCandidates500_7 9496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_8

def innerTail500_6 : List Nat := 5693 :: innerTail500_7
def innerCandidates500_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates500_6_checked : candidates edge 4 3593 5693 innerTail500_7 best.bound = innerCandidates500_6 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_6 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_6 best = best := by
  rw [innerTail500_6, scan, if_neg (by decide), innerCandidates500_6_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_7
    (if 3 ≤ innerCandidates500_6.length then search edge 3 [3593, 5693] innerCandidates500_6 9286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_7

def innerTail500_5 : List Nat := 5279 :: innerTail500_6
def innerCandidates500_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates500_5_checked : candidates edge 4 3593 5279 innerTail500_6 best.bound = innerCandidates500_5 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_5 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_5 best = best := by
  rw [innerTail500_5, scan, if_neg (by decide), innerCandidates500_5_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_6
    (if 3 ≤ innerCandidates500_5.length then search edge 3 [3593, 5279] innerCandidates500_5 8872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_6

def innerTail500_4 : List Nat := 5087 :: innerTail500_5
def innerCandidates500_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates500_4_checked : candidates edge 4 3593 5087 innerTail500_5 best.bound = innerCandidates500_4 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_4 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_4 best = best := by
  rw [innerTail500_4, scan, if_neg (by decide), innerCandidates500_4_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_5
    (if 3 ≤ innerCandidates500_4.length then search edge 3 [3593, 5087] innerCandidates500_4 8680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_5

def innerTail500_3 : List Nat := 5021 :: innerTail500_4
def innerCandidates500_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates500_3_checked : candidates edge 4 3593 5021 innerTail500_4 best.bound = innerCandidates500_3 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_3 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_3 best = best := by
  rw [innerTail500_3, scan, if_neg (by decide), innerCandidates500_3_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_4
    (if 3 ≤ innerCandidates500_3.length then search edge 3 [3593, 5021] innerCandidates500_3 8614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_4

def innerTail500_2 : List Nat := 4517 :: innerTail500_3
def innerCandidates500_2 : List Nat := [5087, 5279]
set_option maxRecDepth 4096 in
theorem innerCandidates500_2_checked : candidates edge 4 3593 4517 innerTail500_3 best.bound = innerCandidates500_2 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_2 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_2 best = best := by
  rw [innerTail500_2, scan, if_neg (by decide), innerCandidates500_2_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_3
    (if 3 ≤ innerCandidates500_2.length then search edge 3 [3593, 4517] innerCandidates500_2 8110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_3

def innerTail500_1 : List Nat := 4001 :: innerTail500_2
def innerCandidates500_1 : List Nat := [7877]
set_option maxRecDepth 4096 in
theorem innerCandidates500_1_checked : candidates edge 4 3593 4001 innerTail500_2 best.bound = innerCandidates500_1 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_1 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_1 best = best := by
  rw [innerTail500_1, scan, if_neg (by decide), innerCandidates500_1_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_2
    (if 3 ≤ innerCandidates500_1.length then search edge 3 [3593, 4001] innerCandidates500_1 7594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_2

def innerTail500_0 : List Nat := 3719 :: innerTail500_1
def innerCandidates500_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates500_0_checked : candidates edge 4 3593 3719 innerTail500_1 best.bound = innerCandidates500_0 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan500_0 : scan edge (search edge 3) 4 [3593] 3593 innerTail500_0 best = best := by
  rw [innerTail500_0, scan, if_neg (by decide), innerCandidates500_0_checked]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_1
    (if 3 ≤ innerCandidates500_0.length then search edge 3 [3593, 3719] innerCandidates500_0 7312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan500_1

theorem rootBranch500 : search edge 4 [3593] rootCandidates500 3593 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3593] 3593 innerTail500_0 best = best
  exact innerScan500_0

def rootCandidates499 : List Nat := [4177, 4423, 4651, 5179, 5689, 5857, 5953, 6067, 6247, 7057, 7759, 7951, 8191]
set_option maxRecDepth 4096 in
theorem rootCandidates499_checked : candidates edge 5 0 3583 tail500 best.bound = rootCandidates499 := by
  unfold candidates
  have row : edge 3583 = SearchData.row3583.contains := by funext q; rfl
  rw [row]
  decide

def innerTail499_13 : List Nat := []
theorem innerScan499_13 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_13 best = best := rfl

def innerTail499_12 : List Nat := 8191 :: innerTail499_13
def innerCandidates499_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates499_12_checked : candidates edge 4 3583 8191 innerTail499_13 best.bound = innerCandidates499_12 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_12 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_12 best = best := by
  rw [innerTail499_12, scan, if_neg (by decide), innerCandidates499_12_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_13
    (if 3 ≤ innerCandidates499_12.length then search edge 3 [3583, 8191] innerCandidates499_12 11774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_13

def innerTail499_11 : List Nat := 7951 :: innerTail499_12
def innerCandidates499_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates499_11_checked : candidates edge 4 3583 7951 innerTail499_12 best.bound = innerCandidates499_11 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_11 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_11 best = best := by
  rw [innerTail499_11, scan, if_neg (by decide), innerCandidates499_11_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_12
    (if 3 ≤ innerCandidates499_11.length then search edge 3 [3583, 7951] innerCandidates499_11 11534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_12

def innerTail499_10 : List Nat := 7759 :: innerTail499_11
def innerCandidates499_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates499_10_checked : candidates edge 4 3583 7759 innerTail499_11 best.bound = innerCandidates499_10 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_10 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_10 best = best := by
  rw [innerTail499_10, scan, if_neg (by decide), innerCandidates499_10_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_11
    (if 3 ≤ innerCandidates499_10.length then search edge 3 [3583, 7759] innerCandidates499_10 11342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_11

def innerTail499_9 : List Nat := 7057 :: innerTail499_10
def innerCandidates499_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates499_9_checked : candidates edge 4 3583 7057 innerTail499_10 best.bound = innerCandidates499_9 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_9 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_9 best = best := by
  rw [innerTail499_9, scan, if_neg (by decide), innerCandidates499_9_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_10
    (if 3 ≤ innerCandidates499_9.length then search edge 3 [3583, 7057] innerCandidates499_9 10640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_10

def innerTail499_8 : List Nat := 6247 :: innerTail499_9
def innerCandidates499_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates499_8_checked : candidates edge 4 3583 6247 innerTail499_9 best.bound = innerCandidates499_8 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_8 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_8 best = best := by
  rw [innerTail499_8, scan, if_neg (by decide), innerCandidates499_8_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_9
    (if 3 ≤ innerCandidates499_8.length then search edge 3 [3583, 6247] innerCandidates499_8 9830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_9

def innerTail499_7 : List Nat := 6067 :: innerTail499_8
def innerCandidates499_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates499_7_checked : candidates edge 4 3583 6067 innerTail499_8 best.bound = innerCandidates499_7 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_7 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_7 best = best := by
  rw [innerTail499_7, scan, if_neg (by decide), innerCandidates499_7_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_8
    (if 3 ≤ innerCandidates499_7.length then search edge 3 [3583, 6067] innerCandidates499_7 9650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_8

def innerTail499_6 : List Nat := 5953 :: innerTail499_7
def innerCandidates499_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates499_6_checked : candidates edge 4 3583 5953 innerTail499_7 best.bound = innerCandidates499_6 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_6 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_6 best = best := by
  rw [innerTail499_6, scan, if_neg (by decide), innerCandidates499_6_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_7
    (if 3 ≤ innerCandidates499_6.length then search edge 3 [3583, 5953] innerCandidates499_6 9536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_7

def innerTail499_5 : List Nat := 5857 :: innerTail499_6
def innerCandidates499_5 : List Nat := [5953]
set_option maxRecDepth 4096 in
theorem innerCandidates499_5_checked : candidates edge 4 3583 5857 innerTail499_6 best.bound = innerCandidates499_5 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_5 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_5 best = best := by
  rw [innerTail499_5, scan, if_neg (by decide), innerCandidates499_5_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_6
    (if 3 ≤ innerCandidates499_5.length then search edge 3 [3583, 5857] innerCandidates499_5 9440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_6

def innerTail499_4 : List Nat := 5689 :: innerTail499_5
def innerCandidates499_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates499_4_checked : candidates edge 4 3583 5689 innerTail499_5 best.bound = innerCandidates499_4 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_4 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_4 best = best := by
  rw [innerTail499_4, scan, if_neg (by decide), innerCandidates499_4_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_5
    (if 3 ≤ innerCandidates499_4.length then search edge 3 [3583, 5689] innerCandidates499_4 9272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_5

def innerTail499_3 : List Nat := 5179 :: innerTail499_4
def innerCandidates499_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates499_3_checked : candidates edge 4 3583 5179 innerTail499_4 best.bound = innerCandidates499_3 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_3 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_3 best = best := by
  rw [innerTail499_3, scan, if_neg (by decide), innerCandidates499_3_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_4
    (if 3 ≤ innerCandidates499_3.length then search edge 3 [3583, 5179] innerCandidates499_3 8762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_4

def innerTail499_2 : List Nat := 4651 :: innerTail499_3
def innerCandidates499_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates499_2_checked : candidates edge 4 3583 4651 innerTail499_3 best.bound = innerCandidates499_2 := by
  unfold candidates
  have row : edge 4651 = SearchData.row4651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_2 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_2 best = best := by
  rw [innerTail499_2, scan, if_neg (by decide), innerCandidates499_2_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_3
    (if 3 ≤ innerCandidates499_2.length then search edge 3 [3583, 4651] innerCandidates499_2 8234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_3

def innerTail499_1 : List Nat := 4423 :: innerTail499_2
def innerCandidates499_1 : List Nat := [5689]
set_option maxRecDepth 4096 in
theorem innerCandidates499_1_checked : candidates edge 4 3583 4423 innerTail499_2 best.bound = innerCandidates499_1 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_1 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_1 best = best := by
  rw [innerTail499_1, scan, if_neg (by decide), innerCandidates499_1_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_2
    (if 3 ≤ innerCandidates499_1.length then search edge 3 [3583, 4423] innerCandidates499_1 8006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_2

def innerTail499_0 : List Nat := 4177 :: innerTail499_1
def innerCandidates499_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates499_0_checked : candidates edge 4 3583 4177 innerTail499_1 best.bound = innerCandidates499_0 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan499_0 : scan edge (search edge 3) 4 [3583] 3583 innerTail499_0 best = best := by
  rw [innerTail499_0, scan, if_neg (by decide), innerCandidates499_0_checked]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_1
    (if 3 ≤ innerCandidates499_0.length then search edge 3 [3583, 4177] innerCandidates499_0 7760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan499_1

theorem rootBranch499 : search edge 4 [3583] rootCandidates499 3583 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3583] 3583 innerTail499_0 best = best
  exact innerScan499_0

def rootCandidates498 : List Nat := [3851, 4091, 4133, 4139, 4421, 4643, 4679, 4931, 5303, 5387, 5849, 5903, 6101, 6689, 6701, 6983, 7253, 7829]
set_option maxRecDepth 4096 in
theorem rootCandidates498_checked : candidates edge 5 0 3581 tail499 best.bound = rootCandidates498 := by
  unfold candidates
  have row : edge 3581 = SearchData.row3581.contains := by funext q; rfl
  rw [row]
  decide

def innerTail498_18 : List Nat := []
theorem innerScan498_18 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_18 best = best := rfl

def innerTail498_17 : List Nat := 7829 :: innerTail498_18
def innerCandidates498_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_17_checked : candidates edge 4 3581 7829 innerTail498_18 best.bound = innerCandidates498_17 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_17 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_17 best = best := by
  rw [innerTail498_17, scan, if_neg (by decide), innerCandidates498_17_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_18
    (if 3 ≤ innerCandidates498_17.length then search edge 3 [3581, 7829] innerCandidates498_17 11410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_18

def innerTail498_16 : List Nat := 7253 :: innerTail498_17
def innerCandidates498_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_16_checked : candidates edge 4 3581 7253 innerTail498_17 best.bound = innerCandidates498_16 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_16 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_16 best = best := by
  rw [innerTail498_16, scan, if_neg (by decide), innerCandidates498_16_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_17
    (if 3 ≤ innerCandidates498_16.length then search edge 3 [3581, 7253] innerCandidates498_16 10834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_17

def innerTail498_15 : List Nat := 6983 :: innerTail498_16
def innerCandidates498_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_15_checked : candidates edge 4 3581 6983 innerTail498_16 best.bound = innerCandidates498_15 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_15 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_15 best = best := by
  rw [innerTail498_15, scan, if_neg (by decide), innerCandidates498_15_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_16
    (if 3 ≤ innerCandidates498_15.length then search edge 3 [3581, 6983] innerCandidates498_15 10564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_16

def innerTail498_14 : List Nat := 6701 :: innerTail498_15
def innerCandidates498_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_14_checked : candidates edge 4 3581 6701 innerTail498_15 best.bound = innerCandidates498_14 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_14 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_14 best = best := by
  rw [innerTail498_14, scan, if_neg (by decide), innerCandidates498_14_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_15
    (if 3 ≤ innerCandidates498_14.length then search edge 3 [3581, 6701] innerCandidates498_14 10282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_15

def innerTail498_13 : List Nat := 6689 :: innerTail498_14
def innerCandidates498_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_13_checked : candidates edge 4 3581 6689 innerTail498_14 best.bound = innerCandidates498_13 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_13 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_13 best = best := by
  rw [innerTail498_13, scan, if_neg (by decide), innerCandidates498_13_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_14
    (if 3 ≤ innerCandidates498_13.length then search edge 3 [3581, 6689] innerCandidates498_13 10270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_14

def innerTail498_12 : List Nat := 6101 :: innerTail498_13
def innerCandidates498_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_12_checked : candidates edge 4 3581 6101 innerTail498_13 best.bound = innerCandidates498_12 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_12 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_12 best = best := by
  rw [innerTail498_12, scan, if_neg (by decide), innerCandidates498_12_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_13
    (if 3 ≤ innerCandidates498_12.length then search edge 3 [3581, 6101] innerCandidates498_12 9682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_13

def innerTail498_11 : List Nat := 5903 :: innerTail498_12
def innerCandidates498_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_11_checked : candidates edge 4 3581 5903 innerTail498_12 best.bound = innerCandidates498_11 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_11 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_11 best = best := by
  rw [innerTail498_11, scan, if_neg (by decide), innerCandidates498_11_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_12
    (if 3 ≤ innerCandidates498_11.length then search edge 3 [3581, 5903] innerCandidates498_11 9484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_12

def innerTail498_10 : List Nat := 5849 :: innerTail498_11
def innerCandidates498_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_10_checked : candidates edge 4 3581 5849 innerTail498_11 best.bound = innerCandidates498_10 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_10 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_10 best = best := by
  rw [innerTail498_10, scan, if_neg (by decide), innerCandidates498_10_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_11
    (if 3 ≤ innerCandidates498_10.length then search edge 3 [3581, 5849] innerCandidates498_10 9430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_11

def innerTail498_9 : List Nat := 5387 :: innerTail498_10
def innerCandidates498_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_9_checked : candidates edge 4 3581 5387 innerTail498_10 best.bound = innerCandidates498_9 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_9 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_9 best = best := by
  rw [innerTail498_9, scan, if_neg (by decide), innerCandidates498_9_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_10
    (if 3 ≤ innerCandidates498_9.length then search edge 3 [3581, 5387] innerCandidates498_9 8968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_10

def innerTail498_8 : List Nat := 5303 :: innerTail498_9
def innerCandidates498_8 : List Nat := [5849, 6101]
set_option maxRecDepth 4096 in
theorem innerCandidates498_8_checked : candidates edge 4 3581 5303 innerTail498_9 best.bound = innerCandidates498_8 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_8 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_8 best = best := by
  rw [innerTail498_8, scan, if_neg (by decide), innerCandidates498_8_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_9
    (if 3 ≤ innerCandidates498_8.length then search edge 3 [3581, 5303] innerCandidates498_8 8884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_9

def innerTail498_7 : List Nat := 4931 :: innerTail498_8
def innerCandidates498_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_7_checked : candidates edge 4 3581 4931 innerTail498_8 best.bound = innerCandidates498_7 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_7 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_7 best = best := by
  rw [innerTail498_7, scan, if_neg (by decide), innerCandidates498_7_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_8
    (if 3 ≤ innerCandidates498_7.length then search edge 3 [3581, 4931] innerCandidates498_7 8512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_8

def innerTail498_6 : List Nat := 4679 :: innerTail498_7
def innerCandidates498_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_6_checked : candidates edge 4 3581 4679 innerTail498_7 best.bound = innerCandidates498_6 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_6 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_6 best = best := by
  rw [innerTail498_6, scan, if_neg (by decide), innerCandidates498_6_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_7
    (if 3 ≤ innerCandidates498_6.length then search edge 3 [3581, 4679] innerCandidates498_6 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_7

def innerTail498_5 : List Nat := 4643 :: innerTail498_6
def innerCandidates498_5 : List Nat := [5849]
set_option maxRecDepth 4096 in
theorem innerCandidates498_5_checked : candidates edge 4 3581 4643 innerTail498_6 best.bound = innerCandidates498_5 := by
  unfold candidates
  have row : edge 4643 = SearchData.row4643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_5 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_5 best = best := by
  rw [innerTail498_5, scan, if_neg (by decide), innerCandidates498_5_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_6
    (if 3 ≤ innerCandidates498_5.length then search edge 3 [3581, 4643] innerCandidates498_5 8224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_6

def innerTail498_4 : List Nat := 4421 :: innerTail498_5
def innerCandidates498_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_4_checked : candidates edge 4 3581 4421 innerTail498_5 best.bound = innerCandidates498_4 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_4 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_4 best = best := by
  rw [innerTail498_4, scan, if_neg (by decide), innerCandidates498_4_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_5
    (if 3 ≤ innerCandidates498_4.length then search edge 3 [3581, 4421] innerCandidates498_4 8002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_5

def innerTail498_3 : List Nat := 4139 :: innerTail498_4
def innerCandidates498_3 : List Nat := [7253]
set_option maxRecDepth 4096 in
theorem innerCandidates498_3_checked : candidates edge 4 3581 4139 innerTail498_4 best.bound = innerCandidates498_3 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_3 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_3 best = best := by
  rw [innerTail498_3, scan, if_neg (by decide), innerCandidates498_3_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_4
    (if 3 ≤ innerCandidates498_3.length then search edge 3 [3581, 4139] innerCandidates498_3 7720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_4

def innerTail498_2 : List Nat := 4133 :: innerTail498_3
def innerCandidates498_2 : List Nat := [5387]
set_option maxRecDepth 4096 in
theorem innerCandidates498_2_checked : candidates edge 4 3581 4133 innerTail498_3 best.bound = innerCandidates498_2 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_2 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_2 best = best := by
  rw [innerTail498_2, scan, if_neg (by decide), innerCandidates498_2_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_3
    (if 3 ≤ innerCandidates498_2.length then search edge 3 [3581, 4133] innerCandidates498_2 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_3

def innerTail498_1 : List Nat := 4091 :: innerTail498_2
def innerCandidates498_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates498_1_checked : candidates edge 4 3581 4091 innerTail498_2 best.bound = innerCandidates498_1 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan498_1 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_1 best = best := by
  rw [innerTail498_1, scan, if_neg (by decide), innerCandidates498_1_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_2
    (if 3 ≤ innerCandidates498_1.length then search edge 3 [3581, 4091] innerCandidates498_1 7672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan498_2

def innerTail498_0 : List Nat := 3851 :: innerTail498_1
def innerCandidates498_0 : List Nat := [4139, 5903, 6983]
set_option maxRecDepth 4096 in
theorem innerCandidates498_0_checked : candidates edge 4 3581 3851 innerTail498_1 best.bound = innerCandidates498_0 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild498_0 : search edge 3 [3581, 3851] innerCandidates498_0 7432 best = best := by decide

theorem innerScan498_0 : scan edge (search edge 3) 4 [3581] 3581 innerTail498_0 best = best := by
  rw [innerTail498_0, scan, if_neg (by decide), innerCandidates498_0_checked]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_1
    (if 3 ≤ innerCandidates498_0.length then search edge 3 [3581, 3851] innerCandidates498_0 7432 best else best) = best
  rw [if_pos (by decide), innerChild498_0]
  exact innerScan498_1

theorem rootBranch498 : search edge 4 [3581] rootCandidates498 3581 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3581] 3581 innerTail498_0 best = best
  exact innerScan498_0

end ProjectEulerSolutions.PrimePairUpper.Search
