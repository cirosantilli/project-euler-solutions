import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates106 : List Nat := [719, 941, 1709, 1823, 1907, 3023, 3557, 3671, 3761, 4091, 5081, 5309, 5399, 5573, 6047, 7517, 7841, 7877, 7883, 8039]
set_option maxRecDepth 4096 in
theorem rootCandidates106_checked : candidates edge 5 0 599 tail107 best.bound = rootCandidates106 := by
  unfold candidates
  have row : edge 599 = PrimePairCache.smallRow599.contains := by funext q; rfl
  rw [row]
  decide

def innerTail106_20 : List Nat := []
theorem innerScan106_20 : scan edge (search edge 3) 4 [599] 599 innerTail106_20 best = best := rfl

def innerTail106_19 : List Nat := 8039 :: innerTail106_20
def innerCandidates106_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_19_checked : candidates edge 4 599 8039 innerTail106_20 best.bound = innerCandidates106_19 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_19 : scan edge (search edge 3) 4 [599] 599 innerTail106_19 best = best := by
  rw [innerTail106_19, scan, if_neg (by decide), innerCandidates106_19_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_20
    (if 3 ≤ innerCandidates106_19.length then search edge 3 [599, 8039] innerCandidates106_19 8638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_20

def innerTail106_18 : List Nat := 7883 :: innerTail106_19
def innerCandidates106_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_18_checked : candidates edge 4 599 7883 innerTail106_19 best.bound = innerCandidates106_18 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_18 : scan edge (search edge 3) 4 [599] 599 innerTail106_18 best = best := by
  rw [innerTail106_18, scan, if_neg (by decide), innerCandidates106_18_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_19
    (if 3 ≤ innerCandidates106_18.length then search edge 3 [599, 7883] innerCandidates106_18 8482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_19

def innerTail106_17 : List Nat := 7877 :: innerTail106_18
def innerCandidates106_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_17_checked : candidates edge 4 599 7877 innerTail106_18 best.bound = innerCandidates106_17 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_17 : scan edge (search edge 3) 4 [599] 599 innerTail106_17 best = best := by
  rw [innerTail106_17, scan, if_neg (by decide), innerCandidates106_17_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_18
    (if 3 ≤ innerCandidates106_17.length then search edge 3 [599, 7877] innerCandidates106_17 8476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_18

def innerTail106_16 : List Nat := 7841 :: innerTail106_17
def innerCandidates106_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_16_checked : candidates edge 4 599 7841 innerTail106_17 best.bound = innerCandidates106_16 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_16 : scan edge (search edge 3) 4 [599] 599 innerTail106_16 best = best := by
  rw [innerTail106_16, scan, if_neg (by decide), innerCandidates106_16_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_17
    (if 3 ≤ innerCandidates106_16.length then search edge 3 [599, 7841] innerCandidates106_16 8440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_17

def innerTail106_15 : List Nat := 7517 :: innerTail106_16
def innerCandidates106_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_15_checked : candidates edge 4 599 7517 innerTail106_16 best.bound = innerCandidates106_15 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_15 : scan edge (search edge 3) 4 [599] 599 innerTail106_15 best = best := by
  rw [innerTail106_15, scan, if_neg (by decide), innerCandidates106_15_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_16
    (if 3 ≤ innerCandidates106_15.length then search edge 3 [599, 7517] innerCandidates106_15 8116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_16

def innerTail106_14 : List Nat := 6047 :: innerTail106_15
def innerCandidates106_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_14_checked : candidates edge 4 599 6047 innerTail106_15 best.bound = innerCandidates106_14 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_14 : scan edge (search edge 3) 4 [599] 599 innerTail106_14 best = best := by
  rw [innerTail106_14, scan, if_neg (by decide), innerCandidates106_14_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_15
    (if 3 ≤ innerCandidates106_14.length then search edge 3 [599, 6047] innerCandidates106_14 6646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_15

def innerTail106_13 : List Nat := 5573 :: innerTail106_14
def innerCandidates106_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_13_checked : candidates edge 4 599 5573 innerTail106_14 best.bound = innerCandidates106_13 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_13 : scan edge (search edge 3) 4 [599] 599 innerTail106_13 best = best := by
  rw [innerTail106_13, scan, if_neg (by decide), innerCandidates106_13_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_14
    (if 3 ≤ innerCandidates106_13.length then search edge 3 [599, 5573] innerCandidates106_13 6172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_14

def innerTail106_12 : List Nat := 5399 :: innerTail106_13
def innerCandidates106_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_12_checked : candidates edge 4 599 5399 innerTail106_13 best.bound = innerCandidates106_12 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_12 : scan edge (search edge 3) 4 [599] 599 innerTail106_12 best = best := by
  rw [innerTail106_12, scan, if_neg (by decide), innerCandidates106_12_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_13
    (if 3 ≤ innerCandidates106_12.length then search edge 3 [599, 5399] innerCandidates106_12 5998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_13

def innerTail106_11 : List Nat := 5309 :: innerTail106_12
def innerCandidates106_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_11_checked : candidates edge 4 599 5309 innerTail106_12 best.bound = innerCandidates106_11 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_11 : scan edge (search edge 3) 4 [599] 599 innerTail106_11 best = best := by
  rw [innerTail106_11, scan, if_neg (by decide), innerCandidates106_11_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_12
    (if 3 ≤ innerCandidates106_11.length then search edge 3 [599, 5309] innerCandidates106_11 5908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_12

def innerTail106_10 : List Nat := 5081 :: innerTail106_11
def innerCandidates106_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_10_checked : candidates edge 4 599 5081 innerTail106_11 best.bound = innerCandidates106_10 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_10 : scan edge (search edge 3) 4 [599] 599 innerTail106_10 best = best := by
  rw [innerTail106_10, scan, if_neg (by decide), innerCandidates106_10_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_11
    (if 3 ≤ innerCandidates106_10.length then search edge 3 [599, 5081] innerCandidates106_10 5680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_11

def innerTail106_9 : List Nat := 4091 :: innerTail106_10
def innerCandidates106_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_9_checked : candidates edge 4 599 4091 innerTail106_10 best.bound = innerCandidates106_9 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_9 : scan edge (search edge 3) 4 [599] 599 innerTail106_9 best = best := by
  rw [innerTail106_9, scan, if_neg (by decide), innerCandidates106_9_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_10
    (if 3 ≤ innerCandidates106_9.length then search edge 3 [599, 4091] innerCandidates106_9 4690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_10

def innerTail106_8 : List Nat := 3761 :: innerTail106_9
def innerCandidates106_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_8_checked : candidates edge 4 599 3761 innerTail106_9 best.bound = innerCandidates106_8 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_8 : scan edge (search edge 3) 4 [599] 599 innerTail106_8 best = best := by
  rw [innerTail106_8, scan, if_neg (by decide), innerCandidates106_8_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_9
    (if 3 ≤ innerCandidates106_8.length then search edge 3 [599, 3761] innerCandidates106_8 4360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_9

def innerTail106_7 : List Nat := 3671 :: innerTail106_8
def innerCandidates106_7 : List Nat := [4091]
set_option maxRecDepth 4096 in
theorem innerCandidates106_7_checked : candidates edge 4 599 3671 innerTail106_8 best.bound = innerCandidates106_7 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_7 : scan edge (search edge 3) 4 [599] 599 innerTail106_7 best = best := by
  rw [innerTail106_7, scan, if_neg (by decide), innerCandidates106_7_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_8
    (if 3 ≤ innerCandidates106_7.length then search edge 3 [599, 3671] innerCandidates106_7 4270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_8

def innerTail106_6 : List Nat := 3557 :: innerTail106_7
def innerCandidates106_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_6_checked : candidates edge 4 599 3557 innerTail106_7 best.bound = innerCandidates106_6 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_6 : scan edge (search edge 3) 4 [599] 599 innerTail106_6 best = best := by
  rw [innerTail106_6, scan, if_neg (by decide), innerCandidates106_6_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_7
    (if 3 ≤ innerCandidates106_6.length then search edge 3 [599, 3557] innerCandidates106_6 4156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_7

def innerTail106_5 : List Nat := 3023 :: innerTail106_6
def innerCandidates106_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_5_checked : candidates edge 4 599 3023 innerTail106_6 best.bound = innerCandidates106_5 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_5 : scan edge (search edge 3) 4 [599] 599 innerTail106_5 best = best := by
  rw [innerTail106_5, scan, if_neg (by decide), innerCandidates106_5_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_6
    (if 3 ≤ innerCandidates106_5.length then search edge 3 [599, 3023] innerCandidates106_5 3622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_6

def innerTail106_4 : List Nat := 1907 :: innerTail106_5
def innerCandidates106_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_4_checked : candidates edge 4 599 1907 innerTail106_5 best.bound = innerCandidates106_4 := by
  unfold candidates
  have row : edge 1907 = PrimePairCache.smallRow1907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_4 : scan edge (search edge 3) 4 [599] 599 innerTail106_4 best = best := by
  rw [innerTail106_4, scan, if_neg (by decide), innerCandidates106_4_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_5
    (if 3 ≤ innerCandidates106_4.length then search edge 3 [599, 1907] innerCandidates106_4 2506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_5

def innerTail106_3 : List Nat := 1823 :: innerTail106_4
def innerCandidates106_3 : List Nat := [3023]
set_option maxRecDepth 4096 in
theorem innerCandidates106_3_checked : candidates edge 4 599 1823 innerTail106_4 best.bound = innerCandidates106_3 := by
  unfold candidates
  have row : edge 1823 = PrimePairCache.smallRow1823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_3 : scan edge (search edge 3) 4 [599] 599 innerTail106_3 best = best := by
  rw [innerTail106_3, scan, if_neg (by decide), innerCandidates106_3_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_4
    (if 3 ≤ innerCandidates106_3.length then search edge 3 [599, 1823] innerCandidates106_3 2422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_4

def innerTail106_2 : List Nat := 1709 :: innerTail106_3
def innerCandidates106_2 : List Nat := [7877]
set_option maxRecDepth 4096 in
theorem innerCandidates106_2_checked : candidates edge 4 599 1709 innerTail106_3 best.bound = innerCandidates106_2 := by
  unfold candidates
  have row : edge 1709 = PrimePairCache.smallRow1709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_2 : scan edge (search edge 3) 4 [599] 599 innerTail106_2 best = best := by
  rw [innerTail106_2, scan, if_neg (by decide), innerCandidates106_2_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_3
    (if 3 ≤ innerCandidates106_2.length then search edge 3 [599, 1709] innerCandidates106_2 2308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_3

def innerTail106_1 : List Nat := 941 :: innerTail106_2
def innerCandidates106_1 : List Nat := [1709, 6047]
set_option maxRecDepth 4096 in
theorem innerCandidates106_1_checked : candidates edge 4 599 941 innerTail106_2 best.bound = innerCandidates106_1 := by
  unfold candidates
  have row : edge 941 = PrimePairCache.smallRow941.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_1 : scan edge (search edge 3) 4 [599] 599 innerTail106_1 best = best := by
  rw [innerTail106_1, scan, if_neg (by decide), innerCandidates106_1_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_2
    (if 3 ≤ innerCandidates106_1.length then search edge 3 [599, 941] innerCandidates106_1 1540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_2

def innerTail106_0 : List Nat := 719 :: innerTail106_1
def innerCandidates106_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates106_0_checked : candidates edge 4 599 719 innerTail106_1 best.bound = innerCandidates106_0 := by
  unfold candidates
  have row : edge 719 = PrimePairCache.smallRow719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan106_0 : scan edge (search edge 3) 4 [599] 599 innerTail106_0 best = best := by
  rw [innerTail106_0, scan, if_neg (by decide), innerCandidates106_0_checked]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_1
    (if 3 ≤ innerCandidates106_0.length then search edge 3 [599, 719] innerCandidates106_0 1318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan106_1

theorem rootBranch106 : search edge 4 [599] rootCandidates106 599 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [599] 599 innerTail106_0 best = best
  exact innerScan106_0

def rootCandidates105 : List Nat := [647, 863, 977, 2141, 2411, 2417, 3203, 3251, 3461, 3467, 3539, 3581, 4001, 4133, 4211, 5393, 5399, 5639, 5861, 5879, 5987, 6653, 6857, 7433, 7559, 7577, 7949, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates105_checked : candidates edge 5 0 593 tail106 best.bound = rootCandidates105 := by
  unfold candidates
  have row : edge 593 = PrimePairCache.smallRow593.contains := by funext q; rfl
  rw [row]
  decide

def innerTail105_28 : List Nat := []
theorem innerScan105_28 : scan edge (search edge 3) 4 [593] 593 innerTail105_28 best = best := rfl

def innerTail105_27 : List Nat := 8123 :: innerTail105_28
def innerCandidates105_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_27_checked : candidates edge 4 593 8123 innerTail105_28 best.bound = innerCandidates105_27 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_27 : scan edge (search edge 3) 4 [593] 593 innerTail105_27 best = best := by
  rw [innerTail105_27, scan, if_neg (by decide), innerCandidates105_27_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_28
    (if 3 ≤ innerCandidates105_27.length then search edge 3 [593, 8123] innerCandidates105_27 8716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_28

def innerTail105_26 : List Nat := 7949 :: innerTail105_27
def innerCandidates105_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_26_checked : candidates edge 4 593 7949 innerTail105_27 best.bound = innerCandidates105_26 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_26 : scan edge (search edge 3) 4 [593] 593 innerTail105_26 best = best := by
  rw [innerTail105_26, scan, if_neg (by decide), innerCandidates105_26_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_27
    (if 3 ≤ innerCandidates105_26.length then search edge 3 [593, 7949] innerCandidates105_26 8542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_27

def innerTail105_25 : List Nat := 7577 :: innerTail105_26
def innerCandidates105_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_25_checked : candidates edge 4 593 7577 innerTail105_26 best.bound = innerCandidates105_25 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_25 : scan edge (search edge 3) 4 [593] 593 innerTail105_25 best = best := by
  rw [innerTail105_25, scan, if_neg (by decide), innerCandidates105_25_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_26
    (if 3 ≤ innerCandidates105_25.length then search edge 3 [593, 7577] innerCandidates105_25 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_26

def innerTail105_24 : List Nat := 7559 :: innerTail105_25
def innerCandidates105_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_24_checked : candidates edge 4 593 7559 innerTail105_25 best.bound = innerCandidates105_24 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_24 : scan edge (search edge 3) 4 [593] 593 innerTail105_24 best = best := by
  rw [innerTail105_24, scan, if_neg (by decide), innerCandidates105_24_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_25
    (if 3 ≤ innerCandidates105_24.length then search edge 3 [593, 7559] innerCandidates105_24 8152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_25

def innerTail105_23 : List Nat := 7433 :: innerTail105_24
def innerCandidates105_23 : List Nat := [7559]
set_option maxRecDepth 4096 in
theorem innerCandidates105_23_checked : candidates edge 4 593 7433 innerTail105_24 best.bound = innerCandidates105_23 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_23 : scan edge (search edge 3) 4 [593] 593 innerTail105_23 best = best := by
  rw [innerTail105_23, scan, if_neg (by decide), innerCandidates105_23_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_24
    (if 3 ≤ innerCandidates105_23.length then search edge 3 [593, 7433] innerCandidates105_23 8026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_24

def innerTail105_22 : List Nat := 6857 :: innerTail105_23
def innerCandidates105_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_22_checked : candidates edge 4 593 6857 innerTail105_23 best.bound = innerCandidates105_22 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_22 : scan edge (search edge 3) 4 [593] 593 innerTail105_22 best = best := by
  rw [innerTail105_22, scan, if_neg (by decide), innerCandidates105_22_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_23
    (if 3 ≤ innerCandidates105_22.length then search edge 3 [593, 6857] innerCandidates105_22 7450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_23

def innerTail105_21 : List Nat := 6653 :: innerTail105_22
def innerCandidates105_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_21_checked : candidates edge 4 593 6653 innerTail105_22 best.bound = innerCandidates105_21 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_21 : scan edge (search edge 3) 4 [593] 593 innerTail105_21 best = best := by
  rw [innerTail105_21, scan, if_neg (by decide), innerCandidates105_21_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_22
    (if 3 ≤ innerCandidates105_21.length then search edge 3 [593, 6653] innerCandidates105_21 7246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_22

def innerTail105_20 : List Nat := 5987 :: innerTail105_21
def innerCandidates105_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_20_checked : candidates edge 4 593 5987 innerTail105_21 best.bound = innerCandidates105_20 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_20 : scan edge (search edge 3) 4 [593] 593 innerTail105_20 best = best := by
  rw [innerTail105_20, scan, if_neg (by decide), innerCandidates105_20_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_21
    (if 3 ≤ innerCandidates105_20.length then search edge 3 [593, 5987] innerCandidates105_20 6580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_21

def innerTail105_19 : List Nat := 5879 :: innerTail105_20
def innerCandidates105_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_19_checked : candidates edge 4 593 5879 innerTail105_20 best.bound = innerCandidates105_19 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_19 : scan edge (search edge 3) 4 [593] 593 innerTail105_19 best = best := by
  rw [innerTail105_19, scan, if_neg (by decide), innerCandidates105_19_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_20
    (if 3 ≤ innerCandidates105_19.length then search edge 3 [593, 5879] innerCandidates105_19 6472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_20

def innerTail105_18 : List Nat := 5861 :: innerTail105_19
def innerCandidates105_18 : List Nat := [5987]
set_option maxRecDepth 4096 in
theorem innerCandidates105_18_checked : candidates edge 4 593 5861 innerTail105_19 best.bound = innerCandidates105_18 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_18 : scan edge (search edge 3) 4 [593] 593 innerTail105_18 best = best := by
  rw [innerTail105_18, scan, if_neg (by decide), innerCandidates105_18_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_19
    (if 3 ≤ innerCandidates105_18.length then search edge 3 [593, 5861] innerCandidates105_18 6454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_19

def innerTail105_17 : List Nat := 5639 :: innerTail105_18
def innerCandidates105_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_17_checked : candidates edge 4 593 5639 innerTail105_18 best.bound = innerCandidates105_17 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_17 : scan edge (search edge 3) 4 [593] 593 innerTail105_17 best = best := by
  rw [innerTail105_17, scan, if_neg (by decide), innerCandidates105_17_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_18
    (if 3 ≤ innerCandidates105_17.length then search edge 3 [593, 5639] innerCandidates105_17 6232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_18

def innerTail105_16 : List Nat := 5399 :: innerTail105_17
def innerCandidates105_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_16_checked : candidates edge 4 593 5399 innerTail105_17 best.bound = innerCandidates105_16 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_16 : scan edge (search edge 3) 4 [593] 593 innerTail105_16 best = best := by
  rw [innerTail105_16, scan, if_neg (by decide), innerCandidates105_16_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_17
    (if 3 ≤ innerCandidates105_16.length then search edge 3 [593, 5399] innerCandidates105_16 5992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_17

def innerTail105_15 : List Nat := 5393 :: innerTail105_16
def innerCandidates105_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_15_checked : candidates edge 4 593 5393 innerTail105_16 best.bound = innerCandidates105_15 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_15 : scan edge (search edge 3) 4 [593] 593 innerTail105_15 best = best := by
  rw [innerTail105_15, scan, if_neg (by decide), innerCandidates105_15_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_16
    (if 3 ≤ innerCandidates105_15.length then search edge 3 [593, 5393] innerCandidates105_15 5986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_16

def innerTail105_14 : List Nat := 4211 :: innerTail105_15
def innerCandidates105_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_14_checked : candidates edge 4 593 4211 innerTail105_15 best.bound = innerCandidates105_14 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_14 : scan edge (search edge 3) 4 [593] 593 innerTail105_14 best = best := by
  rw [innerTail105_14, scan, if_neg (by decide), innerCandidates105_14_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_15
    (if 3 ≤ innerCandidates105_14.length then search edge 3 [593, 4211] innerCandidates105_14 4804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_15

def innerTail105_13 : List Nat := 4133 :: innerTail105_14
def innerCandidates105_13 : List Nat := [7433]
set_option maxRecDepth 4096 in
theorem innerCandidates105_13_checked : candidates edge 4 593 4133 innerTail105_14 best.bound = innerCandidates105_13 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_13 : scan edge (search edge 3) 4 [593] 593 innerTail105_13 best = best := by
  rw [innerTail105_13, scan, if_neg (by decide), innerCandidates105_13_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_14
    (if 3 ≤ innerCandidates105_13.length then search edge 3 [593, 4133] innerCandidates105_13 4726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_14

def innerTail105_12 : List Nat := 4001 :: innerTail105_13
def innerCandidates105_12 : List Nat := [7577]
set_option maxRecDepth 4096 in
theorem innerCandidates105_12_checked : candidates edge 4 593 4001 innerTail105_13 best.bound = innerCandidates105_12 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_12 : scan edge (search edge 3) 4 [593] 593 innerTail105_12 best = best := by
  rw [innerTail105_12, scan, if_neg (by decide), innerCandidates105_12_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_13
    (if 3 ≤ innerCandidates105_12.length then search edge 3 [593, 4001] innerCandidates105_12 4594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_13

def innerTail105_11 : List Nat := 3581 :: innerTail105_12
def innerCandidates105_11 : List Nat := [4133]
set_option maxRecDepth 4096 in
theorem innerCandidates105_11_checked : candidates edge 4 593 3581 innerTail105_12 best.bound = innerCandidates105_11 := by
  unfold candidates
  have row : edge 3581 = SearchData.row3581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_11 : scan edge (search edge 3) 4 [593] 593 innerTail105_11 best = best := by
  rw [innerTail105_11, scan, if_neg (by decide), innerCandidates105_11_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_12
    (if 3 ≤ innerCandidates105_11.length then search edge 3 [593, 3581] innerCandidates105_11 4174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_12

def innerTail105_10 : List Nat := 3539 :: innerTail105_11
def innerCandidates105_10 : List Nat := [5861]
set_option maxRecDepth 4096 in
theorem innerCandidates105_10_checked : candidates edge 4 593 3539 innerTail105_11 best.bound = innerCandidates105_10 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_10 : scan edge (search edge 3) 4 [593] 593 innerTail105_10 best = best := by
  rw [innerTail105_10, scan, if_neg (by decide), innerCandidates105_10_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_11
    (if 3 ≤ innerCandidates105_10.length then search edge 3 [593, 3539] innerCandidates105_10 4132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_11

def innerTail105_9 : List Nat := 3467 :: innerTail105_10
def innerCandidates105_9 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates105_9_checked : candidates edge 4 593 3467 innerTail105_10 best.bound = innerCandidates105_9 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_9 : scan edge (search edge 3) 4 [593] 593 innerTail105_9 best = best := by
  rw [innerTail105_9, scan, if_neg (by decide), innerCandidates105_9_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_10
    (if 3 ≤ innerCandidates105_9.length then search edge 3 [593, 3467] innerCandidates105_9 4060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_10

def innerTail105_8 : List Nat := 3461 :: innerTail105_9
def innerCandidates105_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_8_checked : candidates edge 4 593 3461 innerTail105_9 best.bound = innerCandidates105_8 := by
  unfold candidates
  have row : edge 3461 = SearchData.row3461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_8 : scan edge (search edge 3) 4 [593] 593 innerTail105_8 best = best := by
  rw [innerTail105_8, scan, if_neg (by decide), innerCandidates105_8_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_9
    (if 3 ≤ innerCandidates105_8.length then search edge 3 [593, 3461] innerCandidates105_8 4054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_9

def innerTail105_7 : List Nat := 3251 :: innerTail105_8
def innerCandidates105_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_7_checked : candidates edge 4 593 3251 innerTail105_8 best.bound = innerCandidates105_7 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_7 : scan edge (search edge 3) 4 [593] 593 innerTail105_7 best = best := by
  rw [innerTail105_7, scan, if_neg (by decide), innerCandidates105_7_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_8
    (if 3 ≤ innerCandidates105_7.length then search edge 3 [593, 3251] innerCandidates105_7 3844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_8

def innerTail105_6 : List Nat := 3203 :: innerTail105_7
def innerCandidates105_6 : List Nat := [3467, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates105_6_checked : candidates edge 4 593 3203 innerTail105_7 best.bound = innerCandidates105_6 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_6 : scan edge (search edge 3) 4 [593] 593 innerTail105_6 best = best := by
  rw [innerTail105_6, scan, if_neg (by decide), innerCandidates105_6_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_7
    (if 3 ≤ innerCandidates105_6.length then search edge 3 [593, 3203] innerCandidates105_6 3796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_7

def innerTail105_5 : List Nat := 2417 :: innerTail105_6
def innerCandidates105_5 : List Nat := [4001, 6857]
set_option maxRecDepth 4096 in
theorem innerCandidates105_5_checked : candidates edge 4 593 2417 innerTail105_6 best.bound = innerCandidates105_5 := by
  unfold candidates
  have row : edge 2417 = PrimePairCache.smallRow2417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_5 : scan edge (search edge 3) 4 [593] 593 innerTail105_5 best = best := by
  rw [innerTail105_5, scan, if_neg (by decide), innerCandidates105_5_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_6
    (if 3 ≤ innerCandidates105_5.length then search edge 3 [593, 2417] innerCandidates105_5 3010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_6

def innerTail105_4 : List Nat := 2411 :: innerTail105_5
def innerCandidates105_4 : List Nat := [5879, 7433]
set_option maxRecDepth 4096 in
theorem innerCandidates105_4_checked : candidates edge 4 593 2411 innerTail105_5 best.bound = innerCandidates105_4 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_4 : scan edge (search edge 3) 4 [593] 593 innerTail105_4 best = best := by
  rw [innerTail105_4, scan, if_neg (by decide), innerCandidates105_4_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_5
    (if 3 ≤ innerCandidates105_4.length then search edge 3 [593, 2411] innerCandidates105_4 3004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_5

def innerTail105_3 : List Nat := 2141 :: innerTail105_4
def innerCandidates105_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_3_checked : candidates edge 4 593 2141 innerTail105_4 best.bound = innerCandidates105_3 := by
  unfold candidates
  have row : edge 2141 = PrimePairCache.smallRow2141.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_3 : scan edge (search edge 3) 4 [593] 593 innerTail105_3 best = best := by
  rw [innerTail105_3, scan, if_neg (by decide), innerCandidates105_3_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_4
    (if 3 ≤ innerCandidates105_3.length then search edge 3 [593, 2141] innerCandidates105_3 2734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_4

def innerTail105_2 : List Nat := 977 :: innerTail105_3
def innerCandidates105_2 : List Nat := [2141]
set_option maxRecDepth 4096 in
theorem innerCandidates105_2_checked : candidates edge 4 593 977 innerTail105_3 best.bound = innerCandidates105_2 := by
  unfold candidates
  have row : edge 977 = PrimePairCache.smallRow977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_2 : scan edge (search edge 3) 4 [593] 593 innerTail105_2 best = best := by
  rw [innerTail105_2, scan, if_neg (by decide), innerCandidates105_2_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_3
    (if 3 ≤ innerCandidates105_2.length then search edge 3 [593, 977] innerCandidates105_2 1570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_3

def innerTail105_1 : List Nat := 863 :: innerTail105_2
def innerCandidates105_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates105_1_checked : candidates edge 4 593 863 innerTail105_2 best.bound = innerCandidates105_1 := by
  unfold candidates
  have row : edge 863 = PrimePairCache.smallRow863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_1 : scan edge (search edge 3) 4 [593] 593 innerTail105_1 best = best := by
  rw [innerTail105_1, scan, if_neg (by decide), innerCandidates105_1_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_2
    (if 3 ≤ innerCandidates105_1.length then search edge 3 [593, 863] innerCandidates105_1 1456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_2

def innerTail105_0 : List Nat := 647 :: innerTail105_1
def innerCandidates105_0 : List Nat := [5879, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates105_0_checked : candidates edge 4 593 647 innerTail105_1 best.bound = innerCandidates105_0 := by
  unfold candidates
  have row : edge 647 = PrimePairCache.smallRow647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan105_0 : scan edge (search edge 3) 4 [593] 593 innerTail105_0 best = best := by
  rw [innerTail105_0, scan, if_neg (by decide), innerCandidates105_0_checked]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_1
    (if 3 ≤ innerCandidates105_0.length then search edge 3 [593, 647] innerCandidates105_0 1240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan105_1

theorem rootBranch105 : search edge 4 [593] rootCandidates105 593 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [593] 593 innerTail105_0 best = best
  exact innerScan105_0

end ProjectEulerSolutions.PrimePairUpper.Search
