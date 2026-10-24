import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates102 : List Nat := [853, 1051, 1933, 2341, 2383, 2647, 2659, 2953, 3061, 3121, 3163, 4483, 4759, 4987, 5701, 5839, 6553, 7069, 7213, 7297, 7549, 7951, 8233, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates102_checked : candidates edge 5 0 571 tail103 best.bound = rootCandidates102 := by
  unfold candidates
  have row : edge 571 = PrimePairCache.smallRow571.contains := by funext q; rfl
  rw [row]
  decide

def innerTail102_24 : List Nat := []
theorem innerScan102_24 : scan edge (search edge 3) 4 [571] 571 innerTail102_24 best = best := rfl

def innerTail102_23 : List Nat := 8263 :: innerTail102_24
def innerCandidates102_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_23_checked : candidates edge 4 571 8263 innerTail102_24 best.bound = innerCandidates102_23 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_23 : scan edge (search edge 3) 4 [571] 571 innerTail102_23 best = best := by
  rw [innerTail102_23, scan, if_neg (by decide), innerCandidates102_23_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_24
    (if 3 ≤ innerCandidates102_23.length then search edge 3 [571, 8263] innerCandidates102_23 8834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_24

def innerTail102_22 : List Nat := 8233 :: innerTail102_23
def innerCandidates102_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_22_checked : candidates edge 4 571 8233 innerTail102_23 best.bound = innerCandidates102_22 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_22 : scan edge (search edge 3) 4 [571] 571 innerTail102_22 best = best := by
  rw [innerTail102_22, scan, if_neg (by decide), innerCandidates102_22_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_23
    (if 3 ≤ innerCandidates102_22.length then search edge 3 [571, 8233] innerCandidates102_22 8804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_23

def innerTail102_21 : List Nat := 7951 :: innerTail102_22
def innerCandidates102_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_21_checked : candidates edge 4 571 7951 innerTail102_22 best.bound = innerCandidates102_21 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_21 : scan edge (search edge 3) 4 [571] 571 innerTail102_21 best = best := by
  rw [innerTail102_21, scan, if_neg (by decide), innerCandidates102_21_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_22
    (if 3 ≤ innerCandidates102_21.length then search edge 3 [571, 7951] innerCandidates102_21 8522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_22

def innerTail102_20 : List Nat := 7549 :: innerTail102_21
def innerCandidates102_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_20_checked : candidates edge 4 571 7549 innerTail102_21 best.bound = innerCandidates102_20 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_20 : scan edge (search edge 3) 4 [571] 571 innerTail102_20 best = best := by
  rw [innerTail102_20, scan, if_neg (by decide), innerCandidates102_20_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_21
    (if 3 ≤ innerCandidates102_20.length then search edge 3 [571, 7549] innerCandidates102_20 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_21

def innerTail102_19 : List Nat := 7297 :: innerTail102_20
def innerCandidates102_19 : List Nat := [7951]
set_option maxRecDepth 4096 in
theorem innerCandidates102_19_checked : candidates edge 4 571 7297 innerTail102_20 best.bound = innerCandidates102_19 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_19 : scan edge (search edge 3) 4 [571] 571 innerTail102_19 best = best := by
  rw [innerTail102_19, scan, if_neg (by decide), innerCandidates102_19_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_20
    (if 3 ≤ innerCandidates102_19.length then search edge 3 [571, 7297] innerCandidates102_19 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_20

def innerTail102_18 : List Nat := 7213 :: innerTail102_19
def innerCandidates102_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_18_checked : candidates edge 4 571 7213 innerTail102_19 best.bound = innerCandidates102_18 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_18 : scan edge (search edge 3) 4 [571] 571 innerTail102_18 best = best := by
  rw [innerTail102_18, scan, if_neg (by decide), innerCandidates102_18_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_19
    (if 3 ≤ innerCandidates102_18.length then search edge 3 [571, 7213] innerCandidates102_18 7784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_19

def innerTail102_17 : List Nat := 7069 :: innerTail102_18
def innerCandidates102_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_17_checked : candidates edge 4 571 7069 innerTail102_18 best.bound = innerCandidates102_17 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_17 : scan edge (search edge 3) 4 [571] 571 innerTail102_17 best = best := by
  rw [innerTail102_17, scan, if_neg (by decide), innerCandidates102_17_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_18
    (if 3 ≤ innerCandidates102_17.length then search edge 3 [571, 7069] innerCandidates102_17 7640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_18

def innerTail102_16 : List Nat := 6553 :: innerTail102_17
def innerCandidates102_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_16_checked : candidates edge 4 571 6553 innerTail102_17 best.bound = innerCandidates102_16 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_16 : scan edge (search edge 3) 4 [571] 571 innerTail102_16 best = best := by
  rw [innerTail102_16, scan, if_neg (by decide), innerCandidates102_16_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_17
    (if 3 ≤ innerCandidates102_16.length then search edge 3 [571, 6553] innerCandidates102_16 7124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_17

def innerTail102_15 : List Nat := 5839 :: innerTail102_16
def innerCandidates102_15 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates102_15_checked : candidates edge 4 571 5839 innerTail102_16 best.bound = innerCandidates102_15 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_15 : scan edge (search edge 3) 4 [571] 571 innerTail102_15 best = best := by
  rw [innerTail102_15, scan, if_neg (by decide), innerCandidates102_15_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_16
    (if 3 ≤ innerCandidates102_15.length then search edge 3 [571, 5839] innerCandidates102_15 6410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_16

def innerTail102_14 : List Nat := 5701 :: innerTail102_15
def innerCandidates102_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_14_checked : candidates edge 4 571 5701 innerTail102_15 best.bound = innerCandidates102_14 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_14 : scan edge (search edge 3) 4 [571] 571 innerTail102_14 best = best := by
  rw [innerTail102_14, scan, if_neg (by decide), innerCandidates102_14_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_15
    (if 3 ≤ innerCandidates102_14.length then search edge 3 [571, 5701] innerCandidates102_14 6272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_15

def innerTail102_13 : List Nat := 4987 :: innerTail102_14
def innerCandidates102_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_13_checked : candidates edge 4 571 4987 innerTail102_14 best.bound = innerCandidates102_13 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_13 : scan edge (search edge 3) 4 [571] 571 innerTail102_13 best = best := by
  rw [innerTail102_13, scan, if_neg (by decide), innerCandidates102_13_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_14
    (if 3 ≤ innerCandidates102_13.length then search edge 3 [571, 4987] innerCandidates102_13 5558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_14

def innerTail102_12 : List Nat := 4759 :: innerTail102_13
def innerCandidates102_12 : List Nat := [5701]
set_option maxRecDepth 4096 in
theorem innerCandidates102_12_checked : candidates edge 4 571 4759 innerTail102_13 best.bound = innerCandidates102_12 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_12 : scan edge (search edge 3) 4 [571] 571 innerTail102_12 best = best := by
  rw [innerTail102_12, scan, if_neg (by decide), innerCandidates102_12_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_13
    (if 3 ≤ innerCandidates102_12.length then search edge 3 [571, 4759] innerCandidates102_12 5330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_13

def innerTail102_11 : List Nat := 4483 :: innerTail102_12
def innerCandidates102_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_11_checked : candidates edge 4 571 4483 innerTail102_12 best.bound = innerCandidates102_11 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_11 : scan edge (search edge 3) 4 [571] 571 innerTail102_11 best = best := by
  rw [innerTail102_11, scan, if_neg (by decide), innerCandidates102_11_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_12
    (if 3 ≤ innerCandidates102_11.length then search edge 3 [571, 4483] innerCandidates102_11 5054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_12

def innerTail102_10 : List Nat := 3163 :: innerTail102_11
def innerCandidates102_10 : List Nat := [7951]
set_option maxRecDepth 4096 in
theorem innerCandidates102_10_checked : candidates edge 4 571 3163 innerTail102_11 best.bound = innerCandidates102_10 := by
  unfold candidates
  have row : edge 3163 = SearchData.row3163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_10 : scan edge (search edge 3) 4 [571] 571 innerTail102_10 best = best := by
  rw [innerTail102_10, scan, if_neg (by decide), innerCandidates102_10_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_11
    (if 3 ≤ innerCandidates102_10.length then search edge 3 [571, 3163] innerCandidates102_10 3734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_11

def innerTail102_9 : List Nat := 3121 :: innerTail102_10
def innerCandidates102_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_9_checked : candidates edge 4 571 3121 innerTail102_10 best.bound = innerCandidates102_9 := by
  unfold candidates
  have row : edge 3121 = SearchData.row3121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_9 : scan edge (search edge 3) 4 [571] 571 innerTail102_9 best = best := by
  rw [innerTail102_9, scan, if_neg (by decide), innerCandidates102_9_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_10
    (if 3 ≤ innerCandidates102_9.length then search edge 3 [571, 3121] innerCandidates102_9 3692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_10

def innerTail102_8 : List Nat := 3061 :: innerTail102_9
def innerCandidates102_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_8_checked : candidates edge 4 571 3061 innerTail102_9 best.bound = innerCandidates102_8 := by
  unfold candidates
  have row : edge 3061 = SearchData.row3061.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_8 : scan edge (search edge 3) 4 [571] 571 innerTail102_8 best = best := by
  rw [innerTail102_8, scan, if_neg (by decide), innerCandidates102_8_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_9
    (if 3 ≤ innerCandidates102_8.length then search edge 3 [571, 3061] innerCandidates102_8 3632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_9

def innerTail102_7 : List Nat := 2953 :: innerTail102_8
def innerCandidates102_7 : List Nat := [7213]
set_option maxRecDepth 4096 in
theorem innerCandidates102_7_checked : candidates edge 4 571 2953 innerTail102_8 best.bound = innerCandidates102_7 := by
  unfold candidates
  have row : edge 2953 = SearchData.row2953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_7 : scan edge (search edge 3) 4 [571] 571 innerTail102_7 best = best := by
  rw [innerTail102_7, scan, if_neg (by decide), innerCandidates102_7_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_8
    (if 3 ≤ innerCandidates102_7.length then search edge 3 [571, 2953] innerCandidates102_7 3524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_8

def innerTail102_6 : List Nat := 2659 :: innerTail102_7
def innerCandidates102_6 : List Nat := [4759]
set_option maxRecDepth 4096 in
theorem innerCandidates102_6_checked : candidates edge 4 571 2659 innerTail102_7 best.bound = innerCandidates102_6 := by
  unfold candidates
  have row : edge 2659 = PrimePairCache.smallRow2659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_6 : scan edge (search edge 3) 4 [571] 571 innerTail102_6 best = best := by
  rw [innerTail102_6, scan, if_neg (by decide), innerCandidates102_6_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_7
    (if 3 ≤ innerCandidates102_6.length then search edge 3 [571, 2659] innerCandidates102_6 3230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_7

def innerTail102_5 : List Nat := 2647 :: innerTail102_6
def innerCandidates102_5 : List Nat := [6553, 7951]
set_option maxRecDepth 4096 in
theorem innerCandidates102_5_checked : candidates edge 4 571 2647 innerTail102_6 best.bound = innerCandidates102_5 := by
  unfold candidates
  have row : edge 2647 = PrimePairCache.smallRow2647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_5 : scan edge (search edge 3) 4 [571] 571 innerTail102_5 best = best := by
  rw [innerTail102_5, scan, if_neg (by decide), innerCandidates102_5_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_6
    (if 3 ≤ innerCandidates102_5.length then search edge 3 [571, 2647] innerCandidates102_5 3218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_6

def innerTail102_4 : List Nat := 2383 :: innerTail102_5
def innerCandidates102_4 : List Nat := [2659, 8263]
set_option maxRecDepth 4096 in
theorem innerCandidates102_4_checked : candidates edge 4 571 2383 innerTail102_5 best.bound = innerCandidates102_4 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_4 : scan edge (search edge 3) 4 [571] 571 innerTail102_4 best = best := by
  rw [innerTail102_4, scan, if_neg (by decide), innerCandidates102_4_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_5
    (if 3 ≤ innerCandidates102_4.length then search edge 3 [571, 2383] innerCandidates102_4 2954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_5

def innerTail102_3 : List Nat := 2341 :: innerTail102_4
def innerCandidates102_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_3_checked : candidates edge 4 571 2341 innerTail102_4 best.bound = innerCandidates102_3 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_3 : scan edge (search edge 3) 4 [571] 571 innerTail102_3 best = best := by
  rw [innerTail102_3, scan, if_neg (by decide), innerCandidates102_3_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_4
    (if 3 ≤ innerCandidates102_3.length then search edge 3 [571, 2341] innerCandidates102_3 2912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_4

def innerTail102_2 : List Nat := 1933 :: innerTail102_3
def innerCandidates102_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_2_checked : candidates edge 4 571 1933 innerTail102_3 best.bound = innerCandidates102_2 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_2 : scan edge (search edge 3) 4 [571] 571 innerTail102_2 best = best := by
  rw [innerTail102_2, scan, if_neg (by decide), innerCandidates102_2_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_3
    (if 3 ≤ innerCandidates102_2.length then search edge 3 [571, 1933] innerCandidates102_2 2504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_3

def innerTail102_1 : List Nat := 1051 :: innerTail102_2
def innerCandidates102_1 : List Nat := [5839]
set_option maxRecDepth 4096 in
theorem innerCandidates102_1_checked : candidates edge 4 571 1051 innerTail102_2 best.bound = innerCandidates102_1 := by
  unfold candidates
  have row : edge 1051 = PrimePairCache.smallRow1051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_1 : scan edge (search edge 3) 4 [571] 571 innerTail102_1 best = best := by
  rw [innerTail102_1, scan, if_neg (by decide), innerCandidates102_1_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_2
    (if 3 ≤ innerCandidates102_1.length then search edge 3 [571, 1051] innerCandidates102_1 1622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_2

def innerTail102_0 : List Nat := 853 :: innerTail102_1
def innerCandidates102_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates102_0_checked : candidates edge 4 571 853 innerTail102_1 best.bound = innerCandidates102_0 := by
  unfold candidates
  have row : edge 853 = PrimePairCache.smallRow853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan102_0 : scan edge (search edge 3) 4 [571] 571 innerTail102_0 best = best := by
  rw [innerTail102_0, scan, if_neg (by decide), innerCandidates102_0_checked]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_1
    (if 3 ≤ innerCandidates102_0.length then search edge 3 [571, 853] innerCandidates102_0 1424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan102_1

theorem rootBranch102 : search edge 4 [571] rootCandidates102 571 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [571] 571 innerTail102_0 best = best
  exact innerScan102_0

def rootCandidates101 : List Nat := [659, 773, 797, 809, 887, 1151, 1487, 1601, 2657, 2699, 2861, 3203, 3617, 4673, 4919, 5741, 6959, 7673, 8039, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates101_checked : candidates edge 5 0 569 tail102 best.bound = rootCandidates101 := by
  unfold candidates
  have row : edge 569 = PrimePairCache.smallRow569.contains := by funext q; rfl
  rw [row]
  decide

def innerTail101_20 : List Nat := []
theorem innerScan101_20 : scan edge (search edge 3) 4 [569] 569 innerTail101_20 best = best := rfl

def innerTail101_19 : List Nat := 8219 :: innerTail101_20
def innerCandidates101_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_19_checked : candidates edge 4 569 8219 innerTail101_20 best.bound = innerCandidates101_19 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_19 : scan edge (search edge 3) 4 [569] 569 innerTail101_19 best = best := by
  rw [innerTail101_19, scan, if_neg (by decide), innerCandidates101_19_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_20
    (if 3 ≤ innerCandidates101_19.length then search edge 3 [569, 8219] innerCandidates101_19 8788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_20

def innerTail101_18 : List Nat := 8039 :: innerTail101_19
def innerCandidates101_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_18_checked : candidates edge 4 569 8039 innerTail101_19 best.bound = innerCandidates101_18 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_18 : scan edge (search edge 3) 4 [569] 569 innerTail101_18 best = best := by
  rw [innerTail101_18, scan, if_neg (by decide), innerCandidates101_18_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_19
    (if 3 ≤ innerCandidates101_18.length then search edge 3 [569, 8039] innerCandidates101_18 8608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_19

def innerTail101_17 : List Nat := 7673 :: innerTail101_18
def innerCandidates101_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_17_checked : candidates edge 4 569 7673 innerTail101_18 best.bound = innerCandidates101_17 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_17 : scan edge (search edge 3) 4 [569] 569 innerTail101_17 best = best := by
  rw [innerTail101_17, scan, if_neg (by decide), innerCandidates101_17_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_18
    (if 3 ≤ innerCandidates101_17.length then search edge 3 [569, 7673] innerCandidates101_17 8242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_18

def innerTail101_16 : List Nat := 6959 :: innerTail101_17
def innerCandidates101_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_16_checked : candidates edge 4 569 6959 innerTail101_17 best.bound = innerCandidates101_16 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_16 : scan edge (search edge 3) 4 [569] 569 innerTail101_16 best = best := by
  rw [innerTail101_16, scan, if_neg (by decide), innerCandidates101_16_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_17
    (if 3 ≤ innerCandidates101_16.length then search edge 3 [569, 6959] innerCandidates101_16 7528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_17

def innerTail101_15 : List Nat := 5741 :: innerTail101_16
def innerCandidates101_15 : List Nat := [6959, 8219]
set_option maxRecDepth 4096 in
theorem innerCandidates101_15_checked : candidates edge 4 569 5741 innerTail101_16 best.bound = innerCandidates101_15 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_15 : scan edge (search edge 3) 4 [569] 569 innerTail101_15 best = best := by
  rw [innerTail101_15, scan, if_neg (by decide), innerCandidates101_15_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_16
    (if 3 ≤ innerCandidates101_15.length then search edge 3 [569, 5741] innerCandidates101_15 6310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_16

def innerTail101_14 : List Nat := 4919 :: innerTail101_15
def innerCandidates101_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_14_checked : candidates edge 4 569 4919 innerTail101_15 best.bound = innerCandidates101_14 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_14 : scan edge (search edge 3) 4 [569] 569 innerTail101_14 best = best := by
  rw [innerTail101_14, scan, if_neg (by decide), innerCandidates101_14_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_15
    (if 3 ≤ innerCandidates101_14.length then search edge 3 [569, 4919] innerCandidates101_14 5488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_15

def innerTail101_13 : List Nat := 4673 :: innerTail101_14
def innerCandidates101_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_13_checked : candidates edge 4 569 4673 innerTail101_14 best.bound = innerCandidates101_13 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_13 : scan edge (search edge 3) 4 [569] 569 innerTail101_13 best = best := by
  rw [innerTail101_13, scan, if_neg (by decide), innerCandidates101_13_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_14
    (if 3 ≤ innerCandidates101_13.length then search edge 3 [569, 4673] innerCandidates101_13 5242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_14

def innerTail101_12 : List Nat := 3617 :: innerTail101_13
def innerCandidates101_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_12_checked : candidates edge 4 569 3617 innerTail101_13 best.bound = innerCandidates101_12 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_12 : scan edge (search edge 3) 4 [569] 569 innerTail101_12 best = best := by
  rw [innerTail101_12, scan, if_neg (by decide), innerCandidates101_12_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_13
    (if 3 ≤ innerCandidates101_12.length then search edge 3 [569, 3617] innerCandidates101_12 4186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_13

def innerTail101_11 : List Nat := 3203 :: innerTail101_12
def innerCandidates101_11 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates101_11_checked : candidates edge 4 569 3203 innerTail101_12 best.bound = innerCandidates101_11 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_11 : scan edge (search edge 3) 4 [569] 569 innerTail101_11 best = best := by
  rw [innerTail101_11, scan, if_neg (by decide), innerCandidates101_11_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_12
    (if 3 ≤ innerCandidates101_11.length then search edge 3 [569, 3203] innerCandidates101_11 3772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_12

def innerTail101_10 : List Nat := 2861 :: innerTail101_11
def innerCandidates101_10 : List Nat := [4919]
set_option maxRecDepth 4096 in
theorem innerCandidates101_10_checked : candidates edge 4 569 2861 innerTail101_11 best.bound = innerCandidates101_10 := by
  unfold candidates
  have row : edge 2861 = SearchData.row2861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_10 : scan edge (search edge 3) 4 [569] 569 innerTail101_10 best = best := by
  rw [innerTail101_10, scan, if_neg (by decide), innerCandidates101_10_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_11
    (if 3 ≤ innerCandidates101_10.length then search edge 3 [569, 2861] innerCandidates101_10 3430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_11

def innerTail101_9 : List Nat := 2699 :: innerTail101_10
def innerCandidates101_9 : List Nat := [8039]
set_option maxRecDepth 4096 in
theorem innerCandidates101_9_checked : candidates edge 4 569 2699 innerTail101_10 best.bound = innerCandidates101_9 := by
  unfold candidates
  have row : edge 2699 = SearchData.row2699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_9 : scan edge (search edge 3) 4 [569] 569 innerTail101_9 best = best := by
  rw [innerTail101_9, scan, if_neg (by decide), innerCandidates101_9_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_10
    (if 3 ≤ innerCandidates101_9.length then search edge 3 [569, 2699] innerCandidates101_9 3268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_10

def innerTail101_8 : List Nat := 2657 :: innerTail101_9
def innerCandidates101_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_8_checked : candidates edge 4 569 2657 innerTail101_9 best.bound = innerCandidates101_8 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_8 : scan edge (search edge 3) 4 [569] 569 innerTail101_8 best = best := by
  rw [innerTail101_8, scan, if_neg (by decide), innerCandidates101_8_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_9
    (if 3 ≤ innerCandidates101_8.length then search edge 3 [569, 2657] innerCandidates101_8 3226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_9

def innerTail101_7 : List Nat := 1601 :: innerTail101_8
def innerCandidates101_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_7_checked : candidates edge 4 569 1601 innerTail101_8 best.bound = innerCandidates101_7 := by
  unfold candidates
  have row : edge 1601 = PrimePairCache.smallRow1601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_7 : scan edge (search edge 3) 4 [569] 569 innerTail101_7 best = best := by
  rw [innerTail101_7, scan, if_neg (by decide), innerCandidates101_7_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_8
    (if 3 ≤ innerCandidates101_7.length then search edge 3 [569, 1601] innerCandidates101_7 2170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_8

def innerTail101_6 : List Nat := 1487 :: innerTail101_7
def innerCandidates101_6 : List Nat := [1601]
set_option maxRecDepth 4096 in
theorem innerCandidates101_6_checked : candidates edge 4 569 1487 innerTail101_7 best.bound = innerCandidates101_6 := by
  unfold candidates
  have row : edge 1487 = PrimePairCache.smallRow1487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_6 : scan edge (search edge 3) 4 [569] 569 innerTail101_6 best = best := by
  rw [innerTail101_6, scan, if_neg (by decide), innerCandidates101_6_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_7
    (if 3 ≤ innerCandidates101_6.length then search edge 3 [569, 1487] innerCandidates101_6 2056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_7

def innerTail101_5 : List Nat := 1151 :: innerTail101_6
def innerCandidates101_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_5_checked : candidates edge 4 569 1151 innerTail101_6 best.bound = innerCandidates101_5 := by
  unfold candidates
  have row : edge 1151 = PrimePairCache.smallRow1151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_5 : scan edge (search edge 3) 4 [569] 569 innerTail101_5 best = best := by
  rw [innerTail101_5, scan, if_neg (by decide), innerCandidates101_5_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_6
    (if 3 ≤ innerCandidates101_5.length then search edge 3 [569, 1151] innerCandidates101_5 1720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_6

def innerTail101_4 : List Nat := 887 :: innerTail101_5
def innerCandidates101_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_4_checked : candidates edge 4 569 887 innerTail101_5 best.bound = innerCandidates101_4 := by
  unfold candidates
  have row : edge 887 = PrimePairCache.smallRow887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_4 : scan edge (search edge 3) 4 [569] 569 innerTail101_4 best = best := by
  rw [innerTail101_4, scan, if_neg (by decide), innerCandidates101_4_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_5
    (if 3 ≤ innerCandidates101_4.length then search edge 3 [569, 887] innerCandidates101_4 1456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_5

def innerTail101_3 : List Nat := 809 :: innerTail101_4
def innerCandidates101_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_3_checked : candidates edge 4 569 809 innerTail101_4 best.bound = innerCandidates101_3 := by
  unfold candidates
  have row : edge 809 = PrimePairCache.smallRow809.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_3 : scan edge (search edge 3) 4 [569] 569 innerTail101_3 best = best := by
  rw [innerTail101_3, scan, if_neg (by decide), innerCandidates101_3_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_4
    (if 3 ≤ innerCandidates101_3.length then search edge 3 [569, 809] innerCandidates101_3 1378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_4

def innerTail101_2 : List Nat := 797 :: innerTail101_3
def innerCandidates101_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_2_checked : candidates edge 4 569 797 innerTail101_3 best.bound = innerCandidates101_2 := by
  unfold candidates
  have row : edge 797 = PrimePairCache.smallRow797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_2 : scan edge (search edge 3) 4 [569] 569 innerTail101_2 best = best := by
  rw [innerTail101_2, scan, if_neg (by decide), innerCandidates101_2_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_3
    (if 3 ≤ innerCandidates101_2.length then search edge 3 [569, 797] innerCandidates101_2 1366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_3

def innerTail101_1 : List Nat := 773 :: innerTail101_2
def innerCandidates101_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_1_checked : candidates edge 4 569 773 innerTail101_2 best.bound = innerCandidates101_1 := by
  unfold candidates
  have row : edge 773 = PrimePairCache.smallRow773.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_1 : scan edge (search edge 3) 4 [569] 569 innerTail101_1 best = best := by
  rw [innerTail101_1, scan, if_neg (by decide), innerCandidates101_1_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_2
    (if 3 ≤ innerCandidates101_1.length then search edge 3 [569, 773] innerCandidates101_1 1342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_2

def innerTail101_0 : List Nat := 659 :: innerTail101_1
def innerCandidates101_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates101_0_checked : candidates edge 4 569 659 innerTail101_1 best.bound = innerCandidates101_0 := by
  unfold candidates
  have row : edge 659 = PrimePairCache.smallRow659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan101_0 : scan edge (search edge 3) 4 [569] 569 innerTail101_0 best = best := by
  rw [innerTail101_0, scan, if_neg (by decide), innerCandidates101_0_checked]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_1
    (if 3 ≤ innerCandidates101_0.length then search edge 3 [569, 659] innerCandidates101_0 1228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan101_1

theorem rootBranch101 : search edge 4 [569] rootCandidates101 569 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [569] 569 innerTail101_0 best = best
  exact innerScan101_0

end ProjectEulerSolutions.PrimePairUpper.Search
