import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates399 : List Nat := [2927, 2963, 3089, 3203, 3371, 4583, 4799, 5171, 6101, 6287, 6791, 7001, 7457, 7481, 7487, 7529, 7919]
set_option maxRecDepth 4096 in
theorem rootCandidates399_checked : candidates edge 5 0 2753 tail400 best.bound = rootCandidates399 := by
  unfold candidates
  have row : edge 2753 = SearchData.row2753.contains := by funext q; rfl
  rw [row]
  decide

def innerTail399_17 : List Nat := []
theorem innerScan399_17 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_17 best = best := rfl

def innerTail399_16 : List Nat := 7919 :: innerTail399_17
def innerCandidates399_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates399_16_checked : candidates edge 4 2753 7919 innerTail399_17 best.bound = innerCandidates399_16 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_16 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_16 best = best := by
  rw [innerTail399_16, scan, if_neg (by decide), innerCandidates399_16_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_17
    (if 3 ≤ innerCandidates399_16.length then search edge 3 [2753, 7919] innerCandidates399_16 10672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_17

def innerTail399_15 : List Nat := 7529 :: innerTail399_16
def innerCandidates399_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates399_15_checked : candidates edge 4 2753 7529 innerTail399_16 best.bound = innerCandidates399_15 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_15 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_15 best = best := by
  rw [innerTail399_15, scan, if_neg (by decide), innerCandidates399_15_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_16
    (if 3 ≤ innerCandidates399_15.length then search edge 3 [2753, 7529] innerCandidates399_15 10282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_16

def innerTail399_14 : List Nat := 7487 :: innerTail399_15
def innerCandidates399_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates399_14_checked : candidates edge 4 2753 7487 innerTail399_15 best.bound = innerCandidates399_14 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_14 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_14 best = best := by
  rw [innerTail399_14, scan, if_neg (by decide), innerCandidates399_14_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_15
    (if 3 ≤ innerCandidates399_14.length then search edge 3 [2753, 7487] innerCandidates399_14 10240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_15

def innerTail399_13 : List Nat := 7481 :: innerTail399_14
def innerCandidates399_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates399_13_checked : candidates edge 4 2753 7481 innerTail399_14 best.bound = innerCandidates399_13 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_13 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_13 best = best := by
  rw [innerTail399_13, scan, if_neg (by decide), innerCandidates399_13_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_14
    (if 3 ≤ innerCandidates399_13.length then search edge 3 [2753, 7481] innerCandidates399_13 10234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_14

def innerTail399_12 : List Nat := 7457 :: innerTail399_13
def innerCandidates399_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates399_12_checked : candidates edge 4 2753 7457 innerTail399_13 best.bound = innerCandidates399_12 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_12 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_12 best = best := by
  rw [innerTail399_12, scan, if_neg (by decide), innerCandidates399_12_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_13
    (if 3 ≤ innerCandidates399_12.length then search edge 3 [2753, 7457] innerCandidates399_12 10210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_13

def innerTail399_11 : List Nat := 7001 :: innerTail399_12
def innerCandidates399_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates399_11_checked : candidates edge 4 2753 7001 innerTail399_12 best.bound = innerCandidates399_11 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_11 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_11 best = best := by
  rw [innerTail399_11, scan, if_neg (by decide), innerCandidates399_11_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_12
    (if 3 ≤ innerCandidates399_11.length then search edge 3 [2753, 7001] innerCandidates399_11 9754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_12

def innerTail399_10 : List Nat := 6791 :: innerTail399_11
def innerCandidates399_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates399_10_checked : candidates edge 4 2753 6791 innerTail399_11 best.bound = innerCandidates399_10 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_10 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_10 best = best := by
  rw [innerTail399_10, scan, if_neg (by decide), innerCandidates399_10_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_11
    (if 3 ≤ innerCandidates399_10.length then search edge 3 [2753, 6791] innerCandidates399_10 9544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_11

def innerTail399_9 : List Nat := 6287 :: innerTail399_10
def innerCandidates399_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates399_9_checked : candidates edge 4 2753 6287 innerTail399_10 best.bound = innerCandidates399_9 := by
  unfold candidates
  have row : edge 6287 = SearchData.row6287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_9 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_9 best = best := by
  rw [innerTail399_9, scan, if_neg (by decide), innerCandidates399_9_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_10
    (if 3 ≤ innerCandidates399_9.length then search edge 3 [2753, 6287] innerCandidates399_9 9040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_10

def innerTail399_8 : List Nat := 6101 :: innerTail399_9
def innerCandidates399_8 : List Nat := [7487]
set_option maxRecDepth 4096 in
theorem innerCandidates399_8_checked : candidates edge 4 2753 6101 innerTail399_9 best.bound = innerCandidates399_8 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_8 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_8 best = best := by
  rw [innerTail399_8, scan, if_neg (by decide), innerCandidates399_8_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_9
    (if 3 ≤ innerCandidates399_8.length then search edge 3 [2753, 6101] innerCandidates399_8 8854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_9

def innerTail399_7 : List Nat := 5171 :: innerTail399_8
def innerCandidates399_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates399_7_checked : candidates edge 4 2753 5171 innerTail399_8 best.bound = innerCandidates399_7 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_7 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_7 best = best := by
  rw [innerTail399_7, scan, if_neg (by decide), innerCandidates399_7_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_8
    (if 3 ≤ innerCandidates399_7.length then search edge 3 [2753, 5171] innerCandidates399_7 7924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_8

def innerTail399_6 : List Nat := 4799 :: innerTail399_7
def innerCandidates399_6 : List Nat := [7487]
set_option maxRecDepth 4096 in
theorem innerCandidates399_6_checked : candidates edge 4 2753 4799 innerTail399_7 best.bound = innerCandidates399_6 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_6 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_6 best = best := by
  rw [innerTail399_6, scan, if_neg (by decide), innerCandidates399_6_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_7
    (if 3 ≤ innerCandidates399_6.length then search edge 3 [2753, 4799] innerCandidates399_6 7552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_7

def innerTail399_5 : List Nat := 4583 :: innerTail399_6
def innerCandidates399_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates399_5_checked : candidates edge 4 2753 4583 innerTail399_6 best.bound = innerCandidates399_5 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_5 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_5 best = best := by
  rw [innerTail399_5, scan, if_neg (by decide), innerCandidates399_5_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_6
    (if 3 ≤ innerCandidates399_5.length then search edge 3 [2753, 4583] innerCandidates399_5 7336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_6

def innerTail399_4 : List Nat := 3371 :: innerTail399_5
def innerCandidates399_4 : List Nat := [4799]
set_option maxRecDepth 4096 in
theorem innerCandidates399_4_checked : candidates edge 4 2753 3371 innerTail399_5 best.bound = innerCandidates399_4 := by
  unfold candidates
  have row : edge 3371 = SearchData.row3371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_4 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_4 best = best := by
  rw [innerTail399_4, scan, if_neg (by decide), innerCandidates399_4_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_5
    (if 3 ≤ innerCandidates399_4.length then search edge 3 [2753, 3371] innerCandidates399_4 6124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_5

def innerTail399_3 : List Nat := 3203 :: innerTail399_4
def innerCandidates399_3 : List Nat := [6101]
set_option maxRecDepth 4096 in
theorem innerCandidates399_3_checked : candidates edge 4 2753 3203 innerTail399_4 best.bound = innerCandidates399_3 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_3 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_3 best = best := by
  rw [innerTail399_3, scan, if_neg (by decide), innerCandidates399_3_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_4
    (if 3 ≤ innerCandidates399_3.length then search edge 3 [2753, 3203] innerCandidates399_3 5956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_4

def innerTail399_2 : List Nat := 3089 :: innerTail399_3
def innerCandidates399_2 : List Nat := [7457]
set_option maxRecDepth 4096 in
theorem innerCandidates399_2_checked : candidates edge 4 2753 3089 innerTail399_3 best.bound = innerCandidates399_2 := by
  unfold candidates
  have row : edge 3089 = SearchData.row3089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_2 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_2 best = best := by
  rw [innerTail399_2, scan, if_neg (by decide), innerCandidates399_2_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_3
    (if 3 ≤ innerCandidates399_2.length then search edge 3 [2753, 3089] innerCandidates399_2 5842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_3

def innerTail399_1 : List Nat := 2963 :: innerTail399_2
def innerCandidates399_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates399_1_checked : candidates edge 4 2753 2963 innerTail399_2 best.bound = innerCandidates399_1 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_1 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_1 best = best := by
  rw [innerTail399_1, scan, if_neg (by decide), innerCandidates399_1_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_2
    (if 3 ≤ innerCandidates399_1.length then search edge 3 [2753, 2963] innerCandidates399_1 5716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_2

def innerTail399_0 : List Nat := 2927 :: innerTail399_1
def innerCandidates399_0 : List Nat := [3203]
set_option maxRecDepth 4096 in
theorem innerCandidates399_0_checked : candidates edge 4 2753 2927 innerTail399_1 best.bound = innerCandidates399_0 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan399_0 : scan edge (search edge 3) 4 [2753] 2753 innerTail399_0 best = best := by
  rw [innerTail399_0, scan, if_neg (by decide), innerCandidates399_0_checked]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_1
    (if 3 ≤ innerCandidates399_0.length then search edge 3 [2753, 2927] innerCandidates399_0 5680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan399_1

theorem rootBranch399 : search edge 4 [2753] rootCandidates399 2753 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2753] 2753 innerTail399_0 best = best
  exact innerScan399_0

def rootCandidates398 : List Nat := [3541, 3643, 4021, 4261, 4297, 4993, 6067, 7219, 7741, 8017]
set_option maxRecDepth 4096 in
theorem rootCandidates398_checked : candidates edge 5 0 2749 tail399 best.bound = rootCandidates398 := by
  unfold candidates
  have row : edge 2749 = SearchData.row2749.contains := by funext q; rfl
  rw [row]
  decide

def innerTail398_10 : List Nat := []
theorem innerScan398_10 : scan edge (search edge 3) 4 [2749] 2749 innerTail398_10 best = best := rfl

def innerTail398_9 : List Nat := 8017 :: innerTail398_10
def innerCandidates398_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates398_9_checked : candidates edge 4 2749 8017 innerTail398_10 best.bound = innerCandidates398_9 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan398_9 : scan edge (search edge 3) 4 [2749] 2749 innerTail398_9 best = best := by
  rw [innerTail398_9, scan, if_neg (by decide), innerCandidates398_9_checked]
  change scan edge (search edge 3) 4 [2749] 2749 innerTail398_10
    (if 3 ≤ innerCandidates398_9.length then search edge 3 [2749, 8017] innerCandidates398_9 10766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan398_10

def innerTail398_8 : List Nat := 7741 :: innerTail398_9
def innerCandidates398_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates398_8_checked : candidates edge 4 2749 7741 innerTail398_9 best.bound = innerCandidates398_8 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan398_8 : scan edge (search edge 3) 4 [2749] 2749 innerTail398_8 best = best := by
  rw [innerTail398_8, scan, if_neg (by decide), innerCandidates398_8_checked]
  change scan edge (search edge 3) 4 [2749] 2749 innerTail398_9
    (if 3 ≤ innerCandidates398_8.length then search edge 3 [2749, 7741] innerCandidates398_8 10490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan398_9

def innerTail398_7 : List Nat := 7219 :: innerTail398_8
def innerCandidates398_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates398_7_checked : candidates edge 4 2749 7219 innerTail398_8 best.bound = innerCandidates398_7 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan398_7 : scan edge (search edge 3) 4 [2749] 2749 innerTail398_7 best = best := by
  rw [innerTail398_7, scan, if_neg (by decide), innerCandidates398_7_checked]
  change scan edge (search edge 3) 4 [2749] 2749 innerTail398_8
    (if 3 ≤ innerCandidates398_7.length then search edge 3 [2749, 7219] innerCandidates398_7 9968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan398_8

def innerTail398_6 : List Nat := 6067 :: innerTail398_7
def innerCandidates398_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates398_6_checked : candidates edge 4 2749 6067 innerTail398_7 best.bound = innerCandidates398_6 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan398_6 : scan edge (search edge 3) 4 [2749] 2749 innerTail398_6 best = best := by
  rw [innerTail398_6, scan, if_neg (by decide), innerCandidates398_6_checked]
  change scan edge (search edge 3) 4 [2749] 2749 innerTail398_7
    (if 3 ≤ innerCandidates398_6.length then search edge 3 [2749, 6067] innerCandidates398_6 8816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan398_7

def innerTail398_5 : List Nat := 4993 :: innerTail398_6
def innerCandidates398_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates398_5_checked : candidates edge 4 2749 4993 innerTail398_6 best.bound = innerCandidates398_5 := by
  unfold candidates
  have row : edge 4993 = SearchData.row4993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan398_5 : scan edge (search edge 3) 4 [2749] 2749 innerTail398_5 best = best := by
  rw [innerTail398_5, scan, if_neg (by decide), innerCandidates398_5_checked]
  change scan edge (search edge 3) 4 [2749] 2749 innerTail398_6
    (if 3 ≤ innerCandidates398_5.length then search edge 3 [2749, 4993] innerCandidates398_5 7742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan398_6

def innerTail398_4 : List Nat := 4297 :: innerTail398_5
def innerCandidates398_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates398_4_checked : candidates edge 4 2749 4297 innerTail398_5 best.bound = innerCandidates398_4 := by
  unfold candidates
  have row : edge 4297 = SearchData.row4297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan398_4 : scan edge (search edge 3) 4 [2749] 2749 innerTail398_4 best = best := by
  rw [innerTail398_4, scan, if_neg (by decide), innerCandidates398_4_checked]
  change scan edge (search edge 3) 4 [2749] 2749 innerTail398_5
    (if 3 ≤ innerCandidates398_4.length then search edge 3 [2749, 4297] innerCandidates398_4 7046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan398_5

def innerTail398_3 : List Nat := 4261 :: innerTail398_4
def innerCandidates398_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates398_3_checked : candidates edge 4 2749 4261 innerTail398_4 best.bound = innerCandidates398_3 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan398_3 : scan edge (search edge 3) 4 [2749] 2749 innerTail398_3 best = best := by
  rw [innerTail398_3, scan, if_neg (by decide), innerCandidates398_3_checked]
  change scan edge (search edge 3) 4 [2749] 2749 innerTail398_4
    (if 3 ≤ innerCandidates398_3.length then search edge 3 [2749, 4261] innerCandidates398_3 7010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan398_4

def innerTail398_2 : List Nat := 4021 :: innerTail398_3
def innerCandidates398_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates398_2_checked : candidates edge 4 2749 4021 innerTail398_3 best.bound = innerCandidates398_2 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan398_2 : scan edge (search edge 3) 4 [2749] 2749 innerTail398_2 best = best := by
  rw [innerTail398_2, scan, if_neg (by decide), innerCandidates398_2_checked]
  change scan edge (search edge 3) 4 [2749] 2749 innerTail398_3
    (if 3 ≤ innerCandidates398_2.length then search edge 3 [2749, 4021] innerCandidates398_2 6770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan398_3

def innerTail398_1 : List Nat := 3643 :: innerTail398_2
def innerCandidates398_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates398_1_checked : candidates edge 4 2749 3643 innerTail398_2 best.bound = innerCandidates398_1 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan398_1 : scan edge (search edge 3) 4 [2749] 2749 innerTail398_1 best = best := by
  rw [innerTail398_1, scan, if_neg (by decide), innerCandidates398_1_checked]
  change scan edge (search edge 3) 4 [2749] 2749 innerTail398_2
    (if 3 ≤ innerCandidates398_1.length then search edge 3 [2749, 3643] innerCandidates398_1 6392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan398_2

def innerTail398_0 : List Nat := 3541 :: innerTail398_1
def innerCandidates398_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates398_0_checked : candidates edge 4 2749 3541 innerTail398_1 best.bound = innerCandidates398_0 := by
  unfold candidates
  have row : edge 3541 = SearchData.row3541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan398_0 : scan edge (search edge 3) 4 [2749] 2749 innerTail398_0 best = best := by
  rw [innerTail398_0, scan, if_neg (by decide), innerCandidates398_0_checked]
  change scan edge (search edge 3) 4 [2749] 2749 innerTail398_1
    (if 3 ≤ innerCandidates398_0.length then search edge 3 [2749, 3541] innerCandidates398_0 6290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan398_1

theorem rootBranch398 : search edge 4 [2749] rootCandidates398 2749 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2749] 2749 innerTail398_0 best = best
  exact innerScan398_0

def rootCandidates397 : List Nat := [2837, 2963, 3119, 3329, 3389, 3917, 4133, 4253, 4649, 4799, 6113, 6353, 6689, 6737, 6899, 6911, 7331, 7457, 7757]
set_option maxRecDepth 4096 in
theorem rootCandidates397_checked : candidates edge 5 0 2741 tail398 best.bound = rootCandidates397 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext q; rfl
  rw [row]
  decide

def innerTail397_19 : List Nat := []
theorem innerScan397_19 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_19 best = best := rfl

def innerTail397_18 : List Nat := 7757 :: innerTail397_19
def innerCandidates397_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates397_18_checked : candidates edge 4 2741 7757 innerTail397_19 best.bound = innerCandidates397_18 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_18 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_18 best = best := by
  rw [innerTail397_18, scan, if_neg (by decide), innerCandidates397_18_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_19
    (if 3 ≤ innerCandidates397_18.length then search edge 3 [2741, 7757] innerCandidates397_18 10498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_19

def innerTail397_17 : List Nat := 7457 :: innerTail397_18
def innerCandidates397_17 : List Nat := [7757]
set_option maxRecDepth 4096 in
theorem innerCandidates397_17_checked : candidates edge 4 2741 7457 innerTail397_18 best.bound = innerCandidates397_17 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_17 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_17 best = best := by
  rw [innerTail397_17, scan, if_neg (by decide), innerCandidates397_17_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_18
    (if 3 ≤ innerCandidates397_17.length then search edge 3 [2741, 7457] innerCandidates397_17 10198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_18

def innerTail397_16 : List Nat := 7331 :: innerTail397_17
def innerCandidates397_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates397_16_checked : candidates edge 4 2741 7331 innerTail397_17 best.bound = innerCandidates397_16 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_16 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_16 best = best := by
  rw [innerTail397_16, scan, if_neg (by decide), innerCandidates397_16_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_17
    (if 3 ≤ innerCandidates397_16.length then search edge 3 [2741, 7331] innerCandidates397_16 10072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_17

def innerTail397_15 : List Nat := 6911 :: innerTail397_16
def innerCandidates397_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates397_15_checked : candidates edge 4 2741 6911 innerTail397_16 best.bound = innerCandidates397_15 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_15 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_15 best = best := by
  rw [innerTail397_15, scan, if_neg (by decide), innerCandidates397_15_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_16
    (if 3 ≤ innerCandidates397_15.length then search edge 3 [2741, 6911] innerCandidates397_15 9652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_16

def innerTail397_14 : List Nat := 6899 :: innerTail397_15
def innerCandidates397_14 : List Nat := [6911]
set_option maxRecDepth 4096 in
theorem innerCandidates397_14_checked : candidates edge 4 2741 6899 innerTail397_15 best.bound = innerCandidates397_14 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_14 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_14 best = best := by
  rw [innerTail397_14, scan, if_neg (by decide), innerCandidates397_14_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_15
    (if 3 ≤ innerCandidates397_14.length then search edge 3 [2741, 6899] innerCandidates397_14 9640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_15

def innerTail397_13 : List Nat := 6737 :: innerTail397_14
def innerCandidates397_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates397_13_checked : candidates edge 4 2741 6737 innerTail397_14 best.bound = innerCandidates397_13 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_13 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_13 best = best := by
  rw [innerTail397_13, scan, if_neg (by decide), innerCandidates397_13_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_14
    (if 3 ≤ innerCandidates397_13.length then search edge 3 [2741, 6737] innerCandidates397_13 9478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_14

def innerTail397_12 : List Nat := 6689 :: innerTail397_13
def innerCandidates397_12 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates397_12_checked : candidates edge 4 2741 6689 innerTail397_13 best.bound = innerCandidates397_12 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_12 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_12 best = best := by
  rw [innerTail397_12, scan, if_neg (by decide), innerCandidates397_12_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_13
    (if 3 ≤ innerCandidates397_12.length then search edge 3 [2741, 6689] innerCandidates397_12 9430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_13

def innerTail397_11 : List Nat := 6353 :: innerTail397_12
def innerCandidates397_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates397_11_checked : candidates edge 4 2741 6353 innerTail397_12 best.bound = innerCandidates397_11 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_11 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_11 best = best := by
  rw [innerTail397_11, scan, if_neg (by decide), innerCandidates397_11_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_12
    (if 3 ≤ innerCandidates397_11.length then search edge 3 [2741, 6353] innerCandidates397_11 9094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_12

def innerTail397_10 : List Nat := 6113 :: innerTail397_11
def innerCandidates397_10 : List Nat := [6353]
set_option maxRecDepth 4096 in
theorem innerCandidates397_10_checked : candidates edge 4 2741 6113 innerTail397_11 best.bound = innerCandidates397_10 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_10 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_10 best = best := by
  rw [innerTail397_10, scan, if_neg (by decide), innerCandidates397_10_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_11
    (if 3 ≤ innerCandidates397_10.length then search edge 3 [2741, 6113] innerCandidates397_10 8854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_11

def innerTail397_9 : List Nat := 4799 :: innerTail397_10
def innerCandidates397_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates397_9_checked : candidates edge 4 2741 4799 innerTail397_10 best.bound = innerCandidates397_9 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_9 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_9 best = best := by
  rw [innerTail397_9, scan, if_neg (by decide), innerCandidates397_9_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_10
    (if 3 ≤ innerCandidates397_9.length then search edge 3 [2741, 4799] innerCandidates397_9 7540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_10

def innerTail397_8 : List Nat := 4649 :: innerTail397_9
def innerCandidates397_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates397_8_checked : candidates edge 4 2741 4649 innerTail397_9 best.bound = innerCandidates397_8 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_8 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_8 best = best := by
  rw [innerTail397_8, scan, if_neg (by decide), innerCandidates397_8_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_9
    (if 3 ≤ innerCandidates397_8.length then search edge 3 [2741, 4649] innerCandidates397_8 7390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_9

def innerTail397_7 : List Nat := 4253 :: innerTail397_8
def innerCandidates397_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates397_7_checked : candidates edge 4 2741 4253 innerTail397_8 best.bound = innerCandidates397_7 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_7 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_7 best = best := by
  rw [innerTail397_7, scan, if_neg (by decide), innerCandidates397_7_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_8
    (if 3 ≤ innerCandidates397_7.length then search edge 3 [2741, 4253] innerCandidates397_7 6994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_8

def innerTail397_6 : List Nat := 4133 :: innerTail397_7
def innerCandidates397_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates397_6_checked : candidates edge 4 2741 4133 innerTail397_7 best.bound = innerCandidates397_6 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_6 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_6 best = best := by
  rw [innerTail397_6, scan, if_neg (by decide), innerCandidates397_6_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_7
    (if 3 ≤ innerCandidates397_6.length then search edge 3 [2741, 4133] innerCandidates397_6 6874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_7

def innerTail397_5 : List Nat := 3917 :: innerTail397_6
def innerCandidates397_5 : List Nat := [4649, 6899, 7331]
set_option maxRecDepth 4096 in
theorem innerCandidates397_5_checked : candidates edge 4 2741 3917 innerTail397_6 best.bound = innerCandidates397_5 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild397_5 : search edge 3 [2741, 3917] innerCandidates397_5 6658 best = best := by decide

theorem innerScan397_5 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_5 best = best := by
  rw [innerTail397_5, scan, if_neg (by decide), innerCandidates397_5_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_6
    (if 3 ≤ innerCandidates397_5.length then search edge 3 [2741, 3917] innerCandidates397_5 6658 best else best) = best
  rw [if_pos (by decide), innerChild397_5]
  exact innerScan397_6

def innerTail397_4 : List Nat := 3389 :: innerTail397_5
def innerCandidates397_4 : List Nat := [4799]
set_option maxRecDepth 4096 in
theorem innerCandidates397_4_checked : candidates edge 4 2741 3389 innerTail397_5 best.bound = innerCandidates397_4 := by
  unfold candidates
  have row : edge 3389 = SearchData.row3389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_4 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_4 best = best := by
  rw [innerTail397_4, scan, if_neg (by decide), innerCandidates397_4_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_5
    (if 3 ≤ innerCandidates397_4.length then search edge 3 [2741, 3389] innerCandidates397_4 6130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_5

def innerTail397_3 : List Nat := 3329 :: innerTail397_4
def innerCandidates397_3 : List Nat := [4649]
set_option maxRecDepth 4096 in
theorem innerCandidates397_3_checked : candidates edge 4 2741 3329 innerTail397_4 best.bound = innerCandidates397_3 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_3 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_3 best = best := by
  rw [innerTail397_3, scan, if_neg (by decide), innerCandidates397_3_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_4
    (if 3 ≤ innerCandidates397_3.length then search edge 3 [2741, 3329] innerCandidates397_3 6070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_4

def innerTail397_2 : List Nat := 3119 :: innerTail397_3
def innerCandidates397_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates397_2_checked : candidates edge 4 2741 3119 innerTail397_3 best.bound = innerCandidates397_2 := by
  unfold candidates
  have row : edge 3119 = SearchData.row3119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_2 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_2 best = best := by
  rw [innerTail397_2, scan, if_neg (by decide), innerCandidates397_2_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_3
    (if 3 ≤ innerCandidates397_2.length then search edge 3 [2741, 3119] innerCandidates397_2 5860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_3

def innerTail397_1 : List Nat := 2963 :: innerTail397_2
def innerCandidates397_1 : List Nat := [6689]
set_option maxRecDepth 4096 in
theorem innerCandidates397_1_checked : candidates edge 4 2741 2963 innerTail397_2 best.bound = innerCandidates397_1 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_1 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_1 best = best := by
  rw [innerTail397_1, scan, if_neg (by decide), innerCandidates397_1_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_2
    (if 3 ≤ innerCandidates397_1.length then search edge 3 [2741, 2963] innerCandidates397_1 5704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_2

def innerTail397_0 : List Nat := 2837 :: innerTail397_1
def innerCandidates397_0 : List Nat := [7757]
set_option maxRecDepth 4096 in
theorem innerCandidates397_0_checked : candidates edge 4 2741 2837 innerTail397_1 best.bound = innerCandidates397_0 := by
  unfold candidates
  have row : edge 2837 = SearchData.row2837.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan397_0 : scan edge (search edge 3) 4 [2741] 2741 innerTail397_0 best = best := by
  rw [innerTail397_0, scan, if_neg (by decide), innerCandidates397_0_checked]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_1
    (if 3 ≤ innerCandidates397_0.length then search edge 3 [2741, 2837] innerCandidates397_0 5578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan397_1

theorem rootBranch397 : search edge 4 [2741] rootCandidates397 2741 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2741] 2741 innerTail397_0 best = best
  exact innerScan397_0

def rootCandidates396 : List Nat := [3301, 3571, 3613, 4129, 4423, 4657, 4759, 5641, 6451, 7753, 7867, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates396_checked : candidates edge 5 0 2731 tail397 best.bound = rootCandidates396 := by
  unfold candidates
  have row : edge 2731 = SearchData.row2731.contains := by funext q; rfl
  rw [row]
  decide

def innerTail396_12 : List Nat := []
theorem innerScan396_12 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_12 best = best := rfl

def innerTail396_11 : List Nat := 7993 :: innerTail396_12
def innerCandidates396_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates396_11_checked : candidates edge 4 2731 7993 innerTail396_12 best.bound = innerCandidates396_11 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan396_11 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_11 best = best := by
  rw [innerTail396_11, scan, if_neg (by decide), innerCandidates396_11_checked]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_12
    (if 3 ≤ innerCandidates396_11.length then search edge 3 [2731, 7993] innerCandidates396_11 10724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan396_12

def innerTail396_10 : List Nat := 7867 :: innerTail396_11
def innerCandidates396_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates396_10_checked : candidates edge 4 2731 7867 innerTail396_11 best.bound = innerCandidates396_10 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan396_10 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_10 best = best := by
  rw [innerTail396_10, scan, if_neg (by decide), innerCandidates396_10_checked]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_11
    (if 3 ≤ innerCandidates396_10.length then search edge 3 [2731, 7867] innerCandidates396_10 10598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan396_11

def innerTail396_9 : List Nat := 7753 :: innerTail396_10
def innerCandidates396_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates396_9_checked : candidates edge 4 2731 7753 innerTail396_10 best.bound = innerCandidates396_9 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan396_9 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_9 best = best := by
  rw [innerTail396_9, scan, if_neg (by decide), innerCandidates396_9_checked]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_10
    (if 3 ≤ innerCandidates396_9.length then search edge 3 [2731, 7753] innerCandidates396_9 10484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan396_10

def innerTail396_8 : List Nat := 6451 :: innerTail396_9
def innerCandidates396_8 : List Nat := [7753]
set_option maxRecDepth 4096 in
theorem innerCandidates396_8_checked : candidates edge 4 2731 6451 innerTail396_9 best.bound = innerCandidates396_8 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan396_8 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_8 best = best := by
  rw [innerTail396_8, scan, if_neg (by decide), innerCandidates396_8_checked]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_9
    (if 3 ≤ innerCandidates396_8.length then search edge 3 [2731, 6451] innerCandidates396_8 9182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan396_9

def innerTail396_7 : List Nat := 5641 :: innerTail396_8
def innerCandidates396_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates396_7_checked : candidates edge 4 2731 5641 innerTail396_8 best.bound = innerCandidates396_7 := by
  unfold candidates
  have row : edge 5641 = SearchData.row5641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan396_7 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_7 best = best := by
  rw [innerTail396_7, scan, if_neg (by decide), innerCandidates396_7_checked]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_8
    (if 3 ≤ innerCandidates396_7.length then search edge 3 [2731, 5641] innerCandidates396_7 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan396_8

def innerTail396_6 : List Nat := 4759 :: innerTail396_7
def innerCandidates396_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates396_6_checked : candidates edge 4 2731 4759 innerTail396_7 best.bound = innerCandidates396_6 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan396_6 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_6 best = best := by
  rw [innerTail396_6, scan, if_neg (by decide), innerCandidates396_6_checked]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_7
    (if 3 ≤ innerCandidates396_6.length then search edge 3 [2731, 4759] innerCandidates396_6 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan396_7

def innerTail396_5 : List Nat := 4657 :: innerTail396_6
def innerCandidates396_5 : List Nat := [7753]
set_option maxRecDepth 4096 in
theorem innerCandidates396_5_checked : candidates edge 4 2731 4657 innerTail396_6 best.bound = innerCandidates396_5 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan396_5 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_5 best = best := by
  rw [innerTail396_5, scan, if_neg (by decide), innerCandidates396_5_checked]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_6
    (if 3 ≤ innerCandidates396_5.length then search edge 3 [2731, 4657] innerCandidates396_5 7388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan396_6

def innerTail396_4 : List Nat := 4423 :: innerTail396_5
def innerCandidates396_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates396_4_checked : candidates edge 4 2731 4423 innerTail396_5 best.bound = innerCandidates396_4 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan396_4 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_4 best = best := by
  rw [innerTail396_4, scan, if_neg (by decide), innerCandidates396_4_checked]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_5
    (if 3 ≤ innerCandidates396_4.length then search edge 3 [2731, 4423] innerCandidates396_4 7154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan396_5

def innerTail396_3 : List Nat := 4129 :: innerTail396_4
def innerCandidates396_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates396_3_checked : candidates edge 4 2731 4129 innerTail396_4 best.bound = innerCandidates396_3 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan396_3 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_3 best = best := by
  rw [innerTail396_3, scan, if_neg (by decide), innerCandidates396_3_checked]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_4
    (if 3 ≤ innerCandidates396_3.length then search edge 3 [2731, 4129] innerCandidates396_3 6860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan396_4

def innerTail396_2 : List Nat := 3613 :: innerTail396_3
def innerCandidates396_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates396_2_checked : candidates edge 4 2731 3613 innerTail396_3 best.bound = innerCandidates396_2 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan396_2 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_2 best = best := by
  rw [innerTail396_2, scan, if_neg (by decide), innerCandidates396_2_checked]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_3
    (if 3 ≤ innerCandidates396_2.length then search edge 3 [2731, 3613] innerCandidates396_2 6344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan396_3

def innerTail396_1 : List Nat := 3571 :: innerTail396_2
def innerCandidates396_1 : List Nat := [6451]
set_option maxRecDepth 4096 in
theorem innerCandidates396_1_checked : candidates edge 4 2731 3571 innerTail396_2 best.bound = innerCandidates396_1 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan396_1 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_1 best = best := by
  rw [innerTail396_1, scan, if_neg (by decide), innerCandidates396_1_checked]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_2
    (if 3 ≤ innerCandidates396_1.length then search edge 3 [2731, 3571] innerCandidates396_1 6302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan396_2

def innerTail396_0 : List Nat := 3301 :: innerTail396_1
def innerCandidates396_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates396_0_checked : candidates edge 4 2731 3301 innerTail396_1 best.bound = innerCandidates396_0 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan396_0 : scan edge (search edge 3) 4 [2731] 2731 innerTail396_0 best = best := by
  rw [innerTail396_0, scan, if_neg (by decide), innerCandidates396_0_checked]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_1
    (if 3 ≤ innerCandidates396_0.length then search edge 3 [2731, 3301] innerCandidates396_0 6032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan396_1

theorem rootBranch396 : search edge 4 [2731] rootCandidates396 2731 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2731] 2731 innerTail396_0 best = best
  exact innerScan396_0

end ProjectEulerSolutions.PrimePairUpper.Search
