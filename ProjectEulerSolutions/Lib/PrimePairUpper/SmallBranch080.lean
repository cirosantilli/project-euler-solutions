import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates478 : List Nat := [3643, 4051, 4129, 4987, 5233, 5407, 6373, 6451, 6607, 6733, 7129, 7207, 7741, 8167, 8221]
set_option maxRecDepth 4096 in
theorem rootCandidates478_checked : candidates edge 5 0 3433 tail479 best.bound = rootCandidates478 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext q; rfl
  rw [row]
  decide

def innerTail478_15 : List Nat := []
theorem innerScan478_15 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_15 best = best := rfl

def innerTail478_14 : List Nat := 8221 :: innerTail478_15
def innerCandidates478_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates478_14_checked : candidates edge 4 3433 8221 innerTail478_15 best.bound = innerCandidates478_14 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_14 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_14 best = best := by
  rw [innerTail478_14, scan, if_neg (by decide), innerCandidates478_14_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_15
    (if 3 ≤ innerCandidates478_14.length then search edge 3 [3433, 8221] innerCandidates478_14 11654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_15

def innerTail478_13 : List Nat := 8167 :: innerTail478_14
def innerCandidates478_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates478_13_checked : candidates edge 4 3433 8167 innerTail478_14 best.bound = innerCandidates478_13 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_13 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_13 best = best := by
  rw [innerTail478_13, scan, if_neg (by decide), innerCandidates478_13_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_14
    (if 3 ≤ innerCandidates478_13.length then search edge 3 [3433, 8167] innerCandidates478_13 11600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_14

def innerTail478_12 : List Nat := 7741 :: innerTail478_13
def innerCandidates478_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates478_12_checked : candidates edge 4 3433 7741 innerTail478_13 best.bound = innerCandidates478_12 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_12 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_12 best = best := by
  rw [innerTail478_12, scan, if_neg (by decide), innerCandidates478_12_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_13
    (if 3 ≤ innerCandidates478_12.length then search edge 3 [3433, 7741] innerCandidates478_12 11174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_13

def innerTail478_11 : List Nat := 7207 :: innerTail478_12
def innerCandidates478_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates478_11_checked : candidates edge 4 3433 7207 innerTail478_12 best.bound = innerCandidates478_11 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_11 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_11 best = best := by
  rw [innerTail478_11, scan, if_neg (by decide), innerCandidates478_11_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_12
    (if 3 ≤ innerCandidates478_11.length then search edge 3 [3433, 7207] innerCandidates478_11 10640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_12

def innerTail478_10 : List Nat := 7129 :: innerTail478_11
def innerCandidates478_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates478_10_checked : candidates edge 4 3433 7129 innerTail478_11 best.bound = innerCandidates478_10 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_10 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_10 best = best := by
  rw [innerTail478_10, scan, if_neg (by decide), innerCandidates478_10_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_11
    (if 3 ≤ innerCandidates478_10.length then search edge 3 [3433, 7129] innerCandidates478_10 10562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_11

def innerTail478_9 : List Nat := 6733 :: innerTail478_10
def innerCandidates478_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates478_9_checked : candidates edge 4 3433 6733 innerTail478_10 best.bound = innerCandidates478_9 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_9 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_9 best = best := by
  rw [innerTail478_9, scan, if_neg (by decide), innerCandidates478_9_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_10
    (if 3 ≤ innerCandidates478_9.length then search edge 3 [3433, 6733] innerCandidates478_9 10166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_10

def innerTail478_8 : List Nat := 6607 :: innerTail478_9
def innerCandidates478_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates478_8_checked : candidates edge 4 3433 6607 innerTail478_9 best.bound = innerCandidates478_8 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_8 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_8 best = best := by
  rw [innerTail478_8, scan, if_neg (by decide), innerCandidates478_8_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_9
    (if 3 ≤ innerCandidates478_8.length then search edge 3 [3433, 6607] innerCandidates478_8 10040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_9

def innerTail478_7 : List Nat := 6451 :: innerTail478_8
def innerCandidates478_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates478_7_checked : candidates edge 4 3433 6451 innerTail478_8 best.bound = innerCandidates478_7 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_7 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_7 best = best := by
  rw [innerTail478_7, scan, if_neg (by decide), innerCandidates478_7_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_8
    (if 3 ≤ innerCandidates478_7.length then search edge 3 [3433, 6451] innerCandidates478_7 9884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_8

def innerTail478_6 : List Nat := 6373 :: innerTail478_7
def innerCandidates478_6 : List Nat := [8167]
set_option maxRecDepth 4096 in
theorem innerCandidates478_6_checked : candidates edge 4 3433 6373 innerTail478_7 best.bound = innerCandidates478_6 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_6 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_6 best = best := by
  rw [innerTail478_6, scan, if_neg (by decide), innerCandidates478_6_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_7
    (if 3 ≤ innerCandidates478_6.length then search edge 3 [3433, 6373] innerCandidates478_6 9806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_7

def innerTail478_5 : List Nat := 5407 :: innerTail478_6
def innerCandidates478_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates478_5_checked : candidates edge 4 3433 5407 innerTail478_6 best.bound = innerCandidates478_5 := by
  unfold candidates
  have row : edge 5407 = SearchData.row5407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_5 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_5 best = best := by
  rw [innerTail478_5, scan, if_neg (by decide), innerCandidates478_5_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_6
    (if 3 ≤ innerCandidates478_5.length then search edge 3 [3433, 5407] innerCandidates478_5 8840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_6

def innerTail478_4 : List Nat := 5233 :: innerTail478_5
def innerCandidates478_4 : List Nat := [6373, 6451]
set_option maxRecDepth 4096 in
theorem innerCandidates478_4_checked : candidates edge 4 3433 5233 innerTail478_5 best.bound = innerCandidates478_4 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_4 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_4 best = best := by
  rw [innerTail478_4, scan, if_neg (by decide), innerCandidates478_4_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_5
    (if 3 ≤ innerCandidates478_4.length then search edge 3 [3433, 5233] innerCandidates478_4 8666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_5

def innerTail478_3 : List Nat := 4987 :: innerTail478_4
def innerCandidates478_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates478_3_checked : candidates edge 4 3433 4987 innerTail478_4 best.bound = innerCandidates478_3 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_3 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_3 best = best := by
  rw [innerTail478_3, scan, if_neg (by decide), innerCandidates478_3_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_4
    (if 3 ≤ innerCandidates478_3.length then search edge 3 [3433, 4987] innerCandidates478_3 8420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_4

def innerTail478_2 : List Nat := 4129 :: innerTail478_3
def innerCandidates478_2 : List Nat := [8167, 8221]
set_option maxRecDepth 4096 in
theorem innerCandidates478_2_checked : candidates edge 4 3433 4129 innerTail478_3 best.bound = innerCandidates478_2 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_2 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_2 best = best := by
  rw [innerTail478_2, scan, if_neg (by decide), innerCandidates478_2_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_3
    (if 3 ≤ innerCandidates478_2.length then search edge 3 [3433, 4129] innerCandidates478_2 7562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_3

def innerTail478_1 : List Nat := 4051 :: innerTail478_2
def innerCandidates478_1 : List Nat := [4129]
set_option maxRecDepth 4096 in
theorem innerCandidates478_1_checked : candidates edge 4 3433 4051 innerTail478_2 best.bound = innerCandidates478_1 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_1 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_1 best = best := by
  rw [innerTail478_1, scan, if_neg (by decide), innerCandidates478_1_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_2
    (if 3 ≤ innerCandidates478_1.length then search edge 3 [3433, 4051] innerCandidates478_1 7484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_2

def innerTail478_0 : List Nat := 3643 :: innerTail478_1
def innerCandidates478_0 : List Nat := [4987, 6607]
set_option maxRecDepth 4096 in
theorem innerCandidates478_0_checked : candidates edge 4 3433 3643 innerTail478_1 best.bound = innerCandidates478_0 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan478_0 : scan edge (search edge 3) 4 [3433] 3433 innerTail478_0 best = best := by
  rw [innerTail478_0, scan, if_neg (by decide), innerCandidates478_0_checked]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_1
    (if 3 ≤ innerCandidates478_0.length then search edge 3 [3433, 3643] innerCandidates478_0 7076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan478_1

theorem rootBranch478 : search edge 4 [3433] rootCandidates478 3433 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3433] 3433 innerTail478_0 best = best
  exact innerScan478_0

def rootCandidates477 : List Nat := [3761, 3929, 4649, 5009, 5939, 6101, 6449, 6653, 6689, 7001, 7043, 7211, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates477_checked : candidates edge 5 0 3413 tail478 best.bound = rootCandidates477 := by
  unfold candidates
  have row : edge 3413 = SearchData.row3413.contains := by funext q; rfl
  rw [row]
  decide

def innerTail477_13 : List Nat := []
theorem innerScan477_13 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_13 best = best := rfl

def innerTail477_12 : List Nat := 7949 :: innerTail477_13
def innerCandidates477_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates477_12_checked : candidates edge 4 3413 7949 innerTail477_13 best.bound = innerCandidates477_12 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_12 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_12 best = best := by
  rw [innerTail477_12, scan, if_neg (by decide), innerCandidates477_12_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_13
    (if 3 ≤ innerCandidates477_12.length then search edge 3 [3413, 7949] innerCandidates477_12 11362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_13

def innerTail477_11 : List Nat := 7211 :: innerTail477_12
def innerCandidates477_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates477_11_checked : candidates edge 4 3413 7211 innerTail477_12 best.bound = innerCandidates477_11 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_11 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_11 best = best := by
  rw [innerTail477_11, scan, if_neg (by decide), innerCandidates477_11_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_12
    (if 3 ≤ innerCandidates477_11.length then search edge 3 [3413, 7211] innerCandidates477_11 10624 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_12

def innerTail477_10 : List Nat := 7043 :: innerTail477_11
def innerCandidates477_10 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates477_10_checked : candidates edge 4 3413 7043 innerTail477_11 best.bound = innerCandidates477_10 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_10 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_10 best = best := by
  rw [innerTail477_10, scan, if_neg (by decide), innerCandidates477_10_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_11
    (if 3 ≤ innerCandidates477_10.length then search edge 3 [3413, 7043] innerCandidates477_10 10456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_11

def innerTail477_9 : List Nat := 7001 :: innerTail477_10
def innerCandidates477_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates477_9_checked : candidates edge 4 3413 7001 innerTail477_10 best.bound = innerCandidates477_9 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_9 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_9 best = best := by
  rw [innerTail477_9, scan, if_neg (by decide), innerCandidates477_9_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_10
    (if 3 ≤ innerCandidates477_9.length then search edge 3 [3413, 7001] innerCandidates477_9 10414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_10

def innerTail477_8 : List Nat := 6689 :: innerTail477_9
def innerCandidates477_8 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates477_8_checked : candidates edge 4 3413 6689 innerTail477_9 best.bound = innerCandidates477_8 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_8 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_8 best = best := by
  rw [innerTail477_8, scan, if_neg (by decide), innerCandidates477_8_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_9
    (if 3 ≤ innerCandidates477_8.length then search edge 3 [3413, 6689] innerCandidates477_8 10102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_9

def innerTail477_7 : List Nat := 6653 :: innerTail477_8
def innerCandidates477_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates477_7_checked : candidates edge 4 3413 6653 innerTail477_8 best.bound = innerCandidates477_7 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_7 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_7 best = best := by
  rw [innerTail477_7, scan, if_neg (by decide), innerCandidates477_7_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_8
    (if 3 ≤ innerCandidates477_7.length then search edge 3 [3413, 6653] innerCandidates477_7 10066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_8

def innerTail477_6 : List Nat := 6449 :: innerTail477_7
def innerCandidates477_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates477_6_checked : candidates edge 4 3413 6449 innerTail477_7 best.bound = innerCandidates477_6 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_6 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_6 best = best := by
  rw [innerTail477_6, scan, if_neg (by decide), innerCandidates477_6_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_7
    (if 3 ≤ innerCandidates477_6.length then search edge 3 [3413, 6449] innerCandidates477_6 9862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_7

def innerTail477_5 : List Nat := 6101 :: innerTail477_6
def innerCandidates477_5 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates477_5_checked : candidates edge 4 3413 6101 innerTail477_6 best.bound = innerCandidates477_5 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_5 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_5 best = best := by
  rw [innerTail477_5, scan, if_neg (by decide), innerCandidates477_5_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_6
    (if 3 ≤ innerCandidates477_5.length then search edge 3 [3413, 6101] innerCandidates477_5 9514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_6

def innerTail477_4 : List Nat := 5939 :: innerTail477_5
def innerCandidates477_4 : List Nat := [6101]
set_option maxRecDepth 4096 in
theorem innerCandidates477_4_checked : candidates edge 4 3413 5939 innerTail477_5 best.bound = innerCandidates477_4 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_4 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_4 best = best := by
  rw [innerTail477_4, scan, if_neg (by decide), innerCandidates477_4_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_5
    (if 3 ≤ innerCandidates477_4.length then search edge 3 [3413, 5939] innerCandidates477_4 9352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_5

def innerTail477_3 : List Nat := 5009 :: innerTail477_4
def innerCandidates477_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates477_3_checked : candidates edge 4 3413 5009 innerTail477_4 best.bound = innerCandidates477_3 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_3 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_3 best = best := by
  rw [innerTail477_3, scan, if_neg (by decide), innerCandidates477_3_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_4
    (if 3 ≤ innerCandidates477_3.length then search edge 3 [3413, 5009] innerCandidates477_3 8422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_4

def innerTail477_2 : List Nat := 4649 :: innerTail477_3
def innerCandidates477_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates477_2_checked : candidates edge 4 3413 4649 innerTail477_3 best.bound = innerCandidates477_2 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_2 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_2 best = best := by
  rw [innerTail477_2, scan, if_neg (by decide), innerCandidates477_2_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_3
    (if 3 ≤ innerCandidates477_2.length then search edge 3 [3413, 4649] innerCandidates477_2 8062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_3

def innerTail477_1 : List Nat := 3929 :: innerTail477_2
def innerCandidates477_1 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates477_1_checked : candidates edge 4 3413 3929 innerTail477_2 best.bound = innerCandidates477_1 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_1 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_1 best = best := by
  rw [innerTail477_1, scan, if_neg (by decide), innerCandidates477_1_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_2
    (if 3 ≤ innerCandidates477_1.length then search edge 3 [3413, 3929] innerCandidates477_1 7342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_2

def innerTail477_0 : List Nat := 3761 :: innerTail477_1
def innerCandidates477_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates477_0_checked : candidates edge 4 3413 3761 innerTail477_1 best.bound = innerCandidates477_0 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan477_0 : scan edge (search edge 3) 4 [3413] 3413 innerTail477_0 best = best := by
  rw [innerTail477_0, scan, if_neg (by decide), innerCandidates477_0_checked]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_1
    (if 3 ≤ innerCandidates477_0.length then search edge 3 [3413, 3761] innerCandidates477_0 7174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan477_1

theorem rootBranch477 : search edge 4 [3413] rootCandidates477 3413 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3413] 3413 innerTail477_0 best = best
  exact innerScan477_0

def rootCandidates476 : List Nat := [3671, 4751, 4889, 6299, 6389, 6599, 7079, 7691, 7919]
set_option maxRecDepth 4096 in
theorem rootCandidates476_checked : candidates edge 5 0 3407 tail477 best.bound = rootCandidates476 := by
  unfold candidates
  have row : edge 3407 = SearchData.row3407.contains := by funext q; rfl
  rw [row]
  decide

def innerTail476_9 : List Nat := []
theorem innerScan476_9 : scan edge (search edge 3) 4 [3407] 3407 innerTail476_9 best = best := rfl

def innerTail476_8 : List Nat := 7919 :: innerTail476_9
def innerCandidates476_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates476_8_checked : candidates edge 4 3407 7919 innerTail476_9 best.bound = innerCandidates476_8 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan476_8 : scan edge (search edge 3) 4 [3407] 3407 innerTail476_8 best = best := by
  rw [innerTail476_8, scan, if_neg (by decide), innerCandidates476_8_checked]
  change scan edge (search edge 3) 4 [3407] 3407 innerTail476_9
    (if 3 ≤ innerCandidates476_8.length then search edge 3 [3407, 7919] innerCandidates476_8 11326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan476_9

def innerTail476_7 : List Nat := 7691 :: innerTail476_8
def innerCandidates476_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates476_7_checked : candidates edge 4 3407 7691 innerTail476_8 best.bound = innerCandidates476_7 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan476_7 : scan edge (search edge 3) 4 [3407] 3407 innerTail476_7 best = best := by
  rw [innerTail476_7, scan, if_neg (by decide), innerCandidates476_7_checked]
  change scan edge (search edge 3) 4 [3407] 3407 innerTail476_8
    (if 3 ≤ innerCandidates476_7.length then search edge 3 [3407, 7691] innerCandidates476_7 11098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan476_8

def innerTail476_6 : List Nat := 7079 :: innerTail476_7
def innerCandidates476_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates476_6_checked : candidates edge 4 3407 7079 innerTail476_7 best.bound = innerCandidates476_6 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan476_6 : scan edge (search edge 3) 4 [3407] 3407 innerTail476_6 best = best := by
  rw [innerTail476_6, scan, if_neg (by decide), innerCandidates476_6_checked]
  change scan edge (search edge 3) 4 [3407] 3407 innerTail476_7
    (if 3 ≤ innerCandidates476_6.length then search edge 3 [3407, 7079] innerCandidates476_6 10486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan476_7

def innerTail476_5 : List Nat := 6599 :: innerTail476_6
def innerCandidates476_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates476_5_checked : candidates edge 4 3407 6599 innerTail476_6 best.bound = innerCandidates476_5 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan476_5 : scan edge (search edge 3) 4 [3407] 3407 innerTail476_5 best = best := by
  rw [innerTail476_5, scan, if_neg (by decide), innerCandidates476_5_checked]
  change scan edge (search edge 3) 4 [3407] 3407 innerTail476_6
    (if 3 ≤ innerCandidates476_5.length then search edge 3 [3407, 6599] innerCandidates476_5 10006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan476_6

def innerTail476_4 : List Nat := 6389 :: innerTail476_5
def innerCandidates476_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates476_4_checked : candidates edge 4 3407 6389 innerTail476_5 best.bound = innerCandidates476_4 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan476_4 : scan edge (search edge 3) 4 [3407] 3407 innerTail476_4 best = best := by
  rw [innerTail476_4, scan, if_neg (by decide), innerCandidates476_4_checked]
  change scan edge (search edge 3) 4 [3407] 3407 innerTail476_5
    (if 3 ≤ innerCandidates476_4.length then search edge 3 [3407, 6389] innerCandidates476_4 9796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan476_5

def innerTail476_3 : List Nat := 6299 :: innerTail476_4
def innerCandidates476_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates476_3_checked : candidates edge 4 3407 6299 innerTail476_4 best.bound = innerCandidates476_3 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan476_3 : scan edge (search edge 3) 4 [3407] 3407 innerTail476_3 best = best := by
  rw [innerTail476_3, scan, if_neg (by decide), innerCandidates476_3_checked]
  change scan edge (search edge 3) 4 [3407] 3407 innerTail476_4
    (if 3 ≤ innerCandidates476_3.length then search edge 3 [3407, 6299] innerCandidates476_3 9706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan476_4

def innerTail476_2 : List Nat := 4889 :: innerTail476_3
def innerCandidates476_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates476_2_checked : candidates edge 4 3407 4889 innerTail476_3 best.bound = innerCandidates476_2 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan476_2 : scan edge (search edge 3) 4 [3407] 3407 innerTail476_2 best = best := by
  rw [innerTail476_2, scan, if_neg (by decide), innerCandidates476_2_checked]
  change scan edge (search edge 3) 4 [3407] 3407 innerTail476_3
    (if 3 ≤ innerCandidates476_2.length then search edge 3 [3407, 4889] innerCandidates476_2 8296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan476_3

def innerTail476_1 : List Nat := 4751 :: innerTail476_2
def innerCandidates476_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates476_1_checked : candidates edge 4 3407 4751 innerTail476_2 best.bound = innerCandidates476_1 := by
  unfold candidates
  have row : edge 4751 = SearchData.row4751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan476_1 : scan edge (search edge 3) 4 [3407] 3407 innerTail476_1 best = best := by
  rw [innerTail476_1, scan, if_neg (by decide), innerCandidates476_1_checked]
  change scan edge (search edge 3) 4 [3407] 3407 innerTail476_2
    (if 3 ≤ innerCandidates476_1.length then search edge 3 [3407, 4751] innerCandidates476_1 8158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan476_2

def innerTail476_0 : List Nat := 3671 :: innerTail476_1
def innerCandidates476_0 : List Nat := [7691]
set_option maxRecDepth 4096 in
theorem innerCandidates476_0_checked : candidates edge 4 3407 3671 innerTail476_1 best.bound = innerCandidates476_0 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan476_0 : scan edge (search edge 3) 4 [3407] 3407 innerTail476_0 best = best := by
  rw [innerTail476_0, scan, if_neg (by decide), innerCandidates476_0_checked]
  change scan edge (search edge 3) 4 [3407] 3407 innerTail476_1
    (if 3 ≤ innerCandidates476_0.length then search edge 3 [3407, 3671] innerCandidates476_0 7078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan476_1

theorem rootBranch476 : search edge 4 [3407] rootCandidates476 3407 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3407] 3407 innerTail476_0 best = best
  exact innerScan476_0

def rootCandidates475 : List Nat := [3433, 3511, 3643, 5107, 5113, 5227, 5443, 6079, 6199, 6361, 6607, 6733, 6907, 6997, 7237, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates475_checked : candidates edge 5 0 3391 tail476 best.bound = rootCandidates475 := by
  unfold candidates
  have row : edge 3391 = SearchData.row3391.contains := by funext q; rfl
  rw [row]
  decide

def innerTail475_16 : List Nat := []
theorem innerScan475_16 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_16 best = best := rfl

def innerTail475_15 : List Nat := 8329 :: innerTail475_16
def innerCandidates475_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_15_checked : candidates edge 4 3391 8329 innerTail475_16 best.bound = innerCandidates475_15 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_15 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_15 best = best := by
  rw [innerTail475_15, scan, if_neg (by decide), innerCandidates475_15_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_16
    (if 3 ≤ innerCandidates475_15.length then search edge 3 [3391, 8329] innerCandidates475_15 11720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_16

def innerTail475_14 : List Nat := 7237 :: innerTail475_15
def innerCandidates475_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_14_checked : candidates edge 4 3391 7237 innerTail475_15 best.bound = innerCandidates475_14 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_14 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_14 best = best := by
  rw [innerTail475_14, scan, if_neg (by decide), innerCandidates475_14_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_15
    (if 3 ≤ innerCandidates475_14.length then search edge 3 [3391, 7237] innerCandidates475_14 10628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_15

def innerTail475_13 : List Nat := 6997 :: innerTail475_14
def innerCandidates475_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_13_checked : candidates edge 4 3391 6997 innerTail475_14 best.bound = innerCandidates475_13 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_13 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_13 best = best := by
  rw [innerTail475_13, scan, if_neg (by decide), innerCandidates475_13_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_14
    (if 3 ≤ innerCandidates475_13.length then search edge 3 [3391, 6997] innerCandidates475_13 10388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_14

def innerTail475_12 : List Nat := 6907 :: innerTail475_13
def innerCandidates475_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_12_checked : candidates edge 4 3391 6907 innerTail475_13 best.bound = innerCandidates475_12 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_12 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_12 best = best := by
  rw [innerTail475_12, scan, if_neg (by decide), innerCandidates475_12_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_13
    (if 3 ≤ innerCandidates475_12.length then search edge 3 [3391, 6907] innerCandidates475_12 10298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_13

def innerTail475_11 : List Nat := 6733 :: innerTail475_12
def innerCandidates475_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_11_checked : candidates edge 4 3391 6733 innerTail475_12 best.bound = innerCandidates475_11 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_11 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_11 best = best := by
  rw [innerTail475_11, scan, if_neg (by decide), innerCandidates475_11_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_12
    (if 3 ≤ innerCandidates475_11.length then search edge 3 [3391, 6733] innerCandidates475_11 10124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_12

def innerTail475_10 : List Nat := 6607 :: innerTail475_11
def innerCandidates475_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_10_checked : candidates edge 4 3391 6607 innerTail475_11 best.bound = innerCandidates475_10 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_10 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_10 best = best := by
  rw [innerTail475_10, scan, if_neg (by decide), innerCandidates475_10_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_11
    (if 3 ≤ innerCandidates475_10.length then search edge 3 [3391, 6607] innerCandidates475_10 9998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_11

def innerTail475_9 : List Nat := 6361 :: innerTail475_10
def innerCandidates475_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_9_checked : candidates edge 4 3391 6361 innerTail475_10 best.bound = innerCandidates475_9 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_9 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_9 best = best := by
  rw [innerTail475_9, scan, if_neg (by decide), innerCandidates475_9_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_10
    (if 3 ≤ innerCandidates475_9.length then search edge 3 [3391, 6361] innerCandidates475_9 9752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_10

def innerTail475_8 : List Nat := 6199 :: innerTail475_9
def innerCandidates475_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_8_checked : candidates edge 4 3391 6199 innerTail475_9 best.bound = innerCandidates475_8 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_8 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_8 best = best := by
  rw [innerTail475_8, scan, if_neg (by decide), innerCandidates475_8_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_9
    (if 3 ≤ innerCandidates475_8.length then search edge 3 [3391, 6199] innerCandidates475_8 9590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_9

def innerTail475_7 : List Nat := 6079 :: innerTail475_8
def innerCandidates475_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_7_checked : candidates edge 4 3391 6079 innerTail475_8 best.bound = innerCandidates475_7 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_7 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_7 best = best := by
  rw [innerTail475_7, scan, if_neg (by decide), innerCandidates475_7_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_8
    (if 3 ≤ innerCandidates475_7.length then search edge 3 [3391, 6079] innerCandidates475_7 9470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_8

def innerTail475_6 : List Nat := 5443 :: innerTail475_7
def innerCandidates475_6 : List Nat := [6079]
set_option maxRecDepth 4096 in
theorem innerCandidates475_6_checked : candidates edge 4 3391 5443 innerTail475_7 best.bound = innerCandidates475_6 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_6 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_6 best = best := by
  rw [innerTail475_6, scan, if_neg (by decide), innerCandidates475_6_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_7
    (if 3 ≤ innerCandidates475_6.length then search edge 3 [3391, 5443] innerCandidates475_6 8834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_7

def innerTail475_5 : List Nat := 5227 :: innerTail475_6
def innerCandidates475_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_5_checked : candidates edge 4 3391 5227 innerTail475_6 best.bound = innerCandidates475_5 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_5 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_5 best = best := by
  rw [innerTail475_5, scan, if_neg (by decide), innerCandidates475_5_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_6
    (if 3 ≤ innerCandidates475_5.length then search edge 3 [3391, 5227] innerCandidates475_5 8618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_6

def innerTail475_4 : List Nat := 5113 :: innerTail475_5
def innerCandidates475_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_4_checked : candidates edge 4 3391 5113 innerTail475_5 best.bound = innerCandidates475_4 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_4 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_4 best = best := by
  rw [innerTail475_4, scan, if_neg (by decide), innerCandidates475_4_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_5
    (if 3 ≤ innerCandidates475_4.length then search edge 3 [3391, 5113] innerCandidates475_4 8504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_5

def innerTail475_3 : List Nat := 5107 :: innerTail475_4
def innerCandidates475_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_3_checked : candidates edge 4 3391 5107 innerTail475_4 best.bound = innerCandidates475_3 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_3 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_3 best = best := by
  rw [innerTail475_3, scan, if_neg (by decide), innerCandidates475_3_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_4
    (if 3 ≤ innerCandidates475_3.length then search edge 3 [3391, 5107] innerCandidates475_3 8498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_4

def innerTail475_2 : List Nat := 3643 :: innerTail475_3
def innerCandidates475_2 : List Nat := [6199, 6607]
set_option maxRecDepth 4096 in
theorem innerCandidates475_2_checked : candidates edge 4 3391 3643 innerTail475_3 best.bound = innerCandidates475_2 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_2 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_2 best = best := by
  rw [innerTail475_2, scan, if_neg (by decide), innerCandidates475_2_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_3
    (if 3 ≤ innerCandidates475_2.length then search edge 3 [3391, 3643] innerCandidates475_2 7034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_3

def innerTail475_1 : List Nat := 3511 :: innerTail475_2
def innerCandidates475_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates475_1_checked : candidates edge 4 3391 3511 innerTail475_2 best.bound = innerCandidates475_1 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan475_1 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_1 best = best := by
  rw [innerTail475_1, scan, if_neg (by decide), innerCandidates475_1_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_2
    (if 3 ≤ innerCandidates475_1.length then search edge 3 [3391, 3511] innerCandidates475_1 6902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan475_2

def innerTail475_0 : List Nat := 3433 :: innerTail475_1
def innerCandidates475_0 : List Nat := [3643, 6607, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates475_0_checked : candidates edge 4 3391 3433 innerTail475_1 best.bound = innerCandidates475_0 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild475_0 : search edge 3 [3391, 3433] innerCandidates475_0 6824 best = best := by decide

theorem innerScan475_0 : scan edge (search edge 3) 4 [3391] 3391 innerTail475_0 best = best := by
  rw [innerTail475_0, scan, if_neg (by decide), innerCandidates475_0_checked]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_1
    (if 3 ≤ innerCandidates475_0.length then search edge 3 [3391, 3433] innerCandidates475_0 6824 best else best) = best
  rw [if_pos (by decide), innerChild475_0]
  exact innerScan475_1

theorem rootBranch475 : search edge 4 [3391] rootCandidates475 3391 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3391] 3391 innerTail475_0 best = best
  exact innerScan475_0

end ProjectEulerSolutions.PrimePairUpper.Search
