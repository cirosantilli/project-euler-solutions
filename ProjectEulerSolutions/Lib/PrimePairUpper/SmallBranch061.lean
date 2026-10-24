import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates580 : List Nat := [4397, 4463, 4937, 5657, 5939, 6917, 7829]
set_option maxRecDepth 4096 in
theorem rootCandidates580_checked : candidates edge 5 0 4253 tail581 best.bound = rootCandidates580 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext q; rfl
  rw [row]
  decide

def innerTail580_7 : List Nat := []
theorem innerScan580_7 : scan edge (search edge 3) 4 [4253] 4253 innerTail580_7 best = best := rfl

def innerTail580_6 : List Nat := 7829 :: innerTail580_7
def innerCandidates580_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates580_6_checked : candidates edge 4 4253 7829 innerTail580_7 best.bound = innerCandidates580_6 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan580_6 : scan edge (search edge 3) 4 [4253] 4253 innerTail580_6 best = best := by
  rw [innerTail580_6, scan, if_neg (by decide), innerCandidates580_6_checked]
  change scan edge (search edge 3) 4 [4253] 4253 innerTail580_7
    (if 3 ≤ innerCandidates580_6.length then search edge 3 [4253, 7829] innerCandidates580_6 12082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan580_7

def innerTail580_5 : List Nat := 6917 :: innerTail580_6
def innerCandidates580_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates580_5_checked : candidates edge 4 4253 6917 innerTail580_6 best.bound = innerCandidates580_5 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan580_5 : scan edge (search edge 3) 4 [4253] 4253 innerTail580_5 best = best := by
  rw [innerTail580_5, scan, if_neg (by decide), innerCandidates580_5_checked]
  change scan edge (search edge 3) 4 [4253] 4253 innerTail580_6
    (if 3 ≤ innerCandidates580_5.length then search edge 3 [4253, 6917] innerCandidates580_5 11170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan580_6

def innerTail580_4 : List Nat := 5939 :: innerTail580_5
def innerCandidates580_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates580_4_checked : candidates edge 4 4253 5939 innerTail580_5 best.bound = innerCandidates580_4 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan580_4 : scan edge (search edge 3) 4 [4253] 4253 innerTail580_4 best = best := by
  rw [innerTail580_4, scan, if_neg (by decide), innerCandidates580_4_checked]
  change scan edge (search edge 3) 4 [4253] 4253 innerTail580_5
    (if 3 ≤ innerCandidates580_4.length then search edge 3 [4253, 5939] innerCandidates580_4 10192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan580_5

def innerTail580_3 : List Nat := 5657 :: innerTail580_4
def innerCandidates580_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates580_3_checked : candidates edge 4 4253 5657 innerTail580_4 best.bound = innerCandidates580_3 := by
  unfold candidates
  have row : edge 5657 = SearchData.row5657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan580_3 : scan edge (search edge 3) 4 [4253] 4253 innerTail580_3 best = best := by
  rw [innerTail580_3, scan, if_neg (by decide), innerCandidates580_3_checked]
  change scan edge (search edge 3) 4 [4253] 4253 innerTail580_4
    (if 3 ≤ innerCandidates580_3.length then search edge 3 [4253, 5657] innerCandidates580_3 9910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan580_4

def innerTail580_2 : List Nat := 4937 :: innerTail580_3
def innerCandidates580_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates580_2_checked : candidates edge 4 4253 4937 innerTail580_3 best.bound = innerCandidates580_2 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan580_2 : scan edge (search edge 3) 4 [4253] 4253 innerTail580_2 best = best := by
  rw [innerTail580_2, scan, if_neg (by decide), innerCandidates580_2_checked]
  change scan edge (search edge 3) 4 [4253] 4253 innerTail580_3
    (if 3 ≤ innerCandidates580_2.length then search edge 3 [4253, 4937] innerCandidates580_2 9190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan580_3

def innerTail580_1 : List Nat := 4463 :: innerTail580_2
def innerCandidates580_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates580_1_checked : candidates edge 4 4253 4463 innerTail580_2 best.bound = innerCandidates580_1 := by
  unfold candidates
  have row : edge 4463 = SearchData.row4463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan580_1 : scan edge (search edge 3) 4 [4253] 4253 innerTail580_1 best = best := by
  rw [innerTail580_1, scan, if_neg (by decide), innerCandidates580_1_checked]
  change scan edge (search edge 3) 4 [4253] 4253 innerTail580_2
    (if 3 ≤ innerCandidates580_1.length then search edge 3 [4253, 4463] innerCandidates580_1 8716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan580_2

def innerTail580_0 : List Nat := 4397 :: innerTail580_1
def innerCandidates580_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates580_0_checked : candidates edge 4 4253 4397 innerTail580_1 best.bound = innerCandidates580_0 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan580_0 : scan edge (search edge 3) 4 [4253] 4253 innerTail580_0 best = best := by
  rw [innerTail580_0, scan, if_neg (by decide), innerCandidates580_0_checked]
  change scan edge (search edge 3) 4 [4253] 4253 innerTail580_1
    (if 3 ≤ innerCandidates580_0.length then search edge 3 [4253, 4397] innerCandidates580_0 8650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan580_1

theorem rootBranch580 : search edge 4 [4253] rootCandidates580 4253 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4253] 4253 innerTail580_0 best = best
  exact innerScan580_0

def rootCandidates579 : List Nat := [4339, 4423, 4591, 4621, 4759, 4831, 5347, 5527, 6451, 6469, 6553, 6997, 7207, 7669, 7699]
set_option maxRecDepth 4096 in
theorem rootCandidates579_checked : candidates edge 5 0 4243 tail580 best.bound = rootCandidates579 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext q; rfl
  rw [row]
  decide

def innerTail579_15 : List Nat := []
theorem innerScan579_15 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_15 best = best := rfl

def innerTail579_14 : List Nat := 7699 :: innerTail579_15
def innerCandidates579_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates579_14_checked : candidates edge 4 4243 7699 innerTail579_15 best.bound = innerCandidates579_14 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_14 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_14 best = best := by
  rw [innerTail579_14, scan, if_neg (by decide), innerCandidates579_14_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_15
    (if 3 ≤ innerCandidates579_14.length then search edge 3 [4243, 7699] innerCandidates579_14 11942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_15

def innerTail579_13 : List Nat := 7669 :: innerTail579_14
def innerCandidates579_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates579_13_checked : candidates edge 4 4243 7669 innerTail579_14 best.bound = innerCandidates579_13 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_13 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_13 best = best := by
  rw [innerTail579_13, scan, if_neg (by decide), innerCandidates579_13_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_14
    (if 3 ≤ innerCandidates579_13.length then search edge 3 [4243, 7669] innerCandidates579_13 11912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_14

def innerTail579_12 : List Nat := 7207 :: innerTail579_13
def innerCandidates579_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates579_12_checked : candidates edge 4 4243 7207 innerTail579_13 best.bound = innerCandidates579_12 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_12 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_12 best = best := by
  rw [innerTail579_12, scan, if_neg (by decide), innerCandidates579_12_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_13
    (if 3 ≤ innerCandidates579_12.length then search edge 3 [4243, 7207] innerCandidates579_12 11450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_13

def innerTail579_11 : List Nat := 6997 :: innerTail579_12
def innerCandidates579_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates579_11_checked : candidates edge 4 4243 6997 innerTail579_12 best.bound = innerCandidates579_11 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_11 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_11 best = best := by
  rw [innerTail579_11, scan, if_neg (by decide), innerCandidates579_11_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_12
    (if 3 ≤ innerCandidates579_11.length then search edge 3 [4243, 6997] innerCandidates579_11 11240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_12

def innerTail579_10 : List Nat := 6553 :: innerTail579_11
def innerCandidates579_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates579_10_checked : candidates edge 4 4243 6553 innerTail579_11 best.bound = innerCandidates579_10 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_10 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_10 best = best := by
  rw [innerTail579_10, scan, if_neg (by decide), innerCandidates579_10_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_11
    (if 3 ≤ innerCandidates579_10.length then search edge 3 [4243, 6553] innerCandidates579_10 10796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_11

def innerTail579_9 : List Nat := 6469 :: innerTail579_10
def innerCandidates579_9 : List Nat := [6997]
set_option maxRecDepth 4096 in
theorem innerCandidates579_9_checked : candidates edge 4 4243 6469 innerTail579_10 best.bound = innerCandidates579_9 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_9 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_9 best = best := by
  rw [innerTail579_9, scan, if_neg (by decide), innerCandidates579_9_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_10
    (if 3 ≤ innerCandidates579_9.length then search edge 3 [4243, 6469] innerCandidates579_9 10712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_10

def innerTail579_8 : List Nat := 6451 :: innerTail579_9
def innerCandidates579_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates579_8_checked : candidates edge 4 4243 6451 innerTail579_9 best.bound = innerCandidates579_8 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_8 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_8 best = best := by
  rw [innerTail579_8, scan, if_neg (by decide), innerCandidates579_8_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_9
    (if 3 ≤ innerCandidates579_8.length then search edge 3 [4243, 6451] innerCandidates579_8 10694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_9

def innerTail579_7 : List Nat := 5527 :: innerTail579_8
def innerCandidates579_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates579_7_checked : candidates edge 4 4243 5527 innerTail579_8 best.bound = innerCandidates579_7 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_7 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_7 best = best := by
  rw [innerTail579_7, scan, if_neg (by decide), innerCandidates579_7_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_8
    (if 3 ≤ innerCandidates579_7.length then search edge 3 [4243, 5527] innerCandidates579_7 9770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_8

def innerTail579_6 : List Nat := 5347 :: innerTail579_7
def innerCandidates579_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates579_6_checked : candidates edge 4 4243 5347 innerTail579_7 best.bound = innerCandidates579_6 := by
  unfold candidates
  have row : edge 5347 = SearchData.row5347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_6 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_6 best = best := by
  rw [innerTail579_6, scan, if_neg (by decide), innerCandidates579_6_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_7
    (if 3 ≤ innerCandidates579_6.length then search edge 3 [4243, 5347] innerCandidates579_6 9590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_7

def innerTail579_5 : List Nat := 4831 :: innerTail579_6
def innerCandidates579_5 : List Nat := [6451, 6553, 7699]
set_option maxRecDepth 4096 in
theorem innerCandidates579_5_checked : candidates edge 4 4243 4831 innerTail579_6 best.bound = innerCandidates579_5 := by
  unfold candidates
  have row : edge 4831 = SearchData.row4831.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild579_5 : search edge 3 [4243, 4831] innerCandidates579_5 9074 best = best := by decide

theorem innerScan579_5 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_5 best = best := by
  rw [innerTail579_5, scan, if_neg (by decide), innerCandidates579_5_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_6
    (if 3 ≤ innerCandidates579_5.length then search edge 3 [4243, 4831] innerCandidates579_5 9074 best else best) = best
  rw [if_pos (by decide), innerChild579_5]
  exact innerScan579_6

def innerTail579_4 : List Nat := 4759 :: innerTail579_5
def innerCandidates579_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates579_4_checked : candidates edge 4 4243 4759 innerTail579_5 best.bound = innerCandidates579_4 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_4 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_4 best = best := by
  rw [innerTail579_4, scan, if_neg (by decide), innerCandidates579_4_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_5
    (if 3 ≤ innerCandidates579_4.length then search edge 3 [4243, 4759] innerCandidates579_4 9002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_5

def innerTail579_3 : List Nat := 4621 :: innerTail579_4
def innerCandidates579_3 : List Nat := [5347]
set_option maxRecDepth 4096 in
theorem innerCandidates579_3_checked : candidates edge 4 4243 4621 innerTail579_4 best.bound = innerCandidates579_3 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_3 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_3 best = best := by
  rw [innerTail579_3, scan, if_neg (by decide), innerCandidates579_3_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_4
    (if 3 ≤ innerCandidates579_3.length then search edge 3 [4243, 4621] innerCandidates579_3 8864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_4

def innerTail579_2 : List Nat := 4591 :: innerTail579_3
def innerCandidates579_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates579_2_checked : candidates edge 4 4243 4591 innerTail579_3 best.bound = innerCandidates579_2 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_2 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_2 best = best := by
  rw [innerTail579_2, scan, if_neg (by decide), innerCandidates579_2_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_3
    (if 3 ≤ innerCandidates579_2.length then search edge 3 [4243, 4591] innerCandidates579_2 8834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_3

def innerTail579_1 : List Nat := 4423 :: innerTail579_2
def innerCandidates579_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates579_1_checked : candidates edge 4 4243 4423 innerTail579_2 best.bound = innerCandidates579_1 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_1 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_1 best = best := by
  rw [innerTail579_1, scan, if_neg (by decide), innerCandidates579_1_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_2
    (if 3 ≤ innerCandidates579_1.length then search edge 3 [4243, 4423] innerCandidates579_1 8666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_2

def innerTail579_0 : List Nat := 4339 :: innerTail579_1
def innerCandidates579_0 : List Nat := [4831]
set_option maxRecDepth 4096 in
theorem innerCandidates579_0_checked : candidates edge 4 4243 4339 innerTail579_1 best.bound = innerCandidates579_0 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan579_0 : scan edge (search edge 3) 4 [4243] 4243 innerTail579_0 best = best := by
  rw [innerTail579_0, scan, if_neg (by decide), innerCandidates579_0_checked]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_1
    (if 3 ≤ innerCandidates579_0.length then search edge 3 [4243, 4339] innerCandidates579_0 8582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan579_1

theorem rootBranch579 : search edge 4 [4243] rootCandidates579 4243 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4243] 4243 innerTail579_0 best = best
  exact innerScan579_0

def rootCandidates578 : List Nat := [4751, 4793, 5981, 6053, 6389, 6977, 7103, 7559, 7673, 7757, 8093, 8147]
set_option maxRecDepth 4096 in
theorem rootCandidates578_checked : candidates edge 5 0 4241 tail579 best.bound = rootCandidates578 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext q; rfl
  rw [row]
  decide

def innerTail578_12 : List Nat := []
theorem innerScan578_12 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_12 best = best := rfl

def innerTail578_11 : List Nat := 8147 :: innerTail578_12
def innerCandidates578_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates578_11_checked : candidates edge 4 4241 8147 innerTail578_12 best.bound = innerCandidates578_11 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan578_11 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_11 best = best := by
  rw [innerTail578_11, scan, if_neg (by decide), innerCandidates578_11_checked]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_12
    (if 3 ≤ innerCandidates578_11.length then search edge 3 [4241, 8147] innerCandidates578_11 12388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan578_12

def innerTail578_10 : List Nat := 8093 :: innerTail578_11
def innerCandidates578_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates578_10_checked : candidates edge 4 4241 8093 innerTail578_11 best.bound = innerCandidates578_10 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan578_10 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_10 best = best := by
  rw [innerTail578_10, scan, if_neg (by decide), innerCandidates578_10_checked]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_11
    (if 3 ≤ innerCandidates578_10.length then search edge 3 [4241, 8093] innerCandidates578_10 12334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan578_11

def innerTail578_9 : List Nat := 7757 :: innerTail578_10
def innerCandidates578_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates578_9_checked : candidates edge 4 4241 7757 innerTail578_10 best.bound = innerCandidates578_9 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan578_9 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_9 best = best := by
  rw [innerTail578_9, scan, if_neg (by decide), innerCandidates578_9_checked]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_10
    (if 3 ≤ innerCandidates578_9.length then search edge 3 [4241, 7757] innerCandidates578_9 11998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan578_10

def innerTail578_8 : List Nat := 7673 :: innerTail578_9
def innerCandidates578_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates578_8_checked : candidates edge 4 4241 7673 innerTail578_9 best.bound = innerCandidates578_8 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan578_8 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_8 best = best := by
  rw [innerTail578_8, scan, if_neg (by decide), innerCandidates578_8_checked]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_9
    (if 3 ≤ innerCandidates578_8.length then search edge 3 [4241, 7673] innerCandidates578_8 11914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan578_9

def innerTail578_7 : List Nat := 7559 :: innerTail578_8
def innerCandidates578_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates578_7_checked : candidates edge 4 4241 7559 innerTail578_8 best.bound = innerCandidates578_7 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan578_7 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_7 best = best := by
  rw [innerTail578_7, scan, if_neg (by decide), innerCandidates578_7_checked]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_8
    (if 3 ≤ innerCandidates578_7.length then search edge 3 [4241, 7559] innerCandidates578_7 11800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan578_8

def innerTail578_6 : List Nat := 7103 :: innerTail578_7
def innerCandidates578_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates578_6_checked : candidates edge 4 4241 7103 innerTail578_7 best.bound = innerCandidates578_6 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan578_6 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_6 best = best := by
  rw [innerTail578_6, scan, if_neg (by decide), innerCandidates578_6_checked]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_7
    (if 3 ≤ innerCandidates578_6.length then search edge 3 [4241, 7103] innerCandidates578_6 11344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan578_7

def innerTail578_5 : List Nat := 6977 :: innerTail578_6
def innerCandidates578_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates578_5_checked : candidates edge 4 4241 6977 innerTail578_6 best.bound = innerCandidates578_5 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan578_5 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_5 best = best := by
  rw [innerTail578_5, scan, if_neg (by decide), innerCandidates578_5_checked]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_6
    (if 3 ≤ innerCandidates578_5.length then search edge 3 [4241, 6977] innerCandidates578_5 11218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan578_6

def innerTail578_4 : List Nat := 6389 :: innerTail578_5
def innerCandidates578_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates578_4_checked : candidates edge 4 4241 6389 innerTail578_5 best.bound = innerCandidates578_4 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan578_4 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_4 best = best := by
  rw [innerTail578_4, scan, if_neg (by decide), innerCandidates578_4_checked]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_5
    (if 3 ≤ innerCandidates578_4.length then search edge 3 [4241, 6389] innerCandidates578_4 10630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan578_5

def innerTail578_3 : List Nat := 6053 :: innerTail578_4
def innerCandidates578_3 : List Nat := [6977]
set_option maxRecDepth 4096 in
theorem innerCandidates578_3_checked : candidates edge 4 4241 6053 innerTail578_4 best.bound = innerCandidates578_3 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan578_3 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_3 best = best := by
  rw [innerTail578_3, scan, if_neg (by decide), innerCandidates578_3_checked]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_4
    (if 3 ≤ innerCandidates578_3.length then search edge 3 [4241, 6053] innerCandidates578_3 10294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan578_4

def innerTail578_2 : List Nat := 5981 :: innerTail578_3
def innerCandidates578_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates578_2_checked : candidates edge 4 4241 5981 innerTail578_3 best.bound = innerCandidates578_2 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan578_2 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_2 best = best := by
  rw [innerTail578_2, scan, if_neg (by decide), innerCandidates578_2_checked]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_3
    (if 3 ≤ innerCandidates578_2.length then search edge 3 [4241, 5981] innerCandidates578_2 10222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan578_3

def innerTail578_1 : List Nat := 4793 :: innerTail578_2
def innerCandidates578_1 : List Nat := [6389]
set_option maxRecDepth 4096 in
theorem innerCandidates578_1_checked : candidates edge 4 4241 4793 innerTail578_2 best.bound = innerCandidates578_1 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan578_1 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_1 best = best := by
  rw [innerTail578_1, scan, if_neg (by decide), innerCandidates578_1_checked]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_2
    (if 3 ≤ innerCandidates578_1.length then search edge 3 [4241, 4793] innerCandidates578_1 9034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan578_2

def innerTail578_0 : List Nat := 4751 :: innerTail578_1
def innerCandidates578_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates578_0_checked : candidates edge 4 4241 4751 innerTail578_1 best.bound = innerCandidates578_0 := by
  unfold candidates
  have row : edge 4751 = SearchData.row4751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan578_0 : scan edge (search edge 3) 4 [4241] 4241 innerTail578_0 best = best := by
  rw [innerTail578_0, scan, if_neg (by decide), innerCandidates578_0_checked]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_1
    (if 3 ≤ innerCandidates578_0.length then search edge 3 [4241, 4751] innerCandidates578_0 8992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan578_1

theorem rootBranch578 : search edge 4 [4241] rootCandidates578 4241 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4241] 4241 innerTail578_0 best = best
  exact innerScan578_0

def rootCandidates577 : List Nat := [4339, 4933, 5197, 6037, 6151, 6529, 6577, 7027, 7789]
set_option maxRecDepth 4096 in
theorem rootCandidates577_checked : candidates edge 5 0 4231 tail578 best.bound = rootCandidates577 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext q; rfl
  rw [row]
  decide

def innerTail577_9 : List Nat := []
theorem innerScan577_9 : scan edge (search edge 3) 4 [4231] 4231 innerTail577_9 best = best := rfl

def innerTail577_8 : List Nat := 7789 :: innerTail577_9
def innerCandidates577_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates577_8_checked : candidates edge 4 4231 7789 innerTail577_9 best.bound = innerCandidates577_8 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan577_8 : scan edge (search edge 3) 4 [4231] 4231 innerTail577_8 best = best := by
  rw [innerTail577_8, scan, if_neg (by decide), innerCandidates577_8_checked]
  change scan edge (search edge 3) 4 [4231] 4231 innerTail577_9
    (if 3 ≤ innerCandidates577_8.length then search edge 3 [4231, 7789] innerCandidates577_8 12020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan577_9

def innerTail577_7 : List Nat := 7027 :: innerTail577_8
def innerCandidates577_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates577_7_checked : candidates edge 4 4231 7027 innerTail577_8 best.bound = innerCandidates577_7 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan577_7 : scan edge (search edge 3) 4 [4231] 4231 innerTail577_7 best = best := by
  rw [innerTail577_7, scan, if_neg (by decide), innerCandidates577_7_checked]
  change scan edge (search edge 3) 4 [4231] 4231 innerTail577_8
    (if 3 ≤ innerCandidates577_7.length then search edge 3 [4231, 7027] innerCandidates577_7 11258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan577_8

def innerTail577_6 : List Nat := 6577 :: innerTail577_7
def innerCandidates577_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates577_6_checked : candidates edge 4 4231 6577 innerTail577_7 best.bound = innerCandidates577_6 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan577_6 : scan edge (search edge 3) 4 [4231] 4231 innerTail577_6 best = best := by
  rw [innerTail577_6, scan, if_neg (by decide), innerCandidates577_6_checked]
  change scan edge (search edge 3) 4 [4231] 4231 innerTail577_7
    (if 3 ≤ innerCandidates577_6.length then search edge 3 [4231, 6577] innerCandidates577_6 10808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan577_7

def innerTail577_5 : List Nat := 6529 :: innerTail577_6
def innerCandidates577_5 : List Nat := [7027]
set_option maxRecDepth 4096 in
theorem innerCandidates577_5_checked : candidates edge 4 4231 6529 innerTail577_6 best.bound = innerCandidates577_5 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan577_5 : scan edge (search edge 3) 4 [4231] 4231 innerTail577_5 best = best := by
  rw [innerTail577_5, scan, if_neg (by decide), innerCandidates577_5_checked]
  change scan edge (search edge 3) 4 [4231] 4231 innerTail577_6
    (if 3 ≤ innerCandidates577_5.length then search edge 3 [4231, 6529] innerCandidates577_5 10760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan577_6

def innerTail577_4 : List Nat := 6151 :: innerTail577_5
def innerCandidates577_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates577_4_checked : candidates edge 4 4231 6151 innerTail577_5 best.bound = innerCandidates577_4 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan577_4 : scan edge (search edge 3) 4 [4231] 4231 innerTail577_4 best = best := by
  rw [innerTail577_4, scan, if_neg (by decide), innerCandidates577_4_checked]
  change scan edge (search edge 3) 4 [4231] 4231 innerTail577_5
    (if 3 ≤ innerCandidates577_4.length then search edge 3 [4231, 6151] innerCandidates577_4 10382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan577_5

def innerTail577_3 : List Nat := 6037 :: innerTail577_4
def innerCandidates577_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates577_3_checked : candidates edge 4 4231 6037 innerTail577_4 best.bound = innerCandidates577_3 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan577_3 : scan edge (search edge 3) 4 [4231] 4231 innerTail577_3 best = best := by
  rw [innerTail577_3, scan, if_neg (by decide), innerCandidates577_3_checked]
  change scan edge (search edge 3) 4 [4231] 4231 innerTail577_4
    (if 3 ≤ innerCandidates577_3.length then search edge 3 [4231, 6037] innerCandidates577_3 10268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan577_4

def innerTail577_2 : List Nat := 5197 :: innerTail577_3
def innerCandidates577_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates577_2_checked : candidates edge 4 4231 5197 innerTail577_3 best.bound = innerCandidates577_2 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan577_2 : scan edge (search edge 3) 4 [4231] 4231 innerTail577_2 best = best := by
  rw [innerTail577_2, scan, if_neg (by decide), innerCandidates577_2_checked]
  change scan edge (search edge 3) 4 [4231] 4231 innerTail577_3
    (if 3 ≤ innerCandidates577_2.length then search edge 3 [4231, 5197] innerCandidates577_2 9428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan577_3

def innerTail577_1 : List Nat := 4933 :: innerTail577_2
def innerCandidates577_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates577_1_checked : candidates edge 4 4231 4933 innerTail577_2 best.bound = innerCandidates577_1 := by
  unfold candidates
  have row : edge 4933 = SearchData.row4933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan577_1 : scan edge (search edge 3) 4 [4231] 4231 innerTail577_1 best = best := by
  rw [innerTail577_1, scan, if_neg (by decide), innerCandidates577_1_checked]
  change scan edge (search edge 3) 4 [4231] 4231 innerTail577_2
    (if 3 ≤ innerCandidates577_1.length then search edge 3 [4231, 4933] innerCandidates577_1 9164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan577_2

def innerTail577_0 : List Nat := 4339 :: innerTail577_1
def innerCandidates577_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates577_0_checked : candidates edge 4 4231 4339 innerTail577_1 best.bound = innerCandidates577_0 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan577_0 : scan edge (search edge 3) 4 [4231] 4231 innerTail577_0 best = best := by
  rw [innerTail577_0, scan, if_neg (by decide), innerCandidates577_0_checked]
  change scan edge (search edge 3) 4 [4231] 4231 innerTail577_1
    (if 3 ≤ innerCandidates577_0.length then search edge 3 [4231, 4339] innerCandidates577_0 8570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan577_1

theorem rootBranch577 : search edge 4 [4231] rootCandidates577 4231 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4231] 4231 innerTail577_0 best = best
  exact innerScan577_0

def rootCandidates576 : List Nat := [4391, 5039, 5399, 6689, 7589, 7643]
set_option maxRecDepth 4096 in
theorem rootCandidates576_checked : candidates edge 5 0 4229 tail577 best.bound = rootCandidates576 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext q; rfl
  rw [row]
  decide

def innerTail576_6 : List Nat := []
theorem innerScan576_6 : scan edge (search edge 3) 4 [4229] 4229 innerTail576_6 best = best := rfl

def innerTail576_5 : List Nat := 7643 :: innerTail576_6
def innerCandidates576_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates576_5_checked : candidates edge 4 4229 7643 innerTail576_6 best.bound = innerCandidates576_5 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan576_5 : scan edge (search edge 3) 4 [4229] 4229 innerTail576_5 best = best := by
  rw [innerTail576_5, scan, if_neg (by decide), innerCandidates576_5_checked]
  change scan edge (search edge 3) 4 [4229] 4229 innerTail576_6
    (if 3 ≤ innerCandidates576_5.length then search edge 3 [4229, 7643] innerCandidates576_5 11872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan576_6

def innerTail576_4 : List Nat := 7589 :: innerTail576_5
def innerCandidates576_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates576_4_checked : candidates edge 4 4229 7589 innerTail576_5 best.bound = innerCandidates576_4 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan576_4 : scan edge (search edge 3) 4 [4229] 4229 innerTail576_4 best = best := by
  rw [innerTail576_4, scan, if_neg (by decide), innerCandidates576_4_checked]
  change scan edge (search edge 3) 4 [4229] 4229 innerTail576_5
    (if 3 ≤ innerCandidates576_4.length then search edge 3 [4229, 7589] innerCandidates576_4 11818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan576_5

def innerTail576_3 : List Nat := 6689 :: innerTail576_4
def innerCandidates576_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates576_3_checked : candidates edge 4 4229 6689 innerTail576_4 best.bound = innerCandidates576_3 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan576_3 : scan edge (search edge 3) 4 [4229] 4229 innerTail576_3 best = best := by
  rw [innerTail576_3, scan, if_neg (by decide), innerCandidates576_3_checked]
  change scan edge (search edge 3) 4 [4229] 4229 innerTail576_4
    (if 3 ≤ innerCandidates576_3.length then search edge 3 [4229, 6689] innerCandidates576_3 10918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan576_4

def innerTail576_2 : List Nat := 5399 :: innerTail576_3
def innerCandidates576_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates576_2_checked : candidates edge 4 4229 5399 innerTail576_3 best.bound = innerCandidates576_2 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan576_2 : scan edge (search edge 3) 4 [4229] 4229 innerTail576_2 best = best := by
  rw [innerTail576_2, scan, if_neg (by decide), innerCandidates576_2_checked]
  change scan edge (search edge 3) 4 [4229] 4229 innerTail576_3
    (if 3 ≤ innerCandidates576_2.length then search edge 3 [4229, 5399] innerCandidates576_2 9628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan576_3

def innerTail576_1 : List Nat := 5039 :: innerTail576_2
def innerCandidates576_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates576_1_checked : candidates edge 4 4229 5039 innerTail576_2 best.bound = innerCandidates576_1 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan576_1 : scan edge (search edge 3) 4 [4229] 4229 innerTail576_1 best = best := by
  rw [innerTail576_1, scan, if_neg (by decide), innerCandidates576_1_checked]
  change scan edge (search edge 3) 4 [4229] 4229 innerTail576_2
    (if 3 ≤ innerCandidates576_1.length then search edge 3 [4229, 5039] innerCandidates576_1 9268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan576_2

def innerTail576_0 : List Nat := 4391 :: innerTail576_1
def innerCandidates576_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates576_0_checked : candidates edge 4 4229 4391 innerTail576_1 best.bound = innerCandidates576_0 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan576_0 : scan edge (search edge 3) 4 [4229] 4229 innerTail576_0 best = best := by
  rw [innerTail576_0, scan, if_neg (by decide), innerCandidates576_0_checked]
  change scan edge (search edge 3) 4 [4229] 4229 innerTail576_1
    (if 3 ≤ innerCandidates576_0.length then search edge 3 [4229, 4391] innerCandidates576_0 8620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan576_1

theorem rootBranch576 : search edge 4 [4229] rootCandidates576 4229 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4229] 4229 innerTail576_0 best = best
  exact innerScan576_0

def rootCandidates575 : List Nat := [4597, 5113, 5119, 5227, 5521, 5749, 6337, 6379, 6529, 6793, 6883, 7027, 7369, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates575_checked : candidates edge 5 0 4219 tail576 best.bound = rootCandidates575 := by
  unfold candidates
  have row : edge 4219 = SearchData.row4219.contains := by funext q; rfl
  rw [row]
  decide

def innerTail575_14 : List Nat := []
theorem innerScan575_14 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_14 best = best := rfl

def innerTail575_13 : List Nat := 8209 :: innerTail575_14
def innerCandidates575_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates575_13_checked : candidates edge 4 4219 8209 innerTail575_14 best.bound = innerCandidates575_13 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_13 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_13 best = best := by
  rw [innerTail575_13, scan, if_neg (by decide), innerCandidates575_13_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_14
    (if 3 ≤ innerCandidates575_13.length then search edge 3 [4219, 8209] innerCandidates575_13 12428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_14

def innerTail575_12 : List Nat := 7369 :: innerTail575_13
def innerCandidates575_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates575_12_checked : candidates edge 4 4219 7369 innerTail575_13 best.bound = innerCandidates575_12 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_12 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_12 best = best := by
  rw [innerTail575_12, scan, if_neg (by decide), innerCandidates575_12_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_13
    (if 3 ≤ innerCandidates575_12.length then search edge 3 [4219, 7369] innerCandidates575_12 11588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_13

def innerTail575_11 : List Nat := 7027 :: innerTail575_12
def innerCandidates575_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates575_11_checked : candidates edge 4 4219 7027 innerTail575_12 best.bound = innerCandidates575_11 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_11 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_11 best = best := by
  rw [innerTail575_11, scan, if_neg (by decide), innerCandidates575_11_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_12
    (if 3 ≤ innerCandidates575_11.length then search edge 3 [4219, 7027] innerCandidates575_11 11246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_12

def innerTail575_10 : List Nat := 6883 :: innerTail575_11
def innerCandidates575_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates575_10_checked : candidates edge 4 4219 6883 innerTail575_11 best.bound = innerCandidates575_10 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_10 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_10 best = best := by
  rw [innerTail575_10, scan, if_neg (by decide), innerCandidates575_10_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_11
    (if 3 ≤ innerCandidates575_10.length then search edge 3 [4219, 6883] innerCandidates575_10 11102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_11

def innerTail575_9 : List Nat := 6793 :: innerTail575_10
def innerCandidates575_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates575_9_checked : candidates edge 4 4219 6793 innerTail575_10 best.bound = innerCandidates575_9 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_9 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_9 best = best := by
  rw [innerTail575_9, scan, if_neg (by decide), innerCandidates575_9_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_10
    (if 3 ≤ innerCandidates575_9.length then search edge 3 [4219, 6793] innerCandidates575_9 11012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_10

def innerTail575_8 : List Nat := 6529 :: innerTail575_9
def innerCandidates575_8 : List Nat := [7027]
set_option maxRecDepth 4096 in
theorem innerCandidates575_8_checked : candidates edge 4 4219 6529 innerTail575_9 best.bound = innerCandidates575_8 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_8 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_8 best = best := by
  rw [innerTail575_8, scan, if_neg (by decide), innerCandidates575_8_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_9
    (if 3 ≤ innerCandidates575_8.length then search edge 3 [4219, 6529] innerCandidates575_8 10748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_9

def innerTail575_7 : List Nat := 6379 :: innerTail575_8
def innerCandidates575_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates575_7_checked : candidates edge 4 4219 6379 innerTail575_8 best.bound = innerCandidates575_7 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_7 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_7 best = best := by
  rw [innerTail575_7, scan, if_neg (by decide), innerCandidates575_7_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_8
    (if 3 ≤ innerCandidates575_7.length then search edge 3 [4219, 6379] innerCandidates575_7 10598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_8

def innerTail575_6 : List Nat := 6337 :: innerTail575_7
def innerCandidates575_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates575_6_checked : candidates edge 4 4219 6337 innerTail575_7 best.bound = innerCandidates575_6 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_6 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_6 best = best := by
  rw [innerTail575_6, scan, if_neg (by decide), innerCandidates575_6_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_7
    (if 3 ≤ innerCandidates575_6.length then search edge 3 [4219, 6337] innerCandidates575_6 10556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_7

def innerTail575_5 : List Nat := 5749 :: innerTail575_6
def innerCandidates575_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates575_5_checked : candidates edge 4 4219 5749 innerTail575_6 best.bound = innerCandidates575_5 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_5 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_5 best = best := by
  rw [innerTail575_5, scan, if_neg (by decide), innerCandidates575_5_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_6
    (if 3 ≤ innerCandidates575_5.length then search edge 3 [4219, 5749] innerCandidates575_5 9968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_6

def innerTail575_4 : List Nat := 5521 :: innerTail575_5
def innerCandidates575_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates575_4_checked : candidates edge 4 4219 5521 innerTail575_5 best.bound = innerCandidates575_4 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_4 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_4 best = best := by
  rw [innerTail575_4, scan, if_neg (by decide), innerCandidates575_4_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_5
    (if 3 ≤ innerCandidates575_4.length then search edge 3 [4219, 5521] innerCandidates575_4 9740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_5

def innerTail575_3 : List Nat := 5227 :: innerTail575_4
def innerCandidates575_3 : List Nat := [6379]
set_option maxRecDepth 4096 in
theorem innerCandidates575_3_checked : candidates edge 4 4219 5227 innerTail575_4 best.bound = innerCandidates575_3 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_3 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_3 best = best := by
  rw [innerTail575_3, scan, if_neg (by decide), innerCandidates575_3_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_4
    (if 3 ≤ innerCandidates575_3.length then search edge 3 [4219, 5227] innerCandidates575_3 9446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_4

def innerTail575_2 : List Nat := 5119 :: innerTail575_3
def innerCandidates575_2 : List Nat := [6337, 6529]
set_option maxRecDepth 4096 in
theorem innerCandidates575_2_checked : candidates edge 4 4219 5119 innerTail575_3 best.bound = innerCandidates575_2 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_2 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_2 best = best := by
  rw [innerTail575_2, scan, if_neg (by decide), innerCandidates575_2_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_3
    (if 3 ≤ innerCandidates575_2.length then search edge 3 [4219, 5119] innerCandidates575_2 9338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_3

def innerTail575_1 : List Nat := 5113 :: innerTail575_2
def innerCandidates575_1 : List Nat := [6793]
set_option maxRecDepth 4096 in
theorem innerCandidates575_1_checked : candidates edge 4 4219 5113 innerTail575_2 best.bound = innerCandidates575_1 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_1 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_1 best = best := by
  rw [innerTail575_1, scan, if_neg (by decide), innerCandidates575_1_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_2
    (if 3 ≤ innerCandidates575_1.length then search edge 3 [4219, 5113] innerCandidates575_1 9332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_2

def innerTail575_0 : List Nat := 4597 :: innerTail575_1
def innerCandidates575_0 : List Nat := [5113, 5749]
set_option maxRecDepth 4096 in
theorem innerCandidates575_0_checked : candidates edge 4 4219 4597 innerTail575_1 best.bound = innerCandidates575_0 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan575_0 : scan edge (search edge 3) 4 [4219] 4219 innerTail575_0 best = best := by
  rw [innerTail575_0, scan, if_neg (by decide), innerCandidates575_0_checked]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_1
    (if 3 ≤ innerCandidates575_0.length then search edge 3 [4219, 4597] innerCandidates575_0 8816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan575_1

theorem rootBranch575 : search edge 4 [4219] rootCandidates575 4219 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4219] 4219 innerTail575_0 best = best
  exact innerScan575_0

end ProjectEulerSolutions.PrimePairUpper.Search
