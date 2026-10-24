import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates550 : List Nat := [4091, 4217, 4337, 4751, 5417, 5639, 6011, 6053, 6353, 7283, 7853, 7907, 8081, 8243, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates550_checked : candidates edge 5 0 4007 tail551 best.bound = rootCandidates550 := by
  unfold candidates
  have row : edge 4007 = SearchData.row4007.contains := by funext q; rfl
  rw [row]
  decide

def innerTail550_15 : List Nat := []
theorem innerScan550_15 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_15 best = best := rfl

def innerTail550_14 : List Nat := 8369 :: innerTail550_15
def innerCandidates550_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates550_14_checked : candidates edge 4 4007 8369 innerTail550_15 best.bound = innerCandidates550_14 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_14 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_14 best = best := by
  rw [innerTail550_14, scan, if_neg (by decide), innerCandidates550_14_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_15
    (if 3 ≤ innerCandidates550_14.length then search edge 3 [4007, 8369] innerCandidates550_14 12376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_15

def innerTail550_13 : List Nat := 8243 :: innerTail550_14
def innerCandidates550_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates550_13_checked : candidates edge 4 4007 8243 innerTail550_14 best.bound = innerCandidates550_13 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_13 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_13 best = best := by
  rw [innerTail550_13, scan, if_neg (by decide), innerCandidates550_13_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_14
    (if 3 ≤ innerCandidates550_13.length then search edge 3 [4007, 8243] innerCandidates550_13 12250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_14

def innerTail550_12 : List Nat := 8081 :: innerTail550_13
def innerCandidates550_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates550_12_checked : candidates edge 4 4007 8081 innerTail550_13 best.bound = innerCandidates550_12 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_12 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_12 best = best := by
  rw [innerTail550_12, scan, if_neg (by decide), innerCandidates550_12_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_13
    (if 3 ≤ innerCandidates550_12.length then search edge 3 [4007, 8081] innerCandidates550_12 12088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_13

def innerTail550_11 : List Nat := 7907 :: innerTail550_12
def innerCandidates550_11 : List Nat := [8081]
set_option maxRecDepth 4096 in
theorem innerCandidates550_11_checked : candidates edge 4 4007 7907 innerTail550_12 best.bound = innerCandidates550_11 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_11 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_11 best = best := by
  rw [innerTail550_11, scan, if_neg (by decide), innerCandidates550_11_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_12
    (if 3 ≤ innerCandidates550_11.length then search edge 3 [4007, 7907] innerCandidates550_11 11914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_12

def innerTail550_10 : List Nat := 7853 :: innerTail550_11
def innerCandidates550_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates550_10_checked : candidates edge 4 4007 7853 innerTail550_11 best.bound = innerCandidates550_10 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_10 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_10 best = best := by
  rw [innerTail550_10, scan, if_neg (by decide), innerCandidates550_10_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_11
    (if 3 ≤ innerCandidates550_10.length then search edge 3 [4007, 7853] innerCandidates550_10 11860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_11

def innerTail550_9 : List Nat := 7283 :: innerTail550_10
def innerCandidates550_9 : List Nat := [8243]
set_option maxRecDepth 4096 in
theorem innerCandidates550_9_checked : candidates edge 4 4007 7283 innerTail550_10 best.bound = innerCandidates550_9 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_9 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_9 best = best := by
  rw [innerTail550_9, scan, if_neg (by decide), innerCandidates550_9_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_10
    (if 3 ≤ innerCandidates550_9.length then search edge 3 [4007, 7283] innerCandidates550_9 11290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_10

def innerTail550_8 : List Nat := 6353 :: innerTail550_9
def innerCandidates550_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates550_8_checked : candidates edge 4 4007 6353 innerTail550_9 best.bound = innerCandidates550_8 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_8 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_8 best = best := by
  rw [innerTail550_8, scan, if_neg (by decide), innerCandidates550_8_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_9
    (if 3 ≤ innerCandidates550_8.length then search edge 3 [4007, 6353] innerCandidates550_8 10360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_9

def innerTail550_7 : List Nat := 6053 :: innerTail550_8
def innerCandidates550_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates550_7_checked : candidates edge 4 4007 6053 innerTail550_8 best.bound = innerCandidates550_7 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_7 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_7 best = best := by
  rw [innerTail550_7, scan, if_neg (by decide), innerCandidates550_7_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_8
    (if 3 ≤ innerCandidates550_7.length then search edge 3 [4007, 6053] innerCandidates550_7 10060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_8

def innerTail550_6 : List Nat := 6011 :: innerTail550_7
def innerCandidates550_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates550_6_checked : candidates edge 4 4007 6011 innerTail550_7 best.bound = innerCandidates550_6 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_6 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_6 best = best := by
  rw [innerTail550_6, scan, if_neg (by decide), innerCandidates550_6_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_7
    (if 3 ≤ innerCandidates550_6.length then search edge 3 [4007, 6011] innerCandidates550_6 10018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_7

def innerTail550_5 : List Nat := 5639 :: innerTail550_6
def innerCandidates550_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates550_5_checked : candidates edge 4 4007 5639 innerTail550_6 best.bound = innerCandidates550_5 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_5 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_5 best = best := by
  rw [innerTail550_5, scan, if_neg (by decide), innerCandidates550_5_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_6
    (if 3 ≤ innerCandidates550_5.length then search edge 3 [4007, 5639] innerCandidates550_5 9646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_6

def innerTail550_4 : List Nat := 5417 :: innerTail550_5
def innerCandidates550_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates550_4_checked : candidates edge 4 4007 5417 innerTail550_5 best.bound = innerCandidates550_4 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_4 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_4 best = best := by
  rw [innerTail550_4, scan, if_neg (by decide), innerCandidates550_4_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_5
    (if 3 ≤ innerCandidates550_4.length then search edge 3 [4007, 5417] innerCandidates550_4 9424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_5

def innerTail550_3 : List Nat := 4751 :: innerTail550_4
def innerCandidates550_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates550_3_checked : candidates edge 4 4007 4751 innerTail550_4 best.bound = innerCandidates550_3 := by
  unfold candidates
  have row : edge 4751 = SearchData.row4751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_3 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_3 best = best := by
  rw [innerTail550_3, scan, if_neg (by decide), innerCandidates550_3_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_4
    (if 3 ≤ innerCandidates550_3.length then search edge 3 [4007, 4751] innerCandidates550_3 8758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_4

def innerTail550_2 : List Nat := 4337 :: innerTail550_3
def innerCandidates550_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates550_2_checked : candidates edge 4 4007 4337 innerTail550_3 best.bound = innerCandidates550_2 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_2 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_2 best = best := by
  rw [innerTail550_2, scan, if_neg (by decide), innerCandidates550_2_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_3
    (if 3 ≤ innerCandidates550_2.length then search edge 3 [4007, 4337] innerCandidates550_2 8344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_3

def innerTail550_1 : List Nat := 4217 :: innerTail550_2
def innerCandidates550_1 : List Nat := [4337, 4751]
set_option maxRecDepth 4096 in
theorem innerCandidates550_1_checked : candidates edge 4 4007 4217 innerTail550_2 best.bound = innerCandidates550_1 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_1 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_1 best = best := by
  rw [innerTail550_1, scan, if_neg (by decide), innerCandidates550_1_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_2
    (if 3 ≤ innerCandidates550_1.length then search edge 3 [4007, 4217] innerCandidates550_1 8224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_2

def innerTail550_0 : List Nat := 4091 :: innerTail550_1
def innerCandidates550_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates550_0_checked : candidates edge 4 4007 4091 innerTail550_1 best.bound = innerCandidates550_0 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan550_0 : scan edge (search edge 3) 4 [4007] 4007 innerTail550_0 best = best := by
  rw [innerTail550_0, scan, if_neg (by decide), innerCandidates550_0_checked]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_1
    (if 3 ≤ innerCandidates550_0.length then search edge 3 [4007, 4091] innerCandidates550_0 8098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan550_1

theorem rootBranch550 : search edge 4 [4007] rootCandidates550 4007 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4007] 4007 innerTail550_0 best = best
  exact innerScan550_0

def rootCandidates549 : List Nat := [4567, 4597, 4723, 5431, 5503, 5851, 6163]
set_option maxRecDepth 4096 in
theorem rootCandidates549_checked : candidates edge 5 0 4003 tail550 best.bound = rootCandidates549 := by
  unfold candidates
  have row : edge 4003 = SearchData.row4003.contains := by funext q; rfl
  rw [row]
  decide

def innerTail549_7 : List Nat := []
theorem innerScan549_7 : scan edge (search edge 3) 4 [4003] 4003 innerTail549_7 best = best := rfl

def innerTail549_6 : List Nat := 6163 :: innerTail549_7
def innerCandidates549_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates549_6_checked : candidates edge 4 4003 6163 innerTail549_7 best.bound = innerCandidates549_6 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan549_6 : scan edge (search edge 3) 4 [4003] 4003 innerTail549_6 best = best := by
  rw [innerTail549_6, scan, if_neg (by decide), innerCandidates549_6_checked]
  change scan edge (search edge 3) 4 [4003] 4003 innerTail549_7
    (if 3 ≤ innerCandidates549_6.length then search edge 3 [4003, 6163] innerCandidates549_6 10166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan549_7

def innerTail549_5 : List Nat := 5851 :: innerTail549_6
def innerCandidates549_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates549_5_checked : candidates edge 4 4003 5851 innerTail549_6 best.bound = innerCandidates549_5 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan549_5 : scan edge (search edge 3) 4 [4003] 4003 innerTail549_5 best = best := by
  rw [innerTail549_5, scan, if_neg (by decide), innerCandidates549_5_checked]
  change scan edge (search edge 3) 4 [4003] 4003 innerTail549_6
    (if 3 ≤ innerCandidates549_5.length then search edge 3 [4003, 5851] innerCandidates549_5 9854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan549_6

def innerTail549_4 : List Nat := 5503 :: innerTail549_5
def innerCandidates549_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates549_4_checked : candidates edge 4 4003 5503 innerTail549_5 best.bound = innerCandidates549_4 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan549_4 : scan edge (search edge 3) 4 [4003] 4003 innerTail549_4 best = best := by
  rw [innerTail549_4, scan, if_neg (by decide), innerCandidates549_4_checked]
  change scan edge (search edge 3) 4 [4003] 4003 innerTail549_5
    (if 3 ≤ innerCandidates549_4.length then search edge 3 [4003, 5503] innerCandidates549_4 9506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan549_5

def innerTail549_3 : List Nat := 5431 :: innerTail549_4
def innerCandidates549_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates549_3_checked : candidates edge 4 4003 5431 innerTail549_4 best.bound = innerCandidates549_3 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan549_3 : scan edge (search edge 3) 4 [4003] 4003 innerTail549_3 best = best := by
  rw [innerTail549_3, scan, if_neg (by decide), innerCandidates549_3_checked]
  change scan edge (search edge 3) 4 [4003] 4003 innerTail549_4
    (if 3 ≤ innerCandidates549_3.length then search edge 3 [4003, 5431] innerCandidates549_3 9434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan549_4

def innerTail549_2 : List Nat := 4723 :: innerTail549_3
def innerCandidates549_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates549_2_checked : candidates edge 4 4003 4723 innerTail549_3 best.bound = innerCandidates549_2 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan549_2 : scan edge (search edge 3) 4 [4003] 4003 innerTail549_2 best = best := by
  rw [innerTail549_2, scan, if_neg (by decide), innerCandidates549_2_checked]
  change scan edge (search edge 3) 4 [4003] 4003 innerTail549_3
    (if 3 ≤ innerCandidates549_2.length then search edge 3 [4003, 4723] innerCandidates549_2 8726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan549_3

def innerTail549_1 : List Nat := 4597 :: innerTail549_2
def innerCandidates549_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates549_1_checked : candidates edge 4 4003 4597 innerTail549_2 best.bound = innerCandidates549_1 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan549_1 : scan edge (search edge 3) 4 [4003] 4003 innerTail549_1 best = best := by
  rw [innerTail549_1, scan, if_neg (by decide), innerCandidates549_1_checked]
  change scan edge (search edge 3) 4 [4003] 4003 innerTail549_2
    (if 3 ≤ innerCandidates549_1.length then search edge 3 [4003, 4597] innerCandidates549_1 8600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan549_2

def innerTail549_0 : List Nat := 4567 :: innerTail549_1
def innerCandidates549_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates549_0_checked : candidates edge 4 4003 4567 innerTail549_1 best.bound = innerCandidates549_0 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan549_0 : scan edge (search edge 3) 4 [4003] 4003 innerTail549_0 best = best := by
  rw [innerTail549_0, scan, if_neg (by decide), innerCandidates549_0_checked]
  change scan edge (search edge 3) 4 [4003] 4003 innerTail549_1
    (if 3 ≤ innerCandidates549_0.length then search edge 3 [4003, 4567] innerCandidates549_0 8570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan549_1

theorem rootBranch549 : search edge 4 [4003] rootCandidates549 4003 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4003] 4003 innerTail549_0 best = best
  exact innerScan549_0

def rootCandidates548 : List Nat := [4253, 4349, 4391, 5783, 6323, 6329, 6563, 7121, 7577, 7583, 7877, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates548_checked : candidates edge 5 0 4001 tail549 best.bound = rootCandidates548 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext q; rfl
  rw [row]
  decide

def innerTail548_12 : List Nat := []
theorem innerScan548_12 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_12 best = best := rfl

def innerTail548_11 : List Nat := 8219 :: innerTail548_12
def innerCandidates548_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates548_11_checked : candidates edge 4 4001 8219 innerTail548_12 best.bound = innerCandidates548_11 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan548_11 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_11 best = best := by
  rw [innerTail548_11, scan, if_neg (by decide), innerCandidates548_11_checked]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_12
    (if 3 ≤ innerCandidates548_11.length then search edge 3 [4001, 8219] innerCandidates548_11 12220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan548_12

def innerTail548_10 : List Nat := 7877 :: innerTail548_11
def innerCandidates548_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates548_10_checked : candidates edge 4 4001 7877 innerTail548_11 best.bound = innerCandidates548_10 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan548_10 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_10 best = best := by
  rw [innerTail548_10, scan, if_neg (by decide), innerCandidates548_10_checked]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_11
    (if 3 ≤ innerCandidates548_10.length then search edge 3 [4001, 7877] innerCandidates548_10 11878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan548_11

def innerTail548_9 : List Nat := 7583 :: innerTail548_10
def innerCandidates548_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates548_9_checked : candidates edge 4 4001 7583 innerTail548_10 best.bound = innerCandidates548_9 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan548_9 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_9 best = best := by
  rw [innerTail548_9, scan, if_neg (by decide), innerCandidates548_9_checked]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_10
    (if 3 ≤ innerCandidates548_9.length then search edge 3 [4001, 7583] innerCandidates548_9 11584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan548_10

def innerTail548_8 : List Nat := 7577 :: innerTail548_9
def innerCandidates548_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates548_8_checked : candidates edge 4 4001 7577 innerTail548_9 best.bound = innerCandidates548_8 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan548_8 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_8 best = best := by
  rw [innerTail548_8, scan, if_neg (by decide), innerCandidates548_8_checked]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_9
    (if 3 ≤ innerCandidates548_8.length then search edge 3 [4001, 7577] innerCandidates548_8 11578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan548_9

def innerTail548_7 : List Nat := 7121 :: innerTail548_8
def innerCandidates548_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates548_7_checked : candidates edge 4 4001 7121 innerTail548_8 best.bound = innerCandidates548_7 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan548_7 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_7 best = best := by
  rw [innerTail548_7, scan, if_neg (by decide), innerCandidates548_7_checked]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_8
    (if 3 ≤ innerCandidates548_7.length then search edge 3 [4001, 7121] innerCandidates548_7 11122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan548_8

def innerTail548_6 : List Nat := 6563 :: innerTail548_7
def innerCandidates548_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates548_6_checked : candidates edge 4 4001 6563 innerTail548_7 best.bound = innerCandidates548_6 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan548_6 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_6 best = best := by
  rw [innerTail548_6, scan, if_neg (by decide), innerCandidates548_6_checked]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_7
    (if 3 ≤ innerCandidates548_6.length then search edge 3 [4001, 6563] innerCandidates548_6 10564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan548_7

def innerTail548_5 : List Nat := 6329 :: innerTail548_6
def innerCandidates548_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates548_5_checked : candidates edge 4 4001 6329 innerTail548_6 best.bound = innerCandidates548_5 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan548_5 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_5 best = best := by
  rw [innerTail548_5, scan, if_neg (by decide), innerCandidates548_5_checked]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_6
    (if 3 ≤ innerCandidates548_5.length then search edge 3 [4001, 6329] innerCandidates548_5 10330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan548_6

def innerTail548_4 : List Nat := 6323 :: innerTail548_5
def innerCandidates548_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates548_4_checked : candidates edge 4 4001 6323 innerTail548_5 best.bound = innerCandidates548_4 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan548_4 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_4 best = best := by
  rw [innerTail548_4, scan, if_neg (by decide), innerCandidates548_4_checked]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_5
    (if 3 ≤ innerCandidates548_4.length then search edge 3 [4001, 6323] innerCandidates548_4 10324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan548_5

def innerTail548_3 : List Nat := 5783 :: innerTail548_4
def innerCandidates548_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates548_3_checked : candidates edge 4 4001 5783 innerTail548_4 best.bound = innerCandidates548_3 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan548_3 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_3 best = best := by
  rw [innerTail548_3, scan, if_neg (by decide), innerCandidates548_3_checked]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_4
    (if 3 ≤ innerCandidates548_3.length then search edge 3 [4001, 5783] innerCandidates548_3 9784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan548_4

def innerTail548_2 : List Nat := 4391 :: innerTail548_3
def innerCandidates548_2 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates548_2_checked : candidates edge 4 4001 4391 innerTail548_3 best.bound = innerCandidates548_2 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan548_2 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_2 best = best := by
  rw [innerTail548_2, scan, if_neg (by decide), innerCandidates548_2_checked]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_3
    (if 3 ≤ innerCandidates548_2.length then search edge 3 [4001, 4391] innerCandidates548_2 8392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan548_3

def innerTail548_1 : List Nat := 4349 :: innerTail548_2
def innerCandidates548_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates548_1_checked : candidates edge 4 4001 4349 innerTail548_2 best.bound = innerCandidates548_1 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan548_1 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_1 best = best := by
  rw [innerTail548_1, scan, if_neg (by decide), innerCandidates548_1_checked]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_2
    (if 3 ≤ innerCandidates548_1.length then search edge 3 [4001, 4349] innerCandidates548_1 8350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan548_2

def innerTail548_0 : List Nat := 4253 :: innerTail548_1
def innerCandidates548_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates548_0_checked : candidates edge 4 4001 4253 innerTail548_1 best.bound = innerCandidates548_0 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan548_0 : scan edge (search edge 3) 4 [4001] 4001 innerTail548_0 best = best := by
  rw [innerTail548_0, scan, if_neg (by decide), innerCandidates548_0_checked]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_1
    (if 3 ≤ innerCandidates548_0.length then search edge 3 [4001, 4253] innerCandidates548_0 8254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan548_1

theorem rootBranch548 : search edge 4 [4001] rootCandidates548 4001 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4001] 4001 innerTail548_0 best = best
  exact innerScan548_0

def rootCandidates547 : List Nat := [4157, 4931, 5783, 5927, 6581, 6719, 7043, 7703, 8093, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates547_checked : candidates edge 5 0 3989 tail548 best.bound = rootCandidates547 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext q; rfl
  rw [row]
  decide

def innerTail547_10 : List Nat := []
theorem innerScan547_10 : scan edge (search edge 3) 4 [3989] 3989 innerTail547_10 best = best := rfl

def innerTail547_9 : List Nat := 8291 :: innerTail547_10
def innerCandidates547_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates547_9_checked : candidates edge 4 3989 8291 innerTail547_10 best.bound = innerCandidates547_9 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan547_9 : scan edge (search edge 3) 4 [3989] 3989 innerTail547_9 best = best := by
  rw [innerTail547_9, scan, if_neg (by decide), innerCandidates547_9_checked]
  change scan edge (search edge 3) 4 [3989] 3989 innerTail547_10
    (if 3 ≤ innerCandidates547_9.length then search edge 3 [3989, 8291] innerCandidates547_9 12280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan547_10

def innerTail547_8 : List Nat := 8093 :: innerTail547_9
def innerCandidates547_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates547_8_checked : candidates edge 4 3989 8093 innerTail547_9 best.bound = innerCandidates547_8 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan547_8 : scan edge (search edge 3) 4 [3989] 3989 innerTail547_8 best = best := by
  rw [innerTail547_8, scan, if_neg (by decide), innerCandidates547_8_checked]
  change scan edge (search edge 3) 4 [3989] 3989 innerTail547_9
    (if 3 ≤ innerCandidates547_8.length then search edge 3 [3989, 8093] innerCandidates547_8 12082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan547_9

def innerTail547_7 : List Nat := 7703 :: innerTail547_8
def innerCandidates547_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates547_7_checked : candidates edge 4 3989 7703 innerTail547_8 best.bound = innerCandidates547_7 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan547_7 : scan edge (search edge 3) 4 [3989] 3989 innerTail547_7 best = best := by
  rw [innerTail547_7, scan, if_neg (by decide), innerCandidates547_7_checked]
  change scan edge (search edge 3) 4 [3989] 3989 innerTail547_8
    (if 3 ≤ innerCandidates547_7.length then search edge 3 [3989, 7703] innerCandidates547_7 11692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan547_8

def innerTail547_6 : List Nat := 7043 :: innerTail547_7
def innerCandidates547_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates547_6_checked : candidates edge 4 3989 7043 innerTail547_7 best.bound = innerCandidates547_6 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan547_6 : scan edge (search edge 3) 4 [3989] 3989 innerTail547_6 best = best := by
  rw [innerTail547_6, scan, if_neg (by decide), innerCandidates547_6_checked]
  change scan edge (search edge 3) 4 [3989] 3989 innerTail547_7
    (if 3 ≤ innerCandidates547_6.length then search edge 3 [3989, 7043] innerCandidates547_6 11032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan547_7

def innerTail547_5 : List Nat := 6719 :: innerTail547_6
def innerCandidates547_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates547_5_checked : candidates edge 4 3989 6719 innerTail547_6 best.bound = innerCandidates547_5 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan547_5 : scan edge (search edge 3) 4 [3989] 3989 innerTail547_5 best = best := by
  rw [innerTail547_5, scan, if_neg (by decide), innerCandidates547_5_checked]
  change scan edge (search edge 3) 4 [3989] 3989 innerTail547_6
    (if 3 ≤ innerCandidates547_5.length then search edge 3 [3989, 6719] innerCandidates547_5 10708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan547_6

def innerTail547_4 : List Nat := 6581 :: innerTail547_5
def innerCandidates547_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates547_4_checked : candidates edge 4 3989 6581 innerTail547_5 best.bound = innerCandidates547_4 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan547_4 : scan edge (search edge 3) 4 [3989] 3989 innerTail547_4 best = best := by
  rw [innerTail547_4, scan, if_neg (by decide), innerCandidates547_4_checked]
  change scan edge (search edge 3) 4 [3989] 3989 innerTail547_5
    (if 3 ≤ innerCandidates547_4.length then search edge 3 [3989, 6581] innerCandidates547_4 10570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan547_5

def innerTail547_3 : List Nat := 5927 :: innerTail547_4
def innerCandidates547_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates547_3_checked : candidates edge 4 3989 5927 innerTail547_4 best.bound = innerCandidates547_3 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan547_3 : scan edge (search edge 3) 4 [3989] 3989 innerTail547_3 best = best := by
  rw [innerTail547_3, scan, if_neg (by decide), innerCandidates547_3_checked]
  change scan edge (search edge 3) 4 [3989] 3989 innerTail547_4
    (if 3 ≤ innerCandidates547_3.length then search edge 3 [3989, 5927] innerCandidates547_3 9916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan547_4

def innerTail547_2 : List Nat := 5783 :: innerTail547_3
def innerCandidates547_2 : List Nat := [6719]
set_option maxRecDepth 4096 in
theorem innerCandidates547_2_checked : candidates edge 4 3989 5783 innerTail547_3 best.bound = innerCandidates547_2 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan547_2 : scan edge (search edge 3) 4 [3989] 3989 innerTail547_2 best = best := by
  rw [innerTail547_2, scan, if_neg (by decide), innerCandidates547_2_checked]
  change scan edge (search edge 3) 4 [3989] 3989 innerTail547_3
    (if 3 ≤ innerCandidates547_2.length then search edge 3 [3989, 5783] innerCandidates547_2 9772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan547_3

def innerTail547_1 : List Nat := 4931 :: innerTail547_2
def innerCandidates547_1 : List Nat := [7703]
set_option maxRecDepth 4096 in
theorem innerCandidates547_1_checked : candidates edge 4 3989 4931 innerTail547_2 best.bound = innerCandidates547_1 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan547_1 : scan edge (search edge 3) 4 [3989] 3989 innerTail547_1 best = best := by
  rw [innerTail547_1, scan, if_neg (by decide), innerCandidates547_1_checked]
  change scan edge (search edge 3) 4 [3989] 3989 innerTail547_2
    (if 3 ≤ innerCandidates547_1.length then search edge 3 [3989, 4931] innerCandidates547_1 8920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan547_2

def innerTail547_0 : List Nat := 4157 :: innerTail547_1
def innerCandidates547_0 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates547_0_checked : candidates edge 4 3989 4157 innerTail547_1 best.bound = innerCandidates547_0 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan547_0 : scan edge (search edge 3) 4 [3989] 3989 innerTail547_0 best = best := by
  rw [innerTail547_0, scan, if_neg (by decide), innerCandidates547_0_checked]
  change scan edge (search edge 3) 4 [3989] 3989 innerTail547_1
    (if 3 ≤ innerCandidates547_0.length then search edge 3 [3989, 4157] innerCandidates547_0 8146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan547_1

theorem rootBranch547 : search edge 4 [3989] rootCandidates547 3989 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3989] 3989 innerTail547_0 best = best
  exact innerScan547_0

def rootCandidates546 : List Nat := [4177, 4783, 5011, 5791, 5953, 6067, 6211, 6271, 7159, 7177, 7699, 7927, 7933, 8053, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates546_checked : candidates edge 5 0 3967 tail547 best.bound = rootCandidates546 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext q; rfl
  rw [row]
  decide

def innerTail546_15 : List Nat := []
theorem innerScan546_15 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_15 best = best := rfl

def innerTail546_14 : List Nat := 8209 :: innerTail546_15
def innerCandidates546_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates546_14_checked : candidates edge 4 3967 8209 innerTail546_15 best.bound = innerCandidates546_14 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_14 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_14 best = best := by
  rw [innerTail546_14, scan, if_neg (by decide), innerCandidates546_14_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_15
    (if 3 ≤ innerCandidates546_14.length then search edge 3 [3967, 8209] innerCandidates546_14 12176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_15

def innerTail546_13 : List Nat := 8053 :: innerTail546_14
def innerCandidates546_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates546_13_checked : candidates edge 4 3967 8053 innerTail546_14 best.bound = innerCandidates546_13 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_13 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_13 best = best := by
  rw [innerTail546_13, scan, if_neg (by decide), innerCandidates546_13_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_14
    (if 3 ≤ innerCandidates546_13.length then search edge 3 [3967, 8053] innerCandidates546_13 12020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_14

def innerTail546_12 : List Nat := 7933 :: innerTail546_13
def innerCandidates546_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates546_12_checked : candidates edge 4 3967 7933 innerTail546_13 best.bound = innerCandidates546_12 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_12 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_12 best = best := by
  rw [innerTail546_12, scan, if_neg (by decide), innerCandidates546_12_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_13
    (if 3 ≤ innerCandidates546_12.length then search edge 3 [3967, 7933] innerCandidates546_12 11900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_13

def innerTail546_11 : List Nat := 7927 :: innerTail546_12
def innerCandidates546_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates546_11_checked : candidates edge 4 3967 7927 innerTail546_12 best.bound = innerCandidates546_11 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_11 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_11 best = best := by
  rw [innerTail546_11, scan, if_neg (by decide), innerCandidates546_11_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_12
    (if 3 ≤ innerCandidates546_11.length then search edge 3 [3967, 7927] innerCandidates546_11 11894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_12

def innerTail546_10 : List Nat := 7699 :: innerTail546_11
def innerCandidates546_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates546_10_checked : candidates edge 4 3967 7699 innerTail546_11 best.bound = innerCandidates546_10 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_10 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_10 best = best := by
  rw [innerTail546_10, scan, if_neg (by decide), innerCandidates546_10_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_11
    (if 3 ≤ innerCandidates546_10.length then search edge 3 [3967, 7699] innerCandidates546_10 11666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_11

def innerTail546_9 : List Nat := 7177 :: innerTail546_10
def innerCandidates546_9 : List Nat := [7933]
set_option maxRecDepth 4096 in
theorem innerCandidates546_9_checked : candidates edge 4 3967 7177 innerTail546_10 best.bound = innerCandidates546_9 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_9 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_9 best = best := by
  rw [innerTail546_9, scan, if_neg (by decide), innerCandidates546_9_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_10
    (if 3 ≤ innerCandidates546_9.length then search edge 3 [3967, 7177] innerCandidates546_9 11144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_10

def innerTail546_8 : List Nat := 7159 :: innerTail546_9
def innerCandidates546_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates546_8_checked : candidates edge 4 3967 7159 innerTail546_9 best.bound = innerCandidates546_8 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_8 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_8 best = best := by
  rw [innerTail546_8, scan, if_neg (by decide), innerCandidates546_8_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_9
    (if 3 ≤ innerCandidates546_8.length then search edge 3 [3967, 7159] innerCandidates546_8 11126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_9

def innerTail546_7 : List Nat := 6271 :: innerTail546_8
def innerCandidates546_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates546_7_checked : candidates edge 4 3967 6271 innerTail546_8 best.bound = innerCandidates546_7 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_7 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_7 best = best := by
  rw [innerTail546_7, scan, if_neg (by decide), innerCandidates546_7_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_8
    (if 3 ≤ innerCandidates546_7.length then search edge 3 [3967, 6271] innerCandidates546_7 10238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_8

def innerTail546_6 : List Nat := 6211 :: innerTail546_7
def innerCandidates546_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates546_6_checked : candidates edge 4 3967 6211 innerTail546_7 best.bound = innerCandidates546_6 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_6 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_6 best = best := by
  rw [innerTail546_6, scan, if_neg (by decide), innerCandidates546_6_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_7
    (if 3 ≤ innerCandidates546_6.length then search edge 3 [3967, 6211] innerCandidates546_6 10178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_7

def innerTail546_5 : List Nat := 6067 :: innerTail546_6
def innerCandidates546_5 : List Nat := [7699]
set_option maxRecDepth 4096 in
theorem innerCandidates546_5_checked : candidates edge 4 3967 6067 innerTail546_6 best.bound = innerCandidates546_5 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_5 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_5 best = best := by
  rw [innerTail546_5, scan, if_neg (by decide), innerCandidates546_5_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_6
    (if 3 ≤ innerCandidates546_5.length then search edge 3 [3967, 6067] innerCandidates546_5 10034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_6

def innerTail546_4 : List Nat := 5953 :: innerTail546_5
def innerCandidates546_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates546_4_checked : candidates edge 4 3967 5953 innerTail546_5 best.bound = innerCandidates546_4 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_4 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_4 best = best := by
  rw [innerTail546_4, scan, if_neg (by decide), innerCandidates546_4_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_5
    (if 3 ≤ innerCandidates546_4.length then search edge 3 [3967, 5953] innerCandidates546_4 9920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_5

def innerTail546_3 : List Nat := 5791 :: innerTail546_4
def innerCandidates546_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates546_3_checked : candidates edge 4 3967 5791 innerTail546_4 best.bound = innerCandidates546_3 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_3 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_3 best = best := by
  rw [innerTail546_3, scan, if_neg (by decide), innerCandidates546_3_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_4
    (if 3 ≤ innerCandidates546_3.length then search edge 3 [3967, 5791] innerCandidates546_3 9758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_4

def innerTail546_2 : List Nat := 5011 :: innerTail546_3
def innerCandidates546_2 : List Nat := [6271, 8053]
set_option maxRecDepth 4096 in
theorem innerCandidates546_2_checked : candidates edge 4 3967 5011 innerTail546_3 best.bound = innerCandidates546_2 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_2 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_2 best = best := by
  rw [innerTail546_2, scan, if_neg (by decide), innerCandidates546_2_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_3
    (if 3 ≤ innerCandidates546_2.length then search edge 3 [3967, 5011] innerCandidates546_2 8978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_3

def innerTail546_1 : List Nat := 4783 :: innerTail546_2
def innerCandidates546_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates546_1_checked : candidates edge 4 3967 4783 innerTail546_2 best.bound = innerCandidates546_1 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_1 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_1 best = best := by
  rw [innerTail546_1, scan, if_neg (by decide), innerCandidates546_1_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_2
    (if 3 ≤ innerCandidates546_1.length then search edge 3 [3967, 4783] innerCandidates546_1 8750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_2

def innerTail546_0 : List Nat := 4177 :: innerTail546_1
def innerCandidates546_0 : List Nat := [5791, 8053]
set_option maxRecDepth 4096 in
theorem innerCandidates546_0_checked : candidates edge 4 3967 4177 innerTail546_1 best.bound = innerCandidates546_0 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan546_0 : scan edge (search edge 3) 4 [3967] 3967 innerTail546_0 best = best := by
  rw [innerTail546_0, scan, if_neg (by decide), innerCandidates546_0_checked]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_1
    (if 3 ≤ innerCandidates546_0.length then search edge 3 [3967, 4177] innerCandidates546_0 8144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan546_1

theorem rootBranch546 : search edge 4 [3967] rootCandidates546 3967 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3967] 3967 innerTail546_0 best = best
  exact innerScan546_0

end ProjectEulerSolutions.PrimePairUpper.Search
