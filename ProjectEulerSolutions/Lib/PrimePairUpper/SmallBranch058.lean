import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates600 : List Nat := [5227, 5347, 5743, 5953, 6247, 6271, 6367, 6373, 6481, 6949, 7759, 7873]
set_option maxRecDepth 4096 in
theorem rootCandidates600_checked : candidates edge 5 0 4441 tail601 best.bound = rootCandidates600 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext q; rfl
  rw [row]
  decide

def innerTail600_12 : List Nat := []
theorem innerScan600_12 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_12 best = best := rfl

def innerTail600_11 : List Nat := 7873 :: innerTail600_12
def innerCandidates600_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates600_11_checked : candidates edge 4 4441 7873 innerTail600_12 best.bound = innerCandidates600_11 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan600_11 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_11 best = best := by
  rw [innerTail600_11, scan, if_neg (by decide), innerCandidates600_11_checked]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_12
    (if 3 ≤ innerCandidates600_11.length then search edge 3 [4441, 7873] innerCandidates600_11 12314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan600_12

def innerTail600_10 : List Nat := 7759 :: innerTail600_11
def innerCandidates600_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates600_10_checked : candidates edge 4 4441 7759 innerTail600_11 best.bound = innerCandidates600_10 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan600_10 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_10 best = best := by
  rw [innerTail600_10, scan, if_neg (by decide), innerCandidates600_10_checked]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_11
    (if 3 ≤ innerCandidates600_10.length then search edge 3 [4441, 7759] innerCandidates600_10 12200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan600_11

def innerTail600_9 : List Nat := 6949 :: innerTail600_10
def innerCandidates600_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates600_9_checked : candidates edge 4 4441 6949 innerTail600_10 best.bound = innerCandidates600_9 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan600_9 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_9 best = best := by
  rw [innerTail600_9, scan, if_neg (by decide), innerCandidates600_9_checked]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_10
    (if 3 ≤ innerCandidates600_9.length then search edge 3 [4441, 6949] innerCandidates600_9 11390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan600_10

def innerTail600_8 : List Nat := 6481 :: innerTail600_9
def innerCandidates600_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates600_8_checked : candidates edge 4 4441 6481 innerTail600_9 best.bound = innerCandidates600_8 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan600_8 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_8 best = best := by
  rw [innerTail600_8, scan, if_neg (by decide), innerCandidates600_8_checked]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_9
    (if 3 ≤ innerCandidates600_8.length then search edge 3 [4441, 6481] innerCandidates600_8 10922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan600_9

def innerTail600_7 : List Nat := 6373 :: innerTail600_8
def innerCandidates600_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates600_7_checked : candidates edge 4 4441 6373 innerTail600_8 best.bound = innerCandidates600_7 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan600_7 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_7 best = best := by
  rw [innerTail600_7, scan, if_neg (by decide), innerCandidates600_7_checked]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_8
    (if 3 ≤ innerCandidates600_7.length then search edge 3 [4441, 6373] innerCandidates600_7 10814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan600_8

def innerTail600_6 : List Nat := 6367 :: innerTail600_7
def innerCandidates600_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates600_6_checked : candidates edge 4 4441 6367 innerTail600_7 best.bound = innerCandidates600_6 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan600_6 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_6 best = best := by
  rw [innerTail600_6, scan, if_neg (by decide), innerCandidates600_6_checked]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_7
    (if 3 ≤ innerCandidates600_6.length then search edge 3 [4441, 6367] innerCandidates600_6 10808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan600_7

def innerTail600_5 : List Nat := 6271 :: innerTail600_6
def innerCandidates600_5 : List Nat := [6373]
set_option maxRecDepth 4096 in
theorem innerCandidates600_5_checked : candidates edge 4 4441 6271 innerTail600_6 best.bound = innerCandidates600_5 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan600_5 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_5 best = best := by
  rw [innerTail600_5, scan, if_neg (by decide), innerCandidates600_5_checked]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_6
    (if 3 ≤ innerCandidates600_5.length then search edge 3 [4441, 6271] innerCandidates600_5 10712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan600_6

def innerTail600_4 : List Nat := 6247 :: innerTail600_5
def innerCandidates600_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates600_4_checked : candidates edge 4 4441 6247 innerTail600_5 best.bound = innerCandidates600_4 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan600_4 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_4 best = best := by
  rw [innerTail600_4, scan, if_neg (by decide), innerCandidates600_4_checked]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_5
    (if 3 ≤ innerCandidates600_4.length then search edge 3 [4441, 6247] innerCandidates600_4 10688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan600_5

def innerTail600_3 : List Nat := 5953 :: innerTail600_4
def innerCandidates600_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates600_3_checked : candidates edge 4 4441 5953 innerTail600_4 best.bound = innerCandidates600_3 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan600_3 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_3 best = best := by
  rw [innerTail600_3, scan, if_neg (by decide), innerCandidates600_3_checked]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_4
    (if 3 ≤ innerCandidates600_3.length then search edge 3 [4441, 5953] innerCandidates600_3 10394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan600_4

def innerTail600_2 : List Nat := 5743 :: innerTail600_3
def innerCandidates600_2 : List Nat := [6271, 7759]
set_option maxRecDepth 4096 in
theorem innerCandidates600_2_checked : candidates edge 4 4441 5743 innerTail600_3 best.bound = innerCandidates600_2 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan600_2 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_2 best = best := by
  rw [innerTail600_2, scan, if_neg (by decide), innerCandidates600_2_checked]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_3
    (if 3 ≤ innerCandidates600_2.length then search edge 3 [4441, 5743] innerCandidates600_2 10184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan600_3

def innerTail600_1 : List Nat := 5347 :: innerTail600_2
def innerCandidates600_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates600_1_checked : candidates edge 4 4441 5347 innerTail600_2 best.bound = innerCandidates600_1 := by
  unfold candidates
  have row : edge 5347 = SearchData.row5347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan600_1 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_1 best = best := by
  rw [innerTail600_1, scan, if_neg (by decide), innerCandidates600_1_checked]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_2
    (if 3 ≤ innerCandidates600_1.length then search edge 3 [4441, 5347] innerCandidates600_1 9788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan600_2

def innerTail600_0 : List Nat := 5227 :: innerTail600_1
def innerCandidates600_0 : List Nat := [6481]
set_option maxRecDepth 4096 in
theorem innerCandidates600_0_checked : candidates edge 4 4441 5227 innerTail600_1 best.bound = innerCandidates600_0 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan600_0 : scan edge (search edge 3) 4 [4441] 4441 innerTail600_0 best = best := by
  rw [innerTail600_0, scan, if_neg (by decide), innerCandidates600_0_checked]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_1
    (if 3 ≤ innerCandidates600_0.length then search edge 3 [4441, 5227] innerCandidates600_0 9668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan600_1

theorem rootBranch600 : search edge 4 [4441] rootCandidates600 4441 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4441] 4441 innerTail600_0 best = best
  exact innerScan600_0

def rootCandidates599 : List Nat := [5449, 5689, 5869, 6133, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates599_checked : candidates edge 5 0 4423 tail600 best.bound = rootCandidates599 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext q; rfl
  rw [row]
  decide

def innerTail599_5 : List Nat := []
theorem innerScan599_5 : scan edge (search edge 3) 4 [4423] 4423 innerTail599_5 best = best := rfl

def innerTail599_4 : List Nat := 8317 :: innerTail599_5
def innerCandidates599_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates599_4_checked : candidates edge 4 4423 8317 innerTail599_5 best.bound = innerCandidates599_4 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan599_4 : scan edge (search edge 3) 4 [4423] 4423 innerTail599_4 best = best := by
  rw [innerTail599_4, scan, if_neg (by decide), innerCandidates599_4_checked]
  change scan edge (search edge 3) 4 [4423] 4423 innerTail599_5
    (if 3 ≤ innerCandidates599_4.length then search edge 3 [4423, 8317] innerCandidates599_4 12740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan599_5

def innerTail599_3 : List Nat := 6133 :: innerTail599_4
def innerCandidates599_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates599_3_checked : candidates edge 4 4423 6133 innerTail599_4 best.bound = innerCandidates599_3 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan599_3 : scan edge (search edge 3) 4 [4423] 4423 innerTail599_3 best = best := by
  rw [innerTail599_3, scan, if_neg (by decide), innerCandidates599_3_checked]
  change scan edge (search edge 3) 4 [4423] 4423 innerTail599_4
    (if 3 ≤ innerCandidates599_3.length then search edge 3 [4423, 6133] innerCandidates599_3 10556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan599_4

def innerTail599_2 : List Nat := 5869 :: innerTail599_3
def innerCandidates599_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates599_2_checked : candidates edge 4 4423 5869 innerTail599_3 best.bound = innerCandidates599_2 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan599_2 : scan edge (search edge 3) 4 [4423] 4423 innerTail599_2 best = best := by
  rw [innerTail599_2, scan, if_neg (by decide), innerCandidates599_2_checked]
  change scan edge (search edge 3) 4 [4423] 4423 innerTail599_3
    (if 3 ≤ innerCandidates599_2.length then search edge 3 [4423, 5869] innerCandidates599_2 10292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan599_3

def innerTail599_1 : List Nat := 5689 :: innerTail599_2
def innerCandidates599_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates599_1_checked : candidates edge 4 4423 5689 innerTail599_2 best.bound = innerCandidates599_1 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan599_1 : scan edge (search edge 3) 4 [4423] 4423 innerTail599_1 best = best := by
  rw [innerTail599_1, scan, if_neg (by decide), innerCandidates599_1_checked]
  change scan edge (search edge 3) 4 [4423] 4423 innerTail599_2
    (if 3 ≤ innerCandidates599_1.length then search edge 3 [4423, 5689] innerCandidates599_1 10112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan599_2

def innerTail599_0 : List Nat := 5449 :: innerTail599_1
def innerCandidates599_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates599_0_checked : candidates edge 4 4423 5449 innerTail599_1 best.bound = innerCandidates599_0 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan599_0 : scan edge (search edge 3) 4 [4423] 4423 innerTail599_0 best = best := by
  rw [innerTail599_0, scan, if_neg (by decide), innerCandidates599_0_checked]
  change scan edge (search edge 3) 4 [4423] 4423 innerTail599_1
    (if 3 ≤ innerCandidates599_0.length then search edge 3 [4423, 5449] innerCandidates599_0 9872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan599_1

theorem rootBranch599 : search edge 4 [4423] rootCandidates599 4423 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4423] 4423 innerTail599_0 best = best
  exact innerScan599_0

def rootCandidates598 : List Nat := [5783, 6113, 6563, 6791, 6869, 6899, 7577, 7673, 8039]
set_option maxRecDepth 4096 in
theorem rootCandidates598_checked : candidates edge 5 0 4421 tail599 best.bound = rootCandidates598 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext q; rfl
  rw [row]
  decide

def innerTail598_9 : List Nat := []
theorem innerScan598_9 : scan edge (search edge 3) 4 [4421] 4421 innerTail598_9 best = best := rfl

def innerTail598_8 : List Nat := 8039 :: innerTail598_9
def innerCandidates598_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates598_8_checked : candidates edge 4 4421 8039 innerTail598_9 best.bound = innerCandidates598_8 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan598_8 : scan edge (search edge 3) 4 [4421] 4421 innerTail598_8 best = best := by
  rw [innerTail598_8, scan, if_neg (by decide), innerCandidates598_8_checked]
  change scan edge (search edge 3) 4 [4421] 4421 innerTail598_9
    (if 3 ≤ innerCandidates598_8.length then search edge 3 [4421, 8039] innerCandidates598_8 12460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan598_9

def innerTail598_7 : List Nat := 7673 :: innerTail598_8
def innerCandidates598_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates598_7_checked : candidates edge 4 4421 7673 innerTail598_8 best.bound = innerCandidates598_7 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan598_7 : scan edge (search edge 3) 4 [4421] 4421 innerTail598_7 best = best := by
  rw [innerTail598_7, scan, if_neg (by decide), innerCandidates598_7_checked]
  change scan edge (search edge 3) 4 [4421] 4421 innerTail598_8
    (if 3 ≤ innerCandidates598_7.length then search edge 3 [4421, 7673] innerCandidates598_7 12094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan598_8

def innerTail598_6 : List Nat := 7577 :: innerTail598_7
def innerCandidates598_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates598_6_checked : candidates edge 4 4421 7577 innerTail598_7 best.bound = innerCandidates598_6 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan598_6 : scan edge (search edge 3) 4 [4421] 4421 innerTail598_6 best = best := by
  rw [innerTail598_6, scan, if_neg (by decide), innerCandidates598_6_checked]
  change scan edge (search edge 3) 4 [4421] 4421 innerTail598_7
    (if 3 ≤ innerCandidates598_6.length then search edge 3 [4421, 7577] innerCandidates598_6 11998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan598_7

def innerTail598_5 : List Nat := 6899 :: innerTail598_6
def innerCandidates598_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates598_5_checked : candidates edge 4 4421 6899 innerTail598_6 best.bound = innerCandidates598_5 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan598_5 : scan edge (search edge 3) 4 [4421] 4421 innerTail598_5 best = best := by
  rw [innerTail598_5, scan, if_neg (by decide), innerCandidates598_5_checked]
  change scan edge (search edge 3) 4 [4421] 4421 innerTail598_6
    (if 3 ≤ innerCandidates598_5.length then search edge 3 [4421, 6899] innerCandidates598_5 11320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan598_6

def innerTail598_4 : List Nat := 6869 :: innerTail598_5
def innerCandidates598_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates598_4_checked : candidates edge 4 4421 6869 innerTail598_5 best.bound = innerCandidates598_4 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan598_4 : scan edge (search edge 3) 4 [4421] 4421 innerTail598_4 best = best := by
  rw [innerTail598_4, scan, if_neg (by decide), innerCandidates598_4_checked]
  change scan edge (search edge 3) 4 [4421] 4421 innerTail598_5
    (if 3 ≤ innerCandidates598_4.length then search edge 3 [4421, 6869] innerCandidates598_4 11290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan598_5

def innerTail598_3 : List Nat := 6791 :: innerTail598_4
def innerCandidates598_3 : List Nat := [6899]
set_option maxRecDepth 4096 in
theorem innerCandidates598_3_checked : candidates edge 4 4421 6791 innerTail598_4 best.bound = innerCandidates598_3 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan598_3 : scan edge (search edge 3) 4 [4421] 4421 innerTail598_3 best = best := by
  rw [innerTail598_3, scan, if_neg (by decide), innerCandidates598_3_checked]
  change scan edge (search edge 3) 4 [4421] 4421 innerTail598_4
    (if 3 ≤ innerCandidates598_3.length then search edge 3 [4421, 6791] innerCandidates598_3 11212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan598_4

def innerTail598_2 : List Nat := 6563 :: innerTail598_3
def innerCandidates598_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates598_2_checked : candidates edge 4 4421 6563 innerTail598_3 best.bound = innerCandidates598_2 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan598_2 : scan edge (search edge 3) 4 [4421] 4421 innerTail598_2 best = best := by
  rw [innerTail598_2, scan, if_neg (by decide), innerCandidates598_2_checked]
  change scan edge (search edge 3) 4 [4421] 4421 innerTail598_3
    (if 3 ≤ innerCandidates598_2.length then search edge 3 [4421, 6563] innerCandidates598_2 10984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan598_3

def innerTail598_1 : List Nat := 6113 :: innerTail598_2
def innerCandidates598_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates598_1_checked : candidates edge 4 4421 6113 innerTail598_2 best.bound = innerCandidates598_1 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan598_1 : scan edge (search edge 3) 4 [4421] 4421 innerTail598_1 best = best := by
  rw [innerTail598_1, scan, if_neg (by decide), innerCandidates598_1_checked]
  change scan edge (search edge 3) 4 [4421] 4421 innerTail598_2
    (if 3 ≤ innerCandidates598_1.length then search edge 3 [4421, 6113] innerCandidates598_1 10534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan598_2

def innerTail598_0 : List Nat := 5783 :: innerTail598_1
def innerCandidates598_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates598_0_checked : candidates edge 4 4421 5783 innerTail598_1 best.bound = innerCandidates598_0 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan598_0 : scan edge (search edge 3) 4 [4421] 4421 innerTail598_0 best = best := by
  rw [innerTail598_0, scan, if_neg (by decide), innerCandidates598_0_checked]
  change scan edge (search edge 3) 4 [4421] 4421 innerTail598_1
    (if 3 ≤ innerCandidates598_0.length then search edge 3 [4421, 5783] innerCandidates598_0 10204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan598_1

theorem rootBranch598 : search edge 4 [4421] rootCandidates598 4421 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4421] 4421 innerTail598_0 best = best
  exact innerScan598_0

def rootCandidates597 : List Nat := [5087, 5279, 5477, 5867, 6719, 6737, 6863, 7253]
set_option maxRecDepth 4096 in
theorem rootCandidates597_checked : candidates edge 5 0 4409 tail598 best.bound = rootCandidates597 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext q; rfl
  rw [row]
  decide

def innerTail597_8 : List Nat := []
theorem innerScan597_8 : scan edge (search edge 3) 4 [4409] 4409 innerTail597_8 best = best := rfl

def innerTail597_7 : List Nat := 7253 :: innerTail597_8
def innerCandidates597_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates597_7_checked : candidates edge 4 4409 7253 innerTail597_8 best.bound = innerCandidates597_7 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan597_7 : scan edge (search edge 3) 4 [4409] 4409 innerTail597_7 best = best := by
  rw [innerTail597_7, scan, if_neg (by decide), innerCandidates597_7_checked]
  change scan edge (search edge 3) 4 [4409] 4409 innerTail597_8
    (if 3 ≤ innerCandidates597_7.length then search edge 3 [4409, 7253] innerCandidates597_7 11662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan597_8

def innerTail597_6 : List Nat := 6863 :: innerTail597_7
def innerCandidates597_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates597_6_checked : candidates edge 4 4409 6863 innerTail597_7 best.bound = innerCandidates597_6 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan597_6 : scan edge (search edge 3) 4 [4409] 4409 innerTail597_6 best = best := by
  rw [innerTail597_6, scan, if_neg (by decide), innerCandidates597_6_checked]
  change scan edge (search edge 3) 4 [4409] 4409 innerTail597_7
    (if 3 ≤ innerCandidates597_6.length then search edge 3 [4409, 6863] innerCandidates597_6 11272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan597_7

def innerTail597_5 : List Nat := 6737 :: innerTail597_6
def innerCandidates597_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates597_5_checked : candidates edge 4 4409 6737 innerTail597_6 best.bound = innerCandidates597_5 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan597_5 : scan edge (search edge 3) 4 [4409] 4409 innerTail597_5 best = best := by
  rw [innerTail597_5, scan, if_neg (by decide), innerCandidates597_5_checked]
  change scan edge (search edge 3) 4 [4409] 4409 innerTail597_6
    (if 3 ≤ innerCandidates597_5.length then search edge 3 [4409, 6737] innerCandidates597_5 11146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan597_6

def innerTail597_4 : List Nat := 6719 :: innerTail597_5
def innerCandidates597_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates597_4_checked : candidates edge 4 4409 6719 innerTail597_5 best.bound = innerCandidates597_4 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan597_4 : scan edge (search edge 3) 4 [4409] 4409 innerTail597_4 best = best := by
  rw [innerTail597_4, scan, if_neg (by decide), innerCandidates597_4_checked]
  change scan edge (search edge 3) 4 [4409] 4409 innerTail597_5
    (if 3 ≤ innerCandidates597_4.length then search edge 3 [4409, 6719] innerCandidates597_4 11128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan597_5

def innerTail597_3 : List Nat := 5867 :: innerTail597_4
def innerCandidates597_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates597_3_checked : candidates edge 4 4409 5867 innerTail597_4 best.bound = innerCandidates597_3 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan597_3 : scan edge (search edge 3) 4 [4409] 4409 innerTail597_3 best = best := by
  rw [innerTail597_3, scan, if_neg (by decide), innerCandidates597_3_checked]
  change scan edge (search edge 3) 4 [4409] 4409 innerTail597_4
    (if 3 ≤ innerCandidates597_3.length then search edge 3 [4409, 5867] innerCandidates597_3 10276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan597_4

def innerTail597_2 : List Nat := 5477 :: innerTail597_3
def innerCandidates597_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates597_2_checked : candidates edge 4 4409 5477 innerTail597_3 best.bound = innerCandidates597_2 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan597_2 : scan edge (search edge 3) 4 [4409] 4409 innerTail597_2 best = best := by
  rw [innerTail597_2, scan, if_neg (by decide), innerCandidates597_2_checked]
  change scan edge (search edge 3) 4 [4409] 4409 innerTail597_3
    (if 3 ≤ innerCandidates597_2.length then search edge 3 [4409, 5477] innerCandidates597_2 9886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan597_3

def innerTail597_1 : List Nat := 5279 :: innerTail597_2
def innerCandidates597_1 : List Nat := [6719, 7253]
set_option maxRecDepth 4096 in
theorem innerCandidates597_1_checked : candidates edge 4 4409 5279 innerTail597_2 best.bound = innerCandidates597_1 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan597_1 : scan edge (search edge 3) 4 [4409] 4409 innerTail597_1 best = best := by
  rw [innerTail597_1, scan, if_neg (by decide), innerCandidates597_1_checked]
  change scan edge (search edge 3) 4 [4409] 4409 innerTail597_2
    (if 3 ≤ innerCandidates597_1.length then search edge 3 [4409, 5279] innerCandidates597_1 9688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan597_2

def innerTail597_0 : List Nat := 5087 :: innerTail597_1
def innerCandidates597_0 : List Nat := [6719]
set_option maxRecDepth 4096 in
theorem innerCandidates597_0_checked : candidates edge 4 4409 5087 innerTail597_1 best.bound = innerCandidates597_0 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan597_0 : scan edge (search edge 3) 4 [4409] 4409 innerTail597_0 best = best := by
  rw [innerTail597_0, scan, if_neg (by decide), innerCandidates597_0_checked]
  change scan edge (search edge 3) 4 [4409] 4409 innerTail597_1
    (if 3 ≤ innerCandidates597_0.length then search edge 3 [4409, 5087] innerCandidates597_0 9496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan597_1

theorem rootBranch597 : search edge 4 [4409] rootCandidates597 4409 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4409] 4409 innerTail597_0 best = best
  exact innerScan597_0

def rootCandidates596 : List Nat := [4421, 4691, 4703, 4871, 6311, 6563, 6701, 6833, 7013, 7823, 7841, 8087]
set_option maxRecDepth 4096 in
theorem rootCandidates596_checked : candidates edge 5 0 4397 tail597 best.bound = rootCandidates596 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext q; rfl
  rw [row]
  decide

def innerTail596_12 : List Nat := []
theorem innerScan596_12 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_12 best = best := rfl

def innerTail596_11 : List Nat := 8087 :: innerTail596_12
def innerCandidates596_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates596_11_checked : candidates edge 4 4397 8087 innerTail596_12 best.bound = innerCandidates596_11 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan596_11 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_11 best = best := by
  rw [innerTail596_11, scan, if_neg (by decide), innerCandidates596_11_checked]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_12
    (if 3 ≤ innerCandidates596_11.length then search edge 3 [4397, 8087] innerCandidates596_11 12484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan596_12

def innerTail596_10 : List Nat := 7841 :: innerTail596_11
def innerCandidates596_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates596_10_checked : candidates edge 4 4397 7841 innerTail596_11 best.bound = innerCandidates596_10 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan596_10 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_10 best = best := by
  rw [innerTail596_10, scan, if_neg (by decide), innerCandidates596_10_checked]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_11
    (if 3 ≤ innerCandidates596_10.length then search edge 3 [4397, 7841] innerCandidates596_10 12238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan596_11

def innerTail596_9 : List Nat := 7823 :: innerTail596_10
def innerCandidates596_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates596_9_checked : candidates edge 4 4397 7823 innerTail596_10 best.bound = innerCandidates596_9 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan596_9 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_9 best = best := by
  rw [innerTail596_9, scan, if_neg (by decide), innerCandidates596_9_checked]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_10
    (if 3 ≤ innerCandidates596_9.length then search edge 3 [4397, 7823] innerCandidates596_9 12220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan596_10

def innerTail596_8 : List Nat := 7013 :: innerTail596_9
def innerCandidates596_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates596_8_checked : candidates edge 4 4397 7013 innerTail596_9 best.bound = innerCandidates596_8 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan596_8 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_8 best = best := by
  rw [innerTail596_8, scan, if_neg (by decide), innerCandidates596_8_checked]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_9
    (if 3 ≤ innerCandidates596_8.length then search edge 3 [4397, 7013] innerCandidates596_8 11410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan596_9

def innerTail596_7 : List Nat := 6833 :: innerTail596_8
def innerCandidates596_7 : List Nat := [8087]
set_option maxRecDepth 4096 in
theorem innerCandidates596_7_checked : candidates edge 4 4397 6833 innerTail596_8 best.bound = innerCandidates596_7 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan596_7 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_7 best = best := by
  rw [innerTail596_7, scan, if_neg (by decide), innerCandidates596_7_checked]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_8
    (if 3 ≤ innerCandidates596_7.length then search edge 3 [4397, 6833] innerCandidates596_7 11230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan596_8

def innerTail596_6 : List Nat := 6701 :: innerTail596_7
def innerCandidates596_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates596_6_checked : candidates edge 4 4397 6701 innerTail596_7 best.bound = innerCandidates596_6 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan596_6 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_6 best = best := by
  rw [innerTail596_6, scan, if_neg (by decide), innerCandidates596_6_checked]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_7
    (if 3 ≤ innerCandidates596_6.length then search edge 3 [4397, 6701] innerCandidates596_6 11098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan596_7

def innerTail596_5 : List Nat := 6563 :: innerTail596_6
def innerCandidates596_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates596_5_checked : candidates edge 4 4397 6563 innerTail596_6 best.bound = innerCandidates596_5 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan596_5 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_5 best = best := by
  rw [innerTail596_5, scan, if_neg (by decide), innerCandidates596_5_checked]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_6
    (if 3 ≤ innerCandidates596_5.length then search edge 3 [4397, 6563] innerCandidates596_5 10960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan596_6

def innerTail596_4 : List Nat := 6311 :: innerTail596_5
def innerCandidates596_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates596_4_checked : candidates edge 4 4397 6311 innerTail596_5 best.bound = innerCandidates596_4 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan596_4 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_4 best = best := by
  rw [innerTail596_4, scan, if_neg (by decide), innerCandidates596_4_checked]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_5
    (if 3 ≤ innerCandidates596_4.length then search edge 3 [4397, 6311] innerCandidates596_4 10708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan596_5

def innerTail596_3 : List Nat := 4871 :: innerTail596_4
def innerCandidates596_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates596_3_checked : candidates edge 4 4397 4871 innerTail596_4 best.bound = innerCandidates596_3 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan596_3 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_3 best = best := by
  rw [innerTail596_3, scan, if_neg (by decide), innerCandidates596_3_checked]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_4
    (if 3 ≤ innerCandidates596_3.length then search edge 3 [4397, 4871] innerCandidates596_3 9268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan596_4

def innerTail596_2 : List Nat := 4703 :: innerTail596_3
def innerCandidates596_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates596_2_checked : candidates edge 4 4397 4703 innerTail596_3 best.bound = innerCandidates596_2 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan596_2 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_2 best = best := by
  rw [innerTail596_2, scan, if_neg (by decide), innerCandidates596_2_checked]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_3
    (if 3 ≤ innerCandidates596_2.length then search edge 3 [4397, 4703] innerCandidates596_2 9100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan596_3

def innerTail596_1 : List Nat := 4691 :: innerTail596_2
def innerCandidates596_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates596_1_checked : candidates edge 4 4397 4691 innerTail596_2 best.bound = innerCandidates596_1 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan596_1 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_1 best = best := by
  rw [innerTail596_1, scan, if_neg (by decide), innerCandidates596_1_checked]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_2
    (if 3 ≤ innerCandidates596_1.length then search edge 3 [4397, 4691] innerCandidates596_1 9088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan596_2

def innerTail596_0 : List Nat := 4421 :: innerTail596_1
def innerCandidates596_0 : List Nat := [6563]
set_option maxRecDepth 4096 in
theorem innerCandidates596_0_checked : candidates edge 4 4397 4421 innerTail596_1 best.bound = innerCandidates596_0 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan596_0 : scan edge (search edge 3) 4 [4397] 4397 innerTail596_0 best = best := by
  rw [innerTail596_0, scan, if_neg (by decide), innerCandidates596_0_checked]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_1
    (if 3 ≤ innerCandidates596_0.length then search edge 3 [4397, 4421] innerCandidates596_0 8818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan596_1

theorem rootBranch596 : search edge 4 [4397] rootCandidates596 4397 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4397] 4397 innerTail596_0 best = best
  exact innerScan596_0

def rootCandidates595 : List Nat := [5639, 6263, 6869, 7901, 8111, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates595_checked : candidates edge 5 0 4391 tail596 best.bound = rootCandidates595 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext q; rfl
  rw [row]
  decide

def innerTail595_6 : List Nat := []
theorem innerScan595_6 : scan edge (search edge 3) 4 [4391] 4391 innerTail595_6 best = best := rfl

def innerTail595_5 : List Nat := 8219 :: innerTail595_6
def innerCandidates595_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates595_5_checked : candidates edge 4 4391 8219 innerTail595_6 best.bound = innerCandidates595_5 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan595_5 : scan edge (search edge 3) 4 [4391] 4391 innerTail595_5 best = best := by
  rw [innerTail595_5, scan, if_neg (by decide), innerCandidates595_5_checked]
  change scan edge (search edge 3) 4 [4391] 4391 innerTail595_6
    (if 3 ≤ innerCandidates595_5.length then search edge 3 [4391, 8219] innerCandidates595_5 12610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan595_6

def innerTail595_4 : List Nat := 8111 :: innerTail595_5
def innerCandidates595_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates595_4_checked : candidates edge 4 4391 8111 innerTail595_5 best.bound = innerCandidates595_4 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan595_4 : scan edge (search edge 3) 4 [4391] 4391 innerTail595_4 best = best := by
  rw [innerTail595_4, scan, if_neg (by decide), innerCandidates595_4_checked]
  change scan edge (search edge 3) 4 [4391] 4391 innerTail595_5
    (if 3 ≤ innerCandidates595_4.length then search edge 3 [4391, 8111] innerCandidates595_4 12502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan595_5

def innerTail595_3 : List Nat := 7901 :: innerTail595_4
def innerCandidates595_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates595_3_checked : candidates edge 4 4391 7901 innerTail595_4 best.bound = innerCandidates595_3 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan595_3 : scan edge (search edge 3) 4 [4391] 4391 innerTail595_3 best = best := by
  rw [innerTail595_3, scan, if_neg (by decide), innerCandidates595_3_checked]
  change scan edge (search edge 3) 4 [4391] 4391 innerTail595_4
    (if 3 ≤ innerCandidates595_3.length then search edge 3 [4391, 7901] innerCandidates595_3 12292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan595_4

def innerTail595_2 : List Nat := 6869 :: innerTail595_3
def innerCandidates595_2 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates595_2_checked : candidates edge 4 4391 6869 innerTail595_3 best.bound = innerCandidates595_2 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan595_2 : scan edge (search edge 3) 4 [4391] 4391 innerTail595_2 best = best := by
  rw [innerTail595_2, scan, if_neg (by decide), innerCandidates595_2_checked]
  change scan edge (search edge 3) 4 [4391] 4391 innerTail595_3
    (if 3 ≤ innerCandidates595_2.length then search edge 3 [4391, 6869] innerCandidates595_2 11260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan595_3

def innerTail595_1 : List Nat := 6263 :: innerTail595_2
def innerCandidates595_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates595_1_checked : candidates edge 4 4391 6263 innerTail595_2 best.bound = innerCandidates595_1 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan595_1 : scan edge (search edge 3) 4 [4391] 4391 innerTail595_1 best = best := by
  rw [innerTail595_1, scan, if_neg (by decide), innerCandidates595_1_checked]
  change scan edge (search edge 3) 4 [4391] 4391 innerTail595_2
    (if 3 ≤ innerCandidates595_1.length then search edge 3 [4391, 6263] innerCandidates595_1 10654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan595_2

def innerTail595_0 : List Nat := 5639 :: innerTail595_1
def innerCandidates595_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates595_0_checked : candidates edge 4 4391 5639 innerTail595_1 best.bound = innerCandidates595_0 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan595_0 : scan edge (search edge 3) 4 [4391] 4391 innerTail595_0 best = best := by
  rw [innerTail595_0, scan, if_neg (by decide), innerCandidates595_0_checked]
  change scan edge (search edge 3) 4 [4391] 4391 innerTail595_1
    (if 3 ≤ innerCandidates595_0.length then search edge 3 [4391, 5639] innerCandidates595_0 10030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan595_1

theorem rootBranch595 : search edge 4 [4391] rootCandidates595 4391 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4391] 4391 innerTail595_0 best = best
  exact innerScan595_0

def rootCandidates594 : List Nat := [5189, 5297, 5693, 5801, 5927, 7019, 7121, 7541, 7919, 8039, 8147, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates594_checked : candidates edge 5 0 4373 tail595 best.bound = rootCandidates594 := by
  unfold candidates
  have row : edge 4373 = SearchData.row4373.contains := by funext q; rfl
  rw [row]
  decide

def innerTail594_12 : List Nat := []
theorem innerScan594_12 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_12 best = best := rfl

def innerTail594_11 : List Nat := 8237 :: innerTail594_12
def innerCandidates594_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates594_11_checked : candidates edge 4 4373 8237 innerTail594_12 best.bound = innerCandidates594_11 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan594_11 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_11 best = best := by
  rw [innerTail594_11, scan, if_neg (by decide), innerCandidates594_11_checked]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_12
    (if 3 ≤ innerCandidates594_11.length then search edge 3 [4373, 8237] innerCandidates594_11 12610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan594_12

def innerTail594_10 : List Nat := 8147 :: innerTail594_11
def innerCandidates594_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates594_10_checked : candidates edge 4 4373 8147 innerTail594_11 best.bound = innerCandidates594_10 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan594_10 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_10 best = best := by
  rw [innerTail594_10, scan, if_neg (by decide), innerCandidates594_10_checked]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_11
    (if 3 ≤ innerCandidates594_10.length then search edge 3 [4373, 8147] innerCandidates594_10 12520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan594_11

def innerTail594_9 : List Nat := 8039 :: innerTail594_10
def innerCandidates594_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates594_9_checked : candidates edge 4 4373 8039 innerTail594_10 best.bound = innerCandidates594_9 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan594_9 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_9 best = best := by
  rw [innerTail594_9, scan, if_neg (by decide), innerCandidates594_9_checked]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_10
    (if 3 ≤ innerCandidates594_9.length then search edge 3 [4373, 8039] innerCandidates594_9 12412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan594_10

def innerTail594_8 : List Nat := 7919 :: innerTail594_9
def innerCandidates594_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates594_8_checked : candidates edge 4 4373 7919 innerTail594_9 best.bound = innerCandidates594_8 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan594_8 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_8 best = best := by
  rw [innerTail594_8, scan, if_neg (by decide), innerCandidates594_8_checked]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_9
    (if 3 ≤ innerCandidates594_8.length then search edge 3 [4373, 7919] innerCandidates594_8 12292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan594_9

def innerTail594_7 : List Nat := 7541 :: innerTail594_8
def innerCandidates594_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates594_7_checked : candidates edge 4 4373 7541 innerTail594_8 best.bound = innerCandidates594_7 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan594_7 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_7 best = best := by
  rw [innerTail594_7, scan, if_neg (by decide), innerCandidates594_7_checked]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_8
    (if 3 ≤ innerCandidates594_7.length then search edge 3 [4373, 7541] innerCandidates594_7 11914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan594_8

def innerTail594_6 : List Nat := 7121 :: innerTail594_7
def innerCandidates594_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates594_6_checked : candidates edge 4 4373 7121 innerTail594_7 best.bound = innerCandidates594_6 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan594_6 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_6 best = best := by
  rw [innerTail594_6, scan, if_neg (by decide), innerCandidates594_6_checked]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_7
    (if 3 ≤ innerCandidates594_6.length then search edge 3 [4373, 7121] innerCandidates594_6 11494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan594_7

def innerTail594_5 : List Nat := 7019 :: innerTail594_6
def innerCandidates594_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates594_5_checked : candidates edge 4 4373 7019 innerTail594_6 best.bound = innerCandidates594_5 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan594_5 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_5 best = best := by
  rw [innerTail594_5, scan, if_neg (by decide), innerCandidates594_5_checked]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_6
    (if 3 ≤ innerCandidates594_5.length then search edge 3 [4373, 7019] innerCandidates594_5 11392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan594_6

def innerTail594_4 : List Nat := 5927 :: innerTail594_5
def innerCandidates594_4 : List Nat := [8237]
set_option maxRecDepth 4096 in
theorem innerCandidates594_4_checked : candidates edge 4 4373 5927 innerTail594_5 best.bound = innerCandidates594_4 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan594_4 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_4 best = best := by
  rw [innerTail594_4, scan, if_neg (by decide), innerCandidates594_4_checked]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_5
    (if 3 ≤ innerCandidates594_4.length then search edge 3 [4373, 5927] innerCandidates594_4 10300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan594_5

def innerTail594_3 : List Nat := 5801 :: innerTail594_4
def innerCandidates594_3 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates594_3_checked : candidates edge 4 4373 5801 innerTail594_4 best.bound = innerCandidates594_3 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan594_3 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_3 best = best := by
  rw [innerTail594_3, scan, if_neg (by decide), innerCandidates594_3_checked]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_4
    (if 3 ≤ innerCandidates594_3.length then search edge 3 [4373, 5801] innerCandidates594_3 10174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan594_4

def innerTail594_2 : List Nat := 5693 :: innerTail594_3
def innerCandidates594_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates594_2_checked : candidates edge 4 4373 5693 innerTail594_3 best.bound = innerCandidates594_2 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan594_2 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_2 best = best := by
  rw [innerTail594_2, scan, if_neg (by decide), innerCandidates594_2_checked]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_3
    (if 3 ≤ innerCandidates594_2.length then search edge 3 [4373, 5693] innerCandidates594_2 10066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan594_3

def innerTail594_1 : List Nat := 5297 :: innerTail594_2
def innerCandidates594_1 : List Nat := [7541]
set_option maxRecDepth 4096 in
theorem innerCandidates594_1_checked : candidates edge 4 4373 5297 innerTail594_2 best.bound = innerCandidates594_1 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan594_1 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_1 best = best := by
  rw [innerTail594_1, scan, if_neg (by decide), innerCandidates594_1_checked]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_2
    (if 3 ≤ innerCandidates594_1.length then search edge 3 [4373, 5297] innerCandidates594_1 9670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan594_2

def innerTail594_0 : List Nat := 5189 :: innerTail594_1
def innerCandidates594_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates594_0_checked : candidates edge 4 4373 5189 innerTail594_1 best.bound = innerCandidates594_0 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan594_0 : scan edge (search edge 3) 4 [4373] 4373 innerTail594_0 best = best := by
  rw [innerTail594_0, scan, if_neg (by decide), innerCandidates594_0_checked]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_1
    (if 3 ≤ innerCandidates594_0.length then search edge 3 [4373, 5189] innerCandidates594_0 9562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan594_1

theorem rootBranch594 : search edge 4 [4373] rootCandidates594 4373 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4373] 4373 innerTail594_0 best = best
  exact innerScan594_0

end ProjectEulerSolutions.PrimePairUpper.Search
