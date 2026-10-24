import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates461 : List Nat := [3517, 4363, 4447, 4567, 4651, 5779, 5881, 6343, 6961, 6967, 7297]
set_option maxRecDepth 4096 in
theorem rootCandidates461_checked : candidates edge 5 0 3301 tail462 best.bound = rootCandidates461 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext q; rfl
  rw [row]
  decide

def innerTail461_11 : List Nat := []
theorem innerScan461_11 : scan edge (search edge 3) 4 [3301] 3301 innerTail461_11 best = best := rfl

def innerTail461_10 : List Nat := 7297 :: innerTail461_11
def innerCandidates461_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates461_10_checked : candidates edge 4 3301 7297 innerTail461_11 best.bound = innerCandidates461_10 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan461_10 : scan edge (search edge 3) 4 [3301] 3301 innerTail461_10 best = best := by
  rw [innerTail461_10, scan, if_neg (by decide), innerCandidates461_10_checked]
  change scan edge (search edge 3) 4 [3301] 3301 innerTail461_11
    (if 3 ≤ innerCandidates461_10.length then search edge 3 [3301, 7297] innerCandidates461_10 10598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan461_11

def innerTail461_9 : List Nat := 6967 :: innerTail461_10
def innerCandidates461_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates461_9_checked : candidates edge 4 3301 6967 innerTail461_10 best.bound = innerCandidates461_9 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan461_9 : scan edge (search edge 3) 4 [3301] 3301 innerTail461_9 best = best := by
  rw [innerTail461_9, scan, if_neg (by decide), innerCandidates461_9_checked]
  change scan edge (search edge 3) 4 [3301] 3301 innerTail461_10
    (if 3 ≤ innerCandidates461_9.length then search edge 3 [3301, 6967] innerCandidates461_9 10268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan461_10

def innerTail461_8 : List Nat := 6961 :: innerTail461_9
def innerCandidates461_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates461_8_checked : candidates edge 4 3301 6961 innerTail461_9 best.bound = innerCandidates461_8 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan461_8 : scan edge (search edge 3) 4 [3301] 3301 innerTail461_8 best = best := by
  rw [innerTail461_8, scan, if_neg (by decide), innerCandidates461_8_checked]
  change scan edge (search edge 3) 4 [3301] 3301 innerTail461_9
    (if 3 ≤ innerCandidates461_8.length then search edge 3 [3301, 6961] innerCandidates461_8 10262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan461_9

def innerTail461_7 : List Nat := 6343 :: innerTail461_8
def innerCandidates461_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates461_7_checked : candidates edge 4 3301 6343 innerTail461_8 best.bound = innerCandidates461_7 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan461_7 : scan edge (search edge 3) 4 [3301] 3301 innerTail461_7 best = best := by
  rw [innerTail461_7, scan, if_neg (by decide), innerCandidates461_7_checked]
  change scan edge (search edge 3) 4 [3301] 3301 innerTail461_8
    (if 3 ≤ innerCandidates461_7.length then search edge 3 [3301, 6343] innerCandidates461_7 9644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan461_8

def innerTail461_6 : List Nat := 5881 :: innerTail461_7
def innerCandidates461_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates461_6_checked : candidates edge 4 3301 5881 innerTail461_7 best.bound = innerCandidates461_6 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan461_6 : scan edge (search edge 3) 4 [3301] 3301 innerTail461_6 best = best := by
  rw [innerTail461_6, scan, if_neg (by decide), innerCandidates461_6_checked]
  change scan edge (search edge 3) 4 [3301] 3301 innerTail461_7
    (if 3 ≤ innerCandidates461_6.length then search edge 3 [3301, 5881] innerCandidates461_6 9182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan461_7

def innerTail461_5 : List Nat := 5779 :: innerTail461_6
def innerCandidates461_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates461_5_checked : candidates edge 4 3301 5779 innerTail461_6 best.bound = innerCandidates461_5 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan461_5 : scan edge (search edge 3) 4 [3301] 3301 innerTail461_5 best = best := by
  rw [innerTail461_5, scan, if_neg (by decide), innerCandidates461_5_checked]
  change scan edge (search edge 3) 4 [3301] 3301 innerTail461_6
    (if 3 ≤ innerCandidates461_5.length then search edge 3 [3301, 5779] innerCandidates461_5 9080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan461_6

def innerTail461_4 : List Nat := 4651 :: innerTail461_5
def innerCandidates461_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates461_4_checked : candidates edge 4 3301 4651 innerTail461_5 best.bound = innerCandidates461_4 := by
  unfold candidates
  have row : edge 4651 = SearchData.row4651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan461_4 : scan edge (search edge 3) 4 [3301] 3301 innerTail461_4 best = best := by
  rw [innerTail461_4, scan, if_neg (by decide), innerCandidates461_4_checked]
  change scan edge (search edge 3) 4 [3301] 3301 innerTail461_5
    (if 3 ≤ innerCandidates461_4.length then search edge 3 [3301, 4651] innerCandidates461_4 7952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan461_5

def innerTail461_3 : List Nat := 4567 :: innerTail461_4
def innerCandidates461_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates461_3_checked : candidates edge 4 3301 4567 innerTail461_4 best.bound = innerCandidates461_3 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan461_3 : scan edge (search edge 3) 4 [3301] 3301 innerTail461_3 best = best := by
  rw [innerTail461_3, scan, if_neg (by decide), innerCandidates461_3_checked]
  change scan edge (search edge 3) 4 [3301] 3301 innerTail461_4
    (if 3 ≤ innerCandidates461_3.length then search edge 3 [3301, 4567] innerCandidates461_3 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan461_4

def innerTail461_2 : List Nat := 4447 :: innerTail461_3
def innerCandidates461_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates461_2_checked : candidates edge 4 3301 4447 innerTail461_3 best.bound = innerCandidates461_2 := by
  unfold candidates
  have row : edge 4447 = SearchData.row4447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan461_2 : scan edge (search edge 3) 4 [3301] 3301 innerTail461_2 best = best := by
  rw [innerTail461_2, scan, if_neg (by decide), innerCandidates461_2_checked]
  change scan edge (search edge 3) 4 [3301] 3301 innerTail461_3
    (if 3 ≤ innerCandidates461_2.length then search edge 3 [3301, 4447] innerCandidates461_2 7748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan461_3

def innerTail461_1 : List Nat := 4363 :: innerTail461_2
def innerCandidates461_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates461_1_checked : candidates edge 4 3301 4363 innerTail461_2 best.bound = innerCandidates461_1 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan461_1 : scan edge (search edge 3) 4 [3301] 3301 innerTail461_1 best = best := by
  rw [innerTail461_1, scan, if_neg (by decide), innerCandidates461_1_checked]
  change scan edge (search edge 3) 4 [3301] 3301 innerTail461_2
    (if 3 ≤ innerCandidates461_1.length then search edge 3 [3301, 4363] innerCandidates461_1 7664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan461_2

def innerTail461_0 : List Nat := 3517 :: innerTail461_1
def innerCandidates461_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates461_0_checked : candidates edge 4 3301 3517 innerTail461_1 best.bound = innerCandidates461_0 := by
  unfold candidates
  have row : edge 3517 = SearchData.row3517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan461_0 : scan edge (search edge 3) 4 [3301] 3301 innerTail461_0 best = best := by
  rw [innerTail461_0, scan, if_neg (by decide), innerCandidates461_0_checked]
  change scan edge (search edge 3) 4 [3301] 3301 innerTail461_1
    (if 3 ≤ innerCandidates461_0.length then search edge 3 [3301, 3517] innerCandidates461_0 6818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan461_1

theorem rootBranch461 : search edge 4 [3301] rootCandidates461 3301 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3301] 3301 innerTail461_0 best = best
  exact innerScan461_0

def rootCandidates460 : List Nat := [3767, 4019, 4523, 4817, 6011, 6203, 6869, 6959, 7043, 7499]
set_option maxRecDepth 4096 in
theorem rootCandidates460_checked : candidates edge 5 0 3299 tail461 best.bound = rootCandidates460 := by
  unfold candidates
  have row : edge 3299 = SearchData.row3299.contains := by funext q; rfl
  rw [row]
  decide

def innerTail460_10 : List Nat := []
theorem innerScan460_10 : scan edge (search edge 3) 4 [3299] 3299 innerTail460_10 best = best := rfl

def innerTail460_9 : List Nat := 7499 :: innerTail460_10
def innerCandidates460_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates460_9_checked : candidates edge 4 3299 7499 innerTail460_10 best.bound = innerCandidates460_9 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan460_9 : scan edge (search edge 3) 4 [3299] 3299 innerTail460_9 best = best := by
  rw [innerTail460_9, scan, if_neg (by decide), innerCandidates460_9_checked]
  change scan edge (search edge 3) 4 [3299] 3299 innerTail460_10
    (if 3 ≤ innerCandidates460_9.length then search edge 3 [3299, 7499] innerCandidates460_9 10798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan460_10

def innerTail460_8 : List Nat := 7043 :: innerTail460_9
def innerCandidates460_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates460_8_checked : candidates edge 4 3299 7043 innerTail460_9 best.bound = innerCandidates460_8 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan460_8 : scan edge (search edge 3) 4 [3299] 3299 innerTail460_8 best = best := by
  rw [innerTail460_8, scan, if_neg (by decide), innerCandidates460_8_checked]
  change scan edge (search edge 3) 4 [3299] 3299 innerTail460_9
    (if 3 ≤ innerCandidates460_8.length then search edge 3 [3299, 7043] innerCandidates460_8 10342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan460_9

def innerTail460_7 : List Nat := 6959 :: innerTail460_8
def innerCandidates460_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates460_7_checked : candidates edge 4 3299 6959 innerTail460_8 best.bound = innerCandidates460_7 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan460_7 : scan edge (search edge 3) 4 [3299] 3299 innerTail460_7 best = best := by
  rw [innerTail460_7, scan, if_neg (by decide), innerCandidates460_7_checked]
  change scan edge (search edge 3) 4 [3299] 3299 innerTail460_8
    (if 3 ≤ innerCandidates460_7.length then search edge 3 [3299, 6959] innerCandidates460_7 10258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan460_8

def innerTail460_6 : List Nat := 6869 :: innerTail460_7
def innerCandidates460_6 : List Nat := [6959]
set_option maxRecDepth 4096 in
theorem innerCandidates460_6_checked : candidates edge 4 3299 6869 innerTail460_7 best.bound = innerCandidates460_6 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan460_6 : scan edge (search edge 3) 4 [3299] 3299 innerTail460_6 best = best := by
  rw [innerTail460_6, scan, if_neg (by decide), innerCandidates460_6_checked]
  change scan edge (search edge 3) 4 [3299] 3299 innerTail460_7
    (if 3 ≤ innerCandidates460_6.length then search edge 3 [3299, 6869] innerCandidates460_6 10168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan460_7

def innerTail460_5 : List Nat := 6203 :: innerTail460_6
def innerCandidates460_5 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates460_5_checked : candidates edge 4 3299 6203 innerTail460_6 best.bound = innerCandidates460_5 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan460_5 : scan edge (search edge 3) 4 [3299] 3299 innerTail460_5 best = best := by
  rw [innerTail460_5, scan, if_neg (by decide), innerCandidates460_5_checked]
  change scan edge (search edge 3) 4 [3299] 3299 innerTail460_6
    (if 3 ≤ innerCandidates460_5.length then search edge 3 [3299, 6203] innerCandidates460_5 9502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan460_6

def innerTail460_4 : List Nat := 6011 :: innerTail460_5
def innerCandidates460_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates460_4_checked : candidates edge 4 3299 6011 innerTail460_5 best.bound = innerCandidates460_4 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan460_4 : scan edge (search edge 3) 4 [3299] 3299 innerTail460_4 best = best := by
  rw [innerTail460_4, scan, if_neg (by decide), innerCandidates460_4_checked]
  change scan edge (search edge 3) 4 [3299] 3299 innerTail460_5
    (if 3 ≤ innerCandidates460_4.length then search edge 3 [3299, 6011] innerCandidates460_4 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan460_5

def innerTail460_3 : List Nat := 4817 :: innerTail460_4
def innerCandidates460_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates460_3_checked : candidates edge 4 3299 4817 innerTail460_4 best.bound = innerCandidates460_3 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan460_3 : scan edge (search edge 3) 4 [3299] 3299 innerTail460_3 best = best := by
  rw [innerTail460_3, scan, if_neg (by decide), innerCandidates460_3_checked]
  change scan edge (search edge 3) 4 [3299] 3299 innerTail460_4
    (if 3 ≤ innerCandidates460_3.length then search edge 3 [3299, 4817] innerCandidates460_3 8116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan460_4

def innerTail460_2 : List Nat := 4523 :: innerTail460_3
def innerCandidates460_2 : List Nat := [4817]
set_option maxRecDepth 4096 in
theorem innerCandidates460_2_checked : candidates edge 4 3299 4523 innerTail460_3 best.bound = innerCandidates460_2 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan460_2 : scan edge (search edge 3) 4 [3299] 3299 innerTail460_2 best = best := by
  rw [innerTail460_2, scan, if_neg (by decide), innerCandidates460_2_checked]
  change scan edge (search edge 3) 4 [3299] 3299 innerTail460_3
    (if 3 ≤ innerCandidates460_2.length then search edge 3 [3299, 4523] innerCandidates460_2 7822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan460_3

def innerTail460_1 : List Nat := 4019 :: innerTail460_2
def innerCandidates460_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates460_1_checked : candidates edge 4 3299 4019 innerTail460_2 best.bound = innerCandidates460_1 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan460_1 : scan edge (search edge 3) 4 [3299] 3299 innerTail460_1 best = best := by
  rw [innerTail460_1, scan, if_neg (by decide), innerCandidates460_1_checked]
  change scan edge (search edge 3) 4 [3299] 3299 innerTail460_2
    (if 3 ≤ innerCandidates460_1.length then search edge 3 [3299, 4019] innerCandidates460_1 7318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan460_2

def innerTail460_0 : List Nat := 3767 :: innerTail460_1
def innerCandidates460_0 : List Nat := [4523]
set_option maxRecDepth 4096 in
theorem innerCandidates460_0_checked : candidates edge 4 3299 3767 innerTail460_1 best.bound = innerCandidates460_0 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan460_0 : scan edge (search edge 3) 4 [3299] 3299 innerTail460_0 best = best := by
  rw [innerTail460_0, scan, if_neg (by decide), innerCandidates460_0_checked]
  change scan edge (search edge 3) 4 [3299] 3299 innerTail460_1
    (if 3 ≤ innerCandidates460_0.length then search edge 3 [3299, 3767] innerCandidates460_0 7066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan460_1

theorem rootBranch460 : search edge 4 [3299] rootCandidates460 3299 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3299] 3299 innerTail460_0 best = best
  exact innerScan460_0

def rootCandidates459 : List Nat := [3319, 3361, 4243, 5227, 5653, 5923, 6661, 7603, 7621, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates459_checked : candidates edge 5 0 3271 tail460 best.bound = rootCandidates459 := by
  unfold candidates
  have row : edge 3271 = SearchData.row3271.contains := by funext q; rfl
  rw [row]
  decide

def innerTail459_10 : List Nat := []
theorem innerScan459_10 : scan edge (search edge 3) 4 [3271] 3271 innerTail459_10 best = best := rfl

def innerTail459_9 : List Nat := 8377 :: innerTail459_10
def innerCandidates459_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates459_9_checked : candidates edge 4 3271 8377 innerTail459_10 best.bound = innerCandidates459_9 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan459_9 : scan edge (search edge 3) 4 [3271] 3271 innerTail459_9 best = best := by
  rw [innerTail459_9, scan, if_neg (by decide), innerCandidates459_9_checked]
  change scan edge (search edge 3) 4 [3271] 3271 innerTail459_10
    (if 3 ≤ innerCandidates459_9.length then search edge 3 [3271, 8377] innerCandidates459_9 11648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan459_10

def innerTail459_8 : List Nat := 7621 :: innerTail459_9
def innerCandidates459_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates459_8_checked : candidates edge 4 3271 7621 innerTail459_9 best.bound = innerCandidates459_8 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan459_8 : scan edge (search edge 3) 4 [3271] 3271 innerTail459_8 best = best := by
  rw [innerTail459_8, scan, if_neg (by decide), innerCandidates459_8_checked]
  change scan edge (search edge 3) 4 [3271] 3271 innerTail459_9
    (if 3 ≤ innerCandidates459_8.length then search edge 3 [3271, 7621] innerCandidates459_8 10892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan459_9

def innerTail459_7 : List Nat := 7603 :: innerTail459_8
def innerCandidates459_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates459_7_checked : candidates edge 4 3271 7603 innerTail459_8 best.bound = innerCandidates459_7 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan459_7 : scan edge (search edge 3) 4 [3271] 3271 innerTail459_7 best = best := by
  rw [innerTail459_7, scan, if_neg (by decide), innerCandidates459_7_checked]
  change scan edge (search edge 3) 4 [3271] 3271 innerTail459_8
    (if 3 ≤ innerCandidates459_7.length then search edge 3 [3271, 7603] innerCandidates459_7 10874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan459_8

def innerTail459_6 : List Nat := 6661 :: innerTail459_7
def innerCandidates459_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates459_6_checked : candidates edge 4 3271 6661 innerTail459_7 best.bound = innerCandidates459_6 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan459_6 : scan edge (search edge 3) 4 [3271] 3271 innerTail459_6 best = best := by
  rw [innerTail459_6, scan, if_neg (by decide), innerCandidates459_6_checked]
  change scan edge (search edge 3) 4 [3271] 3271 innerTail459_7
    (if 3 ≤ innerCandidates459_6.length then search edge 3 [3271, 6661] innerCandidates459_6 9932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan459_7

def innerTail459_5 : List Nat := 5923 :: innerTail459_6
def innerCandidates459_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates459_5_checked : candidates edge 4 3271 5923 innerTail459_6 best.bound = innerCandidates459_5 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan459_5 : scan edge (search edge 3) 4 [3271] 3271 innerTail459_5 best = best := by
  rw [innerTail459_5, scan, if_neg (by decide), innerCandidates459_5_checked]
  change scan edge (search edge 3) 4 [3271] 3271 innerTail459_6
    (if 3 ≤ innerCandidates459_5.length then search edge 3 [3271, 5923] innerCandidates459_5 9194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan459_6

def innerTail459_4 : List Nat := 5653 :: innerTail459_5
def innerCandidates459_4 : List Nat := [5923]
set_option maxRecDepth 4096 in
theorem innerCandidates459_4_checked : candidates edge 4 3271 5653 innerTail459_5 best.bound = innerCandidates459_4 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan459_4 : scan edge (search edge 3) 4 [3271] 3271 innerTail459_4 best = best := by
  rw [innerTail459_4, scan, if_neg (by decide), innerCandidates459_4_checked]
  change scan edge (search edge 3) 4 [3271] 3271 innerTail459_5
    (if 3 ≤ innerCandidates459_4.length then search edge 3 [3271, 5653] innerCandidates459_4 8924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan459_5

def innerTail459_3 : List Nat := 5227 :: innerTail459_4
def innerCandidates459_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates459_3_checked : candidates edge 4 3271 5227 innerTail459_4 best.bound = innerCandidates459_3 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan459_3 : scan edge (search edge 3) 4 [3271] 3271 innerTail459_3 best = best := by
  rw [innerTail459_3, scan, if_neg (by decide), innerCandidates459_3_checked]
  change scan edge (search edge 3) 4 [3271] 3271 innerTail459_4
    (if 3 ≤ innerCandidates459_3.length then search edge 3 [3271, 5227] innerCandidates459_3 8498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan459_4

def innerTail459_2 : List Nat := 4243 :: innerTail459_3
def innerCandidates459_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates459_2_checked : candidates edge 4 3271 4243 innerTail459_3 best.bound = innerCandidates459_2 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan459_2 : scan edge (search edge 3) 4 [3271] 3271 innerTail459_2 best = best := by
  rw [innerTail459_2, scan, if_neg (by decide), innerCandidates459_2_checked]
  change scan edge (search edge 3) 4 [3271] 3271 innerTail459_3
    (if 3 ≤ innerCandidates459_2.length then search edge 3 [3271, 4243] innerCandidates459_2 7514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan459_3

def innerTail459_1 : List Nat := 3361 :: innerTail459_2
def innerCandidates459_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates459_1_checked : candidates edge 4 3271 3361 innerTail459_2 best.bound = innerCandidates459_1 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan459_1 : scan edge (search edge 3) 4 [3271] 3271 innerTail459_1 best = best := by
  rw [innerTail459_1, scan, if_neg (by decide), innerCandidates459_1_checked]
  change scan edge (search edge 3) 4 [3271] 3271 innerTail459_2
    (if 3 ≤ innerCandidates459_1.length then search edge 3 [3271, 3361] innerCandidates459_1 6632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan459_2

def innerTail459_0 : List Nat := 3319 :: innerTail459_1
def innerCandidates459_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates459_0_checked : candidates edge 4 3271 3319 innerTail459_1 best.bound = innerCandidates459_0 := by
  unfold candidates
  have row : edge 3319 = SearchData.row3319.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan459_0 : scan edge (search edge 3) 4 [3271] 3271 innerTail459_0 best = best := by
  rw [innerTail459_0, scan, if_neg (by decide), innerCandidates459_0_checked]
  change scan edge (search edge 3) 4 [3271] 3271 innerTail459_1
    (if 3 ≤ innerCandidates459_0.length then search edge 3 [3271, 3319] innerCandidates459_0 6590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan459_1

theorem rootBranch459 : search edge 4 [3271] rootCandidates459 3271 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3271] 3271 innerTail459_0 best = best
  exact innerScan459_0

def rootCandidates458 : List Nat := [3271, 4021, 4327, 4567, 5107, 5503, 5569, 6199, 6247, 7207, 7717, 7963]
set_option maxRecDepth 4096 in
theorem rootCandidates458_checked : candidates edge 5 0 3259 tail459 best.bound = rootCandidates458 := by
  unfold candidates
  have row : edge 3259 = SearchData.row3259.contains := by funext q; rfl
  rw [row]
  decide

def innerTail458_12 : List Nat := []
theorem innerScan458_12 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_12 best = best := rfl

def innerTail458_11 : List Nat := 7963 :: innerTail458_12
def innerCandidates458_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates458_11_checked : candidates edge 4 3259 7963 innerTail458_12 best.bound = innerCandidates458_11 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan458_11 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_11 best = best := by
  rw [innerTail458_11, scan, if_neg (by decide), innerCandidates458_11_checked]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_12
    (if 3 ≤ innerCandidates458_11.length then search edge 3 [3259, 7963] innerCandidates458_11 11222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan458_12

def innerTail458_10 : List Nat := 7717 :: innerTail458_11
def innerCandidates458_10 : List Nat := [7963]
set_option maxRecDepth 4096 in
theorem innerCandidates458_10_checked : candidates edge 4 3259 7717 innerTail458_11 best.bound = innerCandidates458_10 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan458_10 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_10 best = best := by
  rw [innerTail458_10, scan, if_neg (by decide), innerCandidates458_10_checked]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_11
    (if 3 ≤ innerCandidates458_10.length then search edge 3 [3259, 7717] innerCandidates458_10 10976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan458_11

def innerTail458_9 : List Nat := 7207 :: innerTail458_10
def innerCandidates458_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates458_9_checked : candidates edge 4 3259 7207 innerTail458_10 best.bound = innerCandidates458_9 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan458_9 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_9 best = best := by
  rw [innerTail458_9, scan, if_neg (by decide), innerCandidates458_9_checked]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_10
    (if 3 ≤ innerCandidates458_9.length then search edge 3 [3259, 7207] innerCandidates458_9 10466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan458_10

def innerTail458_8 : List Nat := 6247 :: innerTail458_9
def innerCandidates458_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates458_8_checked : candidates edge 4 3259 6247 innerTail458_9 best.bound = innerCandidates458_8 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan458_8 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_8 best = best := by
  rw [innerTail458_8, scan, if_neg (by decide), innerCandidates458_8_checked]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_9
    (if 3 ≤ innerCandidates458_8.length then search edge 3 [3259, 6247] innerCandidates458_8 9506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan458_9

def innerTail458_7 : List Nat := 6199 :: innerTail458_8
def innerCandidates458_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates458_7_checked : candidates edge 4 3259 6199 innerTail458_8 best.bound = innerCandidates458_7 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan458_7 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_7 best = best := by
  rw [innerTail458_7, scan, if_neg (by decide), innerCandidates458_7_checked]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_8
    (if 3 ≤ innerCandidates458_7.length then search edge 3 [3259, 6199] innerCandidates458_7 9458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan458_8

def innerTail458_6 : List Nat := 5569 :: innerTail458_7
def innerCandidates458_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates458_6_checked : candidates edge 4 3259 5569 innerTail458_7 best.bound = innerCandidates458_6 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan458_6 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_6 best = best := by
  rw [innerTail458_6, scan, if_neg (by decide), innerCandidates458_6_checked]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_7
    (if 3 ≤ innerCandidates458_6.length then search edge 3 [3259, 5569] innerCandidates458_6 8828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan458_7

def innerTail458_5 : List Nat := 5503 :: innerTail458_6
def innerCandidates458_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates458_5_checked : candidates edge 4 3259 5503 innerTail458_6 best.bound = innerCandidates458_5 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan458_5 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_5 best = best := by
  rw [innerTail458_5, scan, if_neg (by decide), innerCandidates458_5_checked]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_6
    (if 3 ≤ innerCandidates458_5.length then search edge 3 [3259, 5503] innerCandidates458_5 8762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan458_6

def innerTail458_4 : List Nat := 5107 :: innerTail458_5
def innerCandidates458_4 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates458_4_checked : candidates edge 4 3259 5107 innerTail458_5 best.bound = innerCandidates458_4 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan458_4 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_4 best = best := by
  rw [innerTail458_4, scan, if_neg (by decide), innerCandidates458_4_checked]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_5
    (if 3 ≤ innerCandidates458_4.length then search edge 3 [3259, 5107] innerCandidates458_4 8366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan458_5

def innerTail458_3 : List Nat := 4567 :: innerTail458_4
def innerCandidates458_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates458_3_checked : candidates edge 4 3259 4567 innerTail458_4 best.bound = innerCandidates458_3 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan458_3 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_3 best = best := by
  rw [innerTail458_3, scan, if_neg (by decide), innerCandidates458_3_checked]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_4
    (if 3 ≤ innerCandidates458_3.length then search edge 3 [3259, 4567] innerCandidates458_3 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan458_4

def innerTail458_2 : List Nat := 4327 :: innerTail458_3
def innerCandidates458_2 : List Nat := [5107, 5503, 7963]
set_option maxRecDepth 4096 in
theorem innerCandidates458_2_checked : candidates edge 4 3259 4327 innerTail458_3 best.bound = innerCandidates458_2 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild458_2 : search edge 3 [3259, 4327] innerCandidates458_2 7586 best = best := by decide

theorem innerScan458_2 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_2 best = best := by
  rw [innerTail458_2, scan, if_neg (by decide), innerCandidates458_2_checked]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_3
    (if 3 ≤ innerCandidates458_2.length then search edge 3 [3259, 4327] innerCandidates458_2 7586 best else best) = best
  rw [if_pos (by decide), innerChild458_2]
  exact innerScan458_3

def innerTail458_1 : List Nat := 4021 :: innerTail458_2
def innerCandidates458_1 : List Nat := [7207]
set_option maxRecDepth 4096 in
theorem innerCandidates458_1_checked : candidates edge 4 3259 4021 innerTail458_2 best.bound = innerCandidates458_1 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan458_1 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_1 best = best := by
  rw [innerTail458_1, scan, if_neg (by decide), innerCandidates458_1_checked]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_2
    (if 3 ≤ innerCandidates458_1.length then search edge 3 [3259, 4021] innerCandidates458_1 7280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan458_2

def innerTail458_0 : List Nat := 3271 :: innerTail458_1
def innerCandidates458_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates458_0_checked : candidates edge 4 3259 3271 innerTail458_1 best.bound = innerCandidates458_0 := by
  unfold candidates
  have row : edge 3271 = SearchData.row3271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan458_0 : scan edge (search edge 3) 4 [3259] 3259 innerTail458_0 best = best := by
  rw [innerTail458_0, scan, if_neg (by decide), innerCandidates458_0_checked]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_1
    (if 3 ≤ innerCandidates458_0.length then search edge 3 [3259, 3271] innerCandidates458_0 6530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan458_1

theorem rootBranch458 : search edge 4 [3259] rootCandidates458 3259 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3259] 3259 innerTail458_0 best = best
  exact innerScan458_0

def rootCandidates457 : List Nat := [3407, 3449, 4547, 4649, 4937, 5051, 5441, 5471, 5507, 6053, 6221, 6491, 6917, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates457_checked : candidates edge 5 0 3257 tail458 best.bound = rootCandidates457 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext q; rfl
  rw [row]
  decide

def innerTail457_14 : List Nat := []
theorem innerScan457_14 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_14 best = best := rfl

def innerTail457_13 : List Nat := 7949 :: innerTail457_14
def innerCandidates457_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates457_13_checked : candidates edge 4 3257 7949 innerTail457_14 best.bound = innerCandidates457_13 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_13 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_13 best = best := by
  rw [innerTail457_13, scan, if_neg (by decide), innerCandidates457_13_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_14
    (if 3 ≤ innerCandidates457_13.length then search edge 3 [3257, 7949] innerCandidates457_13 11206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_14

def innerTail457_12 : List Nat := 6917 :: innerTail457_13
def innerCandidates457_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates457_12_checked : candidates edge 4 3257 6917 innerTail457_13 best.bound = innerCandidates457_12 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_12 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_12 best = best := by
  rw [innerTail457_12, scan, if_neg (by decide), innerCandidates457_12_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_13
    (if 3 ≤ innerCandidates457_12.length then search edge 3 [3257, 6917] innerCandidates457_12 10174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_13

def innerTail457_11 : List Nat := 6491 :: innerTail457_12
def innerCandidates457_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates457_11_checked : candidates edge 4 3257 6491 innerTail457_12 best.bound = innerCandidates457_11 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_11 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_11 best = best := by
  rw [innerTail457_11, scan, if_neg (by decide), innerCandidates457_11_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_12
    (if 3 ≤ innerCandidates457_11.length then search edge 3 [3257, 6491] innerCandidates457_11 9748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_12

def innerTail457_10 : List Nat := 6221 :: innerTail457_11
def innerCandidates457_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates457_10_checked : candidates edge 4 3257 6221 innerTail457_11 best.bound = innerCandidates457_10 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_10 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_10 best = best := by
  rw [innerTail457_10, scan, if_neg (by decide), innerCandidates457_10_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_11
    (if 3 ≤ innerCandidates457_10.length then search edge 3 [3257, 6221] innerCandidates457_10 9478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_11

def innerTail457_9 : List Nat := 6053 :: innerTail457_10
def innerCandidates457_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates457_9_checked : candidates edge 4 3257 6053 innerTail457_10 best.bound = innerCandidates457_9 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_9 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_9 best = best := by
  rw [innerTail457_9, scan, if_neg (by decide), innerCandidates457_9_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_10
    (if 3 ≤ innerCandidates457_9.length then search edge 3 [3257, 6053] innerCandidates457_9 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_10

def innerTail457_8 : List Nat := 5507 :: innerTail457_9
def innerCandidates457_8 : List Nat := [6221]
set_option maxRecDepth 4096 in
theorem innerCandidates457_8_checked : candidates edge 4 3257 5507 innerTail457_9 best.bound = innerCandidates457_8 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_8 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_8 best = best := by
  rw [innerTail457_8, scan, if_neg (by decide), innerCandidates457_8_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_9
    (if 3 ≤ innerCandidates457_8.length then search edge 3 [3257, 5507] innerCandidates457_8 8764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_9

def innerTail457_7 : List Nat := 5471 :: innerTail457_8
def innerCandidates457_7 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates457_7_checked : candidates edge 4 3257 5471 innerTail457_8 best.bound = innerCandidates457_7 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_7 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_7 best = best := by
  rw [innerTail457_7, scan, if_neg (by decide), innerCandidates457_7_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_8
    (if 3 ≤ innerCandidates457_7.length then search edge 3 [3257, 5471] innerCandidates457_7 8728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_8

def innerTail457_6 : List Nat := 5441 :: innerTail457_7
def innerCandidates457_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates457_6_checked : candidates edge 4 3257 5441 innerTail457_7 best.bound = innerCandidates457_6 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_6 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_6 best = best := by
  rw [innerTail457_6, scan, if_neg (by decide), innerCandidates457_6_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_7
    (if 3 ≤ innerCandidates457_6.length then search edge 3 [3257, 5441] innerCandidates457_6 8698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_7

def innerTail457_5 : List Nat := 5051 :: innerTail457_6
def innerCandidates457_5 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates457_5_checked : candidates edge 4 3257 5051 innerTail457_6 best.bound = innerCandidates457_5 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_5 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_5 best = best := by
  rw [innerTail457_5, scan, if_neg (by decide), innerCandidates457_5_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_6
    (if 3 ≤ innerCandidates457_5.length then search edge 3 [3257, 5051] innerCandidates457_5 8308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_6

def innerTail457_4 : List Nat := 4937 :: innerTail457_5
def innerCandidates457_4 : List Nat := [6053, 6491]
set_option maxRecDepth 4096 in
theorem innerCandidates457_4_checked : candidates edge 4 3257 4937 innerTail457_5 best.bound = innerCandidates457_4 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_4 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_4 best = best := by
  rw [innerTail457_4, scan, if_neg (by decide), innerCandidates457_4_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_5
    (if 3 ≤ innerCandidates457_4.length then search edge 3 [3257, 4937] innerCandidates457_4 8194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_5

def innerTail457_3 : List Nat := 4649 :: innerTail457_4
def innerCandidates457_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates457_3_checked : candidates edge 4 3257 4649 innerTail457_4 best.bound = innerCandidates457_3 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_3 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_3 best = best := by
  rw [innerTail457_3, scan, if_neg (by decide), innerCandidates457_3_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_4
    (if 3 ≤ innerCandidates457_3.length then search edge 3 [3257, 4649] innerCandidates457_3 7906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_4

def innerTail457_2 : List Nat := 4547 :: innerTail457_3
def innerCandidates457_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates457_2_checked : candidates edge 4 3257 4547 innerTail457_3 best.bound = innerCandidates457_2 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_2 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_2 best = best := by
  rw [innerTail457_2, scan, if_neg (by decide), innerCandidates457_2_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_3
    (if 3 ≤ innerCandidates457_2.length then search edge 3 [3257, 4547] innerCandidates457_2 7804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_3

def innerTail457_1 : List Nat := 3449 :: innerTail457_2
def innerCandidates457_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates457_1_checked : candidates edge 4 3257 3449 innerTail457_2 best.bound = innerCandidates457_1 := by
  unfold candidates
  have row : edge 3449 = SearchData.row3449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_1 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_1 best = best := by
  rw [innerTail457_1, scan, if_neg (by decide), innerCandidates457_1_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_2
    (if 3 ≤ innerCandidates457_1.length then search edge 3 [3257, 3449] innerCandidates457_1 6706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_2

def innerTail457_0 : List Nat := 3407 :: innerTail457_1
def innerCandidates457_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates457_0_checked : candidates edge 4 3257 3407 innerTail457_1 best.bound = innerCandidates457_0 := by
  unfold candidates
  have row : edge 3407 = SearchData.row3407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan457_0 : scan edge (search edge 3) 4 [3257] 3257 innerTail457_0 best = best := by
  rw [innerTail457_0, scan, if_neg (by decide), innerCandidates457_0_checked]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_1
    (if 3 ≤ innerCandidates457_0.length then search edge 3 [3257, 3407] innerCandidates457_0 6664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan457_1

theorem rootBranch457 : search edge 4 [3257] rootCandidates457 3257 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3257] 3257 innerTail457_0 best = best
  exact innerScan457_0

end ProjectEulerSolutions.PrimePairUpper.Search
