import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates736 : List Nat := [5821, 6421, 6547, 6907, 6949, 7639, 7681, 8311, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates736_checked : candidates edge 5 0 5623 tail737 best.bound = rootCandidates736 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext q; rfl
  rw [row]
  decide

def innerTail736_9 : List Nat := []
theorem innerScan736_9 : scan edge (search edge 3) 4 [5623] 5623 innerTail736_9 best = best := rfl

def innerTail736_8 : List Nat := 8317 :: innerTail736_9
def innerCandidates736_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates736_8_checked : candidates edge 4 5623 8317 innerTail736_9 best.bound = innerCandidates736_8 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan736_8 : scan edge (search edge 3) 4 [5623] 5623 innerTail736_8 best = best := by
  rw [innerTail736_8, scan, if_neg (by decide), innerCandidates736_8_checked]
  change scan edge (search edge 3) 4 [5623] 5623 innerTail736_9
    (if 3 ≤ innerCandidates736_8.length then search edge 3 [5623, 8317] innerCandidates736_8 13940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan736_9

def innerTail736_7 : List Nat := 8311 :: innerTail736_8
def innerCandidates736_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates736_7_checked : candidates edge 4 5623 8311 innerTail736_8 best.bound = innerCandidates736_7 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan736_7 : scan edge (search edge 3) 4 [5623] 5623 innerTail736_7 best = best := by
  rw [innerTail736_7, scan, if_neg (by decide), innerCandidates736_7_checked]
  change scan edge (search edge 3) 4 [5623] 5623 innerTail736_8
    (if 3 ≤ innerCandidates736_7.length then search edge 3 [5623, 8311] innerCandidates736_7 13934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan736_8

def innerTail736_6 : List Nat := 7681 :: innerTail736_7
def innerCandidates736_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates736_6_checked : candidates edge 4 5623 7681 innerTail736_7 best.bound = innerCandidates736_6 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan736_6 : scan edge (search edge 3) 4 [5623] 5623 innerTail736_6 best = best := by
  rw [innerTail736_6, scan, if_neg (by decide), innerCandidates736_6_checked]
  change scan edge (search edge 3) 4 [5623] 5623 innerTail736_7
    (if 3 ≤ innerCandidates736_6.length then search edge 3 [5623, 7681] innerCandidates736_6 13304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan736_7

def innerTail736_5 : List Nat := 7639 :: innerTail736_6
def innerCandidates736_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates736_5_checked : candidates edge 4 5623 7639 innerTail736_6 best.bound = innerCandidates736_5 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan736_5 : scan edge (search edge 3) 4 [5623] 5623 innerTail736_5 best = best := by
  rw [innerTail736_5, scan, if_neg (by decide), innerCandidates736_5_checked]
  change scan edge (search edge 3) 4 [5623] 5623 innerTail736_6
    (if 3 ≤ innerCandidates736_5.length then search edge 3 [5623, 7639] innerCandidates736_5 13262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan736_6

def innerTail736_4 : List Nat := 6949 :: innerTail736_5
def innerCandidates736_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates736_4_checked : candidates edge 4 5623 6949 innerTail736_5 best.bound = innerCandidates736_4 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan736_4 : scan edge (search edge 3) 4 [5623] 5623 innerTail736_4 best = best := by
  rw [innerTail736_4, scan, if_neg (by decide), innerCandidates736_4_checked]
  change scan edge (search edge 3) 4 [5623] 5623 innerTail736_5
    (if 3 ≤ innerCandidates736_4.length then search edge 3 [5623, 6949] innerCandidates736_4 12572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan736_5

def innerTail736_3 : List Nat := 6907 :: innerTail736_4
def innerCandidates736_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates736_3_checked : candidates edge 4 5623 6907 innerTail736_4 best.bound = innerCandidates736_3 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan736_3 : scan edge (search edge 3) 4 [5623] 5623 innerTail736_3 best = best := by
  rw [innerTail736_3, scan, if_neg (by decide), innerCandidates736_3_checked]
  change scan edge (search edge 3) 4 [5623] 5623 innerTail736_4
    (if 3 ≤ innerCandidates736_3.length then search edge 3 [5623, 6907] innerCandidates736_3 12530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan736_4

def innerTail736_2 : List Nat := 6547 :: innerTail736_3
def innerCandidates736_2 : List Nat := [7681]
set_option maxRecDepth 4096 in
theorem innerCandidates736_2_checked : candidates edge 4 5623 6547 innerTail736_3 best.bound = innerCandidates736_2 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan736_2 : scan edge (search edge 3) 4 [5623] 5623 innerTail736_2 best = best := by
  rw [innerTail736_2, scan, if_neg (by decide), innerCandidates736_2_checked]
  change scan edge (search edge 3) 4 [5623] 5623 innerTail736_3
    (if 3 ≤ innerCandidates736_2.length then search edge 3 [5623, 6547] innerCandidates736_2 12170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan736_3

def innerTail736_1 : List Nat := 6421 :: innerTail736_2
def innerCandidates736_1 : List Nat := [6949]
set_option maxRecDepth 4096 in
theorem innerCandidates736_1_checked : candidates edge 4 5623 6421 innerTail736_2 best.bound = innerCandidates736_1 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan736_1 : scan edge (search edge 3) 4 [5623] 5623 innerTail736_1 best = best := by
  rw [innerTail736_1, scan, if_neg (by decide), innerCandidates736_1_checked]
  change scan edge (search edge 3) 4 [5623] 5623 innerTail736_2
    (if 3 ≤ innerCandidates736_1.length then search edge 3 [5623, 6421] innerCandidates736_1 12044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan736_2

def innerTail736_0 : List Nat := 5821 :: innerTail736_1
def innerCandidates736_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates736_0_checked : candidates edge 4 5623 5821 innerTail736_1 best.bound = innerCandidates736_0 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan736_0 : scan edge (search edge 3) 4 [5623] 5623 innerTail736_0 best = best := by
  rw [innerTail736_0, scan, if_neg (by decide), innerCandidates736_0_checked]
  change scan edge (search edge 3) 4 [5623] 5623 innerTail736_1
    (if 3 ≤ innerCandidates736_0.length then search edge 3 [5623, 5821] innerCandidates736_0 11444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan736_1

theorem rootBranch736 : search edge 4 [5623] rootCandidates736 5623 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5623] 5623 innerTail736_0 best = best
  exact innerScan736_0

def rootCandidates735 : List Nat := [6173, 6701, 6737]
set_option maxRecDepth 4096 in
theorem rootCandidates735_checked : candidates edge 5 0 5591 tail736 best.bound = rootCandidates735 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates734 : List Nat := [5653, 5737, 5839, 6067, 6991, 7351, 7561, 8017, 8101]
set_option maxRecDepth 4096 in
theorem rootCandidates734_checked : candidates edge 5 0 5581 tail735 best.bound = rootCandidates734 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext q; rfl
  rw [row]
  decide

def innerTail734_9 : List Nat := []
theorem innerScan734_9 : scan edge (search edge 3) 4 [5581] 5581 innerTail734_9 best = best := rfl

def innerTail734_8 : List Nat := 8101 :: innerTail734_9
def innerCandidates734_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates734_8_checked : candidates edge 4 5581 8101 innerTail734_9 best.bound = innerCandidates734_8 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan734_8 : scan edge (search edge 3) 4 [5581] 5581 innerTail734_8 best = best := by
  rw [innerTail734_8, scan, if_neg (by decide), innerCandidates734_8_checked]
  change scan edge (search edge 3) 4 [5581] 5581 innerTail734_9
    (if 3 ≤ innerCandidates734_8.length then search edge 3 [5581, 8101] innerCandidates734_8 13682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan734_9

def innerTail734_7 : List Nat := 8017 :: innerTail734_8
def innerCandidates734_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates734_7_checked : candidates edge 4 5581 8017 innerTail734_8 best.bound = innerCandidates734_7 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan734_7 : scan edge (search edge 3) 4 [5581] 5581 innerTail734_7 best = best := by
  rw [innerTail734_7, scan, if_neg (by decide), innerCandidates734_7_checked]
  change scan edge (search edge 3) 4 [5581] 5581 innerTail734_8
    (if 3 ≤ innerCandidates734_7.length then search edge 3 [5581, 8017] innerCandidates734_7 13598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan734_8

def innerTail734_6 : List Nat := 7561 :: innerTail734_7
def innerCandidates734_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates734_6_checked : candidates edge 4 5581 7561 innerTail734_7 best.bound = innerCandidates734_6 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan734_6 : scan edge (search edge 3) 4 [5581] 5581 innerTail734_6 best = best := by
  rw [innerTail734_6, scan, if_neg (by decide), innerCandidates734_6_checked]
  change scan edge (search edge 3) 4 [5581] 5581 innerTail734_7
    (if 3 ≤ innerCandidates734_6.length then search edge 3 [5581, 7561] innerCandidates734_6 13142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan734_7

def innerTail734_5 : List Nat := 7351 :: innerTail734_6
def innerCandidates734_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates734_5_checked : candidates edge 4 5581 7351 innerTail734_6 best.bound = innerCandidates734_5 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan734_5 : scan edge (search edge 3) 4 [5581] 5581 innerTail734_5 best = best := by
  rw [innerTail734_5, scan, if_neg (by decide), innerCandidates734_5_checked]
  change scan edge (search edge 3) 4 [5581] 5581 innerTail734_6
    (if 3 ≤ innerCandidates734_5.length then search edge 3 [5581, 7351] innerCandidates734_5 12932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan734_6

def innerTail734_4 : List Nat := 6991 :: innerTail734_5
def innerCandidates734_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates734_4_checked : candidates edge 4 5581 6991 innerTail734_5 best.bound = innerCandidates734_4 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan734_4 : scan edge (search edge 3) 4 [5581] 5581 innerTail734_4 best = best := by
  rw [innerTail734_4, scan, if_neg (by decide), innerCandidates734_4_checked]
  change scan edge (search edge 3) 4 [5581] 5581 innerTail734_5
    (if 3 ≤ innerCandidates734_4.length then search edge 3 [5581, 6991] innerCandidates734_4 12572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan734_5

def innerTail734_3 : List Nat := 6067 :: innerTail734_4
def innerCandidates734_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates734_3_checked : candidates edge 4 5581 6067 innerTail734_4 best.bound = innerCandidates734_3 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan734_3 : scan edge (search edge 3) 4 [5581] 5581 innerTail734_3 best = best := by
  rw [innerTail734_3, scan, if_neg (by decide), innerCandidates734_3_checked]
  change scan edge (search edge 3) 4 [5581] 5581 innerTail734_4
    (if 3 ≤ innerCandidates734_3.length then search edge 3 [5581, 6067] innerCandidates734_3 11648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan734_4

def innerTail734_2 : List Nat := 5839 :: innerTail734_3
def innerCandidates734_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates734_2_checked : candidates edge 4 5581 5839 innerTail734_3 best.bound = innerCandidates734_2 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan734_2 : scan edge (search edge 3) 4 [5581] 5581 innerTail734_2 best = best := by
  rw [innerTail734_2, scan, if_neg (by decide), innerCandidates734_2_checked]
  change scan edge (search edge 3) 4 [5581] 5581 innerTail734_3
    (if 3 ≤ innerCandidates734_2.length then search edge 3 [5581, 5839] innerCandidates734_2 11420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan734_3

def innerTail734_1 : List Nat := 5737 :: innerTail734_2
def innerCandidates734_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates734_1_checked : candidates edge 4 5581 5737 innerTail734_2 best.bound = innerCandidates734_1 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan734_1 : scan edge (search edge 3) 4 [5581] 5581 innerTail734_1 best = best := by
  rw [innerTail734_1, scan, if_neg (by decide), innerCandidates734_1_checked]
  change scan edge (search edge 3) 4 [5581] 5581 innerTail734_2
    (if 3 ≤ innerCandidates734_1.length then search edge 3 [5581, 5737] innerCandidates734_1 11318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan734_2

def innerTail734_0 : List Nat := 5653 :: innerTail734_1
def innerCandidates734_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates734_0_checked : candidates edge 4 5581 5653 innerTail734_1 best.bound = innerCandidates734_0 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan734_0 : scan edge (search edge 3) 4 [5581] 5581 innerTail734_0 best = best := by
  rw [innerTail734_0, scan, if_neg (by decide), innerCandidates734_0_checked]
  change scan edge (search edge 3) 4 [5581] 5581 innerTail734_1
    (if 3 ≤ innerCandidates734_0.length then search edge 3 [5581, 5653] innerCandidates734_0 11234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan734_1

theorem rootBranch734 : search edge 4 [5581] rootCandidates734 5581 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5581] 5581 innerTail734_0 best = best
  exact innerScan734_0

def rootCandidates733 : List Nat := [6833, 7457, 7937, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates733_checked : candidates edge 5 0 5573 tail734 best.bound = rootCandidates733 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext q; rfl
  rw [row]
  decide

def innerTail733_4 : List Nat := []
theorem innerScan733_4 : scan edge (search edge 3) 4 [5573] 5573 innerTail733_4 best = best := rfl

def innerTail733_3 : List Nat := 8219 :: innerTail733_4
def innerCandidates733_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates733_3_checked : candidates edge 4 5573 8219 innerTail733_4 best.bound = innerCandidates733_3 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan733_3 : scan edge (search edge 3) 4 [5573] 5573 innerTail733_3 best = best := by
  rw [innerTail733_3, scan, if_neg (by decide), innerCandidates733_3_checked]
  change scan edge (search edge 3) 4 [5573] 5573 innerTail733_4
    (if 3 ≤ innerCandidates733_3.length then search edge 3 [5573, 8219] innerCandidates733_3 13792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan733_4

def innerTail733_2 : List Nat := 7937 :: innerTail733_3
def innerCandidates733_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates733_2_checked : candidates edge 4 5573 7937 innerTail733_3 best.bound = innerCandidates733_2 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan733_2 : scan edge (search edge 3) 4 [5573] 5573 innerTail733_2 best = best := by
  rw [innerTail733_2, scan, if_neg (by decide), innerCandidates733_2_checked]
  change scan edge (search edge 3) 4 [5573] 5573 innerTail733_3
    (if 3 ≤ innerCandidates733_2.length then search edge 3 [5573, 7937] innerCandidates733_2 13510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan733_3

def innerTail733_1 : List Nat := 7457 :: innerTail733_2
def innerCandidates733_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates733_1_checked : candidates edge 4 5573 7457 innerTail733_2 best.bound = innerCandidates733_1 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan733_1 : scan edge (search edge 3) 4 [5573] 5573 innerTail733_1 best = best := by
  rw [innerTail733_1, scan, if_neg (by decide), innerCandidates733_1_checked]
  change scan edge (search edge 3) 4 [5573] 5573 innerTail733_2
    (if 3 ≤ innerCandidates733_1.length then search edge 3 [5573, 7457] innerCandidates733_1 13030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan733_2

def innerTail733_0 : List Nat := 6833 :: innerTail733_1
def innerCandidates733_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates733_0_checked : candidates edge 4 5573 6833 innerTail733_1 best.bound = innerCandidates733_0 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan733_0 : scan edge (search edge 3) 4 [5573] 5573 innerTail733_0 best = best := by
  rw [innerTail733_0, scan, if_neg (by decide), innerCandidates733_0_checked]
  change scan edge (search edge 3) 4 [5573] 5573 innerTail733_1
    (if 3 ≤ innerCandidates733_0.length then search edge 3 [5573, 6833] innerCandidates733_0 12406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan733_1

theorem rootBranch733 : search edge 4 [5573] rootCandidates733 5573 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5573] 5573 innerTail733_0 best = best
  exact innerScan733_0

def rootCandidates732 : List Nat := [5923, 6373, 6451, 6469, 6577, 7309, 7687]
set_option maxRecDepth 4096 in
theorem rootCandidates732_checked : candidates edge 5 0 5569 tail733 best.bound = rootCandidates732 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext q; rfl
  rw [row]
  decide

def innerTail732_7 : List Nat := []
theorem innerScan732_7 : scan edge (search edge 3) 4 [5569] 5569 innerTail732_7 best = best := rfl

def innerTail732_6 : List Nat := 7687 :: innerTail732_7
def innerCandidates732_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates732_6_checked : candidates edge 4 5569 7687 innerTail732_7 best.bound = innerCandidates732_6 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan732_6 : scan edge (search edge 3) 4 [5569] 5569 innerTail732_6 best = best := by
  rw [innerTail732_6, scan, if_neg (by decide), innerCandidates732_6_checked]
  change scan edge (search edge 3) 4 [5569] 5569 innerTail732_7
    (if 3 ≤ innerCandidates732_6.length then search edge 3 [5569, 7687] innerCandidates732_6 13256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan732_7

def innerTail732_5 : List Nat := 7309 :: innerTail732_6
def innerCandidates732_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates732_5_checked : candidates edge 4 5569 7309 innerTail732_6 best.bound = innerCandidates732_5 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan732_5 : scan edge (search edge 3) 4 [5569] 5569 innerTail732_5 best = best := by
  rw [innerTail732_5, scan, if_neg (by decide), innerCandidates732_5_checked]
  change scan edge (search edge 3) 4 [5569] 5569 innerTail732_6
    (if 3 ≤ innerCandidates732_5.length then search edge 3 [5569, 7309] innerCandidates732_5 12878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan732_6

def innerTail732_4 : List Nat := 6577 :: innerTail732_5
def innerCandidates732_4 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates732_4_checked : candidates edge 4 5569 6577 innerTail732_5 best.bound = innerCandidates732_4 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan732_4 : scan edge (search edge 3) 4 [5569] 5569 innerTail732_4 best = best := by
  rw [innerTail732_4, scan, if_neg (by decide), innerCandidates732_4_checked]
  change scan edge (search edge 3) 4 [5569] 5569 innerTail732_5
    (if 3 ≤ innerCandidates732_4.length then search edge 3 [5569, 6577] innerCandidates732_4 12146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan732_5

def innerTail732_3 : List Nat := 6469 :: innerTail732_4
def innerCandidates732_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates732_3_checked : candidates edge 4 5569 6469 innerTail732_4 best.bound = innerCandidates732_3 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan732_3 : scan edge (search edge 3) 4 [5569] 5569 innerTail732_3 best = best := by
  rw [innerTail732_3, scan, if_neg (by decide), innerCandidates732_3_checked]
  change scan edge (search edge 3) 4 [5569] 5569 innerTail732_4
    (if 3 ≤ innerCandidates732_3.length then search edge 3 [5569, 6469] innerCandidates732_3 12038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan732_4

def innerTail732_2 : List Nat := 6451 :: innerTail732_3
def innerCandidates732_2 : List Nat := [7309]
set_option maxRecDepth 4096 in
theorem innerCandidates732_2_checked : candidates edge 4 5569 6451 innerTail732_3 best.bound = innerCandidates732_2 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan732_2 : scan edge (search edge 3) 4 [5569] 5569 innerTail732_2 best = best := by
  rw [innerTail732_2, scan, if_neg (by decide), innerCandidates732_2_checked]
  change scan edge (search edge 3) 4 [5569] 5569 innerTail732_3
    (if 3 ≤ innerCandidates732_2.length then search edge 3 [5569, 6451] innerCandidates732_2 12020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan732_3

def innerTail732_1 : List Nat := 6373 :: innerTail732_2
def innerCandidates732_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates732_1_checked : candidates edge 4 5569 6373 innerTail732_2 best.bound = innerCandidates732_1 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan732_1 : scan edge (search edge 3) 4 [5569] 5569 innerTail732_1 best = best := by
  rw [innerTail732_1, scan, if_neg (by decide), innerCandidates732_1_checked]
  change scan edge (search edge 3) 4 [5569] 5569 innerTail732_2
    (if 3 ≤ innerCandidates732_1.length then search edge 3 [5569, 6373] innerCandidates732_1 11942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan732_2

def innerTail732_0 : List Nat := 5923 :: innerTail732_1
def innerCandidates732_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates732_0_checked : candidates edge 4 5569 5923 innerTail732_1 best.bound = innerCandidates732_0 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan732_0 : scan edge (search edge 3) 4 [5569] 5569 innerTail732_0 best = best := by
  rw [innerTail732_0, scan, if_neg (by decide), innerCandidates732_0_checked]
  change scan edge (search edge 3) 4 [5569] 5569 innerTail732_1
    (if 3 ≤ innerCandidates732_0.length then search edge 3 [5569, 5923] innerCandidates732_0 11492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan732_1

theorem rootBranch732 : search edge 4 [5569] rootCandidates732 5569 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5569] 5569 innerTail732_0 best = best
  exact innerScan732_0

def rootCandidates731 : List Nat := [5791, 6037, 6079, 6781, 7951, 8053]
set_option maxRecDepth 4096 in
theorem rootCandidates731_checked : candidates edge 5 0 5563 tail732 best.bound = rootCandidates731 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext q; rfl
  rw [row]
  decide

def innerTail731_6 : List Nat := []
theorem innerScan731_6 : scan edge (search edge 3) 4 [5563] 5563 innerTail731_6 best = best := rfl

def innerTail731_5 : List Nat := 8053 :: innerTail731_6
def innerCandidates731_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates731_5_checked : candidates edge 4 5563 8053 innerTail731_6 best.bound = innerCandidates731_5 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan731_5 : scan edge (search edge 3) 4 [5563] 5563 innerTail731_5 best = best := by
  rw [innerTail731_5, scan, if_neg (by decide), innerCandidates731_5_checked]
  change scan edge (search edge 3) 4 [5563] 5563 innerTail731_6
    (if 3 ≤ innerCandidates731_5.length then search edge 3 [5563, 8053] innerCandidates731_5 13616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan731_6

def innerTail731_4 : List Nat := 7951 :: innerTail731_5
def innerCandidates731_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates731_4_checked : candidates edge 4 5563 7951 innerTail731_5 best.bound = innerCandidates731_4 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan731_4 : scan edge (search edge 3) 4 [5563] 5563 innerTail731_4 best = best := by
  rw [innerTail731_4, scan, if_neg (by decide), innerCandidates731_4_checked]
  change scan edge (search edge 3) 4 [5563] 5563 innerTail731_5
    (if 3 ≤ innerCandidates731_4.length then search edge 3 [5563, 7951] innerCandidates731_4 13514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan731_5

def innerTail731_3 : List Nat := 6781 :: innerTail731_4
def innerCandidates731_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates731_3_checked : candidates edge 4 5563 6781 innerTail731_4 best.bound = innerCandidates731_3 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan731_3 : scan edge (search edge 3) 4 [5563] 5563 innerTail731_3 best = best := by
  rw [innerTail731_3, scan, if_neg (by decide), innerCandidates731_3_checked]
  change scan edge (search edge 3) 4 [5563] 5563 innerTail731_4
    (if 3 ≤ innerCandidates731_3.length then search edge 3 [5563, 6781] innerCandidates731_3 12344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan731_4

def innerTail731_2 : List Nat := 6079 :: innerTail731_3
def innerCandidates731_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates731_2_checked : candidates edge 4 5563 6079 innerTail731_3 best.bound = innerCandidates731_2 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan731_2 : scan edge (search edge 3) 4 [5563] 5563 innerTail731_2 best = best := by
  rw [innerTail731_2, scan, if_neg (by decide), innerCandidates731_2_checked]
  change scan edge (search edge 3) 4 [5563] 5563 innerTail731_3
    (if 3 ≤ innerCandidates731_2.length then search edge 3 [5563, 6079] innerCandidates731_2 11642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan731_3

def innerTail731_1 : List Nat := 6037 :: innerTail731_2
def innerCandidates731_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates731_1_checked : candidates edge 4 5563 6037 innerTail731_2 best.bound = innerCandidates731_1 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan731_1 : scan edge (search edge 3) 4 [5563] 5563 innerTail731_1 best = best := by
  rw [innerTail731_1, scan, if_neg (by decide), innerCandidates731_1_checked]
  change scan edge (search edge 3) 4 [5563] 5563 innerTail731_2
    (if 3 ≤ innerCandidates731_1.length then search edge 3 [5563, 6037] innerCandidates731_1 11600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan731_2

def innerTail731_0 : List Nat := 5791 :: innerTail731_1
def innerCandidates731_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates731_0_checked : candidates edge 4 5563 5791 innerTail731_1 best.bound = innerCandidates731_0 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan731_0 : scan edge (search edge 3) 4 [5563] 5563 innerTail731_0 best = best := by
  rw [innerTail731_0, scan, if_neg (by decide), innerCandidates731_0_checked]
  change scan edge (search edge 3) 4 [5563] 5563 innerTail731_1
    (if 3 ≤ innerCandidates731_0.length then search edge 3 [5563, 5791] innerCandidates731_0 11354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan731_1

theorem rootBranch731 : search edge 4 [5563] rootCandidates731 5563 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5563] 5563 innerTail731_0 best = best
  exact innerScan731_0

def rootCandidates730 : List Nat := [6067, 6529, 7393, 7507, 7537, 7789, 8053, 8167, 8233, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates730_checked : candidates edge 5 0 5557 tail731 best.bound = rootCandidates730 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext q; rfl
  rw [row]
  decide

def innerTail730_10 : List Nat := []
theorem innerScan730_10 : scan edge (search edge 3) 4 [5557] 5557 innerTail730_10 best = best := rfl

def innerTail730_9 : List Nat := 8293 :: innerTail730_10
def innerCandidates730_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates730_9_checked : candidates edge 4 5557 8293 innerTail730_10 best.bound = innerCandidates730_9 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan730_9 : scan edge (search edge 3) 4 [5557] 5557 innerTail730_9 best = best := by
  rw [innerTail730_9, scan, if_neg (by decide), innerCandidates730_9_checked]
  change scan edge (search edge 3) 4 [5557] 5557 innerTail730_10
    (if 3 ≤ innerCandidates730_9.length then search edge 3 [5557, 8293] innerCandidates730_9 13850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan730_10

def innerTail730_8 : List Nat := 8233 :: innerTail730_9
def innerCandidates730_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates730_8_checked : candidates edge 4 5557 8233 innerTail730_9 best.bound = innerCandidates730_8 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan730_8 : scan edge (search edge 3) 4 [5557] 5557 innerTail730_8 best = best := by
  rw [innerTail730_8, scan, if_neg (by decide), innerCandidates730_8_checked]
  change scan edge (search edge 3) 4 [5557] 5557 innerTail730_9
    (if 3 ≤ innerCandidates730_8.length then search edge 3 [5557, 8233] innerCandidates730_8 13790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan730_9

def innerTail730_7 : List Nat := 8167 :: innerTail730_8
def innerCandidates730_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates730_7_checked : candidates edge 4 5557 8167 innerTail730_8 best.bound = innerCandidates730_7 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan730_7 : scan edge (search edge 3) 4 [5557] 5557 innerTail730_7 best = best := by
  rw [innerTail730_7, scan, if_neg (by decide), innerCandidates730_7_checked]
  change scan edge (search edge 3) 4 [5557] 5557 innerTail730_8
    (if 3 ≤ innerCandidates730_7.length then search edge 3 [5557, 8167] innerCandidates730_7 13724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan730_8

def innerTail730_6 : List Nat := 8053 :: innerTail730_7
def innerCandidates730_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates730_6_checked : candidates edge 4 5557 8053 innerTail730_7 best.bound = innerCandidates730_6 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan730_6 : scan edge (search edge 3) 4 [5557] 5557 innerTail730_6 best = best := by
  rw [innerTail730_6, scan, if_neg (by decide), innerCandidates730_6_checked]
  change scan edge (search edge 3) 4 [5557] 5557 innerTail730_7
    (if 3 ≤ innerCandidates730_6.length then search edge 3 [5557, 8053] innerCandidates730_6 13610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan730_7

def innerTail730_5 : List Nat := 7789 :: innerTail730_6
def innerCandidates730_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates730_5_checked : candidates edge 4 5557 7789 innerTail730_6 best.bound = innerCandidates730_5 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan730_5 : scan edge (search edge 3) 4 [5557] 5557 innerTail730_5 best = best := by
  rw [innerTail730_5, scan, if_neg (by decide), innerCandidates730_5_checked]
  change scan edge (search edge 3) 4 [5557] 5557 innerTail730_6
    (if 3 ≤ innerCandidates730_5.length then search edge 3 [5557, 7789] innerCandidates730_5 13346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan730_6

def innerTail730_4 : List Nat := 7537 :: innerTail730_5
def innerCandidates730_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates730_4_checked : candidates edge 4 5557 7537 innerTail730_5 best.bound = innerCandidates730_4 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan730_4 : scan edge (search edge 3) 4 [5557] 5557 innerTail730_4 best = best := by
  rw [innerTail730_4, scan, if_neg (by decide), innerCandidates730_4_checked]
  change scan edge (search edge 3) 4 [5557] 5557 innerTail730_5
    (if 3 ≤ innerCandidates730_4.length then search edge 3 [5557, 7537] innerCandidates730_4 13094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan730_5

def innerTail730_3 : List Nat := 7507 :: innerTail730_4
def innerCandidates730_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates730_3_checked : candidates edge 4 5557 7507 innerTail730_4 best.bound = innerCandidates730_3 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan730_3 : scan edge (search edge 3) 4 [5557] 5557 innerTail730_3 best = best := by
  rw [innerTail730_3, scan, if_neg (by decide), innerCandidates730_3_checked]
  change scan edge (search edge 3) 4 [5557] 5557 innerTail730_4
    (if 3 ≤ innerCandidates730_3.length then search edge 3 [5557, 7507] innerCandidates730_3 13064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan730_4

def innerTail730_2 : List Nat := 7393 :: innerTail730_3
def innerCandidates730_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates730_2_checked : candidates edge 4 5557 7393 innerTail730_3 best.bound = innerCandidates730_2 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan730_2 : scan edge (search edge 3) 4 [5557] 5557 innerTail730_2 best = best := by
  rw [innerTail730_2, scan, if_neg (by decide), innerCandidates730_2_checked]
  change scan edge (search edge 3) 4 [5557] 5557 innerTail730_3
    (if 3 ≤ innerCandidates730_2.length then search edge 3 [5557, 7393] innerCandidates730_2 12950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan730_3

def innerTail730_1 : List Nat := 6529 :: innerTail730_2
def innerCandidates730_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates730_1_checked : candidates edge 4 5557 6529 innerTail730_2 best.bound = innerCandidates730_1 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan730_1 : scan edge (search edge 3) 4 [5557] 5557 innerTail730_1 best = best := by
  rw [innerTail730_1, scan, if_neg (by decide), innerCandidates730_1_checked]
  change scan edge (search edge 3) 4 [5557] 5557 innerTail730_2
    (if 3 ≤ innerCandidates730_1.length then search edge 3 [5557, 6529] innerCandidates730_1 12086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan730_2

def innerTail730_0 : List Nat := 6067 :: innerTail730_1
def innerCandidates730_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates730_0_checked : candidates edge 4 5557 6067 innerTail730_1 best.bound = innerCandidates730_0 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan730_0 : scan edge (search edge 3) 4 [5557] 5557 innerTail730_0 best = best := by
  rw [innerTail730_0, scan, if_neg (by decide), innerCandidates730_0_checked]
  change scan edge (search edge 3) 4 [5557] 5557 innerTail730_1
    (if 3 ≤ innerCandidates730_0.length then search edge 3 [5557, 6067] innerCandidates730_0 11624 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan730_1

theorem rootBranch730 : search edge 4 [5557] rootCandidates730 5557 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5557] 5557 innerTail730_0 best = best
  exact innerScan730_0

def rootCandidates729 : List Nat := [6089, 7121, 7673, 7883, 8009, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates729_checked : candidates edge 5 0 5531 tail730 best.bound = rootCandidates729 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext q; rfl
  rw [row]
  decide

def innerTail729_6 : List Nat := []
theorem innerScan729_6 : scan edge (search edge 3) 4 [5531] 5531 innerTail729_6 best = best := rfl

def innerTail729_5 : List Nat := 8387 :: innerTail729_6
def innerCandidates729_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates729_5_checked : candidates edge 4 5531 8387 innerTail729_6 best.bound = innerCandidates729_5 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan729_5 : scan edge (search edge 3) 4 [5531] 5531 innerTail729_5 best = best := by
  rw [innerTail729_5, scan, if_neg (by decide), innerCandidates729_5_checked]
  change scan edge (search edge 3) 4 [5531] 5531 innerTail729_6
    (if 3 ≤ innerCandidates729_5.length then search edge 3 [5531, 8387] innerCandidates729_5 13918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan729_6

def innerTail729_4 : List Nat := 8009 :: innerTail729_5
def innerCandidates729_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates729_4_checked : candidates edge 4 5531 8009 innerTail729_5 best.bound = innerCandidates729_4 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan729_4 : scan edge (search edge 3) 4 [5531] 5531 innerTail729_4 best = best := by
  rw [innerTail729_4, scan, if_neg (by decide), innerCandidates729_4_checked]
  change scan edge (search edge 3) 4 [5531] 5531 innerTail729_5
    (if 3 ≤ innerCandidates729_4.length then search edge 3 [5531, 8009] innerCandidates729_4 13540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan729_5

def innerTail729_3 : List Nat := 7883 :: innerTail729_4
def innerCandidates729_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates729_3_checked : candidates edge 4 5531 7883 innerTail729_4 best.bound = innerCandidates729_3 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan729_3 : scan edge (search edge 3) 4 [5531] 5531 innerTail729_3 best = best := by
  rw [innerTail729_3, scan, if_neg (by decide), innerCandidates729_3_checked]
  change scan edge (search edge 3) 4 [5531] 5531 innerTail729_4
    (if 3 ≤ innerCandidates729_3.length then search edge 3 [5531, 7883] innerCandidates729_3 13414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan729_4

def innerTail729_2 : List Nat := 7673 :: innerTail729_3
def innerCandidates729_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates729_2_checked : candidates edge 4 5531 7673 innerTail729_3 best.bound = innerCandidates729_2 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan729_2 : scan edge (search edge 3) 4 [5531] 5531 innerTail729_2 best = best := by
  rw [innerTail729_2, scan, if_neg (by decide), innerCandidates729_2_checked]
  change scan edge (search edge 3) 4 [5531] 5531 innerTail729_3
    (if 3 ≤ innerCandidates729_2.length then search edge 3 [5531, 7673] innerCandidates729_2 13204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan729_3

def innerTail729_1 : List Nat := 7121 :: innerTail729_2
def innerCandidates729_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates729_1_checked : candidates edge 4 5531 7121 innerTail729_2 best.bound = innerCandidates729_1 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan729_1 : scan edge (search edge 3) 4 [5531] 5531 innerTail729_1 best = best := by
  rw [innerTail729_1, scan, if_neg (by decide), innerCandidates729_1_checked]
  change scan edge (search edge 3) 4 [5531] 5531 innerTail729_2
    (if 3 ≤ innerCandidates729_1.length then search edge 3 [5531, 7121] innerCandidates729_1 12652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan729_2

def innerTail729_0 : List Nat := 6089 :: innerTail729_1
def innerCandidates729_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates729_0_checked : candidates edge 4 5531 6089 innerTail729_1 best.bound = innerCandidates729_0 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan729_0 : scan edge (search edge 3) 4 [5531] 5531 innerTail729_0 best = best := by
  rw [innerTail729_0, scan, if_neg (by decide), innerCandidates729_0_checked]
  change scan edge (search edge 3) 4 [5531] 5531 innerTail729_1
    (if 3 ≤ innerCandidates729_0.length then search edge 3 [5531, 6089] innerCandidates729_0 11620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan729_1

theorem rootBranch729 : search edge 4 [5531] rootCandidates729 5531 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5531] 5531 innerTail729_0 best = best
  exact innerScan729_0

end ProjectEulerSolutions.PrimePairUpper.Search
