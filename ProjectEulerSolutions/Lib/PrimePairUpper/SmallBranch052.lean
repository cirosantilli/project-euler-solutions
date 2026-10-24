import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates639 : List Nat := [4999, 5167, 5779, 6073, 6781, 7219, 7873, 7879]
set_option maxRecDepth 4096 in
theorem rootCandidates639_checked : candidates edge 5 0 4783 tail640 best.bound = rootCandidates639 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext q; rfl
  rw [row]
  decide

def innerTail639_8 : List Nat := []
theorem innerScan639_8 : scan edge (search edge 3) 4 [4783] 4783 innerTail639_8 best = best := rfl

def innerTail639_7 : List Nat := 7879 :: innerTail639_8
def innerCandidates639_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates639_7_checked : candidates edge 4 4783 7879 innerTail639_8 best.bound = innerCandidates639_7 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan639_7 : scan edge (search edge 3) 4 [4783] 4783 innerTail639_7 best = best := by
  rw [innerTail639_7, scan, if_neg (by decide), innerCandidates639_7_checked]
  change scan edge (search edge 3) 4 [4783] 4783 innerTail639_8
    (if 3 ≤ innerCandidates639_7.length then search edge 3 [4783, 7879] innerCandidates639_7 12662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan639_8

def innerTail639_6 : List Nat := 7873 :: innerTail639_7
def innerCandidates639_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates639_6_checked : candidates edge 4 4783 7873 innerTail639_7 best.bound = innerCandidates639_6 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan639_6 : scan edge (search edge 3) 4 [4783] 4783 innerTail639_6 best = best := by
  rw [innerTail639_6, scan, if_neg (by decide), innerCandidates639_6_checked]
  change scan edge (search edge 3) 4 [4783] 4783 innerTail639_7
    (if 3 ≤ innerCandidates639_6.length then search edge 3 [4783, 7873] innerCandidates639_6 12656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan639_7

def innerTail639_5 : List Nat := 7219 :: innerTail639_6
def innerCandidates639_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates639_5_checked : candidates edge 4 4783 7219 innerTail639_6 best.bound = innerCandidates639_5 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan639_5 : scan edge (search edge 3) 4 [4783] 4783 innerTail639_5 best = best := by
  rw [innerTail639_5, scan, if_neg (by decide), innerCandidates639_5_checked]
  change scan edge (search edge 3) 4 [4783] 4783 innerTail639_6
    (if 3 ≤ innerCandidates639_5.length then search edge 3 [4783, 7219] innerCandidates639_5 12002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan639_6

def innerTail639_4 : List Nat := 6781 :: innerTail639_5
def innerCandidates639_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates639_4_checked : candidates edge 4 4783 6781 innerTail639_5 best.bound = innerCandidates639_4 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan639_4 : scan edge (search edge 3) 4 [4783] 4783 innerTail639_4 best = best := by
  rw [innerTail639_4, scan, if_neg (by decide), innerCandidates639_4_checked]
  change scan edge (search edge 3) 4 [4783] 4783 innerTail639_5
    (if 3 ≤ innerCandidates639_4.length then search edge 3 [4783, 6781] innerCandidates639_4 11564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan639_5

def innerTail639_3 : List Nat := 6073 :: innerTail639_4
def innerCandidates639_3 : List Nat := [7219]
set_option maxRecDepth 4096 in
theorem innerCandidates639_3_checked : candidates edge 4 4783 6073 innerTail639_4 best.bound = innerCandidates639_3 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan639_3 : scan edge (search edge 3) 4 [4783] 4783 innerTail639_3 best = best := by
  rw [innerTail639_3, scan, if_neg (by decide), innerCandidates639_3_checked]
  change scan edge (search edge 3) 4 [4783] 4783 innerTail639_4
    (if 3 ≤ innerCandidates639_3.length then search edge 3 [4783, 6073] innerCandidates639_3 10856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan639_4

def innerTail639_2 : List Nat := 5779 :: innerTail639_3
def innerCandidates639_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates639_2_checked : candidates edge 4 4783 5779 innerTail639_3 best.bound = innerCandidates639_2 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan639_2 : scan edge (search edge 3) 4 [4783] 4783 innerTail639_2 best = best := by
  rw [innerTail639_2, scan, if_neg (by decide), innerCandidates639_2_checked]
  change scan edge (search edge 3) 4 [4783] 4783 innerTail639_3
    (if 3 ≤ innerCandidates639_2.length then search edge 3 [4783, 5779] innerCandidates639_2 10562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan639_3

def innerTail639_1 : List Nat := 5167 :: innerTail639_2
def innerCandidates639_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates639_1_checked : candidates edge 4 4783 5167 innerTail639_2 best.bound = innerCandidates639_1 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan639_1 : scan edge (search edge 3) 4 [4783] 4783 innerTail639_1 best = best := by
  rw [innerTail639_1, scan, if_neg (by decide), innerCandidates639_1_checked]
  change scan edge (search edge 3) 4 [4783] 4783 innerTail639_2
    (if 3 ≤ innerCandidates639_1.length then search edge 3 [4783, 5167] innerCandidates639_1 9950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan639_2

def innerTail639_0 : List Nat := 4999 :: innerTail639_1
def innerCandidates639_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates639_0_checked : candidates edge 4 4783 4999 innerTail639_1 best.bound = innerCandidates639_0 := by
  unfold candidates
  have row : edge 4999 = SearchData.row4999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan639_0 : scan edge (search edge 3) 4 [4783] 4783 innerTail639_0 best = best := by
  rw [innerTail639_0, scan, if_neg (by decide), innerCandidates639_0_checked]
  change scan edge (search edge 3) 4 [4783] 4783 innerTail639_1
    (if 3 ≤ innerCandidates639_0.length then search edge 3 [4783, 4999] innerCandidates639_0 9782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan639_1

theorem rootBranch639 : search edge 4 [4783] rootCandidates639 4783 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4783] 4783 innerTail639_0 best = best
  exact innerScan639_0

def rootCandidates638 : List Nat := [4903, 5227, 5701, 6151, 6823]
set_option maxRecDepth 4096 in
theorem rootCandidates638_checked : candidates edge 5 0 4759 tail639 best.bound = rootCandidates638 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext q; rfl
  rw [row]
  decide

def innerTail638_5 : List Nat := []
theorem innerScan638_5 : scan edge (search edge 3) 4 [4759] 4759 innerTail638_5 best = best := rfl

def innerTail638_4 : List Nat := 6823 :: innerTail638_5
def innerCandidates638_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates638_4_checked : candidates edge 4 4759 6823 innerTail638_5 best.bound = innerCandidates638_4 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan638_4 : scan edge (search edge 3) 4 [4759] 4759 innerTail638_4 best = best := by
  rw [innerTail638_4, scan, if_neg (by decide), innerCandidates638_4_checked]
  change scan edge (search edge 3) 4 [4759] 4759 innerTail638_5
    (if 3 ≤ innerCandidates638_4.length then search edge 3 [4759, 6823] innerCandidates638_4 11582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan638_5

def innerTail638_3 : List Nat := 6151 :: innerTail638_4
def innerCandidates638_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates638_3_checked : candidates edge 4 4759 6151 innerTail638_4 best.bound = innerCandidates638_3 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan638_3 : scan edge (search edge 3) 4 [4759] 4759 innerTail638_3 best = best := by
  rw [innerTail638_3, scan, if_neg (by decide), innerCandidates638_3_checked]
  change scan edge (search edge 3) 4 [4759] 4759 innerTail638_4
    (if 3 ≤ innerCandidates638_3.length then search edge 3 [4759, 6151] innerCandidates638_3 10910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan638_4

def innerTail638_2 : List Nat := 5701 :: innerTail638_3
def innerCandidates638_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates638_2_checked : candidates edge 4 4759 5701 innerTail638_3 best.bound = innerCandidates638_2 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan638_2 : scan edge (search edge 3) 4 [4759] 4759 innerTail638_2 best = best := by
  rw [innerTail638_2, scan, if_neg (by decide), innerCandidates638_2_checked]
  change scan edge (search edge 3) 4 [4759] 4759 innerTail638_3
    (if 3 ≤ innerCandidates638_2.length then search edge 3 [4759, 5701] innerCandidates638_2 10460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan638_3

def innerTail638_1 : List Nat := 5227 :: innerTail638_2
def innerCandidates638_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates638_1_checked : candidates edge 4 4759 5227 innerTail638_2 best.bound = innerCandidates638_1 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan638_1 : scan edge (search edge 3) 4 [4759] 4759 innerTail638_1 best = best := by
  rw [innerTail638_1, scan, if_neg (by decide), innerCandidates638_1_checked]
  change scan edge (search edge 3) 4 [4759] 4759 innerTail638_2
    (if 3 ≤ innerCandidates638_1.length then search edge 3 [4759, 5227] innerCandidates638_1 9986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan638_2

def innerTail638_0 : List Nat := 4903 :: innerTail638_1
def innerCandidates638_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates638_0_checked : candidates edge 4 4759 4903 innerTail638_1 best.bound = innerCandidates638_0 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan638_0 : scan edge (search edge 3) 4 [4759] 4759 innerTail638_0 best = best := by
  rw [innerTail638_0, scan, if_neg (by decide), innerCandidates638_0_checked]
  change scan edge (search edge 3) 4 [4759] 4759 innerTail638_1
    (if 3 ≤ innerCandidates638_0.length then search edge 3 [4759, 4903] innerCandidates638_0 9662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan638_1

theorem rootBranch638 : search edge 4 [4759] rootCandidates638 4759 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4759] 4759 innerTail638_0 best = best
  exact innerScan638_0

def rootCandidates637 : List Nat := [5099, 5297, 5711, 5879, 6113, 6317, 6659, 7331, 7607, 7649]
set_option maxRecDepth 4096 in
theorem rootCandidates637_checked : candidates edge 5 0 4751 tail638 best.bound = rootCandidates637 := by
  unfold candidates
  have row : edge 4751 = SearchData.row4751.contains := by funext q; rfl
  rw [row]
  decide

def innerTail637_10 : List Nat := []
theorem innerScan637_10 : scan edge (search edge 3) 4 [4751] 4751 innerTail637_10 best = best := rfl

def innerTail637_9 : List Nat := 7649 :: innerTail637_10
def innerCandidates637_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates637_9_checked : candidates edge 4 4751 7649 innerTail637_10 best.bound = innerCandidates637_9 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan637_9 : scan edge (search edge 3) 4 [4751] 4751 innerTail637_9 best = best := by
  rw [innerTail637_9, scan, if_neg (by decide), innerCandidates637_9_checked]
  change scan edge (search edge 3) 4 [4751] 4751 innerTail637_10
    (if 3 ≤ innerCandidates637_9.length then search edge 3 [4751, 7649] innerCandidates637_9 12400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan637_10

def innerTail637_8 : List Nat := 7607 :: innerTail637_9
def innerCandidates637_8 : List Nat := [7649]
set_option maxRecDepth 4096 in
theorem innerCandidates637_8_checked : candidates edge 4 4751 7607 innerTail637_9 best.bound = innerCandidates637_8 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan637_8 : scan edge (search edge 3) 4 [4751] 4751 innerTail637_8 best = best := by
  rw [innerTail637_8, scan, if_neg (by decide), innerCandidates637_8_checked]
  change scan edge (search edge 3) 4 [4751] 4751 innerTail637_9
    (if 3 ≤ innerCandidates637_8.length then search edge 3 [4751, 7607] innerCandidates637_8 12358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan637_9

def innerTail637_7 : List Nat := 7331 :: innerTail637_8
def innerCandidates637_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates637_7_checked : candidates edge 4 4751 7331 innerTail637_8 best.bound = innerCandidates637_7 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan637_7 : scan edge (search edge 3) 4 [4751] 4751 innerTail637_7 best = best := by
  rw [innerTail637_7, scan, if_neg (by decide), innerCandidates637_7_checked]
  change scan edge (search edge 3) 4 [4751] 4751 innerTail637_8
    (if 3 ≤ innerCandidates637_7.length then search edge 3 [4751, 7331] innerCandidates637_7 12082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan637_8

def innerTail637_6 : List Nat := 6659 :: innerTail637_7
def innerCandidates637_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates637_6_checked : candidates edge 4 4751 6659 innerTail637_7 best.bound = innerCandidates637_6 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan637_6 : scan edge (search edge 3) 4 [4751] 4751 innerTail637_6 best = best := by
  rw [innerTail637_6, scan, if_neg (by decide), innerCandidates637_6_checked]
  change scan edge (search edge 3) 4 [4751] 4751 innerTail637_7
    (if 3 ≤ innerCandidates637_6.length then search edge 3 [4751, 6659] innerCandidates637_6 11410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan637_7

def innerTail637_5 : List Nat := 6317 :: innerTail637_6
def innerCandidates637_5 : List Nat := [7607]
set_option maxRecDepth 4096 in
theorem innerCandidates637_5_checked : candidates edge 4 4751 6317 innerTail637_6 best.bound = innerCandidates637_5 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan637_5 : scan edge (search edge 3) 4 [4751] 4751 innerTail637_5 best = best := by
  rw [innerTail637_5, scan, if_neg (by decide), innerCandidates637_5_checked]
  change scan edge (search edge 3) 4 [4751] 4751 innerTail637_6
    (if 3 ≤ innerCandidates637_5.length then search edge 3 [4751, 6317] innerCandidates637_5 11068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan637_6

def innerTail637_4 : List Nat := 6113 :: innerTail637_5
def innerCandidates637_4 : List Nat := [6659]
set_option maxRecDepth 4096 in
theorem innerCandidates637_4_checked : candidates edge 4 4751 6113 innerTail637_5 best.bound = innerCandidates637_4 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan637_4 : scan edge (search edge 3) 4 [4751] 4751 innerTail637_4 best = best := by
  rw [innerTail637_4, scan, if_neg (by decide), innerCandidates637_4_checked]
  change scan edge (search edge 3) 4 [4751] 4751 innerTail637_5
    (if 3 ≤ innerCandidates637_4.length then search edge 3 [4751, 6113] innerCandidates637_4 10864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan637_5

def innerTail637_3 : List Nat := 5879 :: innerTail637_4
def innerCandidates637_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates637_3_checked : candidates edge 4 4751 5879 innerTail637_4 best.bound = innerCandidates637_3 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan637_3 : scan edge (search edge 3) 4 [4751] 4751 innerTail637_3 best = best := by
  rw [innerTail637_3, scan, if_neg (by decide), innerCandidates637_3_checked]
  change scan edge (search edge 3) 4 [4751] 4751 innerTail637_4
    (if 3 ≤ innerCandidates637_3.length then search edge 3 [4751, 5879] innerCandidates637_3 10630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan637_4

def innerTail637_2 : List Nat := 5711 :: innerTail637_3
def innerCandidates637_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates637_2_checked : candidates edge 4 4751 5711 innerTail637_3 best.bound = innerCandidates637_2 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan637_2 : scan edge (search edge 3) 4 [4751] 4751 innerTail637_2 best = best := by
  rw [innerTail637_2, scan, if_neg (by decide), innerCandidates637_2_checked]
  change scan edge (search edge 3) 4 [4751] 4751 innerTail637_3
    (if 3 ≤ innerCandidates637_2.length then search edge 3 [4751, 5711] innerCandidates637_2 10462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan637_3

def innerTail637_1 : List Nat := 5297 :: innerTail637_2
def innerCandidates637_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates637_1_checked : candidates edge 4 4751 5297 innerTail637_2 best.bound = innerCandidates637_1 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan637_1 : scan edge (search edge 3) 4 [4751] 4751 innerTail637_1 best = best := by
  rw [innerTail637_1, scan, if_neg (by decide), innerCandidates637_1_checked]
  change scan edge (search edge 3) 4 [4751] 4751 innerTail637_2
    (if 3 ≤ innerCandidates637_1.length then search edge 3 [4751, 5297] innerCandidates637_1 10048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan637_2

def innerTail637_0 : List Nat := 5099 :: innerTail637_1
def innerCandidates637_0 : List Nat := [7649]
set_option maxRecDepth 4096 in
theorem innerCandidates637_0_checked : candidates edge 4 4751 5099 innerTail637_1 best.bound = innerCandidates637_0 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan637_0 : scan edge (search edge 3) 4 [4751] 4751 innerTail637_0 best = best := by
  rw [innerTail637_0, scan, if_neg (by decide), innerCandidates637_0_checked]
  change scan edge (search edge 3) 4 [4751] 4751 innerTail637_1
    (if 3 ≤ innerCandidates637_0.length then search edge 3 [4751, 5099] innerCandidates637_0 9850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan637_1

theorem rootBranch637 : search edge 4 [4751] rootCandidates637 4751 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4751] 4751 innerTail637_0 best = best
  exact innerScan637_0

def rootCandidates636 : List Nat := [5237, 6269, 6689, 6911, 7079, 7211, 7331, 7517, 7793, 7907, 7919]
set_option maxRecDepth 4096 in
theorem rootCandidates636_checked : candidates edge 5 0 4733 tail637 best.bound = rootCandidates636 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext q; rfl
  rw [row]
  decide

def innerTail636_11 : List Nat := []
theorem innerScan636_11 : scan edge (search edge 3) 4 [4733] 4733 innerTail636_11 best = best := rfl

def innerTail636_10 : List Nat := 7919 :: innerTail636_11
def innerCandidates636_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates636_10_checked : candidates edge 4 4733 7919 innerTail636_11 best.bound = innerCandidates636_10 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan636_10 : scan edge (search edge 3) 4 [4733] 4733 innerTail636_10 best = best := by
  rw [innerTail636_10, scan, if_neg (by decide), innerCandidates636_10_checked]
  change scan edge (search edge 3) 4 [4733] 4733 innerTail636_11
    (if 3 ≤ innerCandidates636_10.length then search edge 3 [4733, 7919] innerCandidates636_10 12652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan636_11

def innerTail636_9 : List Nat := 7907 :: innerTail636_10
def innerCandidates636_9 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates636_9_checked : candidates edge 4 4733 7907 innerTail636_10 best.bound = innerCandidates636_9 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan636_9 : scan edge (search edge 3) 4 [4733] 4733 innerTail636_9 best = best := by
  rw [innerTail636_9, scan, if_neg (by decide), innerCandidates636_9_checked]
  change scan edge (search edge 3) 4 [4733] 4733 innerTail636_10
    (if 3 ≤ innerCandidates636_9.length then search edge 3 [4733, 7907] innerCandidates636_9 12640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan636_10

def innerTail636_8 : List Nat := 7793 :: innerTail636_9
def innerCandidates636_8 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates636_8_checked : candidates edge 4 4733 7793 innerTail636_9 best.bound = innerCandidates636_8 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan636_8 : scan edge (search edge 3) 4 [4733] 4733 innerTail636_8 best = best := by
  rw [innerTail636_8, scan, if_neg (by decide), innerCandidates636_8_checked]
  change scan edge (search edge 3) 4 [4733] 4733 innerTail636_9
    (if 3 ≤ innerCandidates636_8.length then search edge 3 [4733, 7793] innerCandidates636_8 12526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan636_9

def innerTail636_7 : List Nat := 7517 :: innerTail636_8
def innerCandidates636_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates636_7_checked : candidates edge 4 4733 7517 innerTail636_8 best.bound = innerCandidates636_7 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan636_7 : scan edge (search edge 3) 4 [4733] 4733 innerTail636_7 best = best := by
  rw [innerTail636_7, scan, if_neg (by decide), innerCandidates636_7_checked]
  change scan edge (search edge 3) 4 [4733] 4733 innerTail636_8
    (if 3 ≤ innerCandidates636_7.length then search edge 3 [4733, 7517] innerCandidates636_7 12250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan636_8

def innerTail636_6 : List Nat := 7331 :: innerTail636_7
def innerCandidates636_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates636_6_checked : candidates edge 4 4733 7331 innerTail636_7 best.bound = innerCandidates636_6 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan636_6 : scan edge (search edge 3) 4 [4733] 4733 innerTail636_6 best = best := by
  rw [innerTail636_6, scan, if_neg (by decide), innerCandidates636_6_checked]
  change scan edge (search edge 3) 4 [4733] 4733 innerTail636_7
    (if 3 ≤ innerCandidates636_6.length then search edge 3 [4733, 7331] innerCandidates636_6 12064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan636_7

def innerTail636_5 : List Nat := 7211 :: innerTail636_6
def innerCandidates636_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates636_5_checked : candidates edge 4 4733 7211 innerTail636_6 best.bound = innerCandidates636_5 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan636_5 : scan edge (search edge 3) 4 [4733] 4733 innerTail636_5 best = best := by
  rw [innerTail636_5, scan, if_neg (by decide), innerCandidates636_5_checked]
  change scan edge (search edge 3) 4 [4733] 4733 innerTail636_6
    (if 3 ≤ innerCandidates636_5.length then search edge 3 [4733, 7211] innerCandidates636_5 11944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan636_6

def innerTail636_4 : List Nat := 7079 :: innerTail636_5
def innerCandidates636_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates636_4_checked : candidates edge 4 4733 7079 innerTail636_5 best.bound = innerCandidates636_4 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan636_4 : scan edge (search edge 3) 4 [4733] 4733 innerTail636_4 best = best := by
  rw [innerTail636_4, scan, if_neg (by decide), innerCandidates636_4_checked]
  change scan edge (search edge 3) 4 [4733] 4733 innerTail636_5
    (if 3 ≤ innerCandidates636_4.length then search edge 3 [4733, 7079] innerCandidates636_4 11812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan636_5

def innerTail636_3 : List Nat := 6911 :: innerTail636_4
def innerCandidates636_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates636_3_checked : candidates edge 4 4733 6911 innerTail636_4 best.bound = innerCandidates636_3 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan636_3 : scan edge (search edge 3) 4 [4733] 4733 innerTail636_3 best = best := by
  rw [innerTail636_3, scan, if_neg (by decide), innerCandidates636_3_checked]
  change scan edge (search edge 3) 4 [4733] 4733 innerTail636_4
    (if 3 ≤ innerCandidates636_3.length then search edge 3 [4733, 6911] innerCandidates636_3 11644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan636_4

def innerTail636_2 : List Nat := 6689 :: innerTail636_3
def innerCandidates636_2 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates636_2_checked : candidates edge 4 4733 6689 innerTail636_3 best.bound = innerCandidates636_2 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan636_2 : scan edge (search edge 3) 4 [4733] 4733 innerTail636_2 best = best := by
  rw [innerTail636_2, scan, if_neg (by decide), innerCandidates636_2_checked]
  change scan edge (search edge 3) 4 [4733] 4733 innerTail636_3
    (if 3 ≤ innerCandidates636_2.length then search edge 3 [4733, 6689] innerCandidates636_2 11422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan636_3

def innerTail636_1 : List Nat := 6269 :: innerTail636_2
def innerCandidates636_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates636_1_checked : candidates edge 4 4733 6269 innerTail636_2 best.bound = innerCandidates636_1 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan636_1 : scan edge (search edge 3) 4 [4733] 4733 innerTail636_1 best = best := by
  rw [innerTail636_1, scan, if_neg (by decide), innerCandidates636_1_checked]
  change scan edge (search edge 3) 4 [4733] 4733 innerTail636_2
    (if 3 ≤ innerCandidates636_1.length then search edge 3 [4733, 6269] innerCandidates636_1 11002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan636_2

def innerTail636_0 : List Nat := 5237 :: innerTail636_1
def innerCandidates636_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates636_0_checked : candidates edge 4 4733 5237 innerTail636_1 best.bound = innerCandidates636_0 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan636_0 : scan edge (search edge 3) 4 [4733] 4733 innerTail636_0 best = best := by
  rw [innerTail636_0, scan, if_neg (by decide), innerCandidates636_0_checked]
  change scan edge (search edge 3) 4 [4733] 4733 innerTail636_1
    (if 3 ≤ innerCandidates636_0.length then search edge 3 [4733, 5237] innerCandidates636_0 9970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan636_1

theorem rootBranch636 : search edge 4 [4733] rootCandidates636 4733 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4733] 4733 innerTail636_0 best = best
  exact innerScan636_0

def rootCandidates635 : List Nat := [4909, 4999, 5077, 5407, 5749, 5791, 6961, 7207, 7603, 7963]
set_option maxRecDepth 4096 in
theorem rootCandidates635_checked : candidates edge 5 0 4729 tail636 best.bound = rootCandidates635 := by
  unfold candidates
  have row : edge 4729 = SearchData.row4729.contains := by funext q; rfl
  rw [row]
  decide

def innerTail635_10 : List Nat := []
theorem innerScan635_10 : scan edge (search edge 3) 4 [4729] 4729 innerTail635_10 best = best := rfl

def innerTail635_9 : List Nat := 7963 :: innerTail635_10
def innerCandidates635_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates635_9_checked : candidates edge 4 4729 7963 innerTail635_10 best.bound = innerCandidates635_9 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan635_9 : scan edge (search edge 3) 4 [4729] 4729 innerTail635_9 best = best := by
  rw [innerTail635_9, scan, if_neg (by decide), innerCandidates635_9_checked]
  change scan edge (search edge 3) 4 [4729] 4729 innerTail635_10
    (if 3 ≤ innerCandidates635_9.length then search edge 3 [4729, 7963] innerCandidates635_9 12692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan635_10

def innerTail635_8 : List Nat := 7603 :: innerTail635_9
def innerCandidates635_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates635_8_checked : candidates edge 4 4729 7603 innerTail635_9 best.bound = innerCandidates635_8 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan635_8 : scan edge (search edge 3) 4 [4729] 4729 innerTail635_8 best = best := by
  rw [innerTail635_8, scan, if_neg (by decide), innerCandidates635_8_checked]
  change scan edge (search edge 3) 4 [4729] 4729 innerTail635_9
    (if 3 ≤ innerCandidates635_8.length then search edge 3 [4729, 7603] innerCandidates635_8 12332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan635_9

def innerTail635_7 : List Nat := 7207 :: innerTail635_8
def innerCandidates635_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates635_7_checked : candidates edge 4 4729 7207 innerTail635_8 best.bound = innerCandidates635_7 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan635_7 : scan edge (search edge 3) 4 [4729] 4729 innerTail635_7 best = best := by
  rw [innerTail635_7, scan, if_neg (by decide), innerCandidates635_7_checked]
  change scan edge (search edge 3) 4 [4729] 4729 innerTail635_8
    (if 3 ≤ innerCandidates635_7.length then search edge 3 [4729, 7207] innerCandidates635_7 11936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan635_8

def innerTail635_6 : List Nat := 6961 :: innerTail635_7
def innerCandidates635_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates635_6_checked : candidates edge 4 4729 6961 innerTail635_7 best.bound = innerCandidates635_6 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan635_6 : scan edge (search edge 3) 4 [4729] 4729 innerTail635_6 best = best := by
  rw [innerTail635_6, scan, if_neg (by decide), innerCandidates635_6_checked]
  change scan edge (search edge 3) 4 [4729] 4729 innerTail635_7
    (if 3 ≤ innerCandidates635_6.length then search edge 3 [4729, 6961] innerCandidates635_6 11690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan635_7

def innerTail635_5 : List Nat := 5791 :: innerTail635_6
def innerCandidates635_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates635_5_checked : candidates edge 4 4729 5791 innerTail635_6 best.bound = innerCandidates635_5 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan635_5 : scan edge (search edge 3) 4 [4729] 4729 innerTail635_5 best = best := by
  rw [innerTail635_5, scan, if_neg (by decide), innerCandidates635_5_checked]
  change scan edge (search edge 3) 4 [4729] 4729 innerTail635_6
    (if 3 ≤ innerCandidates635_5.length then search edge 3 [4729, 5791] innerCandidates635_5 10520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan635_6

def innerTail635_4 : List Nat := 5749 :: innerTail635_5
def innerCandidates635_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates635_4_checked : candidates edge 4 4729 5749 innerTail635_5 best.bound = innerCandidates635_4 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan635_4 : scan edge (search edge 3) 4 [4729] 4729 innerTail635_4 best = best := by
  rw [innerTail635_4, scan, if_neg (by decide), innerCandidates635_4_checked]
  change scan edge (search edge 3) 4 [4729] 4729 innerTail635_5
    (if 3 ≤ innerCandidates635_4.length then search edge 3 [4729, 5749] innerCandidates635_4 10478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan635_5

def innerTail635_3 : List Nat := 5407 :: innerTail635_4
def innerCandidates635_3 : List Nat := [6961]
set_option maxRecDepth 4096 in
theorem innerCandidates635_3_checked : candidates edge 4 4729 5407 innerTail635_4 best.bound = innerCandidates635_3 := by
  unfold candidates
  have row : edge 5407 = SearchData.row5407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan635_3 : scan edge (search edge 3) 4 [4729] 4729 innerTail635_3 best = best := by
  rw [innerTail635_3, scan, if_neg (by decide), innerCandidates635_3_checked]
  change scan edge (search edge 3) 4 [4729] 4729 innerTail635_4
    (if 3 ≤ innerCandidates635_3.length then search edge 3 [4729, 5407] innerCandidates635_3 10136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan635_4

def innerTail635_2 : List Nat := 5077 :: innerTail635_3
def innerCandidates635_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates635_2_checked : candidates edge 4 4729 5077 innerTail635_3 best.bound = innerCandidates635_2 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan635_2 : scan edge (search edge 3) 4 [4729] 4729 innerTail635_2 best = best := by
  rw [innerTail635_2, scan, if_neg (by decide), innerCandidates635_2_checked]
  change scan edge (search edge 3) 4 [4729] 4729 innerTail635_3
    (if 3 ≤ innerCandidates635_2.length then search edge 3 [4729, 5077] innerCandidates635_2 9806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan635_3

def innerTail635_1 : List Nat := 4999 :: innerTail635_2
def innerCandidates635_1 : List Nat := [5791]
set_option maxRecDepth 4096 in
theorem innerCandidates635_1_checked : candidates edge 4 4729 4999 innerTail635_2 best.bound = innerCandidates635_1 := by
  unfold candidates
  have row : edge 4999 = SearchData.row4999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan635_1 : scan edge (search edge 3) 4 [4729] 4729 innerTail635_1 best = best := by
  rw [innerTail635_1, scan, if_neg (by decide), innerCandidates635_1_checked]
  change scan edge (search edge 3) 4 [4729] 4729 innerTail635_2
    (if 3 ≤ innerCandidates635_1.length then search edge 3 [4729, 4999] innerCandidates635_1 9728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan635_2

def innerTail635_0 : List Nat := 4909 :: innerTail635_1
def innerCandidates635_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates635_0_checked : candidates edge 4 4729 4909 innerTail635_1 best.bound = innerCandidates635_0 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan635_0 : scan edge (search edge 3) 4 [4729] 4729 innerTail635_0 best = best := by
  rw [innerTail635_0, scan, if_neg (by decide), innerCandidates635_0_checked]
  change scan edge (search edge 3) 4 [4729] 4729 innerTail635_1
    (if 3 ≤ innerCandidates635_0.length then search edge 3 [4729, 4909] innerCandidates635_0 9638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan635_1

theorem rootBranch635 : search edge 4 [4729] rootCandidates635 4729 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4729] 4729 innerTail635_0 best = best
  exact innerScan635_0

def rootCandidates634 : List Nat := [5521, 5623, 5869, 6361, 7243, 7369, 8221]
set_option maxRecDepth 4096 in
theorem rootCandidates634_checked : candidates edge 5 0 4723 tail635 best.bound = rootCandidates634 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext q; rfl
  rw [row]
  decide

def innerTail634_7 : List Nat := []
theorem innerScan634_7 : scan edge (search edge 3) 4 [4723] 4723 innerTail634_7 best = best := rfl

def innerTail634_6 : List Nat := 8221 :: innerTail634_7
def innerCandidates634_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates634_6_checked : candidates edge 4 4723 8221 innerTail634_7 best.bound = innerCandidates634_6 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan634_6 : scan edge (search edge 3) 4 [4723] 4723 innerTail634_6 best = best := by
  rw [innerTail634_6, scan, if_neg (by decide), innerCandidates634_6_checked]
  change scan edge (search edge 3) 4 [4723] 4723 innerTail634_7
    (if 3 ≤ innerCandidates634_6.length then search edge 3 [4723, 8221] innerCandidates634_6 12944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan634_7

def innerTail634_5 : List Nat := 7369 :: innerTail634_6
def innerCandidates634_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates634_5_checked : candidates edge 4 4723 7369 innerTail634_6 best.bound = innerCandidates634_5 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan634_5 : scan edge (search edge 3) 4 [4723] 4723 innerTail634_5 best = best := by
  rw [innerTail634_5, scan, if_neg (by decide), innerCandidates634_5_checked]
  change scan edge (search edge 3) 4 [4723] 4723 innerTail634_6
    (if 3 ≤ innerCandidates634_5.length then search edge 3 [4723, 7369] innerCandidates634_5 12092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan634_6

def innerTail634_4 : List Nat := 7243 :: innerTail634_5
def innerCandidates634_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates634_4_checked : candidates edge 4 4723 7243 innerTail634_5 best.bound = innerCandidates634_4 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan634_4 : scan edge (search edge 3) 4 [4723] 4723 innerTail634_4 best = best := by
  rw [innerTail634_4, scan, if_neg (by decide), innerCandidates634_4_checked]
  change scan edge (search edge 3) 4 [4723] 4723 innerTail634_5
    (if 3 ≤ innerCandidates634_4.length then search edge 3 [4723, 7243] innerCandidates634_4 11966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan634_5

def innerTail634_3 : List Nat := 6361 :: innerTail634_4
def innerCandidates634_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates634_3_checked : candidates edge 4 4723 6361 innerTail634_4 best.bound = innerCandidates634_3 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan634_3 : scan edge (search edge 3) 4 [4723] 4723 innerTail634_3 best = best := by
  rw [innerTail634_3, scan, if_neg (by decide), innerCandidates634_3_checked]
  change scan edge (search edge 3) 4 [4723] 4723 innerTail634_4
    (if 3 ≤ innerCandidates634_3.length then search edge 3 [4723, 6361] innerCandidates634_3 11084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan634_4

def innerTail634_2 : List Nat := 5869 :: innerTail634_3
def innerCandidates634_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates634_2_checked : candidates edge 4 4723 5869 innerTail634_3 best.bound = innerCandidates634_2 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan634_2 : scan edge (search edge 3) 4 [4723] 4723 innerTail634_2 best = best := by
  rw [innerTail634_2, scan, if_neg (by decide), innerCandidates634_2_checked]
  change scan edge (search edge 3) 4 [4723] 4723 innerTail634_3
    (if 3 ≤ innerCandidates634_2.length then search edge 3 [4723, 5869] innerCandidates634_2 10592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan634_3

def innerTail634_1 : List Nat := 5623 :: innerTail634_2
def innerCandidates634_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates634_1_checked : candidates edge 4 4723 5623 innerTail634_2 best.bound = innerCandidates634_1 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan634_1 : scan edge (search edge 3) 4 [4723] 4723 innerTail634_1 best = best := by
  rw [innerTail634_1, scan, if_neg (by decide), innerCandidates634_1_checked]
  change scan edge (search edge 3) 4 [4723] 4723 innerTail634_2
    (if 3 ≤ innerCandidates634_1.length then search edge 3 [4723, 5623] innerCandidates634_1 10346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan634_2

def innerTail634_0 : List Nat := 5521 :: innerTail634_1
def innerCandidates634_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates634_0_checked : candidates edge 4 4723 5521 innerTail634_1 best.bound = innerCandidates634_0 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan634_0 : scan edge (search edge 3) 4 [4723] 4723 innerTail634_0 best = best := by
  rw [innerTail634_0, scan, if_neg (by decide), innerCandidates634_0_checked]
  change scan edge (search edge 3) 4 [4723] 4723 innerTail634_1
    (if 3 ≤ innerCandidates634_0.length then search edge 3 [4723, 5521] innerCandidates634_0 10244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan634_1

theorem rootBranch634 : search edge 4 [4723] rootCandidates634 4723 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4723] 4723 innerTail634_0 best = best
  exact innerScan634_0

def rootCandidates633 : List Nat := [4877, 4919, 5417, 5501, 5813, 6971, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates633_checked : candidates edge 5 0 4721 tail634 best.bound = rootCandidates633 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext q; rfl
  rw [row]
  decide

def innerTail633_7 : List Nat := []
theorem innerScan633_7 : scan edge (search edge 3) 4 [4721] 4721 innerTail633_7 best = best := rfl

def innerTail633_6 : List Nat := 7949 :: innerTail633_7
def innerCandidates633_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates633_6_checked : candidates edge 4 4721 7949 innerTail633_7 best.bound = innerCandidates633_6 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan633_6 : scan edge (search edge 3) 4 [4721] 4721 innerTail633_6 best = best := by
  rw [innerTail633_6, scan, if_neg (by decide), innerCandidates633_6_checked]
  change scan edge (search edge 3) 4 [4721] 4721 innerTail633_7
    (if 3 ≤ innerCandidates633_6.length then search edge 3 [4721, 7949] innerCandidates633_6 12670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan633_7

def innerTail633_5 : List Nat := 6971 :: innerTail633_6
def innerCandidates633_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates633_5_checked : candidates edge 4 4721 6971 innerTail633_6 best.bound = innerCandidates633_5 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan633_5 : scan edge (search edge 3) 4 [4721] 4721 innerTail633_5 best = best := by
  rw [innerTail633_5, scan, if_neg (by decide), innerCandidates633_5_checked]
  change scan edge (search edge 3) 4 [4721] 4721 innerTail633_6
    (if 3 ≤ innerCandidates633_5.length then search edge 3 [4721, 6971] innerCandidates633_5 11692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan633_6

def innerTail633_4 : List Nat := 5813 :: innerTail633_5
def innerCandidates633_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates633_4_checked : candidates edge 4 4721 5813 innerTail633_5 best.bound = innerCandidates633_4 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan633_4 : scan edge (search edge 3) 4 [4721] 4721 innerTail633_4 best = best := by
  rw [innerTail633_4, scan, if_neg (by decide), innerCandidates633_4_checked]
  change scan edge (search edge 3) 4 [4721] 4721 innerTail633_5
    (if 3 ≤ innerCandidates633_4.length then search edge 3 [4721, 5813] innerCandidates633_4 10534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan633_5

def innerTail633_3 : List Nat := 5501 :: innerTail633_4
def innerCandidates633_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates633_3_checked : candidates edge 4 4721 5501 innerTail633_4 best.bound = innerCandidates633_3 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan633_3 : scan edge (search edge 3) 4 [4721] 4721 innerTail633_3 best = best := by
  rw [innerTail633_3, scan, if_neg (by decide), innerCandidates633_3_checked]
  change scan edge (search edge 3) 4 [4721] 4721 innerTail633_4
    (if 3 ≤ innerCandidates633_3.length then search edge 3 [4721, 5501] innerCandidates633_3 10222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan633_4

def innerTail633_2 : List Nat := 5417 :: innerTail633_3
def innerCandidates633_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates633_2_checked : candidates edge 4 4721 5417 innerTail633_3 best.bound = innerCandidates633_2 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan633_2 : scan edge (search edge 3) 4 [4721] 4721 innerTail633_2 best = best := by
  rw [innerTail633_2, scan, if_neg (by decide), innerCandidates633_2_checked]
  change scan edge (search edge 3) 4 [4721] 4721 innerTail633_3
    (if 3 ≤ innerCandidates633_2.length then search edge 3 [4721, 5417] innerCandidates633_2 10138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan633_3

def innerTail633_1 : List Nat := 4919 :: innerTail633_2
def innerCandidates633_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates633_1_checked : candidates edge 4 4721 4919 innerTail633_2 best.bound = innerCandidates633_1 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan633_1 : scan edge (search edge 3) 4 [4721] 4721 innerTail633_1 best = best := by
  rw [innerTail633_1, scan, if_neg (by decide), innerCandidates633_1_checked]
  change scan edge (search edge 3) 4 [4721] 4721 innerTail633_2
    (if 3 ≤ innerCandidates633_1.length then search edge 3 [4721, 4919] innerCandidates633_1 9640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan633_2

def innerTail633_0 : List Nat := 4877 :: innerTail633_1
def innerCandidates633_0 : List Nat := [5501, 5813]
set_option maxRecDepth 4096 in
theorem innerCandidates633_0_checked : candidates edge 4 4721 4877 innerTail633_1 best.bound = innerCandidates633_0 := by
  unfold candidates
  have row : edge 4877 = SearchData.row4877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan633_0 : scan edge (search edge 3) 4 [4721] 4721 innerTail633_0 best = best := by
  rw [innerTail633_0, scan, if_neg (by decide), innerCandidates633_0_checked]
  change scan edge (search edge 3) 4 [4721] 4721 innerTail633_1
    (if 3 ≤ innerCandidates633_0.length then search edge 3 [4721, 4877] innerCandidates633_0 9598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan633_1

theorem rootBranch633 : search edge 4 [4721] rootCandidates633 4721 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4721] 4721 innerTail633_0 best = best
  exact innerScan633_0

end ProjectEulerSolutions.PrimePairUpper.Search
