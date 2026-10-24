import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates743 : List Nat := [5701, 5779, 5953, 6121, 6151, 6271, 6577, 6619, 6703, 7069, 7759, 7867, 7927, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates743_checked : candidates edge 5 0 5659 tail744 best.bound = rootCandidates743 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext q; rfl
  rw [row]
  decide

def innerTail743_14 : List Nat := []
theorem innerScan743_14 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_14 best = best := rfl

def innerTail743_13 : List Nat := 7993 :: innerTail743_14
def innerCandidates743_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates743_13_checked : candidates edge 4 5659 7993 innerTail743_14 best.bound = innerCandidates743_13 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_13 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_13 best = best := by
  rw [innerTail743_13, scan, if_neg (by decide), innerCandidates743_13_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_14
    (if 3 ≤ innerCandidates743_13.length then search edge 3 [5659, 7993] innerCandidates743_13 13652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_14

def innerTail743_12 : List Nat := 7927 :: innerTail743_13
def innerCandidates743_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates743_12_checked : candidates edge 4 5659 7927 innerTail743_13 best.bound = innerCandidates743_12 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_12 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_12 best = best := by
  rw [innerTail743_12, scan, if_neg (by decide), innerCandidates743_12_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_13
    (if 3 ≤ innerCandidates743_12.length then search edge 3 [5659, 7927] innerCandidates743_12 13586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_13

def innerTail743_11 : List Nat := 7867 :: innerTail743_12
def innerCandidates743_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates743_11_checked : candidates edge 4 5659 7867 innerTail743_12 best.bound = innerCandidates743_11 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_11 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_11 best = best := by
  rw [innerTail743_11, scan, if_neg (by decide), innerCandidates743_11_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_12
    (if 3 ≤ innerCandidates743_11.length then search edge 3 [5659, 7867] innerCandidates743_11 13526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_12

def innerTail743_10 : List Nat := 7759 :: innerTail743_11
def innerCandidates743_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates743_10_checked : candidates edge 4 5659 7759 innerTail743_11 best.bound = innerCandidates743_10 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_10 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_10 best = best := by
  rw [innerTail743_10, scan, if_neg (by decide), innerCandidates743_10_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_11
    (if 3 ≤ innerCandidates743_10.length then search edge 3 [5659, 7759] innerCandidates743_10 13418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_11

def innerTail743_9 : List Nat := 7069 :: innerTail743_10
def innerCandidates743_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates743_9_checked : candidates edge 4 5659 7069 innerTail743_10 best.bound = innerCandidates743_9 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_9 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_9 best = best := by
  rw [innerTail743_9, scan, if_neg (by decide), innerCandidates743_9_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_10
    (if 3 ≤ innerCandidates743_9.length then search edge 3 [5659, 7069] innerCandidates743_9 12728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_10

def innerTail743_8 : List Nat := 6703 :: innerTail743_9
def innerCandidates743_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates743_8_checked : candidates edge 4 5659 6703 innerTail743_9 best.bound = innerCandidates743_8 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_8 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_8 best = best := by
  rw [innerTail743_8, scan, if_neg (by decide), innerCandidates743_8_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_9
    (if 3 ≤ innerCandidates743_8.length then search edge 3 [5659, 6703] innerCandidates743_8 12362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_9

def innerTail743_7 : List Nat := 6619 :: innerTail743_8
def innerCandidates743_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates743_7_checked : candidates edge 4 5659 6619 innerTail743_8 best.bound = innerCandidates743_7 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_7 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_7 best = best := by
  rw [innerTail743_7, scan, if_neg (by decide), innerCandidates743_7_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_8
    (if 3 ≤ innerCandidates743_7.length then search edge 3 [5659, 6619] innerCandidates743_7 12278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_8

def innerTail743_6 : List Nat := 6577 :: innerTail743_7
def innerCandidates743_6 : List Nat := [6703]
set_option maxRecDepth 4096 in
theorem innerCandidates743_6_checked : candidates edge 4 5659 6577 innerTail743_7 best.bound = innerCandidates743_6 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_6 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_6 best = best := by
  rw [innerTail743_6, scan, if_neg (by decide), innerCandidates743_6_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_7
    (if 3 ≤ innerCandidates743_6.length then search edge 3 [5659, 6577] innerCandidates743_6 12236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_7

def innerTail743_5 : List Nat := 6271 :: innerTail743_6
def innerCandidates743_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates743_5_checked : candidates edge 4 5659 6271 innerTail743_6 best.bound = innerCandidates743_5 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_5 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_5 best = best := by
  rw [innerTail743_5, scan, if_neg (by decide), innerCandidates743_5_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_6
    (if 3 ≤ innerCandidates743_5.length then search edge 3 [5659, 6271] innerCandidates743_5 11930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_6

def innerTail743_4 : List Nat := 6151 :: innerTail743_5
def innerCandidates743_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates743_4_checked : candidates edge 4 5659 6151 innerTail743_5 best.bound = innerCandidates743_4 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_4 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_4 best = best := by
  rw [innerTail743_4, scan, if_neg (by decide), innerCandidates743_4_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_5
    (if 3 ≤ innerCandidates743_4.length then search edge 3 [5659, 6151] innerCandidates743_4 11810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_5

def innerTail743_3 : List Nat := 6121 :: innerTail743_4
def innerCandidates743_3 : List Nat := [6619, 7993]
set_option maxRecDepth 4096 in
theorem innerCandidates743_3_checked : candidates edge 4 5659 6121 innerTail743_4 best.bound = innerCandidates743_3 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_3 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_3 best = best := by
  rw [innerTail743_3, scan, if_neg (by decide), innerCandidates743_3_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_4
    (if 3 ≤ innerCandidates743_3.length then search edge 3 [5659, 6121] innerCandidates743_3 11780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_4

def innerTail743_2 : List Nat := 5953 :: innerTail743_3
def innerCandidates743_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates743_2_checked : candidates edge 4 5659 5953 innerTail743_3 best.bound = innerCandidates743_2 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_2 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_2 best = best := by
  rw [innerTail743_2, scan, if_neg (by decide), innerCandidates743_2_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_3
    (if 3 ≤ innerCandidates743_2.length then search edge 3 [5659, 5953] innerCandidates743_2 11612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_3

def innerTail743_1 : List Nat := 5779 :: innerTail743_2
def innerCandidates743_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates743_1_checked : candidates edge 4 5659 5779 innerTail743_2 best.bound = innerCandidates743_1 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_1 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_1 best = best := by
  rw [innerTail743_1, scan, if_neg (by decide), innerCandidates743_1_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_2
    (if 3 ≤ innerCandidates743_1.length then search edge 3 [5659, 5779] innerCandidates743_1 11438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_2

def innerTail743_0 : List Nat := 5701 :: innerTail743_1
def innerCandidates743_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates743_0_checked : candidates edge 4 5659 5701 innerTail743_1 best.bound = innerCandidates743_0 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan743_0 : scan edge (search edge 3) 4 [5659] 5659 innerTail743_0 best = best := by
  rw [innerTail743_0, scan, if_neg (by decide), innerCandidates743_0_checked]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_1
    (if 3 ≤ innerCandidates743_0.length then search edge 3 [5659, 5701] innerCandidates743_0 11360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan743_1

theorem rootBranch743 : search edge 4 [5659] rootCandidates743 5659 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5659] 5659 innerTail743_0 best = best
  exact innerScan743_0

def rootCandidates742 : List Nat := [6791, 6983, 7529, 7673, 7823, 8009, 8369, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates742_checked : candidates edge 5 0 5657 tail743 best.bound = rootCandidates742 := by
  unfold candidates
  have row : edge 5657 = SearchData.row5657.contains := by funext q; rfl
  rw [row]
  decide

def innerTail742_8 : List Nat := []
theorem innerScan742_8 : scan edge (search edge 3) 4 [5657] 5657 innerTail742_8 best = best := rfl

def innerTail742_7 : List Nat := 8387 :: innerTail742_8
def innerCandidates742_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates742_7_checked : candidates edge 4 5657 8387 innerTail742_8 best.bound = innerCandidates742_7 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan742_7 : scan edge (search edge 3) 4 [5657] 5657 innerTail742_7 best = best := by
  rw [innerTail742_7, scan, if_neg (by decide), innerCandidates742_7_checked]
  change scan edge (search edge 3) 4 [5657] 5657 innerTail742_8
    (if 3 ≤ innerCandidates742_7.length then search edge 3 [5657, 8387] innerCandidates742_7 14044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan742_8

def innerTail742_6 : List Nat := 8369 :: innerTail742_7
def innerCandidates742_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates742_6_checked : candidates edge 4 5657 8369 innerTail742_7 best.bound = innerCandidates742_6 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan742_6 : scan edge (search edge 3) 4 [5657] 5657 innerTail742_6 best = best := by
  rw [innerTail742_6, scan, if_neg (by decide), innerCandidates742_6_checked]
  change scan edge (search edge 3) 4 [5657] 5657 innerTail742_7
    (if 3 ≤ innerCandidates742_6.length then search edge 3 [5657, 8369] innerCandidates742_6 14026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan742_7

def innerTail742_5 : List Nat := 8009 :: innerTail742_6
def innerCandidates742_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates742_5_checked : candidates edge 4 5657 8009 innerTail742_6 best.bound = innerCandidates742_5 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan742_5 : scan edge (search edge 3) 4 [5657] 5657 innerTail742_5 best = best := by
  rw [innerTail742_5, scan, if_neg (by decide), innerCandidates742_5_checked]
  change scan edge (search edge 3) 4 [5657] 5657 innerTail742_6
    (if 3 ≤ innerCandidates742_5.length then search edge 3 [5657, 8009] innerCandidates742_5 13666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan742_6

def innerTail742_4 : List Nat := 7823 :: innerTail742_5
def innerCandidates742_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates742_4_checked : candidates edge 4 5657 7823 innerTail742_5 best.bound = innerCandidates742_4 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan742_4 : scan edge (search edge 3) 4 [5657] 5657 innerTail742_4 best = best := by
  rw [innerTail742_4, scan, if_neg (by decide), innerCandidates742_4_checked]
  change scan edge (search edge 3) 4 [5657] 5657 innerTail742_5
    (if 3 ≤ innerCandidates742_4.length then search edge 3 [5657, 7823] innerCandidates742_4 13480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan742_5

def innerTail742_3 : List Nat := 7673 :: innerTail742_4
def innerCandidates742_3 : List Nat := [7823]
set_option maxRecDepth 4096 in
theorem innerCandidates742_3_checked : candidates edge 4 5657 7673 innerTail742_4 best.bound = innerCandidates742_3 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan742_3 : scan edge (search edge 3) 4 [5657] 5657 innerTail742_3 best = best := by
  rw [innerTail742_3, scan, if_neg (by decide), innerCandidates742_3_checked]
  change scan edge (search edge 3) 4 [5657] 5657 innerTail742_4
    (if 3 ≤ innerCandidates742_3.length then search edge 3 [5657, 7673] innerCandidates742_3 13330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan742_4

def innerTail742_2 : List Nat := 7529 :: innerTail742_3
def innerCandidates742_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates742_2_checked : candidates edge 4 5657 7529 innerTail742_3 best.bound = innerCandidates742_2 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan742_2 : scan edge (search edge 3) 4 [5657] 5657 innerTail742_2 best = best := by
  rw [innerTail742_2, scan, if_neg (by decide), innerCandidates742_2_checked]
  change scan edge (search edge 3) 4 [5657] 5657 innerTail742_3
    (if 3 ≤ innerCandidates742_2.length then search edge 3 [5657, 7529] innerCandidates742_2 13186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan742_3

def innerTail742_1 : List Nat := 6983 :: innerTail742_2
def innerCandidates742_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates742_1_checked : candidates edge 4 5657 6983 innerTail742_2 best.bound = innerCandidates742_1 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan742_1 : scan edge (search edge 3) 4 [5657] 5657 innerTail742_1 best = best := by
  rw [innerTail742_1, scan, if_neg (by decide), innerCandidates742_1_checked]
  change scan edge (search edge 3) 4 [5657] 5657 innerTail742_2
    (if 3 ≤ innerCandidates742_1.length then search edge 3 [5657, 6983] innerCandidates742_1 12640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan742_2

def innerTail742_0 : List Nat := 6791 :: innerTail742_1
def innerCandidates742_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates742_0_checked : candidates edge 4 5657 6791 innerTail742_1 best.bound = innerCandidates742_0 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan742_0 : scan edge (search edge 3) 4 [5657] 5657 innerTail742_0 best = best := by
  rw [innerTail742_0, scan, if_neg (by decide), innerCandidates742_0_checked]
  change scan edge (search edge 3) 4 [5657] 5657 innerTail742_1
    (if 3 ≤ innerCandidates742_0.length then search edge 3 [5657, 6791] innerCandidates742_0 12448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan742_1

theorem rootBranch742 : search edge 4 [5657] rootCandidates742 5657 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5657] 5657 innerTail742_0 best = best
  exact innerScan742_0

def rootCandidates741 : List Nat := [5827, 5923, 6079, 6133, 6211, 6247, 6367, 6619, 6793, 7333, 7549, 8269, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates741_checked : candidates edge 5 0 5653 tail742 best.bound = rootCandidates741 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext q; rfl
  rw [row]
  decide

def innerTail741_13 : List Nat := []
theorem innerScan741_13 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_13 best = best := rfl

def innerTail741_12 : List Nat := 8353 :: innerTail741_13
def innerCandidates741_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates741_12_checked : candidates edge 4 5653 8353 innerTail741_13 best.bound = innerCandidates741_12 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_12 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_12 best = best := by
  rw [innerTail741_12, scan, if_neg (by decide), innerCandidates741_12_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_13
    (if 3 ≤ innerCandidates741_12.length then search edge 3 [5653, 8353] innerCandidates741_12 14006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_13

def innerTail741_11 : List Nat := 8269 :: innerTail741_12
def innerCandidates741_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates741_11_checked : candidates edge 4 5653 8269 innerTail741_12 best.bound = innerCandidates741_11 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_11 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_11 best = best := by
  rw [innerTail741_11, scan, if_neg (by decide), innerCandidates741_11_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_12
    (if 3 ≤ innerCandidates741_11.length then search edge 3 [5653, 8269] innerCandidates741_11 13922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_12

def innerTail741_10 : List Nat := 7549 :: innerTail741_11
def innerCandidates741_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates741_10_checked : candidates edge 4 5653 7549 innerTail741_11 best.bound = innerCandidates741_10 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_10 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_10 best = best := by
  rw [innerTail741_10, scan, if_neg (by decide), innerCandidates741_10_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_11
    (if 3 ≤ innerCandidates741_10.length then search edge 3 [5653, 7549] innerCandidates741_10 13202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_11

def innerTail741_9 : List Nat := 7333 :: innerTail741_10
def innerCandidates741_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates741_9_checked : candidates edge 4 5653 7333 innerTail741_10 best.bound = innerCandidates741_9 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_9 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_9 best = best := by
  rw [innerTail741_9, scan, if_neg (by decide), innerCandidates741_9_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_10
    (if 3 ≤ innerCandidates741_9.length then search edge 3 [5653, 7333] innerCandidates741_9 12986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_10

def innerTail741_8 : List Nat := 6793 :: innerTail741_9
def innerCandidates741_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates741_8_checked : candidates edge 4 5653 6793 innerTail741_9 best.bound = innerCandidates741_8 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_8 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_8 best = best := by
  rw [innerTail741_8, scan, if_neg (by decide), innerCandidates741_8_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_9
    (if 3 ≤ innerCandidates741_8.length then search edge 3 [5653, 6793] innerCandidates741_8 12446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_9

def innerTail741_7 : List Nat := 6619 :: innerTail741_8
def innerCandidates741_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates741_7_checked : candidates edge 4 5653 6619 innerTail741_8 best.bound = innerCandidates741_7 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_7 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_7 best = best := by
  rw [innerTail741_7, scan, if_neg (by decide), innerCandidates741_7_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_8
    (if 3 ≤ innerCandidates741_7.length then search edge 3 [5653, 6619] innerCandidates741_7 12272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_8

def innerTail741_6 : List Nat := 6367 :: innerTail741_7
def innerCandidates741_6 : List Nat := [6793, 8353]
set_option maxRecDepth 4096 in
theorem innerCandidates741_6_checked : candidates edge 4 5653 6367 innerTail741_7 best.bound = innerCandidates741_6 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_6 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_6 best = best := by
  rw [innerTail741_6, scan, if_neg (by decide), innerCandidates741_6_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_7
    (if 3 ≤ innerCandidates741_6.length then search edge 3 [5653, 6367] innerCandidates741_6 12020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_7

def innerTail741_5 : List Nat := 6247 :: innerTail741_6
def innerCandidates741_5 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem innerCandidates741_5_checked : candidates edge 4 5653 6247 innerTail741_6 best.bound = innerCandidates741_5 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_5 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_5 best = best := by
  rw [innerTail741_5, scan, if_neg (by decide), innerCandidates741_5_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_6
    (if 3 ≤ innerCandidates741_5.length then search edge 3 [5653, 6247] innerCandidates741_5 11900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_6

def innerTail741_4 : List Nat := 6211 :: innerTail741_5
def innerCandidates741_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates741_4_checked : candidates edge 4 5653 6211 innerTail741_5 best.bound = innerCandidates741_4 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_4 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_4 best = best := by
  rw [innerTail741_4, scan, if_neg (by decide), innerCandidates741_4_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_5
    (if 3 ≤ innerCandidates741_4.length then search edge 3 [5653, 6211] innerCandidates741_4 11864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_5

def innerTail741_3 : List Nat := 6133 :: innerTail741_4
def innerCandidates741_3 : List Nat := [8353]
set_option maxRecDepth 4096 in
theorem innerCandidates741_3_checked : candidates edge 4 5653 6133 innerTail741_4 best.bound = innerCandidates741_3 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_3 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_3 best = best := by
  rw [innerTail741_3, scan, if_neg (by decide), innerCandidates741_3_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_4
    (if 3 ≤ innerCandidates741_3.length then search edge 3 [5653, 6133] innerCandidates741_3 11786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_4

def innerTail741_2 : List Nat := 6079 :: innerTail741_3
def innerCandidates741_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates741_2_checked : candidates edge 4 5653 6079 innerTail741_3 best.bound = innerCandidates741_2 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_2 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_2 best = best := by
  rw [innerTail741_2, scan, if_neg (by decide), innerCandidates741_2_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_3
    (if 3 ≤ innerCandidates741_2.length then search edge 3 [5653, 6079] innerCandidates741_2 11732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_3

def innerTail741_1 : List Nat := 5923 :: innerTail741_2
def innerCandidates741_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates741_1_checked : candidates edge 4 5653 5923 innerTail741_2 best.bound = innerCandidates741_1 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_1 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_1 best = best := by
  rw [innerTail741_1, scan, if_neg (by decide), innerCandidates741_1_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_2
    (if 3 ≤ innerCandidates741_1.length then search edge 3 [5653, 5923] innerCandidates741_1 11576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_2

def innerTail741_0 : List Nat := 5827 :: innerTail741_1
def innerCandidates741_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates741_0_checked : candidates edge 4 5653 5827 innerTail741_1 best.bound = innerCandidates741_0 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan741_0 : scan edge (search edge 3) 4 [5653] 5653 innerTail741_0 best = best := by
  rw [innerTail741_0, scan, if_neg (by decide), innerCandidates741_0_checked]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_1
    (if 3 ≤ innerCandidates741_0.length then search edge 3 [5653, 5827] innerCandidates741_0 11480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan741_1

theorem rootBranch741 : search edge 4 [5653] rootCandidates741 5653 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5653] 5653 innerTail741_0 best = best
  exact innerScan741_0

def rootCandidates740 : List Nat := [5783, 6011, 7499, 7691, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates740_checked : candidates edge 5 0 5651 tail741 best.bound = rootCandidates740 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext q; rfl
  rw [row]
  decide

def innerTail740_5 : List Nat := []
theorem innerScan740_5 : scan edge (search edge 3) 4 [5651] 5651 innerTail740_5 best = best := rfl

def innerTail740_4 : List Nat := 8093 :: innerTail740_5
def innerCandidates740_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates740_4_checked : candidates edge 4 5651 8093 innerTail740_5 best.bound = innerCandidates740_4 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan740_4 : scan edge (search edge 3) 4 [5651] 5651 innerTail740_4 best = best := by
  rw [innerTail740_4, scan, if_neg (by decide), innerCandidates740_4_checked]
  change scan edge (search edge 3) 4 [5651] 5651 innerTail740_5
    (if 3 ≤ innerCandidates740_4.length then search edge 3 [5651, 8093] innerCandidates740_4 13744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan740_5

def innerTail740_3 : List Nat := 7691 :: innerTail740_4
def innerCandidates740_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates740_3_checked : candidates edge 4 5651 7691 innerTail740_4 best.bound = innerCandidates740_3 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan740_3 : scan edge (search edge 3) 4 [5651] 5651 innerTail740_3 best = best := by
  rw [innerTail740_3, scan, if_neg (by decide), innerCandidates740_3_checked]
  change scan edge (search edge 3) 4 [5651] 5651 innerTail740_4
    (if 3 ≤ innerCandidates740_3.length then search edge 3 [5651, 7691] innerCandidates740_3 13342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan740_4

def innerTail740_2 : List Nat := 7499 :: innerTail740_3
def innerCandidates740_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates740_2_checked : candidates edge 4 5651 7499 innerTail740_3 best.bound = innerCandidates740_2 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan740_2 : scan edge (search edge 3) 4 [5651] 5651 innerTail740_2 best = best := by
  rw [innerTail740_2, scan, if_neg (by decide), innerCandidates740_2_checked]
  change scan edge (search edge 3) 4 [5651] 5651 innerTail740_3
    (if 3 ≤ innerCandidates740_2.length then search edge 3 [5651, 7499] innerCandidates740_2 13150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan740_3

def innerTail740_1 : List Nat := 6011 :: innerTail740_2
def innerCandidates740_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates740_1_checked : candidates edge 4 5651 6011 innerTail740_2 best.bound = innerCandidates740_1 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan740_1 : scan edge (search edge 3) 4 [5651] 5651 innerTail740_1 best = best := by
  rw [innerTail740_1, scan, if_neg (by decide), innerCandidates740_1_checked]
  change scan edge (search edge 3) 4 [5651] 5651 innerTail740_2
    (if 3 ≤ innerCandidates740_1.length then search edge 3 [5651, 6011] innerCandidates740_1 11662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan740_2

def innerTail740_0 : List Nat := 5783 :: innerTail740_1
def innerCandidates740_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates740_0_checked : candidates edge 4 5651 5783 innerTail740_1 best.bound = innerCandidates740_0 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan740_0 : scan edge (search edge 3) 4 [5651] 5651 innerTail740_0 best = best := by
  rw [innerTail740_0, scan, if_neg (by decide), innerCandidates740_0_checked]
  change scan edge (search edge 3) 4 [5651] 5651 innerTail740_1
    (if 3 ≤ innerCandidates740_0.length then search edge 3 [5651, 5783] innerCandidates740_0 11434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan740_1

theorem rootBranch740 : search edge 4 [5651] rootCandidates740 5651 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5651] 5651 innerTail740_0 best = best
  exact innerScan740_0

def rootCandidates739 : List Nat := [5659, 6373, 6733, 7129, 7741, 7867]
set_option maxRecDepth 4096 in
theorem rootCandidates739_checked : candidates edge 5 0 5647 tail740 best.bound = rootCandidates739 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext q; rfl
  rw [row]
  decide

def innerTail739_6 : List Nat := []
theorem innerScan739_6 : scan edge (search edge 3) 4 [5647] 5647 innerTail739_6 best = best := rfl

def innerTail739_5 : List Nat := 7867 :: innerTail739_6
def innerCandidates739_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates739_5_checked : candidates edge 4 5647 7867 innerTail739_6 best.bound = innerCandidates739_5 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan739_5 : scan edge (search edge 3) 4 [5647] 5647 innerTail739_5 best = best := by
  rw [innerTail739_5, scan, if_neg (by decide), innerCandidates739_5_checked]
  change scan edge (search edge 3) 4 [5647] 5647 innerTail739_6
    (if 3 ≤ innerCandidates739_5.length then search edge 3 [5647, 7867] innerCandidates739_5 13514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan739_6

def innerTail739_4 : List Nat := 7741 :: innerTail739_5
def innerCandidates739_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates739_4_checked : candidates edge 4 5647 7741 innerTail739_5 best.bound = innerCandidates739_4 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan739_4 : scan edge (search edge 3) 4 [5647] 5647 innerTail739_4 best = best := by
  rw [innerTail739_4, scan, if_neg (by decide), innerCandidates739_4_checked]
  change scan edge (search edge 3) 4 [5647] 5647 innerTail739_5
    (if 3 ≤ innerCandidates739_4.length then search edge 3 [5647, 7741] innerCandidates739_4 13388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan739_5

def innerTail739_3 : List Nat := 7129 :: innerTail739_4
def innerCandidates739_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates739_3_checked : candidates edge 4 5647 7129 innerTail739_4 best.bound = innerCandidates739_3 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan739_3 : scan edge (search edge 3) 4 [5647] 5647 innerTail739_3 best = best := by
  rw [innerTail739_3, scan, if_neg (by decide), innerCandidates739_3_checked]
  change scan edge (search edge 3) 4 [5647] 5647 innerTail739_4
    (if 3 ≤ innerCandidates739_3.length then search edge 3 [5647, 7129] innerCandidates739_3 12776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan739_4

def innerTail739_2 : List Nat := 6733 :: innerTail739_3
def innerCandidates739_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates739_2_checked : candidates edge 4 5647 6733 innerTail739_3 best.bound = innerCandidates739_2 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan739_2 : scan edge (search edge 3) 4 [5647] 5647 innerTail739_2 best = best := by
  rw [innerTail739_2, scan, if_neg (by decide), innerCandidates739_2_checked]
  change scan edge (search edge 3) 4 [5647] 5647 innerTail739_3
    (if 3 ≤ innerCandidates739_2.length then search edge 3 [5647, 6733] innerCandidates739_2 12380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan739_3

def innerTail739_1 : List Nat := 6373 :: innerTail739_2
def innerCandidates739_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates739_1_checked : candidates edge 4 5647 6373 innerTail739_2 best.bound = innerCandidates739_1 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan739_1 : scan edge (search edge 3) 4 [5647] 5647 innerTail739_1 best = best := by
  rw [innerTail739_1, scan, if_neg (by decide), innerCandidates739_1_checked]
  change scan edge (search edge 3) 4 [5647] 5647 innerTail739_2
    (if 3 ≤ innerCandidates739_1.length then search edge 3 [5647, 6373] innerCandidates739_1 12020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan739_2

def innerTail739_0 : List Nat := 5659 :: innerTail739_1
def innerCandidates739_0 : List Nat := [7867]
set_option maxRecDepth 4096 in
theorem innerCandidates739_0_checked : candidates edge 4 5647 5659 innerTail739_1 best.bound = innerCandidates739_0 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan739_0 : scan edge (search edge 3) 4 [5647] 5647 innerTail739_0 best = best := by
  rw [innerTail739_0, scan, if_neg (by decide), innerCandidates739_0_checked]
  change scan edge (search edge 3) 4 [5647] 5647 innerTail739_1
    (if 3 ≤ innerCandidates739_0.length then search edge 3 [5647, 5659] innerCandidates739_0 11306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan739_1

theorem rootBranch739 : search edge 4 [5647] rootCandidates739 5647 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5647] 5647 innerTail739_0 best = best
  exact innerScan739_0

def rootCandidates738 : List Nat := [5743, 6007, 6637, 6793, 7027, 7687, 7759, 8191, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates738_checked : candidates edge 5 0 5641 tail739 best.bound = rootCandidates738 := by
  unfold candidates
  have row : edge 5641 = SearchData.row5641.contains := by funext q; rfl
  rw [row]
  decide

def innerTail738_9 : List Nat := []
theorem innerScan738_9 : scan edge (search edge 3) 4 [5641] 5641 innerTail738_9 best = best := rfl

def innerTail738_8 : List Nat := 8209 :: innerTail738_9
def innerCandidates738_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates738_8_checked : candidates edge 4 5641 8209 innerTail738_9 best.bound = innerCandidates738_8 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan738_8 : scan edge (search edge 3) 4 [5641] 5641 innerTail738_8 best = best := by
  rw [innerTail738_8, scan, if_neg (by decide), innerCandidates738_8_checked]
  change scan edge (search edge 3) 4 [5641] 5641 innerTail738_9
    (if 3 ≤ innerCandidates738_8.length then search edge 3 [5641, 8209] innerCandidates738_8 13850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan738_9

def innerTail738_7 : List Nat := 8191 :: innerTail738_8
def innerCandidates738_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates738_7_checked : candidates edge 4 5641 8191 innerTail738_8 best.bound = innerCandidates738_7 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan738_7 : scan edge (search edge 3) 4 [5641] 5641 innerTail738_7 best = best := by
  rw [innerTail738_7, scan, if_neg (by decide), innerCandidates738_7_checked]
  change scan edge (search edge 3) 4 [5641] 5641 innerTail738_8
    (if 3 ≤ innerCandidates738_7.length then search edge 3 [5641, 8191] innerCandidates738_7 13832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan738_8

def innerTail738_6 : List Nat := 7759 :: innerTail738_7
def innerCandidates738_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates738_6_checked : candidates edge 4 5641 7759 innerTail738_7 best.bound = innerCandidates738_6 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan738_6 : scan edge (search edge 3) 4 [5641] 5641 innerTail738_6 best = best := by
  rw [innerTail738_6, scan, if_neg (by decide), innerCandidates738_6_checked]
  change scan edge (search edge 3) 4 [5641] 5641 innerTail738_7
    (if 3 ≤ innerCandidates738_6.length then search edge 3 [5641, 7759] innerCandidates738_6 13400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan738_7

def innerTail738_5 : List Nat := 7687 :: innerTail738_6
def innerCandidates738_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates738_5_checked : candidates edge 4 5641 7687 innerTail738_6 best.bound = innerCandidates738_5 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan738_5 : scan edge (search edge 3) 4 [5641] 5641 innerTail738_5 best = best := by
  rw [innerTail738_5, scan, if_neg (by decide), innerCandidates738_5_checked]
  change scan edge (search edge 3) 4 [5641] 5641 innerTail738_6
    (if 3 ≤ innerCandidates738_5.length then search edge 3 [5641, 7687] innerCandidates738_5 13328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan738_6

def innerTail738_4 : List Nat := 7027 :: innerTail738_5
def innerCandidates738_4 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates738_4_checked : candidates edge 4 5641 7027 innerTail738_5 best.bound = innerCandidates738_4 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan738_4 : scan edge (search edge 3) 4 [5641] 5641 innerTail738_4 best = best := by
  rw [innerTail738_4, scan, if_neg (by decide), innerCandidates738_4_checked]
  change scan edge (search edge 3) 4 [5641] 5641 innerTail738_5
    (if 3 ≤ innerCandidates738_4.length then search edge 3 [5641, 7027] innerCandidates738_4 12668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan738_5

def innerTail738_3 : List Nat := 6793 :: innerTail738_4
def innerCandidates738_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates738_3_checked : candidates edge 4 5641 6793 innerTail738_4 best.bound = innerCandidates738_3 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan738_3 : scan edge (search edge 3) 4 [5641] 5641 innerTail738_3 best = best := by
  rw [innerTail738_3, scan, if_neg (by decide), innerCandidates738_3_checked]
  change scan edge (search edge 3) 4 [5641] 5641 innerTail738_4
    (if 3 ≤ innerCandidates738_3.length then search edge 3 [5641, 6793] innerCandidates738_3 12434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan738_4

def innerTail738_2 : List Nat := 6637 :: innerTail738_3
def innerCandidates738_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates738_2_checked : candidates edge 4 5641 6637 innerTail738_3 best.bound = innerCandidates738_2 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan738_2 : scan edge (search edge 3) 4 [5641] 5641 innerTail738_2 best = best := by
  rw [innerTail738_2, scan, if_neg (by decide), innerCandidates738_2_checked]
  change scan edge (search edge 3) 4 [5641] 5641 innerTail738_3
    (if 3 ≤ innerCandidates738_2.length then search edge 3 [5641, 6637] innerCandidates738_2 12278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan738_3

def innerTail738_1 : List Nat := 6007 :: innerTail738_2
def innerCandidates738_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates738_1_checked : candidates edge 4 5641 6007 innerTail738_2 best.bound = innerCandidates738_1 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan738_1 : scan edge (search edge 3) 4 [5641] 5641 innerTail738_1 best = best := by
  rw [innerTail738_1, scan, if_neg (by decide), innerCandidates738_1_checked]
  change scan edge (search edge 3) 4 [5641] 5641 innerTail738_2
    (if 3 ≤ innerCandidates738_1.length then search edge 3 [5641, 6007] innerCandidates738_1 11648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan738_2

def innerTail738_0 : List Nat := 5743 :: innerTail738_1
def innerCandidates738_0 : List Nat := [7027, 7759]
set_option maxRecDepth 4096 in
theorem innerCandidates738_0_checked : candidates edge 4 5641 5743 innerTail738_1 best.bound = innerCandidates738_0 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan738_0 : scan edge (search edge 3) 4 [5641] 5641 innerTail738_0 best = best := by
  rw [innerTail738_0, scan, if_neg (by decide), innerCandidates738_0_checked]
  change scan edge (search edge 3) 4 [5641] 5641 innerTail738_1
    (if 3 ≤ innerCandidates738_0.length then search edge 3 [5641, 5743] innerCandidates738_0 11384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan738_1

theorem rootBranch738 : search edge 4 [5641] rootCandidates738 5641 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5641] 5641 innerTail738_0 best = best
  exact innerScan738_0

def rootCandidates737 : List Nat := [6563, 6737, 6827, 6833, 6983, 7079, 7877, 8039]
set_option maxRecDepth 4096 in
theorem rootCandidates737_checked : candidates edge 5 0 5639 tail738 best.bound = rootCandidates737 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext q; rfl
  rw [row]
  decide

def innerTail737_8 : List Nat := []
theorem innerScan737_8 : scan edge (search edge 3) 4 [5639] 5639 innerTail737_8 best = best := rfl

def innerTail737_7 : List Nat := 8039 :: innerTail737_8
def innerCandidates737_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates737_7_checked : candidates edge 4 5639 8039 innerTail737_8 best.bound = innerCandidates737_7 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan737_7 : scan edge (search edge 3) 4 [5639] 5639 innerTail737_7 best = best := by
  rw [innerTail737_7, scan, if_neg (by decide), innerCandidates737_7_checked]
  change scan edge (search edge 3) 4 [5639] 5639 innerTail737_8
    (if 3 ≤ innerCandidates737_7.length then search edge 3 [5639, 8039] innerCandidates737_7 13678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan737_8

def innerTail737_6 : List Nat := 7877 :: innerTail737_7
def innerCandidates737_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates737_6_checked : candidates edge 4 5639 7877 innerTail737_7 best.bound = innerCandidates737_6 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan737_6 : scan edge (search edge 3) 4 [5639] 5639 innerTail737_6 best = best := by
  rw [innerTail737_6, scan, if_neg (by decide), innerCandidates737_6_checked]
  change scan edge (search edge 3) 4 [5639] 5639 innerTail737_7
    (if 3 ≤ innerCandidates737_6.length then search edge 3 [5639, 7877] innerCandidates737_6 13516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan737_7

def innerTail737_5 : List Nat := 7079 :: innerTail737_6
def innerCandidates737_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates737_5_checked : candidates edge 4 5639 7079 innerTail737_6 best.bound = innerCandidates737_5 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan737_5 : scan edge (search edge 3) 4 [5639] 5639 innerTail737_5 best = best := by
  rw [innerTail737_5, scan, if_neg (by decide), innerCandidates737_5_checked]
  change scan edge (search edge 3) 4 [5639] 5639 innerTail737_6
    (if 3 ≤ innerCandidates737_5.length then search edge 3 [5639, 7079] innerCandidates737_5 12718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan737_6

def innerTail737_4 : List Nat := 6983 :: innerTail737_5
def innerCandidates737_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates737_4_checked : candidates edge 4 5639 6983 innerTail737_5 best.bound = innerCandidates737_4 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan737_4 : scan edge (search edge 3) 4 [5639] 5639 innerTail737_4 best = best := by
  rw [innerTail737_4, scan, if_neg (by decide), innerCandidates737_4_checked]
  change scan edge (search edge 3) 4 [5639] 5639 innerTail737_5
    (if 3 ≤ innerCandidates737_4.length then search edge 3 [5639, 6983] innerCandidates737_4 12622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan737_5

def innerTail737_3 : List Nat := 6833 :: innerTail737_4
def innerCandidates737_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates737_3_checked : candidates edge 4 5639 6833 innerTail737_4 best.bound = innerCandidates737_3 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan737_3 : scan edge (search edge 3) 4 [5639] 5639 innerTail737_3 best = best := by
  rw [innerTail737_3, scan, if_neg (by decide), innerCandidates737_3_checked]
  change scan edge (search edge 3) 4 [5639] 5639 innerTail737_4
    (if 3 ≤ innerCandidates737_3.length then search edge 3 [5639, 6833] innerCandidates737_3 12472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan737_4

def innerTail737_2 : List Nat := 6827 :: innerTail737_3
def innerCandidates737_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates737_2_checked : candidates edge 4 5639 6827 innerTail737_3 best.bound = innerCandidates737_2 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan737_2 : scan edge (search edge 3) 4 [5639] 5639 innerTail737_2 best = best := by
  rw [innerTail737_2, scan, if_neg (by decide), innerCandidates737_2_checked]
  change scan edge (search edge 3) 4 [5639] 5639 innerTail737_3
    (if 3 ≤ innerCandidates737_2.length then search edge 3 [5639, 6827] innerCandidates737_2 12466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan737_3

def innerTail737_1 : List Nat := 6737 :: innerTail737_2
def innerCandidates737_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates737_1_checked : candidates edge 4 5639 6737 innerTail737_2 best.bound = innerCandidates737_1 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan737_1 : scan edge (search edge 3) 4 [5639] 5639 innerTail737_1 best = best := by
  rw [innerTail737_1, scan, if_neg (by decide), innerCandidates737_1_checked]
  change scan edge (search edge 3) 4 [5639] 5639 innerTail737_2
    (if 3 ≤ innerCandidates737_1.length then search edge 3 [5639, 6737] innerCandidates737_1 12376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan737_2

def innerTail737_0 : List Nat := 6563 :: innerTail737_1
def innerCandidates737_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates737_0_checked : candidates edge 4 5639 6563 innerTail737_1 best.bound = innerCandidates737_0 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan737_0 : scan edge (search edge 3) 4 [5639] 5639 innerTail737_0 best = best := by
  rw [innerTail737_0, scan, if_neg (by decide), innerCandidates737_0_checked]
  change scan edge (search edge 3) 4 [5639] 5639 innerTail737_1
    (if 3 ≤ innerCandidates737_0.length then search edge 3 [5639, 6563] innerCandidates737_0 12202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan737_1

theorem rootBranch737 : search edge 4 [5639] rootCandidates737 5639 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5639] 5639 innerTail737_0 best = best
  exact innerScan737_0

end ProjectEulerSolutions.PrimePairUpper.Search
