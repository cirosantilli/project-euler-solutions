import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates520 : List Nat := [3767, 4229, 4391, 4799, 5807, 5843, 6287, 7499, 7523, 7793, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates520_checked : candidates edge 5 0 3761 tail521 best.bound = rootCandidates520 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext q; rfl
  rw [row]
  decide

def innerTail520_11 : List Nat := []
theorem innerScan520_11 : scan edge (search edge 3) 4 [3761] 3761 innerTail520_11 best = best := rfl

def innerTail520_10 : List Nat := 8369 :: innerTail520_11
def innerCandidates520_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates520_10_checked : candidates edge 4 3761 8369 innerTail520_11 best.bound = innerCandidates520_10 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan520_10 : scan edge (search edge 3) 4 [3761] 3761 innerTail520_10 best = best := by
  rw [innerTail520_10, scan, if_neg (by decide), innerCandidates520_10_checked]
  change scan edge (search edge 3) 4 [3761] 3761 innerTail520_11
    (if 3 ≤ innerCandidates520_10.length then search edge 3 [3761, 8369] innerCandidates520_10 12130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan520_11

def innerTail520_9 : List Nat := 7793 :: innerTail520_10
def innerCandidates520_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates520_9_checked : candidates edge 4 3761 7793 innerTail520_10 best.bound = innerCandidates520_9 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan520_9 : scan edge (search edge 3) 4 [3761] 3761 innerTail520_9 best = best := by
  rw [innerTail520_9, scan, if_neg (by decide), innerCandidates520_9_checked]
  change scan edge (search edge 3) 4 [3761] 3761 innerTail520_10
    (if 3 ≤ innerCandidates520_9.length then search edge 3 [3761, 7793] innerCandidates520_9 11554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan520_10

def innerTail520_8 : List Nat := 7523 :: innerTail520_9
def innerCandidates520_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates520_8_checked : candidates edge 4 3761 7523 innerTail520_9 best.bound = innerCandidates520_8 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan520_8 : scan edge (search edge 3) 4 [3761] 3761 innerTail520_8 best = best := by
  rw [innerTail520_8, scan, if_neg (by decide), innerCandidates520_8_checked]
  change scan edge (search edge 3) 4 [3761] 3761 innerTail520_9
    (if 3 ≤ innerCandidates520_8.length then search edge 3 [3761, 7523] innerCandidates520_8 11284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan520_9

def innerTail520_7 : List Nat := 7499 :: innerTail520_8
def innerCandidates520_7 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates520_7_checked : candidates edge 4 3761 7499 innerTail520_8 best.bound = innerCandidates520_7 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan520_7 : scan edge (search edge 3) 4 [3761] 3761 innerTail520_7 best = best := by
  rw [innerTail520_7, scan, if_neg (by decide), innerCandidates520_7_checked]
  change scan edge (search edge 3) 4 [3761] 3761 innerTail520_8
    (if 3 ≤ innerCandidates520_7.length then search edge 3 [3761, 7499] innerCandidates520_7 11260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan520_8

def innerTail520_6 : List Nat := 6287 :: innerTail520_7
def innerCandidates520_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates520_6_checked : candidates edge 4 3761 6287 innerTail520_7 best.bound = innerCandidates520_6 := by
  unfold candidates
  have row : edge 6287 = SearchData.row6287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan520_6 : scan edge (search edge 3) 4 [3761] 3761 innerTail520_6 best = best := by
  rw [innerTail520_6, scan, if_neg (by decide), innerCandidates520_6_checked]
  change scan edge (search edge 3) 4 [3761] 3761 innerTail520_7
    (if 3 ≤ innerCandidates520_6.length then search edge 3 [3761, 6287] innerCandidates520_6 10048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan520_7

def innerTail520_5 : List Nat := 5843 :: innerTail520_6
def innerCandidates520_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates520_5_checked : candidates edge 4 3761 5843 innerTail520_6 best.bound = innerCandidates520_5 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan520_5 : scan edge (search edge 3) 4 [3761] 3761 innerTail520_5 best = best := by
  rw [innerTail520_5, scan, if_neg (by decide), innerCandidates520_5_checked]
  change scan edge (search edge 3) 4 [3761] 3761 innerTail520_6
    (if 3 ≤ innerCandidates520_5.length then search edge 3 [3761, 5843] innerCandidates520_5 9604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan520_6

def innerTail520_4 : List Nat := 5807 :: innerTail520_5
def innerCandidates520_4 : List Nat := [7499]
set_option maxRecDepth 4096 in
theorem innerCandidates520_4_checked : candidates edge 4 3761 5807 innerTail520_5 best.bound = innerCandidates520_4 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan520_4 : scan edge (search edge 3) 4 [3761] 3761 innerTail520_4 best = best := by
  rw [innerTail520_4, scan, if_neg (by decide), innerCandidates520_4_checked]
  change scan edge (search edge 3) 4 [3761] 3761 innerTail520_5
    (if 3 ≤ innerCandidates520_4.length then search edge 3 [3761, 5807] innerCandidates520_4 9568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan520_5

def innerTail520_3 : List Nat := 4799 :: innerTail520_4
def innerCandidates520_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates520_3_checked : candidates edge 4 3761 4799 innerTail520_4 best.bound = innerCandidates520_3 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan520_3 : scan edge (search edge 3) 4 [3761] 3761 innerTail520_3 best = best := by
  rw [innerTail520_3, scan, if_neg (by decide), innerCandidates520_3_checked]
  change scan edge (search edge 3) 4 [3761] 3761 innerTail520_4
    (if 3 ≤ innerCandidates520_3.length then search edge 3 [3761, 4799] innerCandidates520_3 8560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan520_4

def innerTail520_2 : List Nat := 4391 :: innerTail520_3
def innerCandidates520_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates520_2_checked : candidates edge 4 3761 4391 innerTail520_3 best.bound = innerCandidates520_2 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan520_2 : scan edge (search edge 3) 4 [3761] 3761 innerTail520_2 best = best := by
  rw [innerTail520_2, scan, if_neg (by decide), innerCandidates520_2_checked]
  change scan edge (search edge 3) 4 [3761] 3761 innerTail520_3
    (if 3 ≤ innerCandidates520_2.length then search edge 3 [3761, 4391] innerCandidates520_2 8152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan520_3

def innerTail520_1 : List Nat := 4229 :: innerTail520_2
def innerCandidates520_1 : List Nat := [4391]
set_option maxRecDepth 4096 in
theorem innerCandidates520_1_checked : candidates edge 4 3761 4229 innerTail520_2 best.bound = innerCandidates520_1 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan520_1 : scan edge (search edge 3) 4 [3761] 3761 innerTail520_1 best = best := by
  rw [innerTail520_1, scan, if_neg (by decide), innerCandidates520_1_checked]
  change scan edge (search edge 3) 4 [3761] 3761 innerTail520_2
    (if 3 ≤ innerCandidates520_1.length then search edge 3 [3761, 4229] innerCandidates520_1 7990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan520_2

def innerTail520_0 : List Nat := 3767 :: innerTail520_1
def innerCandidates520_0 : List Nat := [7793]
set_option maxRecDepth 4096 in
theorem innerCandidates520_0_checked : candidates edge 4 3761 3767 innerTail520_1 best.bound = innerCandidates520_0 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan520_0 : scan edge (search edge 3) 4 [3761] 3761 innerTail520_0 best = best := by
  rw [innerTail520_0, scan, if_neg (by decide), innerCandidates520_0_checked]
  change scan edge (search edge 3) 4 [3761] 3761 innerTail520_1
    (if 3 ≤ innerCandidates520_0.length then search edge 3 [3761, 3767] innerCandidates520_0 7528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan520_1

theorem rootBranch520 : search edge 4 [3761] rootCandidates520 3761 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3761] 3761 innerTail520_0 best = best
  exact innerScan520_0

def rootCandidates519 : List Nat := [4099, 5107, 5233, 5851, 6637, 6679, 6763, 6871, 8233, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates519_checked : candidates edge 5 0 3739 tail520 best.bound = rootCandidates519 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext q; rfl
  rw [row]
  decide

def innerTail519_10 : List Nat := []
theorem innerScan519_10 : scan edge (search edge 3) 4 [3739] 3739 innerTail519_10 best = best := rfl

def innerTail519_9 : List Nat := 8353 :: innerTail519_10
def innerCandidates519_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates519_9_checked : candidates edge 4 3739 8353 innerTail519_10 best.bound = innerCandidates519_9 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan519_9 : scan edge (search edge 3) 4 [3739] 3739 innerTail519_9 best = best := by
  rw [innerTail519_9, scan, if_neg (by decide), innerCandidates519_9_checked]
  change scan edge (search edge 3) 4 [3739] 3739 innerTail519_10
    (if 3 ≤ innerCandidates519_9.length then search edge 3 [3739, 8353] innerCandidates519_9 12092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan519_10

def innerTail519_8 : List Nat := 8233 :: innerTail519_9
def innerCandidates519_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates519_8_checked : candidates edge 4 3739 8233 innerTail519_9 best.bound = innerCandidates519_8 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan519_8 : scan edge (search edge 3) 4 [3739] 3739 innerTail519_8 best = best := by
  rw [innerTail519_8, scan, if_neg (by decide), innerCandidates519_8_checked]
  change scan edge (search edge 3) 4 [3739] 3739 innerTail519_9
    (if 3 ≤ innerCandidates519_8.length then search edge 3 [3739, 8233] innerCandidates519_8 11972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan519_9

def innerTail519_7 : List Nat := 6871 :: innerTail519_8
def innerCandidates519_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates519_7_checked : candidates edge 4 3739 6871 innerTail519_8 best.bound = innerCandidates519_7 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan519_7 : scan edge (search edge 3) 4 [3739] 3739 innerTail519_7 best = best := by
  rw [innerTail519_7, scan, if_neg (by decide), innerCandidates519_7_checked]
  change scan edge (search edge 3) 4 [3739] 3739 innerTail519_8
    (if 3 ≤ innerCandidates519_7.length then search edge 3 [3739, 6871] innerCandidates519_7 10610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan519_8

def innerTail519_6 : List Nat := 6763 :: innerTail519_7
def innerCandidates519_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates519_6_checked : candidates edge 4 3739 6763 innerTail519_7 best.bound = innerCandidates519_6 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan519_6 : scan edge (search edge 3) 4 [3739] 3739 innerTail519_6 best = best := by
  rw [innerTail519_6, scan, if_neg (by decide), innerCandidates519_6_checked]
  change scan edge (search edge 3) 4 [3739] 3739 innerTail519_7
    (if 3 ≤ innerCandidates519_6.length then search edge 3 [3739, 6763] innerCandidates519_6 10502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan519_7

def innerTail519_5 : List Nat := 6679 :: innerTail519_6
def innerCandidates519_5 : List Nat := [8353]
set_option maxRecDepth 4096 in
theorem innerCandidates519_5_checked : candidates edge 4 3739 6679 innerTail519_6 best.bound = innerCandidates519_5 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan519_5 : scan edge (search edge 3) 4 [3739] 3739 innerTail519_5 best = best := by
  rw [innerTail519_5, scan, if_neg (by decide), innerCandidates519_5_checked]
  change scan edge (search edge 3) 4 [3739] 3739 innerTail519_6
    (if 3 ≤ innerCandidates519_5.length then search edge 3 [3739, 6679] innerCandidates519_5 10418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan519_6

def innerTail519_4 : List Nat := 6637 :: innerTail519_5
def innerCandidates519_4 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates519_4_checked : candidates edge 4 3739 6637 innerTail519_5 best.bound = innerCandidates519_4 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan519_4 : scan edge (search edge 3) 4 [3739] 3739 innerTail519_4 best = best := by
  rw [innerTail519_4, scan, if_neg (by decide), innerCandidates519_4_checked]
  change scan edge (search edge 3) 4 [3739] 3739 innerTail519_5
    (if 3 ≤ innerCandidates519_4.length then search edge 3 [3739, 6637] innerCandidates519_4 10376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan519_5

def innerTail519_3 : List Nat := 5851 :: innerTail519_4
def innerCandidates519_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates519_3_checked : candidates edge 4 3739 5851 innerTail519_4 best.bound = innerCandidates519_3 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan519_3 : scan edge (search edge 3) 4 [3739] 3739 innerTail519_3 best = best := by
  rw [innerTail519_3, scan, if_neg (by decide), innerCandidates519_3_checked]
  change scan edge (search edge 3) 4 [3739] 3739 innerTail519_4
    (if 3 ≤ innerCandidates519_3.length then search edge 3 [3739, 5851] innerCandidates519_3 9590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan519_4

def innerTail519_2 : List Nat := 5233 :: innerTail519_3
def innerCandidates519_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates519_2_checked : candidates edge 4 3739 5233 innerTail519_3 best.bound = innerCandidates519_2 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan519_2 : scan edge (search edge 3) 4 [3739] 3739 innerTail519_2 best = best := by
  rw [innerTail519_2, scan, if_neg (by decide), innerCandidates519_2_checked]
  change scan edge (search edge 3) 4 [3739] 3739 innerTail519_3
    (if 3 ≤ innerCandidates519_2.length then search edge 3 [3739, 5233] innerCandidates519_2 8972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan519_3

def innerTail519_1 : List Nat := 5107 :: innerTail519_2
def innerCandidates519_1 : List Nat := [6763]
set_option maxRecDepth 4096 in
theorem innerCandidates519_1_checked : candidates edge 4 3739 5107 innerTail519_2 best.bound = innerCandidates519_1 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan519_1 : scan edge (search edge 3) 4 [3739] 3739 innerTail519_1 best = best := by
  rw [innerTail519_1, scan, if_neg (by decide), innerCandidates519_1_checked]
  change scan edge (search edge 3) 4 [3739] 3739 innerTail519_2
    (if 3 ≤ innerCandidates519_1.length then search edge 3 [3739, 5107] innerCandidates519_1 8846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan519_2

def innerTail519_0 : List Nat := 4099 :: innerTail519_1
def innerCandidates519_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates519_0_checked : candidates edge 4 3739 4099 innerTail519_1 best.bound = innerCandidates519_0 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan519_0 : scan edge (search edge 3) 4 [3739] 3739 innerTail519_0 best = best := by
  rw [innerTail519_0, scan, if_neg (by decide), innerCandidates519_0_checked]
  change scan edge (search edge 3) 4 [3739] 3739 innerTail519_1
    (if 3 ≤ innerCandidates519_0.length then search edge 3 [3739, 4099] innerCandidates519_0 7838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan519_1

theorem rootBranch519 : search edge 4 [3739] rootCandidates519 3739 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3739] 3739 innerTail519_0 best = best
  exact innerScan519_0

def rootCandidates518 : List Nat := [4657, 5521, 6151, 6421, 6451, 6529, 7687, 8089]
set_option maxRecDepth 4096 in
theorem rootCandidates518_checked : candidates edge 5 0 3733 tail519 best.bound = rootCandidates518 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext q; rfl
  rw [row]
  decide

def innerTail518_8 : List Nat := []
theorem innerScan518_8 : scan edge (search edge 3) 4 [3733] 3733 innerTail518_8 best = best := rfl

def innerTail518_7 : List Nat := 8089 :: innerTail518_8
def innerCandidates518_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates518_7_checked : candidates edge 4 3733 8089 innerTail518_8 best.bound = innerCandidates518_7 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan518_7 : scan edge (search edge 3) 4 [3733] 3733 innerTail518_7 best = best := by
  rw [innerTail518_7, scan, if_neg (by decide), innerCandidates518_7_checked]
  change scan edge (search edge 3) 4 [3733] 3733 innerTail518_8
    (if 3 ≤ innerCandidates518_7.length then search edge 3 [3733, 8089] innerCandidates518_7 11822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan518_8

def innerTail518_6 : List Nat := 7687 :: innerTail518_7
def innerCandidates518_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates518_6_checked : candidates edge 4 3733 7687 innerTail518_7 best.bound = innerCandidates518_6 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan518_6 : scan edge (search edge 3) 4 [3733] 3733 innerTail518_6 best = best := by
  rw [innerTail518_6, scan, if_neg (by decide), innerCandidates518_6_checked]
  change scan edge (search edge 3) 4 [3733] 3733 innerTail518_7
    (if 3 ≤ innerCandidates518_6.length then search edge 3 [3733, 7687] innerCandidates518_6 11420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan518_7

def innerTail518_5 : List Nat := 6529 :: innerTail518_6
def innerCandidates518_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates518_5_checked : candidates edge 4 3733 6529 innerTail518_6 best.bound = innerCandidates518_5 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan518_5 : scan edge (search edge 3) 4 [3733] 3733 innerTail518_5 best = best := by
  rw [innerTail518_5, scan, if_neg (by decide), innerCandidates518_5_checked]
  change scan edge (search edge 3) 4 [3733] 3733 innerTail518_6
    (if 3 ≤ innerCandidates518_5.length then search edge 3 [3733, 6529] innerCandidates518_5 10262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan518_6

def innerTail518_4 : List Nat := 6451 :: innerTail518_5
def innerCandidates518_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates518_4_checked : candidates edge 4 3733 6451 innerTail518_5 best.bound = innerCandidates518_4 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan518_4 : scan edge (search edge 3) 4 [3733] 3733 innerTail518_4 best = best := by
  rw [innerTail518_4, scan, if_neg (by decide), innerCandidates518_4_checked]
  change scan edge (search edge 3) 4 [3733] 3733 innerTail518_5
    (if 3 ≤ innerCandidates518_4.length then search edge 3 [3733, 6451] innerCandidates518_4 10184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan518_5

def innerTail518_3 : List Nat := 6421 :: innerTail518_4
def innerCandidates518_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates518_3_checked : candidates edge 4 3733 6421 innerTail518_4 best.bound = innerCandidates518_3 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan518_3 : scan edge (search edge 3) 4 [3733] 3733 innerTail518_3 best = best := by
  rw [innerTail518_3, scan, if_neg (by decide), innerCandidates518_3_checked]
  change scan edge (search edge 3) 4 [3733] 3733 innerTail518_4
    (if 3 ≤ innerCandidates518_3.length then search edge 3 [3733, 6421] innerCandidates518_3 10154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan518_4

def innerTail518_2 : List Nat := 6151 :: innerTail518_3
def innerCandidates518_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates518_2_checked : candidates edge 4 3733 6151 innerTail518_3 best.bound = innerCandidates518_2 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan518_2 : scan edge (search edge 3) 4 [3733] 3733 innerTail518_2 best = best := by
  rw [innerTail518_2, scan, if_neg (by decide), innerCandidates518_2_checked]
  change scan edge (search edge 3) 4 [3733] 3733 innerTail518_3
    (if 3 ≤ innerCandidates518_2.length then search edge 3 [3733, 6151] innerCandidates518_2 9884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan518_3

def innerTail518_1 : List Nat := 5521 :: innerTail518_2
def innerCandidates518_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates518_1_checked : candidates edge 4 3733 5521 innerTail518_2 best.bound = innerCandidates518_1 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan518_1 : scan edge (search edge 3) 4 [3733] 3733 innerTail518_1 best = best := by
  rw [innerTail518_1, scan, if_neg (by decide), innerCandidates518_1_checked]
  change scan edge (search edge 3) 4 [3733] 3733 innerTail518_2
    (if 3 ≤ innerCandidates518_1.length then search edge 3 [3733, 5521] innerCandidates518_1 9254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan518_2

def innerTail518_0 : List Nat := 4657 :: innerTail518_1
def innerCandidates518_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates518_0_checked : candidates edge 4 3733 4657 innerTail518_1 best.bound = innerCandidates518_0 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan518_0 : scan edge (search edge 3) 4 [3733] 3733 innerTail518_0 best = best := by
  rw [innerTail518_0, scan, if_neg (by decide), innerCandidates518_0_checked]
  change scan edge (search edge 3) 4 [3733] 3733 innerTail518_1
    (if 3 ≤ innerCandidates518_0.length then search edge 3 [3733, 4657] innerCandidates518_0 8390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan518_1

theorem rootBranch518 : search edge 4 [3733] rootCandidates518 3733 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3733] 3733 innerTail518_0 best = best
  exact innerScan518_0

def rootCandidates517 : List Nat := [4129, 4639, 5011, 5659, 6037, 6949, 6961, 7699]
set_option maxRecDepth 4096 in
theorem rootCandidates517_checked : candidates edge 5 0 3727 tail518 best.bound = rootCandidates517 := by
  unfold candidates
  have row : edge 3727 = SearchData.row3727.contains := by funext q; rfl
  rw [row]
  decide

def innerTail517_8 : List Nat := []
theorem innerScan517_8 : scan edge (search edge 3) 4 [3727] 3727 innerTail517_8 best = best := rfl

def innerTail517_7 : List Nat := 7699 :: innerTail517_8
def innerCandidates517_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates517_7_checked : candidates edge 4 3727 7699 innerTail517_8 best.bound = innerCandidates517_7 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan517_7 : scan edge (search edge 3) 4 [3727] 3727 innerTail517_7 best = best := by
  rw [innerTail517_7, scan, if_neg (by decide), innerCandidates517_7_checked]
  change scan edge (search edge 3) 4 [3727] 3727 innerTail517_8
    (if 3 ≤ innerCandidates517_7.length then search edge 3 [3727, 7699] innerCandidates517_7 11426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan517_8

def innerTail517_6 : List Nat := 6961 :: innerTail517_7
def innerCandidates517_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates517_6_checked : candidates edge 4 3727 6961 innerTail517_7 best.bound = innerCandidates517_6 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan517_6 : scan edge (search edge 3) 4 [3727] 3727 innerTail517_6 best = best := by
  rw [innerTail517_6, scan, if_neg (by decide), innerCandidates517_6_checked]
  change scan edge (search edge 3) 4 [3727] 3727 innerTail517_7
    (if 3 ≤ innerCandidates517_6.length then search edge 3 [3727, 6961] innerCandidates517_6 10688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan517_7

def innerTail517_5 : List Nat := 6949 :: innerTail517_6
def innerCandidates517_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates517_5_checked : candidates edge 4 3727 6949 innerTail517_6 best.bound = innerCandidates517_5 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan517_5 : scan edge (search edge 3) 4 [3727] 3727 innerTail517_5 best = best := by
  rw [innerTail517_5, scan, if_neg (by decide), innerCandidates517_5_checked]
  change scan edge (search edge 3) 4 [3727] 3727 innerTail517_6
    (if 3 ≤ innerCandidates517_5.length then search edge 3 [3727, 6949] innerCandidates517_5 10676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan517_6

def innerTail517_4 : List Nat := 6037 :: innerTail517_5
def innerCandidates517_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates517_4_checked : candidates edge 4 3727 6037 innerTail517_5 best.bound = innerCandidates517_4 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan517_4 : scan edge (search edge 3) 4 [3727] 3727 innerTail517_4 best = best := by
  rw [innerTail517_4, scan, if_neg (by decide), innerCandidates517_4_checked]
  change scan edge (search edge 3) 4 [3727] 3727 innerTail517_5
    (if 3 ≤ innerCandidates517_4.length then search edge 3 [3727, 6037] innerCandidates517_4 9764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan517_5

def innerTail517_3 : List Nat := 5659 :: innerTail517_4
def innerCandidates517_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates517_3_checked : candidates edge 4 3727 5659 innerTail517_4 best.bound = innerCandidates517_3 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan517_3 : scan edge (search edge 3) 4 [3727] 3727 innerTail517_3 best = best := by
  rw [innerTail517_3, scan, if_neg (by decide), innerCandidates517_3_checked]
  change scan edge (search edge 3) 4 [3727] 3727 innerTail517_4
    (if 3 ≤ innerCandidates517_3.length then search edge 3 [3727, 5659] innerCandidates517_3 9386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan517_4

def innerTail517_2 : List Nat := 5011 :: innerTail517_3
def innerCandidates517_2 : List Nat := [6949, 6961]
set_option maxRecDepth 4096 in
theorem innerCandidates517_2_checked : candidates edge 4 3727 5011 innerTail517_3 best.bound = innerCandidates517_2 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan517_2 : scan edge (search edge 3) 4 [3727] 3727 innerTail517_2 best = best := by
  rw [innerTail517_2, scan, if_neg (by decide), innerCandidates517_2_checked]
  change scan edge (search edge 3) 4 [3727] 3727 innerTail517_3
    (if 3 ≤ innerCandidates517_2.length then search edge 3 [3727, 5011] innerCandidates517_2 8738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan517_3

def innerTail517_1 : List Nat := 4639 :: innerTail517_2
def innerCandidates517_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates517_1_checked : candidates edge 4 3727 4639 innerTail517_2 best.bound = innerCandidates517_1 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan517_1 : scan edge (search edge 3) 4 [3727] 3727 innerTail517_1 best = best := by
  rw [innerTail517_1, scan, if_neg (by decide), innerCandidates517_1_checked]
  change scan edge (search edge 3) 4 [3727] 3727 innerTail517_2
    (if 3 ≤ innerCandidates517_1.length then search edge 3 [3727, 4639] innerCandidates517_1 8366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan517_2

def innerTail517_0 : List Nat := 4129 :: innerTail517_1
def innerCandidates517_0 : List Nat := [6949]
set_option maxRecDepth 4096 in
theorem innerCandidates517_0_checked : candidates edge 4 3727 4129 innerTail517_1 best.bound = innerCandidates517_0 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan517_0 : scan edge (search edge 3) 4 [3727] 3727 innerTail517_0 best = best := by
  rw [innerTail517_0, scan, if_neg (by decide), innerCandidates517_0_checked]
  change scan edge (search edge 3) 4 [3727] 3727 innerTail517_1
    (if 3 ≤ innerCandidates517_0.length then search edge 3 [3727, 4129] innerCandidates517_0 7856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan517_1

theorem rootBranch517 : search edge 4 [3727] rootCandidates517 3727 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3727] 3727 innerTail517_0 best = best
  exact innerScan517_0

def rootCandidates516 : List Nat := [4271, 5273, 5297, 5591, 6791]
set_option maxRecDepth 4096 in
theorem rootCandidates516_checked : candidates edge 5 0 3719 tail517 best.bound = rootCandidates516 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext q; rfl
  rw [row]
  decide

def innerTail516_5 : List Nat := []
theorem innerScan516_5 : scan edge (search edge 3) 4 [3719] 3719 innerTail516_5 best = best := rfl

def innerTail516_4 : List Nat := 6791 :: innerTail516_5
def innerCandidates516_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates516_4_checked : candidates edge 4 3719 6791 innerTail516_5 best.bound = innerCandidates516_4 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan516_4 : scan edge (search edge 3) 4 [3719] 3719 innerTail516_4 best = best := by
  rw [innerTail516_4, scan, if_neg (by decide), innerCandidates516_4_checked]
  change scan edge (search edge 3) 4 [3719] 3719 innerTail516_5
    (if 3 ≤ innerCandidates516_4.length then search edge 3 [3719, 6791] innerCandidates516_4 10510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan516_5

def innerTail516_3 : List Nat := 5591 :: innerTail516_4
def innerCandidates516_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates516_3_checked : candidates edge 4 3719 5591 innerTail516_4 best.bound = innerCandidates516_3 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan516_3 : scan edge (search edge 3) 4 [3719] 3719 innerTail516_3 best = best := by
  rw [innerTail516_3, scan, if_neg (by decide), innerCandidates516_3_checked]
  change scan edge (search edge 3) 4 [3719] 3719 innerTail516_4
    (if 3 ≤ innerCandidates516_3.length then search edge 3 [3719, 5591] innerCandidates516_3 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan516_4

def innerTail516_2 : List Nat := 5297 :: innerTail516_3
def innerCandidates516_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates516_2_checked : candidates edge 4 3719 5297 innerTail516_3 best.bound = innerCandidates516_2 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan516_2 : scan edge (search edge 3) 4 [3719] 3719 innerTail516_2 best = best := by
  rw [innerTail516_2, scan, if_neg (by decide), innerCandidates516_2_checked]
  change scan edge (search edge 3) 4 [3719] 3719 innerTail516_3
    (if 3 ≤ innerCandidates516_2.length then search edge 3 [3719, 5297] innerCandidates516_2 9016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan516_3

def innerTail516_1 : List Nat := 5273 :: innerTail516_2
def innerCandidates516_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates516_1_checked : candidates edge 4 3719 5273 innerTail516_2 best.bound = innerCandidates516_1 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan516_1 : scan edge (search edge 3) 4 [3719] 3719 innerTail516_1 best = best := by
  rw [innerTail516_1, scan, if_neg (by decide), innerCandidates516_1_checked]
  change scan edge (search edge 3) 4 [3719] 3719 innerTail516_2
    (if 3 ≤ innerCandidates516_1.length then search edge 3 [3719, 5273] innerCandidates516_1 8992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan516_2

def innerTail516_0 : List Nat := 4271 :: innerTail516_1
def innerCandidates516_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates516_0_checked : candidates edge 4 3719 4271 innerTail516_1 best.bound = innerCandidates516_0 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan516_0 : scan edge (search edge 3) 4 [3719] 3719 innerTail516_0 best = best := by
  rw [innerTail516_0, scan, if_neg (by decide), innerCandidates516_0_checked]
  change scan edge (search edge 3) 4 [3719] 3719 innerTail516_1
    (if 3 ≤ innerCandidates516_0.length then search edge 3 [3719, 4271] innerCandidates516_0 7990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan516_1

theorem rootBranch516 : search edge 4 [3719] rootCandidates516 3719 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3719] 3719 innerTail516_0 best = best
  exact innerScan516_0

def rootCandidates515 : List Nat := [4231, 4261, 4621, 6991, 7369, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates515_checked : candidates edge 5 0 3709 tail516 best.bound = rootCandidates515 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext q; rfl
  rw [row]
  decide

def innerTail515_6 : List Nat := []
theorem innerScan515_6 : scan edge (search edge 3) 4 [3709] 3709 innerTail515_6 best = best := rfl

def innerTail515_5 : List Nat := 8377 :: innerTail515_6
def innerCandidates515_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates515_5_checked : candidates edge 4 3709 8377 innerTail515_6 best.bound = innerCandidates515_5 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan515_5 : scan edge (search edge 3) 4 [3709] 3709 innerTail515_5 best = best := by
  rw [innerTail515_5, scan, if_neg (by decide), innerCandidates515_5_checked]
  change scan edge (search edge 3) 4 [3709] 3709 innerTail515_6
    (if 3 ≤ innerCandidates515_5.length then search edge 3 [3709, 8377] innerCandidates515_5 12086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan515_6

def innerTail515_4 : List Nat := 7369 :: innerTail515_5
def innerCandidates515_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates515_4_checked : candidates edge 4 3709 7369 innerTail515_5 best.bound = innerCandidates515_4 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan515_4 : scan edge (search edge 3) 4 [3709] 3709 innerTail515_4 best = best := by
  rw [innerTail515_4, scan, if_neg (by decide), innerCandidates515_4_checked]
  change scan edge (search edge 3) 4 [3709] 3709 innerTail515_5
    (if 3 ≤ innerCandidates515_4.length then search edge 3 [3709, 7369] innerCandidates515_4 11078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan515_5

def innerTail515_3 : List Nat := 6991 :: innerTail515_4
def innerCandidates515_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates515_3_checked : candidates edge 4 3709 6991 innerTail515_4 best.bound = innerCandidates515_3 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan515_3 : scan edge (search edge 3) 4 [3709] 3709 innerTail515_3 best = best := by
  rw [innerTail515_3, scan, if_neg (by decide), innerCandidates515_3_checked]
  change scan edge (search edge 3) 4 [3709] 3709 innerTail515_4
    (if 3 ≤ innerCandidates515_3.length then search edge 3 [3709, 6991] innerCandidates515_3 10700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan515_4

def innerTail515_2 : List Nat := 4621 :: innerTail515_3
def innerCandidates515_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates515_2_checked : candidates edge 4 3709 4621 innerTail515_3 best.bound = innerCandidates515_2 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan515_2 : scan edge (search edge 3) 4 [3709] 3709 innerTail515_2 best = best := by
  rw [innerTail515_2, scan, if_neg (by decide), innerCandidates515_2_checked]
  change scan edge (search edge 3) 4 [3709] 3709 innerTail515_3
    (if 3 ≤ innerCandidates515_2.length then search edge 3 [3709, 4621] innerCandidates515_2 8330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan515_3

def innerTail515_1 : List Nat := 4261 :: innerTail515_2
def innerCandidates515_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates515_1_checked : candidates edge 4 3709 4261 innerTail515_2 best.bound = innerCandidates515_1 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan515_1 : scan edge (search edge 3) 4 [3709] 3709 innerTail515_1 best = best := by
  rw [innerTail515_1, scan, if_neg (by decide), innerCandidates515_1_checked]
  change scan edge (search edge 3) 4 [3709] 3709 innerTail515_2
    (if 3 ≤ innerCandidates515_1.length then search edge 3 [3709, 4261] innerCandidates515_1 7970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan515_2

def innerTail515_0 : List Nat := 4231 :: innerTail515_1
def innerCandidates515_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates515_0_checked : candidates edge 4 3709 4231 innerTail515_1 best.bound = innerCandidates515_0 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan515_0 : scan edge (search edge 3) 4 [3709] 3709 innerTail515_0 best = best := by
  rw [innerTail515_0, scan, if_neg (by decide), innerCandidates515_0_checked]
  change scan edge (search edge 3) 4 [3709] 3709 innerTail515_1
    (if 3 ≤ innerCandidates515_0.length then search edge 3 [3709, 4231] innerCandidates515_0 7940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan515_1

theorem rootBranch515 : search edge 4 [3709] rootCandidates515 3709 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3709] 3709 innerTail515_0 best = best
  exact innerScan515_0

def rootCandidates514 : List Nat := [3797, 4283, 4493, 4877, 4919, 4931, 4973, 5081, 5309, 5351, 5501, 6299, 6869, 6911, 7829]
set_option maxRecDepth 4096 in
theorem rootCandidates514_checked : candidates edge 5 0 3701 tail515 best.bound = rootCandidates514 := by
  unfold candidates
  have row : edge 3701 = SearchData.row3701.contains := by funext q; rfl
  rw [row]
  decide

def innerTail514_15 : List Nat := []
theorem innerScan514_15 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_15 best = best := rfl

def innerTail514_14 : List Nat := 7829 :: innerTail514_15
def innerCandidates514_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates514_14_checked : candidates edge 4 3701 7829 innerTail514_15 best.bound = innerCandidates514_14 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_14 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_14 best = best := by
  rw [innerTail514_14, scan, if_neg (by decide), innerCandidates514_14_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_15
    (if 3 ≤ innerCandidates514_14.length then search edge 3 [3701, 7829] innerCandidates514_14 11530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_15

def innerTail514_13 : List Nat := 6911 :: innerTail514_14
def innerCandidates514_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates514_13_checked : candidates edge 4 3701 6911 innerTail514_14 best.bound = innerCandidates514_13 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_13 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_13 best = best := by
  rw [innerTail514_13, scan, if_neg (by decide), innerCandidates514_13_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_14
    (if 3 ≤ innerCandidates514_13.length then search edge 3 [3701, 6911] innerCandidates514_13 10612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_14

def innerTail514_12 : List Nat := 6869 :: innerTail514_13
def innerCandidates514_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates514_12_checked : candidates edge 4 3701 6869 innerTail514_13 best.bound = innerCandidates514_12 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_12 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_12 best = best := by
  rw [innerTail514_12, scan, if_neg (by decide), innerCandidates514_12_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_13
    (if 3 ≤ innerCandidates514_12.length then search edge 3 [3701, 6869] innerCandidates514_12 10570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_13

def innerTail514_11 : List Nat := 6299 :: innerTail514_12
def innerCandidates514_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates514_11_checked : candidates edge 4 3701 6299 innerTail514_12 best.bound = innerCandidates514_11 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_11 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_11 best = best := by
  rw [innerTail514_11, scan, if_neg (by decide), innerCandidates514_11_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_12
    (if 3 ≤ innerCandidates514_11.length then search edge 3 [3701, 6299] innerCandidates514_11 10000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_12

def innerTail514_10 : List Nat := 5501 :: innerTail514_11
def innerCandidates514_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates514_10_checked : candidates edge 4 3701 5501 innerTail514_11 best.bound = innerCandidates514_10 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_10 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_10 best = best := by
  rw [innerTail514_10, scan, if_neg (by decide), innerCandidates514_10_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_11
    (if 3 ≤ innerCandidates514_10.length then search edge 3 [3701, 5501] innerCandidates514_10 9202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_11

def innerTail514_9 : List Nat := 5351 :: innerTail514_10
def innerCandidates514_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates514_9_checked : candidates edge 4 3701 5351 innerTail514_10 best.bound = innerCandidates514_9 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_9 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_9 best = best := by
  rw [innerTail514_9, scan, if_neg (by decide), innerCandidates514_9_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_10
    (if 3 ≤ innerCandidates514_9.length then search edge 3 [3701, 5351] innerCandidates514_9 9052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_10

def innerTail514_8 : List Nat := 5309 :: innerTail514_9
def innerCandidates514_8 : List Nat := [5501]
set_option maxRecDepth 4096 in
theorem innerCandidates514_8_checked : candidates edge 4 3701 5309 innerTail514_9 best.bound = innerCandidates514_8 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_8 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_8 best = best := by
  rw [innerTail514_8, scan, if_neg (by decide), innerCandidates514_8_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_9
    (if 3 ≤ innerCandidates514_8.length then search edge 3 [3701, 5309] innerCandidates514_8 9010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_9

def innerTail514_7 : List Nat := 5081 :: innerTail514_8
def innerCandidates514_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates514_7_checked : candidates edge 4 3701 5081 innerTail514_8 best.bound = innerCandidates514_7 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_7 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_7 best = best := by
  rw [innerTail514_7, scan, if_neg (by decide), innerCandidates514_7_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_8
    (if 3 ≤ innerCandidates514_7.length then search edge 3 [3701, 5081] innerCandidates514_7 8782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_8

def innerTail514_6 : List Nat := 4973 :: innerTail514_7
def innerCandidates514_6 : List Nat := [7829]
set_option maxRecDepth 4096 in
theorem innerCandidates514_6_checked : candidates edge 4 3701 4973 innerTail514_7 best.bound = innerCandidates514_6 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_6 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_6 best = best := by
  rw [innerTail514_6, scan, if_neg (by decide), innerCandidates514_6_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_7
    (if 3 ≤ innerCandidates514_6.length then search edge 3 [3701, 4973] innerCandidates514_6 8674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_7

def innerTail514_5 : List Nat := 4931 :: innerTail514_6
def innerCandidates514_5 : List Nat := [5309]
set_option maxRecDepth 4096 in
theorem innerCandidates514_5_checked : candidates edge 4 3701 4931 innerTail514_6 best.bound = innerCandidates514_5 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_5 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_5 best = best := by
  rw [innerTail514_5, scan, if_neg (by decide), innerCandidates514_5_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_6
    (if 3 ≤ innerCandidates514_5.length then search edge 3 [3701, 4931] innerCandidates514_5 8632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_6

def innerTail514_4 : List Nat := 4919 :: innerTail514_5
def innerCandidates514_4 : List Nat := [6869, 7829]
set_option maxRecDepth 4096 in
theorem innerCandidates514_4_checked : candidates edge 4 3701 4919 innerTail514_5 best.bound = innerCandidates514_4 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_4 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_4 best = best := by
  rw [innerTail514_4, scan, if_neg (by decide), innerCandidates514_4_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_5
    (if 3 ≤ innerCandidates514_4.length then search edge 3 [3701, 4919] innerCandidates514_4 8620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_5

def innerTail514_3 : List Nat := 4877 :: innerTail514_4
def innerCandidates514_3 : List Nat := [5501]
set_option maxRecDepth 4096 in
theorem innerCandidates514_3_checked : candidates edge 4 3701 4877 innerTail514_4 best.bound = innerCandidates514_3 := by
  unfold candidates
  have row : edge 4877 = SearchData.row4877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_3 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_3 best = best := by
  rw [innerTail514_3, scan, if_neg (by decide), innerCandidates514_3_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_4
    (if 3 ≤ innerCandidates514_3.length then search edge 3 [3701, 4877] innerCandidates514_3 8578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_4

def innerTail514_2 : List Nat := 4493 :: innerTail514_3
def innerCandidates514_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates514_2_checked : candidates edge 4 3701 4493 innerTail514_3 best.bound = innerCandidates514_2 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_2 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_2 best = best := by
  rw [innerTail514_2, scan, if_neg (by decide), innerCandidates514_2_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_3
    (if 3 ≤ innerCandidates514_2.length then search edge 3 [3701, 4493] innerCandidates514_2 8194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_3

def innerTail514_1 : List Nat := 4283 :: innerTail514_2
def innerCandidates514_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates514_1_checked : candidates edge 4 3701 4283 innerTail514_2 best.bound = innerCandidates514_1 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_1 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_1 best = best := by
  rw [innerTail514_1, scan, if_neg (by decide), innerCandidates514_1_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_2
    (if 3 ≤ innerCandidates514_1.length then search edge 3 [3701, 4283] innerCandidates514_1 7984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_2

def innerTail514_0 : List Nat := 3797 :: innerTail514_1
def innerCandidates514_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates514_0_checked : candidates edge 4 3701 3797 innerTail514_1 best.bound = innerCandidates514_0 := by
  unfold candidates
  have row : edge 3797 = SearchData.row3797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan514_0 : scan edge (search edge 3) 4 [3701] 3701 innerTail514_0 best = best := by
  rw [innerTail514_0, scan, if_neg (by decide), innerCandidates514_0_checked]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_1
    (if 3 ≤ innerCandidates514_0.length then search edge 3 [3701, 3797] innerCandidates514_0 7498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan514_1

theorem rootBranch514 : search edge 4 [3701] rootCandidates514 3701 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3701] 3701 innerTail514_0 best = best
  exact innerScan514_0

end ProjectEulerSolutions.PrimePairUpper.Search
