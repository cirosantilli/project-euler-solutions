import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates606 : List Nat := [5113, 5347, 5737, 6577, 6607, 6793, 7489, 7537, 7717, 7759]
set_option maxRecDepth 4096 in
theorem rootCandidates606_checked : candidates edge 5 0 4483 tail607 best.bound = rootCandidates606 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext q; rfl
  rw [row]
  decide

def innerTail606_10 : List Nat := []
theorem innerScan606_10 : scan edge (search edge 3) 4 [4483] 4483 innerTail606_10 best = best := rfl

def innerTail606_9 : List Nat := 7759 :: innerTail606_10
def innerCandidates606_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates606_9_checked : candidates edge 4 4483 7759 innerTail606_10 best.bound = innerCandidates606_9 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan606_9 : scan edge (search edge 3) 4 [4483] 4483 innerTail606_9 best = best := by
  rw [innerTail606_9, scan, if_neg (by decide), innerCandidates606_9_checked]
  change scan edge (search edge 3) 4 [4483] 4483 innerTail606_10
    (if 3 ≤ innerCandidates606_9.length then search edge 3 [4483, 7759] innerCandidates606_9 12242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan606_10

def innerTail606_8 : List Nat := 7717 :: innerTail606_9
def innerCandidates606_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates606_8_checked : candidates edge 4 4483 7717 innerTail606_9 best.bound = innerCandidates606_8 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan606_8 : scan edge (search edge 3) 4 [4483] 4483 innerTail606_8 best = best := by
  rw [innerTail606_8, scan, if_neg (by decide), innerCandidates606_8_checked]
  change scan edge (search edge 3) 4 [4483] 4483 innerTail606_9
    (if 3 ≤ innerCandidates606_8.length then search edge 3 [4483, 7717] innerCandidates606_8 12200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan606_9

def innerTail606_7 : List Nat := 7537 :: innerTail606_8
def innerCandidates606_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates606_7_checked : candidates edge 4 4483 7537 innerTail606_8 best.bound = innerCandidates606_7 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan606_7 : scan edge (search edge 3) 4 [4483] 4483 innerTail606_7 best = best := by
  rw [innerTail606_7, scan, if_neg (by decide), innerCandidates606_7_checked]
  change scan edge (search edge 3) 4 [4483] 4483 innerTail606_8
    (if 3 ≤ innerCandidates606_7.length then search edge 3 [4483, 7537] innerCandidates606_7 12020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan606_8

def innerTail606_6 : List Nat := 7489 :: innerTail606_7
def innerCandidates606_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates606_6_checked : candidates edge 4 4483 7489 innerTail606_7 best.bound = innerCandidates606_6 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan606_6 : scan edge (search edge 3) 4 [4483] 4483 innerTail606_6 best = best := by
  rw [innerTail606_6, scan, if_neg (by decide), innerCandidates606_6_checked]
  change scan edge (search edge 3) 4 [4483] 4483 innerTail606_7
    (if 3 ≤ innerCandidates606_6.length then search edge 3 [4483, 7489] innerCandidates606_6 11972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan606_7

def innerTail606_5 : List Nat := 6793 :: innerTail606_6
def innerCandidates606_5 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates606_5_checked : candidates edge 4 4483 6793 innerTail606_6 best.bound = innerCandidates606_5 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan606_5 : scan edge (search edge 3) 4 [4483] 4483 innerTail606_5 best = best := by
  rw [innerTail606_5, scan, if_neg (by decide), innerCandidates606_5_checked]
  change scan edge (search edge 3) 4 [4483] 4483 innerTail606_6
    (if 3 ≤ innerCandidates606_5.length then search edge 3 [4483, 6793] innerCandidates606_5 11276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan606_6

def innerTail606_4 : List Nat := 6607 :: innerTail606_5
def innerCandidates606_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates606_4_checked : candidates edge 4 4483 6607 innerTail606_5 best.bound = innerCandidates606_4 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan606_4 : scan edge (search edge 3) 4 [4483] 4483 innerTail606_4 best = best := by
  rw [innerTail606_4, scan, if_neg (by decide), innerCandidates606_4_checked]
  change scan edge (search edge 3) 4 [4483] 4483 innerTail606_5
    (if 3 ≤ innerCandidates606_4.length then search edge 3 [4483, 6607] innerCandidates606_4 11090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan606_5

def innerTail606_3 : List Nat := 6577 :: innerTail606_4
def innerCandidates606_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates606_3_checked : candidates edge 4 4483 6577 innerTail606_4 best.bound = innerCandidates606_3 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan606_3 : scan edge (search edge 3) 4 [4483] 4483 innerTail606_3 best = best := by
  rw [innerTail606_3, scan, if_neg (by decide), innerCandidates606_3_checked]
  change scan edge (search edge 3) 4 [4483] 4483 innerTail606_4
    (if 3 ≤ innerCandidates606_3.length then search edge 3 [4483, 6577] innerCandidates606_3 11060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan606_4

def innerTail606_2 : List Nat := 5737 :: innerTail606_3
def innerCandidates606_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates606_2_checked : candidates edge 4 4483 5737 innerTail606_3 best.bound = innerCandidates606_2 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan606_2 : scan edge (search edge 3) 4 [4483] 4483 innerTail606_2 best = best := by
  rw [innerTail606_2, scan, if_neg (by decide), innerCandidates606_2_checked]
  change scan edge (search edge 3) 4 [4483] 4483 innerTail606_3
    (if 3 ≤ innerCandidates606_2.length then search edge 3 [4483, 5737] innerCandidates606_2 10220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan606_3

def innerTail606_1 : List Nat := 5347 :: innerTail606_2
def innerCandidates606_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates606_1_checked : candidates edge 4 4483 5347 innerTail606_2 best.bound = innerCandidates606_1 := by
  unfold candidates
  have row : edge 5347 = SearchData.row5347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan606_1 : scan edge (search edge 3) 4 [4483] 4483 innerTail606_1 best = best := by
  rw [innerTail606_1, scan, if_neg (by decide), innerCandidates606_1_checked]
  change scan edge (search edge 3) 4 [4483] 4483 innerTail606_2
    (if 3 ≤ innerCandidates606_1.length then search edge 3 [4483, 5347] innerCandidates606_1 9830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan606_2

def innerTail606_0 : List Nat := 5113 :: innerTail606_1
def innerCandidates606_0 : List Nat := [6793]
set_option maxRecDepth 4096 in
theorem innerCandidates606_0_checked : candidates edge 4 4483 5113 innerTail606_1 best.bound = innerCandidates606_0 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan606_0 : scan edge (search edge 3) 4 [4483] 4483 innerTail606_0 best = best := by
  rw [innerTail606_0, scan, if_neg (by decide), innerCandidates606_0_checked]
  change scan edge (search edge 3) 4 [4483] 4483 innerTail606_1
    (if 3 ≤ innerCandidates606_0.length then search edge 3 [4483, 5113] innerCandidates606_0 9596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan606_1

theorem rootBranch606 : search edge 4 [4483] rootCandidates606 4483 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4483] 4483 innerTail606_0 best = best
  exact innerScan606_0

def rootCandidates605 : List Nat := [5189, 6203, 6491, 6791, 7193, 7529, 7583, 7907, 8117]
set_option maxRecDepth 4096 in
theorem rootCandidates605_checked : candidates edge 5 0 4481 tail606 best.bound = rootCandidates605 := by
  unfold candidates
  have row : edge 4481 = SearchData.row4481.contains := by funext q; rfl
  rw [row]
  decide

def innerTail605_9 : List Nat := []
theorem innerScan605_9 : scan edge (search edge 3) 4 [4481] 4481 innerTail605_9 best = best := rfl

def innerTail605_8 : List Nat := 8117 :: innerTail605_9
def innerCandidates605_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates605_8_checked : candidates edge 4 4481 8117 innerTail605_9 best.bound = innerCandidates605_8 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan605_8 : scan edge (search edge 3) 4 [4481] 4481 innerTail605_8 best = best := by
  rw [innerTail605_8, scan, if_neg (by decide), innerCandidates605_8_checked]
  change scan edge (search edge 3) 4 [4481] 4481 innerTail605_9
    (if 3 ≤ innerCandidates605_8.length then search edge 3 [4481, 8117] innerCandidates605_8 12598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan605_9

def innerTail605_7 : List Nat := 7907 :: innerTail605_8
def innerCandidates605_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates605_7_checked : candidates edge 4 4481 7907 innerTail605_8 best.bound = innerCandidates605_7 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan605_7 : scan edge (search edge 3) 4 [4481] 4481 innerTail605_7 best = best := by
  rw [innerTail605_7, scan, if_neg (by decide), innerCandidates605_7_checked]
  change scan edge (search edge 3) 4 [4481] 4481 innerTail605_8
    (if 3 ≤ innerCandidates605_7.length then search edge 3 [4481, 7907] innerCandidates605_7 12388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan605_8

def innerTail605_6 : List Nat := 7583 :: innerTail605_7
def innerCandidates605_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates605_6_checked : candidates edge 4 4481 7583 innerTail605_7 best.bound = innerCandidates605_6 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan605_6 : scan edge (search edge 3) 4 [4481] 4481 innerTail605_6 best = best := by
  rw [innerTail605_6, scan, if_neg (by decide), innerCandidates605_6_checked]
  change scan edge (search edge 3) 4 [4481] 4481 innerTail605_7
    (if 3 ≤ innerCandidates605_6.length then search edge 3 [4481, 7583] innerCandidates605_6 12064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan605_7

def innerTail605_5 : List Nat := 7529 :: innerTail605_6
def innerCandidates605_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates605_5_checked : candidates edge 4 4481 7529 innerTail605_6 best.bound = innerCandidates605_5 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan605_5 : scan edge (search edge 3) 4 [4481] 4481 innerTail605_5 best = best := by
  rw [innerTail605_5, scan, if_neg (by decide), innerCandidates605_5_checked]
  change scan edge (search edge 3) 4 [4481] 4481 innerTail605_6
    (if 3 ≤ innerCandidates605_5.length then search edge 3 [4481, 7529] innerCandidates605_5 12010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan605_6

def innerTail605_4 : List Nat := 7193 :: innerTail605_5
def innerCandidates605_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates605_4_checked : candidates edge 4 4481 7193 innerTail605_5 best.bound = innerCandidates605_4 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan605_4 : scan edge (search edge 3) 4 [4481] 4481 innerTail605_4 best = best := by
  rw [innerTail605_4, scan, if_neg (by decide), innerCandidates605_4_checked]
  change scan edge (search edge 3) 4 [4481] 4481 innerTail605_5
    (if 3 ≤ innerCandidates605_4.length then search edge 3 [4481, 7193] innerCandidates605_4 11674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan605_5

def innerTail605_3 : List Nat := 6791 :: innerTail605_4
def innerCandidates605_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates605_3_checked : candidates edge 4 4481 6791 innerTail605_4 best.bound = innerCandidates605_3 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan605_3 : scan edge (search edge 3) 4 [4481] 4481 innerTail605_3 best = best := by
  rw [innerTail605_3, scan, if_neg (by decide), innerCandidates605_3_checked]
  change scan edge (search edge 3) 4 [4481] 4481 innerTail605_4
    (if 3 ≤ innerCandidates605_3.length then search edge 3 [4481, 6791] innerCandidates605_3 11272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan605_4

def innerTail605_2 : List Nat := 6491 :: innerTail605_3
def innerCandidates605_2 : List Nat := [7583]
set_option maxRecDepth 4096 in
theorem innerCandidates605_2_checked : candidates edge 4 4481 6491 innerTail605_3 best.bound = innerCandidates605_2 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan605_2 : scan edge (search edge 3) 4 [4481] 4481 innerTail605_2 best = best := by
  rw [innerTail605_2, scan, if_neg (by decide), innerCandidates605_2_checked]
  change scan edge (search edge 3) 4 [4481] 4481 innerTail605_3
    (if 3 ≤ innerCandidates605_2.length then search edge 3 [4481, 6491] innerCandidates605_2 10972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan605_3

def innerTail605_1 : List Nat := 6203 :: innerTail605_2
def innerCandidates605_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates605_1_checked : candidates edge 4 4481 6203 innerTail605_2 best.bound = innerCandidates605_1 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan605_1 : scan edge (search edge 3) 4 [4481] 4481 innerTail605_1 best = best := by
  rw [innerTail605_1, scan, if_neg (by decide), innerCandidates605_1_checked]
  change scan edge (search edge 3) 4 [4481] 4481 innerTail605_2
    (if 3 ≤ innerCandidates605_1.length then search edge 3 [4481, 6203] innerCandidates605_1 10684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan605_2

def innerTail605_0 : List Nat := 5189 :: innerTail605_1
def innerCandidates605_0 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates605_0_checked : candidates edge 4 4481 5189 innerTail605_1 best.bound = innerCandidates605_0 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan605_0 : scan edge (search edge 3) 4 [4481] 4481 innerTail605_0 best = best := by
  rw [innerTail605_0, scan, if_neg (by decide), innerCandidates605_0_checked]
  change scan edge (search edge 3) 4 [4481] 4481 innerTail605_1
    (if 3 ≤ innerCandidates605_0.length then search edge 3 [4481, 5189] innerCandidates605_0 9670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan605_1

theorem rootBranch605 : search edge 4 [4481] rootCandidates605 4481 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4481] 4481 innerTail605_0 best = best
  exact innerScan605_0

def rootCandidates604 : List Nat := [5273, 5399, 5441, 5813, 5861, 6521, 6791, 7079, 7877, 8369, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates604_checked : candidates edge 5 0 4463 tail605 best.bound = rootCandidates604 := by
  unfold candidates
  have row : edge 4463 = SearchData.row4463.contains := by funext q; rfl
  rw [row]
  decide

def innerTail604_11 : List Nat := []
theorem innerScan604_11 : scan edge (search edge 3) 4 [4463] 4463 innerTail604_11 best = best := rfl

def innerTail604_10 : List Nat := 8387 :: innerTail604_11
def innerCandidates604_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates604_10_checked : candidates edge 4 4463 8387 innerTail604_11 best.bound = innerCandidates604_10 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan604_10 : scan edge (search edge 3) 4 [4463] 4463 innerTail604_10 best = best := by
  rw [innerTail604_10, scan, if_neg (by decide), innerCandidates604_10_checked]
  change scan edge (search edge 3) 4 [4463] 4463 innerTail604_11
    (if 3 ≤ innerCandidates604_10.length then search edge 3 [4463, 8387] innerCandidates604_10 12850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan604_11

def innerTail604_9 : List Nat := 8369 :: innerTail604_10
def innerCandidates604_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates604_9_checked : candidates edge 4 4463 8369 innerTail604_10 best.bound = innerCandidates604_9 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan604_9 : scan edge (search edge 3) 4 [4463] 4463 innerTail604_9 best = best := by
  rw [innerTail604_9, scan, if_neg (by decide), innerCandidates604_9_checked]
  change scan edge (search edge 3) 4 [4463] 4463 innerTail604_10
    (if 3 ≤ innerCandidates604_9.length then search edge 3 [4463, 8369] innerCandidates604_9 12832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan604_10

def innerTail604_8 : List Nat := 7877 :: innerTail604_9
def innerCandidates604_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates604_8_checked : candidates edge 4 4463 7877 innerTail604_9 best.bound = innerCandidates604_8 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan604_8 : scan edge (search edge 3) 4 [4463] 4463 innerTail604_8 best = best := by
  rw [innerTail604_8, scan, if_neg (by decide), innerCandidates604_8_checked]
  change scan edge (search edge 3) 4 [4463] 4463 innerTail604_9
    (if 3 ≤ innerCandidates604_8.length then search edge 3 [4463, 7877] innerCandidates604_8 12340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan604_9

def innerTail604_7 : List Nat := 7079 :: innerTail604_8
def innerCandidates604_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates604_7_checked : candidates edge 4 4463 7079 innerTail604_8 best.bound = innerCandidates604_7 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan604_7 : scan edge (search edge 3) 4 [4463] 4463 innerTail604_7 best = best := by
  rw [innerTail604_7, scan, if_neg (by decide), innerCandidates604_7_checked]
  change scan edge (search edge 3) 4 [4463] 4463 innerTail604_8
    (if 3 ≤ innerCandidates604_7.length then search edge 3 [4463, 7079] innerCandidates604_7 11542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan604_8

def innerTail604_6 : List Nat := 6791 :: innerTail604_7
def innerCandidates604_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates604_6_checked : candidates edge 4 4463 6791 innerTail604_7 best.bound = innerCandidates604_6 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan604_6 : scan edge (search edge 3) 4 [4463] 4463 innerTail604_6 best = best := by
  rw [innerTail604_6, scan, if_neg (by decide), innerCandidates604_6_checked]
  change scan edge (search edge 3) 4 [4463] 4463 innerTail604_7
    (if 3 ≤ innerCandidates604_6.length then search edge 3 [4463, 6791] innerCandidates604_6 11254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan604_7

def innerTail604_5 : List Nat := 6521 :: innerTail604_6
def innerCandidates604_5 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates604_5_checked : candidates edge 4 4463 6521 innerTail604_6 best.bound = innerCandidates604_5 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan604_5 : scan edge (search edge 3) 4 [4463] 4463 innerTail604_5 best = best := by
  rw [innerTail604_5, scan, if_neg (by decide), innerCandidates604_5_checked]
  change scan edge (search edge 3) 4 [4463] 4463 innerTail604_6
    (if 3 ≤ innerCandidates604_5.length then search edge 3 [4463, 6521] innerCandidates604_5 10984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan604_6

def innerTail604_4 : List Nat := 5861 :: innerTail604_5
def innerCandidates604_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates604_4_checked : candidates edge 4 4463 5861 innerTail604_5 best.bound = innerCandidates604_4 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan604_4 : scan edge (search edge 3) 4 [4463] 4463 innerTail604_4 best = best := by
  rw [innerTail604_4, scan, if_neg (by decide), innerCandidates604_4_checked]
  change scan edge (search edge 3) 4 [4463] 4463 innerTail604_5
    (if 3 ≤ innerCandidates604_4.length then search edge 3 [4463, 5861] innerCandidates604_4 10324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan604_5

def innerTail604_3 : List Nat := 5813 :: innerTail604_4
def innerCandidates604_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates604_3_checked : candidates edge 4 4463 5813 innerTail604_4 best.bound = innerCandidates604_3 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan604_3 : scan edge (search edge 3) 4 [4463] 4463 innerTail604_3 best = best := by
  rw [innerTail604_3, scan, if_neg (by decide), innerCandidates604_3_checked]
  change scan edge (search edge 3) 4 [4463] 4463 innerTail604_4
    (if 3 ≤ innerCandidates604_3.length then search edge 3 [4463, 5813] innerCandidates604_3 10276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan604_4

def innerTail604_2 : List Nat := 5441 :: innerTail604_3
def innerCandidates604_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates604_2_checked : candidates edge 4 4463 5441 innerTail604_3 best.bound = innerCandidates604_2 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan604_2 : scan edge (search edge 3) 4 [4463] 4463 innerTail604_2 best = best := by
  rw [innerTail604_2, scan, if_neg (by decide), innerCandidates604_2_checked]
  change scan edge (search edge 3) 4 [4463] 4463 innerTail604_3
    (if 3 ≤ innerCandidates604_2.length then search edge 3 [4463, 5441] innerCandidates604_2 9904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan604_3

def innerTail604_1 : List Nat := 5399 :: innerTail604_2
def innerCandidates604_1 : List Nat := [7079]
set_option maxRecDepth 4096 in
theorem innerCandidates604_1_checked : candidates edge 4 4463 5399 innerTail604_2 best.bound = innerCandidates604_1 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan604_1 : scan edge (search edge 3) 4 [4463] 4463 innerTail604_1 best = best := by
  rw [innerTail604_1, scan, if_neg (by decide), innerCandidates604_1_checked]
  change scan edge (search edge 3) 4 [4463] 4463 innerTail604_2
    (if 3 ≤ innerCandidates604_1.length then search edge 3 [4463, 5399] innerCandidates604_1 9862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan604_2

def innerTail604_0 : List Nat := 5273 :: innerTail604_1
def innerCandidates604_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates604_0_checked : candidates edge 4 4463 5273 innerTail604_1 best.bound = innerCandidates604_0 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan604_0 : scan edge (search edge 3) 4 [4463] 4463 innerTail604_0 best = best := by
  rw [innerTail604_0, scan, if_neg (by decide), innerCandidates604_0_checked]
  change scan edge (search edge 3) 4 [4463] 4463 innerTail604_1
    (if 3 ≤ innerCandidates604_0.length then search edge 3 [4463, 5273] innerCandidates604_0 9736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan604_1

theorem rootBranch604 : search edge 4 [4463] rootCandidates604 4463 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4463] 4463 innerTail604_0 best = best
  exact innerScan604_0

def rootCandidates603 : List Nat := [4679, 4793, 5393, 5483, 5507, 5591, 5981, 6473, 6491, 7019, 7331, 7433, 7499, 7607, 7853, 7919]
set_option maxRecDepth 4096 in
theorem rootCandidates603_checked : candidates edge 5 0 4457 tail604 best.bound = rootCandidates603 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext q; rfl
  rw [row]
  decide

def innerTail603_16 : List Nat := []
theorem innerScan603_16 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_16 best = best := rfl

def innerTail603_15 : List Nat := 7919 :: innerTail603_16
def innerCandidates603_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates603_15_checked : candidates edge 4 4457 7919 innerTail603_16 best.bound = innerCandidates603_15 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_15 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_15 best = best := by
  rw [innerTail603_15, scan, if_neg (by decide), innerCandidates603_15_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_16
    (if 3 ≤ innerCandidates603_15.length then search edge 3 [4457, 7919] innerCandidates603_15 12376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_16

def innerTail603_14 : List Nat := 7853 :: innerTail603_15
def innerCandidates603_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates603_14_checked : candidates edge 4 4457 7853 innerTail603_15 best.bound = innerCandidates603_14 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_14 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_14 best = best := by
  rw [innerTail603_14, scan, if_neg (by decide), innerCandidates603_14_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_15
    (if 3 ≤ innerCandidates603_14.length then search edge 3 [4457, 7853] innerCandidates603_14 12310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_15

def innerTail603_13 : List Nat := 7607 :: innerTail603_14
def innerCandidates603_13 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates603_13_checked : candidates edge 4 4457 7607 innerTail603_14 best.bound = innerCandidates603_13 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_13 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_13 best = best := by
  rw [innerTail603_13, scan, if_neg (by decide), innerCandidates603_13_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_14
    (if 3 ≤ innerCandidates603_13.length then search edge 3 [4457, 7607] innerCandidates603_13 12064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_14

def innerTail603_12 : List Nat := 7499 :: innerTail603_13
def innerCandidates603_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates603_12_checked : candidates edge 4 4457 7499 innerTail603_13 best.bound = innerCandidates603_12 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_12 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_12 best = best := by
  rw [innerTail603_12, scan, if_neg (by decide), innerCandidates603_12_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_13
    (if 3 ≤ innerCandidates603_12.length then search edge 3 [4457, 7499] innerCandidates603_12 11956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_13

def innerTail603_11 : List Nat := 7433 :: innerTail603_12
def innerCandidates603_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates603_11_checked : candidates edge 4 4457 7433 innerTail603_12 best.bound = innerCandidates603_11 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_11 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_11 best = best := by
  rw [innerTail603_11, scan, if_neg (by decide), innerCandidates603_11_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_12
    (if 3 ≤ innerCandidates603_11.length then search edge 3 [4457, 7433] innerCandidates603_11 11890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_12

def innerTail603_10 : List Nat := 7331 :: innerTail603_11
def innerCandidates603_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates603_10_checked : candidates edge 4 4457 7331 innerTail603_11 best.bound = innerCandidates603_10 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_10 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_10 best = best := by
  rw [innerTail603_10, scan, if_neg (by decide), innerCandidates603_10_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_11
    (if 3 ≤ innerCandidates603_10.length then search edge 3 [4457, 7331] innerCandidates603_10 11788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_11

def innerTail603_9 : List Nat := 7019 :: innerTail603_10
def innerCandidates603_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates603_9_checked : candidates edge 4 4457 7019 innerTail603_10 best.bound = innerCandidates603_9 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_9 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_9 best = best := by
  rw [innerTail603_9, scan, if_neg (by decide), innerCandidates603_9_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_10
    (if 3 ≤ innerCandidates603_9.length then search edge 3 [4457, 7019] innerCandidates603_9 11476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_10

def innerTail603_8 : List Nat := 6491 :: innerTail603_9
def innerCandidates603_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates603_8_checked : candidates edge 4 4457 6491 innerTail603_9 best.bound = innerCandidates603_8 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_8 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_8 best = best := by
  rw [innerTail603_8, scan, if_neg (by decide), innerCandidates603_8_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_9
    (if 3 ≤ innerCandidates603_8.length then search edge 3 [4457, 6491] innerCandidates603_8 10948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_9

def innerTail603_7 : List Nat := 6473 :: innerTail603_8
def innerCandidates603_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates603_7_checked : candidates edge 4 4457 6473 innerTail603_8 best.bound = innerCandidates603_7 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_7 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_7 best = best := by
  rw [innerTail603_7, scan, if_neg (by decide), innerCandidates603_7_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_8
    (if 3 ≤ innerCandidates603_7.length then search edge 3 [4457, 6473] innerCandidates603_7 10930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_8

def innerTail603_6 : List Nat := 5981 :: innerTail603_7
def innerCandidates603_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates603_6_checked : candidates edge 4 4457 5981 innerTail603_7 best.bound = innerCandidates603_6 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_6 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_6 best = best := by
  rw [innerTail603_6, scan, if_neg (by decide), innerCandidates603_6_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_7
    (if 3 ≤ innerCandidates603_6.length then search edge 3 [4457, 5981] innerCandidates603_6 10438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_7

def innerTail603_5 : List Nat := 5591 :: innerTail603_6
def innerCandidates603_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates603_5_checked : candidates edge 4 4457 5591 innerTail603_6 best.bound = innerCandidates603_5 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_5 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_5 best = best := by
  rw [innerTail603_5, scan, if_neg (by decide), innerCandidates603_5_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_6
    (if 3 ≤ innerCandidates603_5.length then search edge 3 [4457, 5591] innerCandidates603_5 10048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_6

def innerTail603_4 : List Nat := 5507 :: innerTail603_5
def innerCandidates603_4 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates603_4_checked : candidates edge 4 4457 5507 innerTail603_5 best.bound = innerCandidates603_4 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_4 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_4 best = best := by
  rw [innerTail603_4, scan, if_neg (by decide), innerCandidates603_4_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_5
    (if 3 ≤ innerCandidates603_4.length then search edge 3 [4457, 5507] innerCandidates603_4 9964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_5

def innerTail603_3 : List Nat := 5483 :: innerTail603_4
def innerCandidates603_3 : List Nat := [5507, 6491]
set_option maxRecDepth 4096 in
theorem innerCandidates603_3_checked : candidates edge 4 4457 5483 innerTail603_4 best.bound = innerCandidates603_3 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_3 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_3 best = best := by
  rw [innerTail603_3, scan, if_neg (by decide), innerCandidates603_3_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_4
    (if 3 ≤ innerCandidates603_3.length then search edge 3 [4457, 5483] innerCandidates603_3 9940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_4

def innerTail603_2 : List Nat := 5393 :: innerTail603_3
def innerCandidates603_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates603_2_checked : candidates edge 4 4457 5393 innerTail603_3 best.bound = innerCandidates603_2 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_2 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_2 best = best := by
  rw [innerTail603_2, scan, if_neg (by decide), innerCandidates603_2_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_3
    (if 3 ≤ innerCandidates603_2.length then search edge 3 [4457, 5393] innerCandidates603_2 9850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_3

def innerTail603_1 : List Nat := 4793 :: innerTail603_2
def innerCandidates603_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates603_1_checked : candidates edge 4 4457 4793 innerTail603_2 best.bound = innerCandidates603_1 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_1 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_1 best = best := by
  rw [innerTail603_1, scan, if_neg (by decide), innerCandidates603_1_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_2
    (if 3 ≤ innerCandidates603_1.length then search edge 3 [4457, 4793] innerCandidates603_1 9250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_2

def innerTail603_0 : List Nat := 4679 :: innerTail603_1
def innerCandidates603_0 : List Nat := [5393]
set_option maxRecDepth 4096 in
theorem innerCandidates603_0_checked : candidates edge 4 4457 4679 innerTail603_1 best.bound = innerCandidates603_0 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan603_0 : scan edge (search edge 3) 4 [4457] 4457 innerTail603_0 best = best := by
  rw [innerTail603_0, scan, if_neg (by decide), innerCandidates603_0_checked]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_1
    (if 3 ≤ innerCandidates603_0.length then search edge 3 [4457, 4679] innerCandidates603_0 9136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan603_1

theorem rootBranch603 : search edge 4 [4457] rootCandidates603 4457 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4457] 4457 innerTail603_0 best = best
  exact innerScan603_0

def rootCandidates602 : List Nat := [5279, 6089, 6257, 6599, 6803, 6833, 7307, 7853]
set_option maxRecDepth 4096 in
theorem rootCandidates602_checked : candidates edge 5 0 4451 tail603 best.bound = rootCandidates602 := by
  unfold candidates
  have row : edge 4451 = SearchData.row4451.contains := by funext q; rfl
  rw [row]
  decide

def innerTail602_8 : List Nat := []
theorem innerScan602_8 : scan edge (search edge 3) 4 [4451] 4451 innerTail602_8 best = best := rfl

def innerTail602_7 : List Nat := 7853 :: innerTail602_8
def innerCandidates602_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates602_7_checked : candidates edge 4 4451 7853 innerTail602_8 best.bound = innerCandidates602_7 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan602_7 : scan edge (search edge 3) 4 [4451] 4451 innerTail602_7 best = best := by
  rw [innerTail602_7, scan, if_neg (by decide), innerCandidates602_7_checked]
  change scan edge (search edge 3) 4 [4451] 4451 innerTail602_8
    (if 3 ≤ innerCandidates602_7.length then search edge 3 [4451, 7853] innerCandidates602_7 12304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan602_8

def innerTail602_6 : List Nat := 7307 :: innerTail602_7
def innerCandidates602_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates602_6_checked : candidates edge 4 4451 7307 innerTail602_7 best.bound = innerCandidates602_6 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan602_6 : scan edge (search edge 3) 4 [4451] 4451 innerTail602_6 best = best := by
  rw [innerTail602_6, scan, if_neg (by decide), innerCandidates602_6_checked]
  change scan edge (search edge 3) 4 [4451] 4451 innerTail602_7
    (if 3 ≤ innerCandidates602_6.length then search edge 3 [4451, 7307] innerCandidates602_6 11758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan602_7

def innerTail602_5 : List Nat := 6833 :: innerTail602_6
def innerCandidates602_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates602_5_checked : candidates edge 4 4451 6833 innerTail602_6 best.bound = innerCandidates602_5 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan602_5 : scan edge (search edge 3) 4 [4451] 4451 innerTail602_5 best = best := by
  rw [innerTail602_5, scan, if_neg (by decide), innerCandidates602_5_checked]
  change scan edge (search edge 3) 4 [4451] 4451 innerTail602_6
    (if 3 ≤ innerCandidates602_5.length then search edge 3 [4451, 6833] innerCandidates602_5 11284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan602_6

def innerTail602_4 : List Nat := 6803 :: innerTail602_5
def innerCandidates602_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates602_4_checked : candidates edge 4 4451 6803 innerTail602_5 best.bound = innerCandidates602_4 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan602_4 : scan edge (search edge 3) 4 [4451] 4451 innerTail602_4 best = best := by
  rw [innerTail602_4, scan, if_neg (by decide), innerCandidates602_4_checked]
  change scan edge (search edge 3) 4 [4451] 4451 innerTail602_5
    (if 3 ≤ innerCandidates602_4.length then search edge 3 [4451, 6803] innerCandidates602_4 11254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan602_5

def innerTail602_3 : List Nat := 6599 :: innerTail602_4
def innerCandidates602_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates602_3_checked : candidates edge 4 4451 6599 innerTail602_4 best.bound = innerCandidates602_3 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan602_3 : scan edge (search edge 3) 4 [4451] 4451 innerTail602_3 best = best := by
  rw [innerTail602_3, scan, if_neg (by decide), innerCandidates602_3_checked]
  change scan edge (search edge 3) 4 [4451] 4451 innerTail602_4
    (if 3 ≤ innerCandidates602_3.length then search edge 3 [4451, 6599] innerCandidates602_3 11050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan602_4

def innerTail602_2 : List Nat := 6257 :: innerTail602_3
def innerCandidates602_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates602_2_checked : candidates edge 4 4451 6257 innerTail602_3 best.bound = innerCandidates602_2 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan602_2 : scan edge (search edge 3) 4 [4451] 4451 innerTail602_2 best = best := by
  rw [innerTail602_2, scan, if_neg (by decide), innerCandidates602_2_checked]
  change scan edge (search edge 3) 4 [4451] 4451 innerTail602_3
    (if 3 ≤ innerCandidates602_2.length then search edge 3 [4451, 6257] innerCandidates602_2 10708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan602_3

def innerTail602_1 : List Nat := 6089 :: innerTail602_2
def innerCandidates602_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates602_1_checked : candidates edge 4 4451 6089 innerTail602_2 best.bound = innerCandidates602_1 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan602_1 : scan edge (search edge 3) 4 [4451] 4451 innerTail602_1 best = best := by
  rw [innerTail602_1, scan, if_neg (by decide), innerCandidates602_1_checked]
  change scan edge (search edge 3) 4 [4451] 4451 innerTail602_2
    (if 3 ≤ innerCandidates602_1.length then search edge 3 [4451, 6089] innerCandidates602_1 10540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan602_2

def innerTail602_0 : List Nat := 5279 :: innerTail602_1
def innerCandidates602_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates602_0_checked : candidates edge 4 4451 5279 innerTail602_1 best.bound = innerCandidates602_0 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan602_0 : scan edge (search edge 3) 4 [4451] 4451 innerTail602_0 best = best := by
  rw [innerTail602_0, scan, if_neg (by decide), innerCandidates602_0_checked]
  change scan edge (search edge 3) 4 [4451] 4451 innerTail602_1
    (if 3 ≤ innerCandidates602_0.length then search edge 3 [4451, 5279] innerCandidates602_0 9730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan602_1

theorem rootBranch602 : search edge 4 [4451] rootCandidates602 4451 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4451] 4451 innerTail602_0 best = best
  exact innerScan602_0

def rootCandidates601 : List Nat := [4789, 5023, 5419, 6067, 6427, 7057, 7207, 8101, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates601_checked : candidates edge 5 0 4447 tail602 best.bound = rootCandidates601 := by
  unfold candidates
  have row : edge 4447 = SearchData.row4447.contains := by funext q; rfl
  rw [row]
  decide

def innerTail601_9 : List Nat := []
theorem innerScan601_9 : scan edge (search edge 3) 4 [4447] 4447 innerTail601_9 best = best := rfl

def innerTail601_8 : List Nat := 8293 :: innerTail601_9
def innerCandidates601_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates601_8_checked : candidates edge 4 4447 8293 innerTail601_9 best.bound = innerCandidates601_8 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan601_8 : scan edge (search edge 3) 4 [4447] 4447 innerTail601_8 best = best := by
  rw [innerTail601_8, scan, if_neg (by decide), innerCandidates601_8_checked]
  change scan edge (search edge 3) 4 [4447] 4447 innerTail601_9
    (if 3 ≤ innerCandidates601_8.length then search edge 3 [4447, 8293] innerCandidates601_8 12740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan601_9

def innerTail601_7 : List Nat := 8101 :: innerTail601_8
def innerCandidates601_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates601_7_checked : candidates edge 4 4447 8101 innerTail601_8 best.bound = innerCandidates601_7 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan601_7 : scan edge (search edge 3) 4 [4447] 4447 innerTail601_7 best = best := by
  rw [innerTail601_7, scan, if_neg (by decide), innerCandidates601_7_checked]
  change scan edge (search edge 3) 4 [4447] 4447 innerTail601_8
    (if 3 ≤ innerCandidates601_7.length then search edge 3 [4447, 8101] innerCandidates601_7 12548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan601_8

def innerTail601_6 : List Nat := 7207 :: innerTail601_7
def innerCandidates601_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates601_6_checked : candidates edge 4 4447 7207 innerTail601_7 best.bound = innerCandidates601_6 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan601_6 : scan edge (search edge 3) 4 [4447] 4447 innerTail601_6 best = best := by
  rw [innerTail601_6, scan, if_neg (by decide), innerCandidates601_6_checked]
  change scan edge (search edge 3) 4 [4447] 4447 innerTail601_7
    (if 3 ≤ innerCandidates601_6.length then search edge 3 [4447, 7207] innerCandidates601_6 11654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan601_7

def innerTail601_5 : List Nat := 7057 :: innerTail601_6
def innerCandidates601_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates601_5_checked : candidates edge 4 4447 7057 innerTail601_6 best.bound = innerCandidates601_5 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan601_5 : scan edge (search edge 3) 4 [4447] 4447 innerTail601_5 best = best := by
  rw [innerTail601_5, scan, if_neg (by decide), innerCandidates601_5_checked]
  change scan edge (search edge 3) 4 [4447] 4447 innerTail601_6
    (if 3 ≤ innerCandidates601_5.length then search edge 3 [4447, 7057] innerCandidates601_5 11504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan601_6

def innerTail601_4 : List Nat := 6427 :: innerTail601_5
def innerCandidates601_4 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates601_4_checked : candidates edge 4 4447 6427 innerTail601_5 best.bound = innerCandidates601_4 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan601_4 : scan edge (search edge 3) 4 [4447] 4447 innerTail601_4 best = best := by
  rw [innerTail601_4, scan, if_neg (by decide), innerCandidates601_4_checked]
  change scan edge (search edge 3) 4 [4447] 4447 innerTail601_5
    (if 3 ≤ innerCandidates601_4.length then search edge 3 [4447, 6427] innerCandidates601_4 10874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan601_5

def innerTail601_3 : List Nat := 6067 :: innerTail601_4
def innerCandidates601_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates601_3_checked : candidates edge 4 4447 6067 innerTail601_4 best.bound = innerCandidates601_3 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan601_3 : scan edge (search edge 3) 4 [4447] 4447 innerTail601_3 best = best := by
  rw [innerTail601_3, scan, if_neg (by decide), innerCandidates601_3_checked]
  change scan edge (search edge 3) 4 [4447] 4447 innerTail601_4
    (if 3 ≤ innerCandidates601_3.length then search edge 3 [4447, 6067] innerCandidates601_3 10514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan601_4

def innerTail601_2 : List Nat := 5419 :: innerTail601_3
def innerCandidates601_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates601_2_checked : candidates edge 4 4447 5419 innerTail601_3 best.bound = innerCandidates601_2 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan601_2 : scan edge (search edge 3) 4 [4447] 4447 innerTail601_2 best = best := by
  rw [innerTail601_2, scan, if_neg (by decide), innerCandidates601_2_checked]
  change scan edge (search edge 3) 4 [4447] 4447 innerTail601_3
    (if 3 ≤ innerCandidates601_2.length then search edge 3 [4447, 5419] innerCandidates601_2 9866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan601_3

def innerTail601_1 : List Nat := 5023 :: innerTail601_2
def innerCandidates601_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates601_1_checked : candidates edge 4 4447 5023 innerTail601_2 best.bound = innerCandidates601_1 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan601_1 : scan edge (search edge 3) 4 [4447] 4447 innerTail601_1 best = best := by
  rw [innerTail601_1, scan, if_neg (by decide), innerCandidates601_1_checked]
  change scan edge (search edge 3) 4 [4447] 4447 innerTail601_2
    (if 3 ≤ innerCandidates601_1.length then search edge 3 [4447, 5023] innerCandidates601_1 9470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan601_2

def innerTail601_0 : List Nat := 4789 :: innerTail601_1
def innerCandidates601_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates601_0_checked : candidates edge 4 4447 4789 innerTail601_1 best.bound = innerCandidates601_0 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan601_0 : scan edge (search edge 3) 4 [4447] 4447 innerTail601_0 best = best := by
  rw [innerTail601_0, scan, if_neg (by decide), innerCandidates601_0_checked]
  change scan edge (search edge 3) 4 [4447] 4447 innerTail601_1
    (if 3 ≤ innerCandidates601_0.length then search edge 3 [4447, 4789] innerCandidates601_0 9236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan601_1

theorem rootBranch601 : search edge 4 [4447] rootCandidates601 4447 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4447] 4447 innerTail601_0 best = best
  exact innerScan601_0

end ProjectEulerSolutions.PrimePairUpper.Search
