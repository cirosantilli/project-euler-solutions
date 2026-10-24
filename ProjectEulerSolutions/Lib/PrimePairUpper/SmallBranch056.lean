import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates612 : List Nat := [4817, 4967, 5039, 5081, 5153, 5639, 6173, 6329, 6551, 6761, 6911, 7457, 7583, 7937, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates612_checked : candidates edge 5 0 4523 tail613 best.bound = rootCandidates612 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext q; rfl
  rw [row]
  decide

def innerTail612_15 : List Nat := []
theorem innerScan612_15 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_15 best = best := rfl

def innerTail612_14 : List Nat := 8093 :: innerTail612_15
def innerCandidates612_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates612_14_checked : candidates edge 4 4523 8093 innerTail612_15 best.bound = innerCandidates612_14 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_14 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_14 best = best := by
  rw [innerTail612_14, scan, if_neg (by decide), innerCandidates612_14_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_15
    (if 3 ≤ innerCandidates612_14.length then search edge 3 [4523, 8093] innerCandidates612_14 12616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_15

def innerTail612_13 : List Nat := 7937 :: innerTail612_14
def innerCandidates612_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates612_13_checked : candidates edge 4 4523 7937 innerTail612_14 best.bound = innerCandidates612_13 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_13 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_13 best = best := by
  rw [innerTail612_13, scan, if_neg (by decide), innerCandidates612_13_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_14
    (if 3 ≤ innerCandidates612_13.length then search edge 3 [4523, 7937] innerCandidates612_13 12460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_14

def innerTail612_12 : List Nat := 7583 :: innerTail612_13
def innerCandidates612_12 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates612_12_checked : candidates edge 4 4523 7583 innerTail612_13 best.bound = innerCandidates612_12 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_12 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_12 best = best := by
  rw [innerTail612_12, scan, if_neg (by decide), innerCandidates612_12_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_13
    (if 3 ≤ innerCandidates612_12.length then search edge 3 [4523, 7583] innerCandidates612_12 12106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_13

def innerTail612_11 : List Nat := 7457 :: innerTail612_12
def innerCandidates612_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates612_11_checked : candidates edge 4 4523 7457 innerTail612_12 best.bound = innerCandidates612_11 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_11 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_11 best = best := by
  rw [innerTail612_11, scan, if_neg (by decide), innerCandidates612_11_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_12
    (if 3 ≤ innerCandidates612_11.length then search edge 3 [4523, 7457] innerCandidates612_11 11980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_12

def innerTail612_10 : List Nat := 6911 :: innerTail612_11
def innerCandidates612_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates612_10_checked : candidates edge 4 4523 6911 innerTail612_11 best.bound = innerCandidates612_10 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_10 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_10 best = best := by
  rw [innerTail612_10, scan, if_neg (by decide), innerCandidates612_10_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_11
    (if 3 ≤ innerCandidates612_10.length then search edge 3 [4523, 6911] innerCandidates612_10 11434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_11

def innerTail612_9 : List Nat := 6761 :: innerTail612_10
def innerCandidates612_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates612_9_checked : candidates edge 4 4523 6761 innerTail612_10 best.bound = innerCandidates612_9 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_9 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_9 best = best := by
  rw [innerTail612_9, scan, if_neg (by decide), innerCandidates612_9_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_10
    (if 3 ≤ innerCandidates612_9.length then search edge 3 [4523, 6761] innerCandidates612_9 11284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_10

def innerTail612_8 : List Nat := 6551 :: innerTail612_9
def innerCandidates612_8 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates612_8_checked : candidates edge 4 4523 6551 innerTail612_9 best.bound = innerCandidates612_8 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_8 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_8 best = best := by
  rw [innerTail612_8, scan, if_neg (by decide), innerCandidates612_8_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_9
    (if 3 ≤ innerCandidates612_8.length then search edge 3 [4523, 6551] innerCandidates612_8 11074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_9

def innerTail612_7 : List Nat := 6329 :: innerTail612_8
def innerCandidates612_7 : List Nat := [6551, 7937]
set_option maxRecDepth 4096 in
theorem innerCandidates612_7_checked : candidates edge 4 4523 6329 innerTail612_8 best.bound = innerCandidates612_7 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_7 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_7 best = best := by
  rw [innerTail612_7, scan, if_neg (by decide), innerCandidates612_7_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_8
    (if 3 ≤ innerCandidates612_7.length then search edge 3 [4523, 6329] innerCandidates612_7 10852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_8

def innerTail612_6 : List Nat := 6173 :: innerTail612_7
def innerCandidates612_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates612_6_checked : candidates edge 4 4523 6173 innerTail612_7 best.bound = innerCandidates612_6 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_6 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_6 best = best := by
  rw [innerTail612_6, scan, if_neg (by decide), innerCandidates612_6_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_7
    (if 3 ≤ innerCandidates612_6.length then search edge 3 [4523, 6173] innerCandidates612_6 10696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_7

def innerTail612_5 : List Nat := 5639 :: innerTail612_6
def innerCandidates612_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates612_5_checked : candidates edge 4 4523 5639 innerTail612_6 best.bound = innerCandidates612_5 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_5 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_5 best = best := by
  rw [innerTail612_5, scan, if_neg (by decide), innerCandidates612_5_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_6
    (if 3 ≤ innerCandidates612_5.length then search edge 3 [4523, 5639] innerCandidates612_5 10162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_6

def innerTail612_4 : List Nat := 5153 :: innerTail612_5
def innerCandidates612_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates612_4_checked : candidates edge 4 4523 5153 innerTail612_5 best.bound = innerCandidates612_4 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_4 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_4 best = best := by
  rw [innerTail612_4, scan, if_neg (by decide), innerCandidates612_4_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_5
    (if 3 ≤ innerCandidates612_4.length then search edge 3 [4523, 5153] innerCandidates612_4 9676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_5

def innerTail612_3 : List Nat := 5081 :: innerTail612_4
def innerCandidates612_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates612_3_checked : candidates edge 4 4523 5081 innerTail612_4 best.bound = innerCandidates612_3 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_3 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_3 best = best := by
  rw [innerTail612_3, scan, if_neg (by decide), innerCandidates612_3_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_4
    (if 3 ≤ innerCandidates612_3.length then search edge 3 [4523, 5081] innerCandidates612_3 9604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_4

def innerTail612_2 : List Nat := 5039 :: innerTail612_3
def innerCandidates612_2 : List Nat := [5153, 6329, 8093]
set_option maxRecDepth 4096 in
theorem innerCandidates612_2_checked : candidates edge 4 4523 5039 innerTail612_3 best.bound = innerCandidates612_2 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild612_2 : search edge 3 [4523, 5039] innerCandidates612_2 9562 best = best := by decide

theorem innerScan612_2 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_2 best = best := by
  rw [innerTail612_2, scan, if_neg (by decide), innerCandidates612_2_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_3
    (if 3 ≤ innerCandidates612_2.length then search edge 3 [4523, 5039] innerCandidates612_2 9562 best else best) = best
  rw [if_pos (by decide), innerChild612_2]
  exact innerScan612_3

def innerTail612_1 : List Nat := 4967 :: innerTail612_2
def innerCandidates612_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates612_1_checked : candidates edge 4 4523 4967 innerTail612_2 best.bound = innerCandidates612_1 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_1 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_1 best = best := by
  rw [innerTail612_1, scan, if_neg (by decide), innerCandidates612_1_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_2
    (if 3 ≤ innerCandidates612_1.length then search edge 3 [4523, 4967] innerCandidates612_1 9490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_2

def innerTail612_0 : List Nat := 4817 :: innerTail612_1
def innerCandidates612_0 : List Nat := [6173]
set_option maxRecDepth 4096 in
theorem innerCandidates612_0_checked : candidates edge 4 4523 4817 innerTail612_1 best.bound = innerCandidates612_0 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan612_0 : scan edge (search edge 3) 4 [4523] 4523 innerTail612_0 best = best := by
  rw [innerTail612_0, scan, if_neg (by decide), innerCandidates612_0_checked]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_1
    (if 3 ≤ innerCandidates612_0.length then search edge 3 [4523, 4817] innerCandidates612_0 9340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan612_1

theorem rootBranch612 : search edge 4 [4523] rootCandidates612 4523 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4523] 4523 innerTail612_0 best = best
  exact innerScan612_0

def rootCandidates611 : List Nat := [4657, 4729, 5209, 5419, 5701, 6043, 6133]
set_option maxRecDepth 4096 in
theorem rootCandidates611_checked : candidates edge 5 0 4519 tail612 best.bound = rootCandidates611 := by
  unfold candidates
  have row : edge 4519 = SearchData.row4519.contains := by funext q; rfl
  rw [row]
  decide

def innerTail611_7 : List Nat := []
theorem innerScan611_7 : scan edge (search edge 3) 4 [4519] 4519 innerTail611_7 best = best := rfl

def innerTail611_6 : List Nat := 6133 :: innerTail611_7
def innerCandidates611_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates611_6_checked : candidates edge 4 4519 6133 innerTail611_7 best.bound = innerCandidates611_6 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan611_6 : scan edge (search edge 3) 4 [4519] 4519 innerTail611_6 best = best := by
  rw [innerTail611_6, scan, if_neg (by decide), innerCandidates611_6_checked]
  change scan edge (search edge 3) 4 [4519] 4519 innerTail611_7
    (if 3 ≤ innerCandidates611_6.length then search edge 3 [4519, 6133] innerCandidates611_6 10652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan611_7

def innerTail611_5 : List Nat := 6043 :: innerTail611_6
def innerCandidates611_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates611_5_checked : candidates edge 4 4519 6043 innerTail611_6 best.bound = innerCandidates611_5 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan611_5 : scan edge (search edge 3) 4 [4519] 4519 innerTail611_5 best = best := by
  rw [innerTail611_5, scan, if_neg (by decide), innerCandidates611_5_checked]
  change scan edge (search edge 3) 4 [4519] 4519 innerTail611_6
    (if 3 ≤ innerCandidates611_5.length then search edge 3 [4519, 6043] innerCandidates611_5 10562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan611_6

def innerTail611_4 : List Nat := 5701 :: innerTail611_5
def innerCandidates611_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates611_4_checked : candidates edge 4 4519 5701 innerTail611_5 best.bound = innerCandidates611_4 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan611_4 : scan edge (search edge 3) 4 [4519] 4519 innerTail611_4 best = best := by
  rw [innerTail611_4, scan, if_neg (by decide), innerCandidates611_4_checked]
  change scan edge (search edge 3) 4 [4519] 4519 innerTail611_5
    (if 3 ≤ innerCandidates611_4.length then search edge 3 [4519, 5701] innerCandidates611_4 10220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan611_5

def innerTail611_3 : List Nat := 5419 :: innerTail611_4
def innerCandidates611_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates611_3_checked : candidates edge 4 4519 5419 innerTail611_4 best.bound = innerCandidates611_3 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan611_3 : scan edge (search edge 3) 4 [4519] 4519 innerTail611_3 best = best := by
  rw [innerTail611_3, scan, if_neg (by decide), innerCandidates611_3_checked]
  change scan edge (search edge 3) 4 [4519] 4519 innerTail611_4
    (if 3 ≤ innerCandidates611_3.length then search edge 3 [4519, 5419] innerCandidates611_3 9938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan611_4

def innerTail611_2 : List Nat := 5209 :: innerTail611_3
def innerCandidates611_2 : List Nat := [5419]
set_option maxRecDepth 4096 in
theorem innerCandidates611_2_checked : candidates edge 4 4519 5209 innerTail611_3 best.bound = innerCandidates611_2 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan611_2 : scan edge (search edge 3) 4 [4519] 4519 innerTail611_2 best = best := by
  rw [innerTail611_2, scan, if_neg (by decide), innerCandidates611_2_checked]
  change scan edge (search edge 3) 4 [4519] 4519 innerTail611_3
    (if 3 ≤ innerCandidates611_2.length then search edge 3 [4519, 5209] innerCandidates611_2 9728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan611_3

def innerTail611_1 : List Nat := 4729 :: innerTail611_2
def innerCandidates611_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates611_1_checked : candidates edge 4 4519 4729 innerTail611_2 best.bound = innerCandidates611_1 := by
  unfold candidates
  have row : edge 4729 = SearchData.row4729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan611_1 : scan edge (search edge 3) 4 [4519] 4519 innerTail611_1 best = best := by
  rw [innerTail611_1, scan, if_neg (by decide), innerCandidates611_1_checked]
  change scan edge (search edge 3) 4 [4519] 4519 innerTail611_2
    (if 3 ≤ innerCandidates611_1.length then search edge 3 [4519, 4729] innerCandidates611_1 9248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan611_2

def innerTail611_0 : List Nat := 4657 :: innerTail611_1
def innerCandidates611_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates611_0_checked : candidates edge 4 4519 4657 innerTail611_1 best.bound = innerCandidates611_0 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan611_0 : scan edge (search edge 3) 4 [4519] 4519 innerTail611_0 best = best := by
  rw [innerTail611_0, scan, if_neg (by decide), innerCandidates611_0_checked]
  change scan edge (search edge 3) 4 [4519] 4519 innerTail611_1
    (if 3 ≤ innerCandidates611_0.length then search edge 3 [4519, 4657] innerCandidates611_0 9176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan611_1

theorem rootBranch611 : search edge 4 [4519] rootCandidates611 4519 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4519] 4519 innerTail611_0 best = best
  exact innerScan611_0

def rootCandidates610 : List Nat := [5087, 5279, 5387, 5849, 6581, 6689, 6869, 6959, 7523, 8087, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates610_checked : candidates edge 5 0 4517 tail611 best.bound = rootCandidates610 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext q; rfl
  rw [row]
  decide

def innerTail610_11 : List Nat := []
theorem innerScan610_11 : scan edge (search edge 3) 4 [4517] 4517 innerTail610_11 best = best := rfl

def innerTail610_10 : List Nat := 8387 :: innerTail610_11
def innerCandidates610_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates610_10_checked : candidates edge 4 4517 8387 innerTail610_11 best.bound = innerCandidates610_10 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan610_10 : scan edge (search edge 3) 4 [4517] 4517 innerTail610_10 best = best := by
  rw [innerTail610_10, scan, if_neg (by decide), innerCandidates610_10_checked]
  change scan edge (search edge 3) 4 [4517] 4517 innerTail610_11
    (if 3 ≤ innerCandidates610_10.length then search edge 3 [4517, 8387] innerCandidates610_10 12904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan610_11

def innerTail610_9 : List Nat := 8087 :: innerTail610_10
def innerCandidates610_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates610_9_checked : candidates edge 4 4517 8087 innerTail610_10 best.bound = innerCandidates610_9 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan610_9 : scan edge (search edge 3) 4 [4517] 4517 innerTail610_9 best = best := by
  rw [innerTail610_9, scan, if_neg (by decide), innerCandidates610_9_checked]
  change scan edge (search edge 3) 4 [4517] 4517 innerTail610_10
    (if 3 ≤ innerCandidates610_9.length then search edge 3 [4517, 8087] innerCandidates610_9 12604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan610_10

def innerTail610_8 : List Nat := 7523 :: innerTail610_9
def innerCandidates610_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates610_8_checked : candidates edge 4 4517 7523 innerTail610_9 best.bound = innerCandidates610_8 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan610_8 : scan edge (search edge 3) 4 [4517] 4517 innerTail610_8 best = best := by
  rw [innerTail610_8, scan, if_neg (by decide), innerCandidates610_8_checked]
  change scan edge (search edge 3) 4 [4517] 4517 innerTail610_9
    (if 3 ≤ innerCandidates610_8.length then search edge 3 [4517, 7523] innerCandidates610_8 12040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan610_9

def innerTail610_7 : List Nat := 6959 :: innerTail610_8
def innerCandidates610_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates610_7_checked : candidates edge 4 4517 6959 innerTail610_8 best.bound = innerCandidates610_7 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan610_7 : scan edge (search edge 3) 4 [4517] 4517 innerTail610_7 best = best := by
  rw [innerTail610_7, scan, if_neg (by decide), innerCandidates610_7_checked]
  change scan edge (search edge 3) 4 [4517] 4517 innerTail610_8
    (if 3 ≤ innerCandidates610_7.length then search edge 3 [4517, 6959] innerCandidates610_7 11476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan610_8

def innerTail610_6 : List Nat := 6869 :: innerTail610_7
def innerCandidates610_6 : List Nat := [6959, 8087]
set_option maxRecDepth 4096 in
theorem innerCandidates610_6_checked : candidates edge 4 4517 6869 innerTail610_7 best.bound = innerCandidates610_6 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan610_6 : scan edge (search edge 3) 4 [4517] 4517 innerTail610_6 best = best := by
  rw [innerTail610_6, scan, if_neg (by decide), innerCandidates610_6_checked]
  change scan edge (search edge 3) 4 [4517] 4517 innerTail610_7
    (if 3 ≤ innerCandidates610_6.length then search edge 3 [4517, 6869] innerCandidates610_6 11386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan610_7

def innerTail610_5 : List Nat := 6689 :: innerTail610_6
def innerCandidates610_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates610_5_checked : candidates edge 4 4517 6689 innerTail610_6 best.bound = innerCandidates610_5 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan610_5 : scan edge (search edge 3) 4 [4517] 4517 innerTail610_5 best = best := by
  rw [innerTail610_5, scan, if_neg (by decide), innerCandidates610_5_checked]
  change scan edge (search edge 3) 4 [4517] 4517 innerTail610_6
    (if 3 ≤ innerCandidates610_5.length then search edge 3 [4517, 6689] innerCandidates610_5 11206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan610_6

def innerTail610_4 : List Nat := 6581 :: innerTail610_5
def innerCandidates610_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates610_4_checked : candidates edge 4 4517 6581 innerTail610_5 best.bound = innerCandidates610_4 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan610_4 : scan edge (search edge 3) 4 [4517] 4517 innerTail610_4 best = best := by
  rw [innerTail610_4, scan, if_neg (by decide), innerCandidates610_4_checked]
  change scan edge (search edge 3) 4 [4517] 4517 innerTail610_5
    (if 3 ≤ innerCandidates610_4.length then search edge 3 [4517, 6581] innerCandidates610_4 11098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan610_5

def innerTail610_3 : List Nat := 5849 :: innerTail610_4
def innerCandidates610_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates610_3_checked : candidates edge 4 4517 5849 innerTail610_4 best.bound = innerCandidates610_3 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan610_3 : scan edge (search edge 3) 4 [4517] 4517 innerTail610_3 best = best := by
  rw [innerTail610_3, scan, if_neg (by decide), innerCandidates610_3_checked]
  change scan edge (search edge 3) 4 [4517] 4517 innerTail610_4
    (if 3 ≤ innerCandidates610_3.length then search edge 3 [4517, 5849] innerCandidates610_3 10366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan610_4

def innerTail610_2 : List Nat := 5387 :: innerTail610_3
def innerCandidates610_2 : List Nat := [6959]
set_option maxRecDepth 4096 in
theorem innerCandidates610_2_checked : candidates edge 4 4517 5387 innerTail610_3 best.bound = innerCandidates610_2 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan610_2 : scan edge (search edge 3) 4 [4517] 4517 innerTail610_2 best = best := by
  rw [innerTail610_2, scan, if_neg (by decide), innerCandidates610_2_checked]
  change scan edge (search edge 3) 4 [4517] 4517 innerTail610_3
    (if 3 ≤ innerCandidates610_2.length then search edge 3 [4517, 5387] innerCandidates610_2 9904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan610_3

def innerTail610_1 : List Nat := 5279 :: innerTail610_2
def innerCandidates610_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates610_1_checked : candidates edge 4 4517 5279 innerTail610_2 best.bound = innerCandidates610_1 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan610_1 : scan edge (search edge 3) 4 [4517] 4517 innerTail610_1 best = best := by
  rw [innerTail610_1, scan, if_neg (by decide), innerCandidates610_1_checked]
  change scan edge (search edge 3) 4 [4517] 4517 innerTail610_2
    (if 3 ≤ innerCandidates610_1.length then search edge 3 [4517, 5279] innerCandidates610_1 9796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan610_2

def innerTail610_0 : List Nat := 5087 :: innerTail610_1
def innerCandidates610_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates610_0_checked : candidates edge 4 4517 5087 innerTail610_1 best.bound = innerCandidates610_0 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan610_0 : scan edge (search edge 3) 4 [4517] 4517 innerTail610_0 best = best := by
  rw [innerTail610_0, scan, if_neg (by decide), innerCandidates610_0_checked]
  change scan edge (search edge 3) 4 [4517] 4517 innerTail610_1
    (if 3 ≤ innerCandidates610_0.length then search edge 3 [4517, 5087] innerCandidates610_0 9604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan610_1

theorem rootBranch610 : search edge 4 [4517] rootCandidates610 4517 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4517] 4517 innerTail610_0 best = best
  exact innerScan610_0

def rootCandidates609 : List Nat := [4591, 5437, 5581, 6163, 6277, 6397, 6841, 7723, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates609_checked : candidates edge 5 0 4513 tail610 best.bound = rootCandidates609 := by
  unfold candidates
  have row : edge 4513 = SearchData.row4513.contains := by funext q; rfl
  rw [row]
  decide

def innerTail609_9 : List Nat := []
theorem innerScan609_9 : scan edge (search edge 3) 4 [4513] 4513 innerTail609_9 best = best := rfl

def innerTail609_8 : List Nat := 8287 :: innerTail609_9
def innerCandidates609_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates609_8_checked : candidates edge 4 4513 8287 innerTail609_9 best.bound = innerCandidates609_8 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan609_8 : scan edge (search edge 3) 4 [4513] 4513 innerTail609_8 best = best := by
  rw [innerTail609_8, scan, if_neg (by decide), innerCandidates609_8_checked]
  change scan edge (search edge 3) 4 [4513] 4513 innerTail609_9
    (if 3 ≤ innerCandidates609_8.length then search edge 3 [4513, 8287] innerCandidates609_8 12800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan609_9

def innerTail609_7 : List Nat := 7723 :: innerTail609_8
def innerCandidates609_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates609_7_checked : candidates edge 4 4513 7723 innerTail609_8 best.bound = innerCandidates609_7 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan609_7 : scan edge (search edge 3) 4 [4513] 4513 innerTail609_7 best = best := by
  rw [innerTail609_7, scan, if_neg (by decide), innerCandidates609_7_checked]
  change scan edge (search edge 3) 4 [4513] 4513 innerTail609_8
    (if 3 ≤ innerCandidates609_7.length then search edge 3 [4513, 7723] innerCandidates609_7 12236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan609_8

def innerTail609_6 : List Nat := 6841 :: innerTail609_7
def innerCandidates609_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates609_6_checked : candidates edge 4 4513 6841 innerTail609_7 best.bound = innerCandidates609_6 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan609_6 : scan edge (search edge 3) 4 [4513] 4513 innerTail609_6 best = best := by
  rw [innerTail609_6, scan, if_neg (by decide), innerCandidates609_6_checked]
  change scan edge (search edge 3) 4 [4513] 4513 innerTail609_7
    (if 3 ≤ innerCandidates609_6.length then search edge 3 [4513, 6841] innerCandidates609_6 11354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan609_7

def innerTail609_5 : List Nat := 6397 :: innerTail609_6
def innerCandidates609_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates609_5_checked : candidates edge 4 4513 6397 innerTail609_6 best.bound = innerCandidates609_5 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan609_5 : scan edge (search edge 3) 4 [4513] 4513 innerTail609_5 best = best := by
  rw [innerTail609_5, scan, if_neg (by decide), innerCandidates609_5_checked]
  change scan edge (search edge 3) 4 [4513] 4513 innerTail609_6
    (if 3 ≤ innerCandidates609_5.length then search edge 3 [4513, 6397] innerCandidates609_5 10910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan609_6

def innerTail609_4 : List Nat := 6277 :: innerTail609_5
def innerCandidates609_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates609_4_checked : candidates edge 4 4513 6277 innerTail609_5 best.bound = innerCandidates609_4 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan609_4 : scan edge (search edge 3) 4 [4513] 4513 innerTail609_4 best = best := by
  rw [innerTail609_4, scan, if_neg (by decide), innerCandidates609_4_checked]
  change scan edge (search edge 3) 4 [4513] 4513 innerTail609_5
    (if 3 ≤ innerCandidates609_4.length then search edge 3 [4513, 6277] innerCandidates609_4 10790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan609_5

def innerTail609_3 : List Nat := 6163 :: innerTail609_4
def innerCandidates609_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates609_3_checked : candidates edge 4 4513 6163 innerTail609_4 best.bound = innerCandidates609_3 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan609_3 : scan edge (search edge 3) 4 [4513] 4513 innerTail609_3 best = best := by
  rw [innerTail609_3, scan, if_neg (by decide), innerCandidates609_3_checked]
  change scan edge (search edge 3) 4 [4513] 4513 innerTail609_4
    (if 3 ≤ innerCandidates609_3.length then search edge 3 [4513, 6163] innerCandidates609_3 10676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan609_4

def innerTail609_2 : List Nat := 5581 :: innerTail609_3
def innerCandidates609_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates609_2_checked : candidates edge 4 4513 5581 innerTail609_3 best.bound = innerCandidates609_2 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan609_2 : scan edge (search edge 3) 4 [4513] 4513 innerTail609_2 best = best := by
  rw [innerTail609_2, scan, if_neg (by decide), innerCandidates609_2_checked]
  change scan edge (search edge 3) 4 [4513] 4513 innerTail609_3
    (if 3 ≤ innerCandidates609_2.length then search edge 3 [4513, 5581] innerCandidates609_2 10094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan609_3

def innerTail609_1 : List Nat := 5437 :: innerTail609_2
def innerCandidates609_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates609_1_checked : candidates edge 4 4513 5437 innerTail609_2 best.bound = innerCandidates609_1 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan609_1 : scan edge (search edge 3) 4 [4513] 4513 innerTail609_1 best = best := by
  rw [innerTail609_1, scan, if_neg (by decide), innerCandidates609_1_checked]
  change scan edge (search edge 3) 4 [4513] 4513 innerTail609_2
    (if 3 ≤ innerCandidates609_1.length then search edge 3 [4513, 5437] innerCandidates609_1 9950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan609_2

def innerTail609_0 : List Nat := 4591 :: innerTail609_1
def innerCandidates609_0 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates609_0_checked : candidates edge 4 4513 4591 innerTail609_1 best.bound = innerCandidates609_0 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan609_0 : scan edge (search edge 3) 4 [4513] 4513 innerTail609_0 best = best := by
  rw [innerTail609_0, scan, if_neg (by decide), innerCandidates609_0_checked]
  change scan edge (search edge 3) 4 [4513] 4513 innerTail609_1
    (if 3 ≤ innerCandidates609_0.length then search edge 3 [4513, 4591] innerCandidates609_0 9104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan609_1

theorem rootBranch609 : search edge 4 [4513] rootCandidates609 4513 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4513] 4513 innerTail609_0 best = best
  exact innerScan609_0

def rootCandidates608 : List Nat := [4789, 4801, 4999, 5419, 5881, 6091, 6217, 6709, 7129, 7603, 7759, 7993, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates608_checked : candidates edge 5 0 4507 tail609 best.bound = rootCandidates608 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext q; rfl
  rw [row]
  decide

def innerTail608_13 : List Nat := []
theorem innerScan608_13 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_13 best = best := rfl

def innerTail608_12 : List Nat := 8377 :: innerTail608_13
def innerCandidates608_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates608_12_checked : candidates edge 4 4507 8377 innerTail608_13 best.bound = innerCandidates608_12 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_12 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_12 best = best := by
  rw [innerTail608_12, scan, if_neg (by decide), innerCandidates608_12_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_13
    (if 3 ≤ innerCandidates608_12.length then search edge 3 [4507, 8377] innerCandidates608_12 12884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_13

def innerTail608_11 : List Nat := 7993 :: innerTail608_12
def innerCandidates608_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates608_11_checked : candidates edge 4 4507 7993 innerTail608_12 best.bound = innerCandidates608_11 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_11 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_11 best = best := by
  rw [innerTail608_11, scan, if_neg (by decide), innerCandidates608_11_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_12
    (if 3 ≤ innerCandidates608_11.length then search edge 3 [4507, 7993] innerCandidates608_11 12500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_12

def innerTail608_10 : List Nat := 7759 :: innerTail608_11
def innerCandidates608_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates608_10_checked : candidates edge 4 4507 7759 innerTail608_11 best.bound = innerCandidates608_10 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_10 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_10 best = best := by
  rw [innerTail608_10, scan, if_neg (by decide), innerCandidates608_10_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_11
    (if 3 ≤ innerCandidates608_10.length then search edge 3 [4507, 7759] innerCandidates608_10 12266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_11

def innerTail608_9 : List Nat := 7603 :: innerTail608_10
def innerCandidates608_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates608_9_checked : candidates edge 4 4507 7603 innerTail608_10 best.bound = innerCandidates608_9 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_9 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_9 best = best := by
  rw [innerTail608_9, scan, if_neg (by decide), innerCandidates608_9_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_10
    (if 3 ≤ innerCandidates608_9.length then search edge 3 [4507, 7603] innerCandidates608_9 12110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_10

def innerTail608_8 : List Nat := 7129 :: innerTail608_9
def innerCandidates608_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates608_8_checked : candidates edge 4 4507 7129 innerTail608_9 best.bound = innerCandidates608_8 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_8 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_8 best = best := by
  rw [innerTail608_8, scan, if_neg (by decide), innerCandidates608_8_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_9
    (if 3 ≤ innerCandidates608_8.length then search edge 3 [4507, 7129] innerCandidates608_8 11636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_9

def innerTail608_7 : List Nat := 6709 :: innerTail608_8
def innerCandidates608_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates608_7_checked : candidates edge 4 4507 6709 innerTail608_8 best.bound = innerCandidates608_7 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_7 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_7 best = best := by
  rw [innerTail608_7, scan, if_neg (by decide), innerCandidates608_7_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_8
    (if 3 ≤ innerCandidates608_7.length then search edge 3 [4507, 6709] innerCandidates608_7 11216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_8

def innerTail608_6 : List Nat := 6217 :: innerTail608_7
def innerCandidates608_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates608_6_checked : candidates edge 4 4507 6217 innerTail608_7 best.bound = innerCandidates608_6 := by
  unfold candidates
  have row : edge 6217 = SearchData.row6217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_6 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_6 best = best := by
  rw [innerTail608_6, scan, if_neg (by decide), innerCandidates608_6_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_7
    (if 3 ≤ innerCandidates608_6.length then search edge 3 [4507, 6217] innerCandidates608_6 10724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_7

def innerTail608_5 : List Nat := 6091 :: innerTail608_6
def innerCandidates608_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates608_5_checked : candidates edge 4 4507 6091 innerTail608_6 best.bound = innerCandidates608_5 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_5 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_5 best = best := by
  rw [innerTail608_5, scan, if_neg (by decide), innerCandidates608_5_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_6
    (if 3 ≤ innerCandidates608_5.length then search edge 3 [4507, 6091] innerCandidates608_5 10598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_6

def innerTail608_4 : List Nat := 5881 :: innerTail608_5
def innerCandidates608_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates608_4_checked : candidates edge 4 4507 5881 innerTail608_5 best.bound = innerCandidates608_4 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_4 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_4 best = best := by
  rw [innerTail608_4, scan, if_neg (by decide), innerCandidates608_4_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_5
    (if 3 ≤ innerCandidates608_4.length then search edge 3 [4507, 5881] innerCandidates608_4 10388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_5

def innerTail608_3 : List Nat := 5419 :: innerTail608_4
def innerCandidates608_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates608_3_checked : candidates edge 4 4507 5419 innerTail608_4 best.bound = innerCandidates608_3 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_3 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_3 best = best := by
  rw [innerTail608_3, scan, if_neg (by decide), innerCandidates608_3_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_4
    (if 3 ≤ innerCandidates608_3.length then search edge 3 [4507, 5419] innerCandidates608_3 9926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_4

def innerTail608_2 : List Nat := 4999 :: innerTail608_3
def innerCandidates608_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates608_2_checked : candidates edge 4 4507 4999 innerTail608_3 best.bound = innerCandidates608_2 := by
  unfold candidates
  have row : edge 4999 = SearchData.row4999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_2 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_2 best = best := by
  rw [innerTail608_2, scan, if_neg (by decide), innerCandidates608_2_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_3
    (if 3 ≤ innerCandidates608_2.length then search edge 3 [4507, 4999] innerCandidates608_2 9506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_3

def innerTail608_1 : List Nat := 4801 :: innerTail608_2
def innerCandidates608_1 : List Nat := [5419, 7993]
set_option maxRecDepth 4096 in
theorem innerCandidates608_1_checked : candidates edge 4 4507 4801 innerTail608_2 best.bound = innerCandidates608_1 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_1 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_1 best = best := by
  rw [innerTail608_1, scan, if_neg (by decide), innerCandidates608_1_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_2
    (if 3 ≤ innerCandidates608_1.length then search edge 3 [4507, 4801] innerCandidates608_1 9308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_2

def innerTail608_0 : List Nat := 4789 :: innerTail608_1
def innerCandidates608_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates608_0_checked : candidates edge 4 4507 4789 innerTail608_1 best.bound = innerCandidates608_0 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan608_0 : scan edge (search edge 3) 4 [4507] 4507 innerTail608_0 best = best := by
  rw [innerTail608_0, scan, if_neg (by decide), innerCandidates608_0_checked]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_1
    (if 3 ≤ innerCandidates608_0.length then search edge 3 [4507, 4789] innerCandidates608_0 9296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan608_1

theorem rootBranch608 : search edge 4 [4507] rootCandidates608 4507 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4507] 4507 innerTail608_0 best = best
  exact innerScan608_0

def rootCandidates607 : List Nat := [4793, 5381, 5507, 5519, 5717, 6053, 6173, 7187, 7349]
set_option maxRecDepth 4096 in
theorem rootCandidates607_checked : candidates edge 5 0 4493 tail608 best.bound = rootCandidates607 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext q; rfl
  rw [row]
  decide

def innerTail607_9 : List Nat := []
theorem innerScan607_9 : scan edge (search edge 3) 4 [4493] 4493 innerTail607_9 best = best := rfl

def innerTail607_8 : List Nat := 7349 :: innerTail607_9
def innerCandidates607_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates607_8_checked : candidates edge 4 4493 7349 innerTail607_9 best.bound = innerCandidates607_8 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan607_8 : scan edge (search edge 3) 4 [4493] 4493 innerTail607_8 best = best := by
  rw [innerTail607_8, scan, if_neg (by decide), innerCandidates607_8_checked]
  change scan edge (search edge 3) 4 [4493] 4493 innerTail607_9
    (if 3 ≤ innerCandidates607_8.length then search edge 3 [4493, 7349] innerCandidates607_8 11842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan607_9

def innerTail607_7 : List Nat := 7187 :: innerTail607_8
def innerCandidates607_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates607_7_checked : candidates edge 4 4493 7187 innerTail607_8 best.bound = innerCandidates607_7 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan607_7 : scan edge (search edge 3) 4 [4493] 4493 innerTail607_7 best = best := by
  rw [innerTail607_7, scan, if_neg (by decide), innerCandidates607_7_checked]
  change scan edge (search edge 3) 4 [4493] 4493 innerTail607_8
    (if 3 ≤ innerCandidates607_7.length then search edge 3 [4493, 7187] innerCandidates607_7 11680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan607_8

def innerTail607_6 : List Nat := 6173 :: innerTail607_7
def innerCandidates607_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates607_6_checked : candidates edge 4 4493 6173 innerTail607_7 best.bound = innerCandidates607_6 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan607_6 : scan edge (search edge 3) 4 [4493] 4493 innerTail607_6 best = best := by
  rw [innerTail607_6, scan, if_neg (by decide), innerCandidates607_6_checked]
  change scan edge (search edge 3) 4 [4493] 4493 innerTail607_7
    (if 3 ≤ innerCandidates607_6.length then search edge 3 [4493, 6173] innerCandidates607_6 10666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan607_7

def innerTail607_5 : List Nat := 6053 :: innerTail607_6
def innerCandidates607_5 : List Nat := [6173]
set_option maxRecDepth 4096 in
theorem innerCandidates607_5_checked : candidates edge 4 4493 6053 innerTail607_6 best.bound = innerCandidates607_5 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan607_5 : scan edge (search edge 3) 4 [4493] 4493 innerTail607_5 best = best := by
  rw [innerTail607_5, scan, if_neg (by decide), innerCandidates607_5_checked]
  change scan edge (search edge 3) 4 [4493] 4493 innerTail607_6
    (if 3 ≤ innerCandidates607_5.length then search edge 3 [4493, 6053] innerCandidates607_5 10546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan607_6

def innerTail607_4 : List Nat := 5717 :: innerTail607_5
def innerCandidates607_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates607_4_checked : candidates edge 4 4493 5717 innerTail607_5 best.bound = innerCandidates607_4 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan607_4 : scan edge (search edge 3) 4 [4493] 4493 innerTail607_4 best = best := by
  rw [innerTail607_4, scan, if_neg (by decide), innerCandidates607_4_checked]
  change scan edge (search edge 3) 4 [4493] 4493 innerTail607_5
    (if 3 ≤ innerCandidates607_4.length then search edge 3 [4493, 5717] innerCandidates607_4 10210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan607_5

def innerTail607_3 : List Nat := 5519 :: innerTail607_4
def innerCandidates607_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates607_3_checked : candidates edge 4 4493 5519 innerTail607_4 best.bound = innerCandidates607_3 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan607_3 : scan edge (search edge 3) 4 [4493] 4493 innerTail607_3 best = best := by
  rw [innerTail607_3, scan, if_neg (by decide), innerCandidates607_3_checked]
  change scan edge (search edge 3) 4 [4493] 4493 innerTail607_4
    (if 3 ≤ innerCandidates607_3.length then search edge 3 [4493, 5519] innerCandidates607_3 10012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan607_4

def innerTail607_2 : List Nat := 5507 :: innerTail607_3
def innerCandidates607_2 : List Nat := [6173]
set_option maxRecDepth 4096 in
theorem innerCandidates607_2_checked : candidates edge 4 4493 5507 innerTail607_3 best.bound = innerCandidates607_2 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan607_2 : scan edge (search edge 3) 4 [4493] 4493 innerTail607_2 best = best := by
  rw [innerTail607_2, scan, if_neg (by decide), innerCandidates607_2_checked]
  change scan edge (search edge 3) 4 [4493] 4493 innerTail607_3
    (if 3 ≤ innerCandidates607_2.length then search edge 3 [4493, 5507] innerCandidates607_2 10000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan607_3

def innerTail607_1 : List Nat := 5381 :: innerTail607_2
def innerCandidates607_1 : List Nat := [5507]
set_option maxRecDepth 4096 in
theorem innerCandidates607_1_checked : candidates edge 4 4493 5381 innerTail607_2 best.bound = innerCandidates607_1 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan607_1 : scan edge (search edge 3) 4 [4493] 4493 innerTail607_1 best = best := by
  rw [innerTail607_1, scan, if_neg (by decide), innerCandidates607_1_checked]
  change scan edge (search edge 3) 4 [4493] 4493 innerTail607_2
    (if 3 ≤ innerCandidates607_1.length then search edge 3 [4493, 5381] innerCandidates607_1 9874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan607_2

def innerTail607_0 : List Nat := 4793 :: innerTail607_1
def innerCandidates607_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates607_0_checked : candidates edge 4 4493 4793 innerTail607_1 best.bound = innerCandidates607_0 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan607_0 : scan edge (search edge 3) 4 [4493] 4493 innerTail607_0 best = best := by
  rw [innerTail607_0, scan, if_neg (by decide), innerCandidates607_0_checked]
  change scan edge (search edge 3) 4 [4493] 4493 innerTail607_1
    (if 3 ≤ innerCandidates607_0.length then search edge 3 [4493, 4793] innerCandidates607_0 9286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan607_1

theorem rootBranch607 : search edge 4 [4493] rootCandidates607 4493 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4493] 4493 innerTail607_0 best = best
  exact innerScan607_0

end ProjectEulerSolutions.PrimePairUpper.Search
