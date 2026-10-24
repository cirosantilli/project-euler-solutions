import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates491 : List Nat := [3593, 4133, 4583, 4679, 4703, 5099, 5483, 6113, 7019, 7079, 7607, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates491_checked : candidates edge 5 0 3533 tail492 best.bound = rootCandidates491 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext q; rfl
  rw [row]
  decide

def innerTail491_12 : List Nat := []
theorem innerScan491_12 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_12 best = best := rfl

def innerTail491_11 : List Nat := 8273 :: innerTail491_12
def innerCandidates491_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates491_11_checked : candidates edge 4 3533 8273 innerTail491_12 best.bound = innerCandidates491_11 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan491_11 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_11 best = best := by
  rw [innerTail491_11, scan, if_neg (by decide), innerCandidates491_11_checked]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_12
    (if 3 ≤ innerCandidates491_11.length then search edge 3 [3533, 8273] innerCandidates491_11 11806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan491_12

def innerTail491_10 : List Nat := 7607 :: innerTail491_11
def innerCandidates491_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates491_10_checked : candidates edge 4 3533 7607 innerTail491_11 best.bound = innerCandidates491_10 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan491_10 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_10 best = best := by
  rw [innerTail491_10, scan, if_neg (by decide), innerCandidates491_10_checked]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_11
    (if 3 ≤ innerCandidates491_10.length then search edge 3 [3533, 7607] innerCandidates491_10 11140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan491_11

def innerTail491_9 : List Nat := 7079 :: innerTail491_10
def innerCandidates491_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates491_9_checked : candidates edge 4 3533 7079 innerTail491_10 best.bound = innerCandidates491_9 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan491_9 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_9 best = best := by
  rw [innerTail491_9, scan, if_neg (by decide), innerCandidates491_9_checked]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_10
    (if 3 ≤ innerCandidates491_9.length then search edge 3 [3533, 7079] innerCandidates491_9 10612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan491_10

def innerTail491_8 : List Nat := 7019 :: innerTail491_9
def innerCandidates491_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates491_8_checked : candidates edge 4 3533 7019 innerTail491_9 best.bound = innerCandidates491_8 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan491_8 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_8 best = best := by
  rw [innerTail491_8, scan, if_neg (by decide), innerCandidates491_8_checked]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_9
    (if 3 ≤ innerCandidates491_8.length then search edge 3 [3533, 7019] innerCandidates491_8 10552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan491_9

def innerTail491_7 : List Nat := 6113 :: innerTail491_8
def innerCandidates491_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates491_7_checked : candidates edge 4 3533 6113 innerTail491_8 best.bound = innerCandidates491_7 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan491_7 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_7 best = best := by
  rw [innerTail491_7, scan, if_neg (by decide), innerCandidates491_7_checked]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_8
    (if 3 ≤ innerCandidates491_7.length then search edge 3 [3533, 6113] innerCandidates491_7 9646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan491_8

def innerTail491_6 : List Nat := 5483 :: innerTail491_7
def innerCandidates491_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates491_6_checked : candidates edge 4 3533 5483 innerTail491_7 best.bound = innerCandidates491_6 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan491_6 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_6 best = best := by
  rw [innerTail491_6, scan, if_neg (by decide), innerCandidates491_6_checked]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_7
    (if 3 ≤ innerCandidates491_6.length then search edge 3 [3533, 5483] innerCandidates491_6 9016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan491_7

def innerTail491_5 : List Nat := 5099 :: innerTail491_6
def innerCandidates491_5 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates491_5_checked : candidates edge 4 3533 5099 innerTail491_6 best.bound = innerCandidates491_5 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan491_5 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_5 best = best := by
  rw [innerTail491_5, scan, if_neg (by decide), innerCandidates491_5_checked]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_6
    (if 3 ≤ innerCandidates491_5.length then search edge 3 [3533, 5099] innerCandidates491_5 8632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan491_6

def innerTail491_4 : List Nat := 4703 :: innerTail491_5
def innerCandidates491_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates491_4_checked : candidates edge 4 3533 4703 innerTail491_5 best.bound = innerCandidates491_4 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan491_4 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_4 best = best := by
  rw [innerTail491_4, scan, if_neg (by decide), innerCandidates491_4_checked]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_5
    (if 3 ≤ innerCandidates491_4.length then search edge 3 [3533, 4703] innerCandidates491_4 8236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan491_5

def innerTail491_3 : List Nat := 4679 :: innerTail491_4
def innerCandidates491_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates491_3_checked : candidates edge 4 3533 4679 innerTail491_4 best.bound = innerCandidates491_3 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan491_3 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_3 best = best := by
  rw [innerTail491_3, scan, if_neg (by decide), innerCandidates491_3_checked]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_4
    (if 3 ≤ innerCandidates491_3.length then search edge 3 [3533, 4679] innerCandidates491_3 8212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan491_4

def innerTail491_2 : List Nat := 4583 :: innerTail491_3
def innerCandidates491_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates491_2_checked : candidates edge 4 3533 4583 innerTail491_3 best.bound = innerCandidates491_2 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan491_2 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_2 best = best := by
  rw [innerTail491_2, scan, if_neg (by decide), innerCandidates491_2_checked]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_3
    (if 3 ≤ innerCandidates491_2.length then search edge 3 [3533, 4583] innerCandidates491_2 8116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan491_3

def innerTail491_1 : List Nat := 4133 :: innerTail491_2
def innerCandidates491_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates491_1_checked : candidates edge 4 3533 4133 innerTail491_2 best.bound = innerCandidates491_1 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan491_1 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_1 best = best := by
  rw [innerTail491_1, scan, if_neg (by decide), innerCandidates491_1_checked]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_2
    (if 3 ≤ innerCandidates491_1.length then search edge 3 [3533, 4133] innerCandidates491_1 7666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan491_2

def innerTail491_0 : List Nat := 3593 :: innerTail491_1
def innerCandidates491_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates491_0_checked : candidates edge 4 3533 3593 innerTail491_1 best.bound = innerCandidates491_0 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan491_0 : scan edge (search edge 3) 4 [3533] 3533 innerTail491_0 best = best := by
  rw [innerTail491_0, scan, if_neg (by decide), innerCandidates491_0_checked]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_1
    (if 3 ≤ innerCandidates491_0.length then search edge 3 [3533, 3593] innerCandidates491_0 7126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan491_1

theorem rootBranch491 : search edge 4 [3533] rootCandidates491 3533 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3533] 3533 innerTail491_0 best = best
  exact innerScan491_0

def rootCandidates490 : List Nat := [3943, 4423, 4657, 4969, 5011, 5527, 5569, 5851, 6397, 6427, 6553, 6733, 7393, 8059]
set_option maxRecDepth 4096 in
theorem rootCandidates490_checked : candidates edge 5 0 3529 tail491 best.bound = rootCandidates490 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext q; rfl
  rw [row]
  decide

def innerTail490_14 : List Nat := []
theorem innerScan490_14 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_14 best = best := rfl

def innerTail490_13 : List Nat := 8059 :: innerTail490_14
def innerCandidates490_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates490_13_checked : candidates edge 4 3529 8059 innerTail490_14 best.bound = innerCandidates490_13 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_13 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_13 best = best := by
  rw [innerTail490_13, scan, if_neg (by decide), innerCandidates490_13_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_14
    (if 3 ≤ innerCandidates490_13.length then search edge 3 [3529, 8059] innerCandidates490_13 11588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_14

def innerTail490_12 : List Nat := 7393 :: innerTail490_13
def innerCandidates490_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates490_12_checked : candidates edge 4 3529 7393 innerTail490_13 best.bound = innerCandidates490_12 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_12 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_12 best = best := by
  rw [innerTail490_12, scan, if_neg (by decide), innerCandidates490_12_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_13
    (if 3 ≤ innerCandidates490_12.length then search edge 3 [3529, 7393] innerCandidates490_12 10922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_13

def innerTail490_11 : List Nat := 6733 :: innerTail490_12
def innerCandidates490_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates490_11_checked : candidates edge 4 3529 6733 innerTail490_12 best.bound = innerCandidates490_11 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_11 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_11 best = best := by
  rw [innerTail490_11, scan, if_neg (by decide), innerCandidates490_11_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_12
    (if 3 ≤ innerCandidates490_11.length then search edge 3 [3529, 6733] innerCandidates490_11 10262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_12

def innerTail490_10 : List Nat := 6553 :: innerTail490_11
def innerCandidates490_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates490_10_checked : candidates edge 4 3529 6553 innerTail490_11 best.bound = innerCandidates490_10 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_10 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_10 best = best := by
  rw [innerTail490_10, scan, if_neg (by decide), innerCandidates490_10_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_11
    (if 3 ≤ innerCandidates490_10.length then search edge 3 [3529, 6553] innerCandidates490_10 10082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_11

def innerTail490_9 : List Nat := 6427 :: innerTail490_10
def innerCandidates490_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates490_9_checked : candidates edge 4 3529 6427 innerTail490_10 best.bound = innerCandidates490_9 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_9 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_9 best = best := by
  rw [innerTail490_9, scan, if_neg (by decide), innerCandidates490_9_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_10
    (if 3 ≤ innerCandidates490_9.length then search edge 3 [3529, 6427] innerCandidates490_9 9956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_10

def innerTail490_8 : List Nat := 6397 :: innerTail490_9
def innerCandidates490_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates490_8_checked : candidates edge 4 3529 6397 innerTail490_9 best.bound = innerCandidates490_8 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_8 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_8 best = best := by
  rw [innerTail490_8, scan, if_neg (by decide), innerCandidates490_8_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_9
    (if 3 ≤ innerCandidates490_8.length then search edge 3 [3529, 6397] innerCandidates490_8 9926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_9

def innerTail490_7 : List Nat := 5851 :: innerTail490_8
def innerCandidates490_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates490_7_checked : candidates edge 4 3529 5851 innerTail490_8 best.bound = innerCandidates490_7 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_7 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_7 best = best := by
  rw [innerTail490_7, scan, if_neg (by decide), innerCandidates490_7_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_8
    (if 3 ≤ innerCandidates490_7.length then search edge 3 [3529, 5851] innerCandidates490_7 9380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_8

def innerTail490_6 : List Nat := 5569 :: innerTail490_7
def innerCandidates490_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates490_6_checked : candidates edge 4 3529 5569 innerTail490_7 best.bound = innerCandidates490_6 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_6 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_6 best = best := by
  rw [innerTail490_6, scan, if_neg (by decide), innerCandidates490_6_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_7
    (if 3 ≤ innerCandidates490_6.length then search edge 3 [3529, 5569] innerCandidates490_6 9098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_7

def innerTail490_5 : List Nat := 5527 :: innerTail490_6
def innerCandidates490_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates490_5_checked : candidates edge 4 3529 5527 innerTail490_6 best.bound = innerCandidates490_5 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_5 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_5 best = best := by
  rw [innerTail490_5, scan, if_neg (by decide), innerCandidates490_5_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_6
    (if 3 ≤ innerCandidates490_5.length then search edge 3 [3529, 5527] innerCandidates490_5 9056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_6

def innerTail490_4 : List Nat := 5011 :: innerTail490_5
def innerCandidates490_4 : List Nat := [6397, 7393]
set_option maxRecDepth 4096 in
theorem innerCandidates490_4_checked : candidates edge 4 3529 5011 innerTail490_5 best.bound = innerCandidates490_4 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_4 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_4 best = best := by
  rw [innerTail490_4, scan, if_neg (by decide), innerCandidates490_4_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_5
    (if 3 ≤ innerCandidates490_4.length then search edge 3 [3529, 5011] innerCandidates490_4 8540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_5

def innerTail490_3 : List Nat := 4969 :: innerTail490_4
def innerCandidates490_3 : List Nat := [6397]
set_option maxRecDepth 4096 in
theorem innerCandidates490_3_checked : candidates edge 4 3529 4969 innerTail490_4 best.bound = innerCandidates490_3 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_3 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_3 best = best := by
  rw [innerTail490_3, scan, if_neg (by decide), innerCandidates490_3_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_4
    (if 3 ≤ innerCandidates490_3.length then search edge 3 [3529, 4969] innerCandidates490_3 8498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_4

def innerTail490_2 : List Nat := 4657 :: innerTail490_3
def innerCandidates490_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates490_2_checked : candidates edge 4 3529 4657 innerTail490_3 best.bound = innerCandidates490_2 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_2 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_2 best = best := by
  rw [innerTail490_2, scan, if_neg (by decide), innerCandidates490_2_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_3
    (if 3 ≤ innerCandidates490_2.length then search edge 3 [3529, 4657] innerCandidates490_2 8186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_3

def innerTail490_1 : List Nat := 4423 :: innerTail490_2
def innerCandidates490_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates490_1_checked : candidates edge 4 3529 4423 innerTail490_2 best.bound = innerCandidates490_1 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_1 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_1 best = best := by
  rw [innerTail490_1, scan, if_neg (by decide), innerCandidates490_1_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_2
    (if 3 ≤ innerCandidates490_1.length then search edge 3 [3529, 4423] innerCandidates490_1 7952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_2

def innerTail490_0 : List Nat := 3943 :: innerTail490_1
def innerCandidates490_0 : List Nat := [5569, 8059]
set_option maxRecDepth 4096 in
theorem innerCandidates490_0_checked : candidates edge 4 3529 3943 innerTail490_1 best.bound = innerCandidates490_0 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan490_0 : scan edge (search edge 3) 4 [3529] 3529 innerTail490_0 best = best := by
  rw [innerTail490_0, scan, if_neg (by decide), innerCandidates490_0_checked]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_1
    (if 3 ≤ innerCandidates490_0.length then search edge 3 [3529, 3943] innerCandidates490_0 7472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan490_1

theorem rootBranch490 : search edge 4 [3529] rootCandidates490 3529 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3529] 3529 innerTail490_0 best = best
  exact innerScan490_0

def rootCandidates489 : List Nat := [3533, 4457, 4649, 6551, 6803, 7013, 7901]
set_option maxRecDepth 4096 in
theorem rootCandidates489_checked : candidates edge 5 0 3527 tail490 best.bound = rootCandidates489 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext q; rfl
  rw [row]
  decide

def innerTail489_7 : List Nat := []
theorem innerScan489_7 : scan edge (search edge 3) 4 [3527] 3527 innerTail489_7 best = best := rfl

def innerTail489_6 : List Nat := 7901 :: innerTail489_7
def innerCandidates489_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates489_6_checked : candidates edge 4 3527 7901 innerTail489_7 best.bound = innerCandidates489_6 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan489_6 : scan edge (search edge 3) 4 [3527] 3527 innerTail489_6 best = best := by
  rw [innerTail489_6, scan, if_neg (by decide), innerCandidates489_6_checked]
  change scan edge (search edge 3) 4 [3527] 3527 innerTail489_7
    (if 3 ≤ innerCandidates489_6.length then search edge 3 [3527, 7901] innerCandidates489_6 11428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan489_7

def innerTail489_5 : List Nat := 7013 :: innerTail489_6
def innerCandidates489_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates489_5_checked : candidates edge 4 3527 7013 innerTail489_6 best.bound = innerCandidates489_5 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan489_5 : scan edge (search edge 3) 4 [3527] 3527 innerTail489_5 best = best := by
  rw [innerTail489_5, scan, if_neg (by decide), innerCandidates489_5_checked]
  change scan edge (search edge 3) 4 [3527] 3527 innerTail489_6
    (if 3 ≤ innerCandidates489_5.length then search edge 3 [3527, 7013] innerCandidates489_5 10540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan489_6

def innerTail489_4 : List Nat := 6803 :: innerTail489_5
def innerCandidates489_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates489_4_checked : candidates edge 4 3527 6803 innerTail489_5 best.bound = innerCandidates489_4 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan489_4 : scan edge (search edge 3) 4 [3527] 3527 innerTail489_4 best = best := by
  rw [innerTail489_4, scan, if_neg (by decide), innerCandidates489_4_checked]
  change scan edge (search edge 3) 4 [3527] 3527 innerTail489_5
    (if 3 ≤ innerCandidates489_4.length then search edge 3 [3527, 6803] innerCandidates489_4 10330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan489_5

def innerTail489_3 : List Nat := 6551 :: innerTail489_4
def innerCandidates489_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates489_3_checked : candidates edge 4 3527 6551 innerTail489_4 best.bound = innerCandidates489_3 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan489_3 : scan edge (search edge 3) 4 [3527] 3527 innerTail489_3 best = best := by
  rw [innerTail489_3, scan, if_neg (by decide), innerCandidates489_3_checked]
  change scan edge (search edge 3) 4 [3527] 3527 innerTail489_4
    (if 3 ≤ innerCandidates489_3.length then search edge 3 [3527, 6551] innerCandidates489_3 10078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan489_4

def innerTail489_2 : List Nat := 4649 :: innerTail489_3
def innerCandidates489_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates489_2_checked : candidates edge 4 3527 4649 innerTail489_3 best.bound = innerCandidates489_2 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan489_2 : scan edge (search edge 3) 4 [3527] 3527 innerTail489_2 best = best := by
  rw [innerTail489_2, scan, if_neg (by decide), innerCandidates489_2_checked]
  change scan edge (search edge 3) 4 [3527] 3527 innerTail489_3
    (if 3 ≤ innerCandidates489_2.length then search edge 3 [3527, 4649] innerCandidates489_2 8176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan489_3

def innerTail489_1 : List Nat := 4457 :: innerTail489_2
def innerCandidates489_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates489_1_checked : candidates edge 4 3527 4457 innerTail489_2 best.bound = innerCandidates489_1 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan489_1 : scan edge (search edge 3) 4 [3527] 3527 innerTail489_1 best = best := by
  rw [innerTail489_1, scan, if_neg (by decide), innerCandidates489_1_checked]
  change scan edge (search edge 3) 4 [3527] 3527 innerTail489_2
    (if 3 ≤ innerCandidates489_1.length then search edge 3 [3527, 4457] innerCandidates489_1 7984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan489_2

def innerTail489_0 : List Nat := 3533 :: innerTail489_1
def innerCandidates489_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates489_0_checked : candidates edge 4 3527 3533 innerTail489_1 best.bound = innerCandidates489_0 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan489_0 : scan edge (search edge 3) 4 [3527] 3527 innerTail489_0 best = best := by
  rw [innerTail489_0, scan, if_neg (by decide), innerCandidates489_0_checked]
  change scan edge (search edge 3) 4 [3527] 3527 innerTail489_1
    (if 3 ≤ innerCandidates489_0.length then search edge 3 [3527, 3533] innerCandidates489_0 7060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan489_1

theorem rootBranch489 : search edge 4 [3527] rootCandidates489 3527 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3527] 3527 innerTail489_0 best = best
  exact innerScan489_0

def rootCandidates488 : List Nat := [4099, 4177, 5659, 6829, 6907, 7159, 7507, 7621, 7669, 7717, 8161, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates488_checked : candidates edge 5 0 3517 tail489 best.bound = rootCandidates488 := by
  unfold candidates
  have row : edge 3517 = SearchData.row3517.contains := by funext q; rfl
  rw [row]
  decide

def innerTail488_12 : List Nat := []
theorem innerScan488_12 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_12 best = best := rfl

def innerTail488_11 : List Nat := 8263 :: innerTail488_12
def innerCandidates488_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates488_11_checked : candidates edge 4 3517 8263 innerTail488_12 best.bound = innerCandidates488_11 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan488_11 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_11 best = best := by
  rw [innerTail488_11, scan, if_neg (by decide), innerCandidates488_11_checked]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_12
    (if 3 ≤ innerCandidates488_11.length then search edge 3 [3517, 8263] innerCandidates488_11 11780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan488_12

def innerTail488_10 : List Nat := 8161 :: innerTail488_11
def innerCandidates488_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates488_10_checked : candidates edge 4 3517 8161 innerTail488_11 best.bound = innerCandidates488_10 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan488_10 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_10 best = best := by
  rw [innerTail488_10, scan, if_neg (by decide), innerCandidates488_10_checked]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_11
    (if 3 ≤ innerCandidates488_10.length then search edge 3 [3517, 8161] innerCandidates488_10 11678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan488_11

def innerTail488_9 : List Nat := 7717 :: innerTail488_10
def innerCandidates488_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates488_9_checked : candidates edge 4 3517 7717 innerTail488_10 best.bound = innerCandidates488_9 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan488_9 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_9 best = best := by
  rw [innerTail488_9, scan, if_neg (by decide), innerCandidates488_9_checked]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_10
    (if 3 ≤ innerCandidates488_9.length then search edge 3 [3517, 7717] innerCandidates488_9 11234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan488_10

def innerTail488_8 : List Nat := 7669 :: innerTail488_9
def innerCandidates488_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates488_8_checked : candidates edge 4 3517 7669 innerTail488_9 best.bound = innerCandidates488_8 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan488_8 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_8 best = best := by
  rw [innerTail488_8, scan, if_neg (by decide), innerCandidates488_8_checked]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_9
    (if 3 ≤ innerCandidates488_8.length then search edge 3 [3517, 7669] innerCandidates488_8 11186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan488_9

def innerTail488_7 : List Nat := 7621 :: innerTail488_8
def innerCandidates488_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates488_7_checked : candidates edge 4 3517 7621 innerTail488_8 best.bound = innerCandidates488_7 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan488_7 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_7 best = best := by
  rw [innerTail488_7, scan, if_neg (by decide), innerCandidates488_7_checked]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_8
    (if 3 ≤ innerCandidates488_7.length then search edge 3 [3517, 7621] innerCandidates488_7 11138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan488_8

def innerTail488_6 : List Nat := 7507 :: innerTail488_7
def innerCandidates488_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates488_6_checked : candidates edge 4 3517 7507 innerTail488_7 best.bound = innerCandidates488_6 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan488_6 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_6 best = best := by
  rw [innerTail488_6, scan, if_neg (by decide), innerCandidates488_6_checked]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_7
    (if 3 ≤ innerCandidates488_6.length then search edge 3 [3517, 7507] innerCandidates488_6 11024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan488_7

def innerTail488_5 : List Nat := 7159 :: innerTail488_6
def innerCandidates488_5 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates488_5_checked : candidates edge 4 3517 7159 innerTail488_6 best.bound = innerCandidates488_5 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan488_5 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_5 best = best := by
  rw [innerTail488_5, scan, if_neg (by decide), innerCandidates488_5_checked]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_6
    (if 3 ≤ innerCandidates488_5.length then search edge 3 [3517, 7159] innerCandidates488_5 10676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan488_6

def innerTail488_4 : List Nat := 6907 :: innerTail488_5
def innerCandidates488_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates488_4_checked : candidates edge 4 3517 6907 innerTail488_5 best.bound = innerCandidates488_4 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan488_4 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_4 best = best := by
  rw [innerTail488_4, scan, if_neg (by decide), innerCandidates488_4_checked]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_5
    (if 3 ≤ innerCandidates488_4.length then search edge 3 [3517, 6907] innerCandidates488_4 10424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan488_5

def innerTail488_3 : List Nat := 6829 :: innerTail488_4
def innerCandidates488_3 : List Nat := [7507]
set_option maxRecDepth 4096 in
theorem innerCandidates488_3_checked : candidates edge 4 3517 6829 innerTail488_4 best.bound = innerCandidates488_3 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan488_3 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_3 best = best := by
  rw [innerTail488_3, scan, if_neg (by decide), innerCandidates488_3_checked]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_4
    (if 3 ≤ innerCandidates488_3.length then search edge 3 [3517, 6829] innerCandidates488_3 10346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan488_4

def innerTail488_2 : List Nat := 5659 :: innerTail488_3
def innerCandidates488_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates488_2_checked : candidates edge 4 3517 5659 innerTail488_3 best.bound = innerCandidates488_2 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan488_2 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_2 best = best := by
  rw [innerTail488_2, scan, if_neg (by decide), innerCandidates488_2_checked]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_3
    (if 3 ≤ innerCandidates488_2.length then search edge 3 [3517, 5659] innerCandidates488_2 9176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan488_3

def innerTail488_1 : List Nat := 4177 :: innerTail488_2
def innerCandidates488_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates488_1_checked : candidates edge 4 3517 4177 innerTail488_2 best.bound = innerCandidates488_1 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan488_1 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_1 best = best := by
  rw [innerTail488_1, scan, if_neg (by decide), innerCandidates488_1_checked]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_2
    (if 3 ≤ innerCandidates488_1.length then search edge 3 [3517, 4177] innerCandidates488_1 7694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan488_2

def innerTail488_0 : List Nat := 4099 :: innerTail488_1
def innerCandidates488_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates488_0_checked : candidates edge 4 3517 4099 innerTail488_1 best.bound = innerCandidates488_0 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan488_0 : scan edge (search edge 3) 4 [3517] 3517 innerTail488_0 best = best := by
  rw [innerTail488_0, scan, if_neg (by decide), innerCandidates488_0_checked]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_1
    (if 3 ≤ innerCandidates488_0.length then search edge 3 [3517, 4099] innerCandidates488_0 7616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan488_1

theorem rootBranch488 : search edge 4 [3517] rootCandidates488 3517 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3517] 3517 innerTail488_0 best = best
  exact innerScan488_0

def rootCandidates487 : List Nat := [3889, 4153, 5233, 6967, 7561, 7879, 7963, 8011, 8089]
set_option maxRecDepth 4096 in
theorem rootCandidates487_checked : candidates edge 5 0 3511 tail488 best.bound = rootCandidates487 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext q; rfl
  rw [row]
  decide

def innerTail487_9 : List Nat := []
theorem innerScan487_9 : scan edge (search edge 3) 4 [3511] 3511 innerTail487_9 best = best := rfl

def innerTail487_8 : List Nat := 8089 :: innerTail487_9
def innerCandidates487_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates487_8_checked : candidates edge 4 3511 8089 innerTail487_9 best.bound = innerCandidates487_8 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan487_8 : scan edge (search edge 3) 4 [3511] 3511 innerTail487_8 best = best := by
  rw [innerTail487_8, scan, if_neg (by decide), innerCandidates487_8_checked]
  change scan edge (search edge 3) 4 [3511] 3511 innerTail487_9
    (if 3 ≤ innerCandidates487_8.length then search edge 3 [3511, 8089] innerCandidates487_8 11600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan487_9

def innerTail487_7 : List Nat := 8011 :: innerTail487_8
def innerCandidates487_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates487_7_checked : candidates edge 4 3511 8011 innerTail487_8 best.bound = innerCandidates487_7 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan487_7 : scan edge (search edge 3) 4 [3511] 3511 innerTail487_7 best = best := by
  rw [innerTail487_7, scan, if_neg (by decide), innerCandidates487_7_checked]
  change scan edge (search edge 3) 4 [3511] 3511 innerTail487_8
    (if 3 ≤ innerCandidates487_7.length then search edge 3 [3511, 8011] innerCandidates487_7 11522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan487_8

def innerTail487_6 : List Nat := 7963 :: innerTail487_7
def innerCandidates487_6 : List Nat := [8011]
set_option maxRecDepth 4096 in
theorem innerCandidates487_6_checked : candidates edge 4 3511 7963 innerTail487_7 best.bound = innerCandidates487_6 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan487_6 : scan edge (search edge 3) 4 [3511] 3511 innerTail487_6 best = best := by
  rw [innerTail487_6, scan, if_neg (by decide), innerCandidates487_6_checked]
  change scan edge (search edge 3) 4 [3511] 3511 innerTail487_7
    (if 3 ≤ innerCandidates487_6.length then search edge 3 [3511, 7963] innerCandidates487_6 11474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan487_7

def innerTail487_5 : List Nat := 7879 :: innerTail487_6
def innerCandidates487_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates487_5_checked : candidates edge 4 3511 7879 innerTail487_6 best.bound = innerCandidates487_5 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan487_5 : scan edge (search edge 3) 4 [3511] 3511 innerTail487_5 best = best := by
  rw [innerTail487_5, scan, if_neg (by decide), innerCandidates487_5_checked]
  change scan edge (search edge 3) 4 [3511] 3511 innerTail487_6
    (if 3 ≤ innerCandidates487_5.length then search edge 3 [3511, 7879] innerCandidates487_5 11390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan487_6

def innerTail487_4 : List Nat := 7561 :: innerTail487_5
def innerCandidates487_4 : List Nat := [7963]
set_option maxRecDepth 4096 in
theorem innerCandidates487_4_checked : candidates edge 4 3511 7561 innerTail487_5 best.bound = innerCandidates487_4 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan487_4 : scan edge (search edge 3) 4 [3511] 3511 innerTail487_4 best = best := by
  rw [innerTail487_4, scan, if_neg (by decide), innerCandidates487_4_checked]
  change scan edge (search edge 3) 4 [3511] 3511 innerTail487_5
    (if 3 ≤ innerCandidates487_4.length then search edge 3 [3511, 7561] innerCandidates487_4 11072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan487_5

def innerTail487_3 : List Nat := 6967 :: innerTail487_4
def innerCandidates487_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates487_3_checked : candidates edge 4 3511 6967 innerTail487_4 best.bound = innerCandidates487_3 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan487_3 : scan edge (search edge 3) 4 [3511] 3511 innerTail487_3 best = best := by
  rw [innerTail487_3, scan, if_neg (by decide), innerCandidates487_3_checked]
  change scan edge (search edge 3) 4 [3511] 3511 innerTail487_4
    (if 3 ≤ innerCandidates487_3.length then search edge 3 [3511, 6967] innerCandidates487_3 10478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan487_4

def innerTail487_2 : List Nat := 5233 :: innerTail487_3
def innerCandidates487_2 : List Nat := [8089]
set_option maxRecDepth 4096 in
theorem innerCandidates487_2_checked : candidates edge 4 3511 5233 innerTail487_3 best.bound = innerCandidates487_2 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan487_2 : scan edge (search edge 3) 4 [3511] 3511 innerTail487_2 best = best := by
  rw [innerTail487_2, scan, if_neg (by decide), innerCandidates487_2_checked]
  change scan edge (search edge 3) 4 [3511] 3511 innerTail487_3
    (if 3 ≤ innerCandidates487_2.length then search edge 3 [3511, 5233] innerCandidates487_2 8744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan487_3

def innerTail487_1 : List Nat := 4153 :: innerTail487_2
def innerCandidates487_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates487_1_checked : candidates edge 4 3511 4153 innerTail487_2 best.bound = innerCandidates487_1 := by
  unfold candidates
  have row : edge 4153 = SearchData.row4153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan487_1 : scan edge (search edge 3) 4 [3511] 3511 innerTail487_1 best = best := by
  rw [innerTail487_1, scan, if_neg (by decide), innerCandidates487_1_checked]
  change scan edge (search edge 3) 4 [3511] 3511 innerTail487_2
    (if 3 ≤ innerCandidates487_1.length then search edge 3 [3511, 4153] innerCandidates487_1 7664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan487_2

def innerTail487_0 : List Nat := 3889 :: innerTail487_1
def innerCandidates487_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates487_0_checked : candidates edge 4 3511 3889 innerTail487_1 best.bound = innerCandidates487_0 := by
  unfold candidates
  have row : edge 3889 = SearchData.row3889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan487_0 : scan edge (search edge 3) 4 [3511] 3511 innerTail487_0 best = best := by
  rw [innerTail487_0, scan, if_neg (by decide), innerCandidates487_0_checked]
  change scan edge (search edge 3) 4 [3511] 3511 innerTail487_1
    (if 3 ≤ innerCandidates487_0.length then search edge 3 [3511, 3889] innerCandidates487_0 7400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan487_1

theorem rootBranch487 : search edge 4 [3511] rootCandidates487 3511 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3511] 3511 innerTail487_0 best = best
  exact innerScan487_0

end ProjectEulerSolutions.PrimePairUpper.Search
