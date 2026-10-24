import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates402 : List Nat := [2861, 3323, 3461, 4007, 4133, 4673, 4703, 5051, 5471, 5639, 6299, 6563, 6689, 7691, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates402_checked : candidates edge 5 0 2789 tail403 best.bound = rootCandidates402 := by
  unfold candidates
  have row : edge 2789 = SearchData.row2789.contains := by funext q; rfl
  rw [row]
  decide

def innerTail402_15 : List Nat := []
theorem innerScan402_15 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_15 best = best := rfl

def innerTail402_14 : List Nat := 8369 :: innerTail402_15
def innerCandidates402_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_14_checked : candidates edge 4 2789 8369 innerTail402_15 best.bound = innerCandidates402_14 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_14 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_14 best = best := by
  rw [innerTail402_14, scan, if_neg (by decide), innerCandidates402_14_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_15
    (if 3 ≤ innerCandidates402_14.length then search edge 3 [2789, 8369] innerCandidates402_14 11158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_15

def innerTail402_13 : List Nat := 7691 :: innerTail402_14
def innerCandidates402_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_13_checked : candidates edge 4 2789 7691 innerTail402_14 best.bound = innerCandidates402_13 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_13 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_13 best = best := by
  rw [innerTail402_13, scan, if_neg (by decide), innerCandidates402_13_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_14
    (if 3 ≤ innerCandidates402_13.length then search edge 3 [2789, 7691] innerCandidates402_13 10480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_14

def innerTail402_12 : List Nat := 6689 :: innerTail402_13
def innerCandidates402_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_12_checked : candidates edge 4 2789 6689 innerTail402_13 best.bound = innerCandidates402_12 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_12 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_12 best = best := by
  rw [innerTail402_12, scan, if_neg (by decide), innerCandidates402_12_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_13
    (if 3 ≤ innerCandidates402_12.length then search edge 3 [2789, 6689] innerCandidates402_12 9478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_13

def innerTail402_11 : List Nat := 6563 :: innerTail402_12
def innerCandidates402_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_11_checked : candidates edge 4 2789 6563 innerTail402_12 best.bound = innerCandidates402_11 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_11 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_11 best = best := by
  rw [innerTail402_11, scan, if_neg (by decide), innerCandidates402_11_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_12
    (if 3 ≤ innerCandidates402_11.length then search edge 3 [2789, 6563] innerCandidates402_11 9352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_12

def innerTail402_10 : List Nat := 6299 :: innerTail402_11
def innerCandidates402_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_10_checked : candidates edge 4 2789 6299 innerTail402_11 best.bound = innerCandidates402_10 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_10 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_10 best = best := by
  rw [innerTail402_10, scan, if_neg (by decide), innerCandidates402_10_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_11
    (if 3 ≤ innerCandidates402_10.length then search edge 3 [2789, 6299] innerCandidates402_10 9088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_11

def innerTail402_9 : List Nat := 5639 :: innerTail402_10
def innerCandidates402_9 : List Nat := [6563]
set_option maxRecDepth 4096 in
theorem innerCandidates402_9_checked : candidates edge 4 2789 5639 innerTail402_10 best.bound = innerCandidates402_9 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_9 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_9 best = best := by
  rw [innerTail402_9, scan, if_neg (by decide), innerCandidates402_9_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_10
    (if 3 ≤ innerCandidates402_9.length then search edge 3 [2789, 5639] innerCandidates402_9 8428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_10

def innerTail402_8 : List Nat := 5471 :: innerTail402_9
def innerCandidates402_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_8_checked : candidates edge 4 2789 5471 innerTail402_9 best.bound = innerCandidates402_8 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_8 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_8 best = best := by
  rw [innerTail402_8, scan, if_neg (by decide), innerCandidates402_8_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_9
    (if 3 ≤ innerCandidates402_8.length then search edge 3 [2789, 5471] innerCandidates402_8 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_9

def innerTail402_7 : List Nat := 5051 :: innerTail402_8
def innerCandidates402_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_7_checked : candidates edge 4 2789 5051 innerTail402_8 best.bound = innerCandidates402_7 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_7 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_7 best = best := by
  rw [innerTail402_7, scan, if_neg (by decide), innerCandidates402_7_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_8
    (if 3 ≤ innerCandidates402_7.length then search edge 3 [2789, 5051] innerCandidates402_7 7840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_8

def innerTail402_6 : List Nat := 4703 :: innerTail402_7
def innerCandidates402_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_6_checked : candidates edge 4 2789 4703 innerTail402_7 best.bound = innerCandidates402_6 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_6 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_6 best = best := by
  rw [innerTail402_6, scan, if_neg (by decide), innerCandidates402_6_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_7
    (if 3 ≤ innerCandidates402_6.length then search edge 3 [2789, 4703] innerCandidates402_6 7492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_7

def innerTail402_5 : List Nat := 4673 :: innerTail402_6
def innerCandidates402_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_5_checked : candidates edge 4 2789 4673 innerTail402_6 best.bound = innerCandidates402_5 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_5 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_5 best = best := by
  rw [innerTail402_5, scan, if_neg (by decide), innerCandidates402_5_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_6
    (if 3 ≤ innerCandidates402_5.length then search edge 3 [2789, 4673] innerCandidates402_5 7462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_6

def innerTail402_4 : List Nat := 4133 :: innerTail402_5
def innerCandidates402_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_4_checked : candidates edge 4 2789 4133 innerTail402_5 best.bound = innerCandidates402_4 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_4 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_4 best = best := by
  rw [innerTail402_4, scan, if_neg (by decide), innerCandidates402_4_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_5
    (if 3 ≤ innerCandidates402_4.length then search edge 3 [2789, 4133] innerCandidates402_4 6922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_5

def innerTail402_3 : List Nat := 4007 :: innerTail402_4
def innerCandidates402_3 : List Nat := [5639, 8369]
set_option maxRecDepth 4096 in
theorem innerCandidates402_3_checked : candidates edge 4 2789 4007 innerTail402_4 best.bound = innerCandidates402_3 := by
  unfold candidates
  have row : edge 4007 = SearchData.row4007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_3 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_3 best = best := by
  rw [innerTail402_3, scan, if_neg (by decide), innerCandidates402_3_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_4
    (if 3 ≤ innerCandidates402_3.length then search edge 3 [2789, 4007] innerCandidates402_3 6796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_4

def innerTail402_2 : List Nat := 3461 :: innerTail402_3
def innerCandidates402_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_2_checked : candidates edge 4 2789 3461 innerTail402_3 best.bound = innerCandidates402_2 := by
  unfold candidates
  have row : edge 3461 = SearchData.row3461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_2 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_2 best = best := by
  rw [innerTail402_2, scan, if_neg (by decide), innerCandidates402_2_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_3
    (if 3 ≤ innerCandidates402_2.length then search edge 3 [2789, 3461] innerCandidates402_2 6250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_3

def innerTail402_1 : List Nat := 3323 :: innerTail402_2
def innerCandidates402_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_1_checked : candidates edge 4 2789 3323 innerTail402_2 best.bound = innerCandidates402_1 := by
  unfold candidates
  have row : edge 3323 = SearchData.row3323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_1 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_1 best = best := by
  rw [innerTail402_1, scan, if_neg (by decide), innerCandidates402_1_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_2
    (if 3 ≤ innerCandidates402_1.length then search edge 3 [2789, 3323] innerCandidates402_1 6112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_2

def innerTail402_0 : List Nat := 2861 :: innerTail402_1
def innerCandidates402_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates402_0_checked : candidates edge 4 2789 2861 innerTail402_1 best.bound = innerCandidates402_0 := by
  unfold candidates
  have row : edge 2861 = SearchData.row2861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan402_0 : scan edge (search edge 3) 4 [2789] 2789 innerTail402_0 best = best := by
  rw [innerTail402_0, scan, if_neg (by decide), innerCandidates402_0_checked]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_1
    (if 3 ≤ innerCandidates402_0.length then search edge 3 [2789, 2861] innerCandidates402_0 5650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan402_1

theorem rootBranch402 : search edge 4 [2789] rootCandidates402 2789 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2789] 2789 innerTail402_0 best = best
  exact innerScan402_0

def rootCandidates401 : List Nat := [4289, 4793, 4973, 5477, 6173, 6473, 6971, 8039, 8087, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates401_checked : candidates edge 5 0 2777 tail402 best.bound = rootCandidates401 := by
  unfold candidates
  have row : edge 2777 = SearchData.row2777.contains := by funext q; rfl
  rw [row]
  decide

def innerTail401_10 : List Nat := []
theorem innerScan401_10 : scan edge (search edge 3) 4 [2777] 2777 innerTail401_10 best = best := rfl

def innerTail401_9 : List Nat := 8291 :: innerTail401_10
def innerCandidates401_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates401_9_checked : candidates edge 4 2777 8291 innerTail401_10 best.bound = innerCandidates401_9 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan401_9 : scan edge (search edge 3) 4 [2777] 2777 innerTail401_9 best = best := by
  rw [innerTail401_9, scan, if_neg (by decide), innerCandidates401_9_checked]
  change scan edge (search edge 3) 4 [2777] 2777 innerTail401_10
    (if 3 ≤ innerCandidates401_9.length then search edge 3 [2777, 8291] innerCandidates401_9 11068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan401_10

def innerTail401_8 : List Nat := 8087 :: innerTail401_9
def innerCandidates401_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates401_8_checked : candidates edge 4 2777 8087 innerTail401_9 best.bound = innerCandidates401_8 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan401_8 : scan edge (search edge 3) 4 [2777] 2777 innerTail401_8 best = best := by
  rw [innerTail401_8, scan, if_neg (by decide), innerCandidates401_8_checked]
  change scan edge (search edge 3) 4 [2777] 2777 innerTail401_9
    (if 3 ≤ innerCandidates401_8.length then search edge 3 [2777, 8087] innerCandidates401_8 10864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan401_9

def innerTail401_7 : List Nat := 8039 :: innerTail401_8
def innerCandidates401_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates401_7_checked : candidates edge 4 2777 8039 innerTail401_8 best.bound = innerCandidates401_7 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan401_7 : scan edge (search edge 3) 4 [2777] 2777 innerTail401_7 best = best := by
  rw [innerTail401_7, scan, if_neg (by decide), innerCandidates401_7_checked]
  change scan edge (search edge 3) 4 [2777] 2777 innerTail401_8
    (if 3 ≤ innerCandidates401_7.length then search edge 3 [2777, 8039] innerCandidates401_7 10816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan401_8

def innerTail401_6 : List Nat := 6971 :: innerTail401_7
def innerCandidates401_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates401_6_checked : candidates edge 4 2777 6971 innerTail401_7 best.bound = innerCandidates401_6 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan401_6 : scan edge (search edge 3) 4 [2777] 2777 innerTail401_6 best = best := by
  rw [innerTail401_6, scan, if_neg (by decide), innerCandidates401_6_checked]
  change scan edge (search edge 3) 4 [2777] 2777 innerTail401_7
    (if 3 ≤ innerCandidates401_6.length then search edge 3 [2777, 6971] innerCandidates401_6 9748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan401_7

def innerTail401_5 : List Nat := 6473 :: innerTail401_6
def innerCandidates401_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates401_5_checked : candidates edge 4 2777 6473 innerTail401_6 best.bound = innerCandidates401_5 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan401_5 : scan edge (search edge 3) 4 [2777] 2777 innerTail401_5 best = best := by
  rw [innerTail401_5, scan, if_neg (by decide), innerCandidates401_5_checked]
  change scan edge (search edge 3) 4 [2777] 2777 innerTail401_6
    (if 3 ≤ innerCandidates401_5.length then search edge 3 [2777, 6473] innerCandidates401_5 9250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan401_6

def innerTail401_4 : List Nat := 6173 :: innerTail401_5
def innerCandidates401_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates401_4_checked : candidates edge 4 2777 6173 innerTail401_5 best.bound = innerCandidates401_4 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan401_4 : scan edge (search edge 3) 4 [2777] 2777 innerTail401_4 best = best := by
  rw [innerTail401_4, scan, if_neg (by decide), innerCandidates401_4_checked]
  change scan edge (search edge 3) 4 [2777] 2777 innerTail401_5
    (if 3 ≤ innerCandidates401_4.length then search edge 3 [2777, 6173] innerCandidates401_4 8950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan401_5

def innerTail401_3 : List Nat := 5477 :: innerTail401_4
def innerCandidates401_3 : List Nat := [8039]
set_option maxRecDepth 4096 in
theorem innerCandidates401_3_checked : candidates edge 4 2777 5477 innerTail401_4 best.bound = innerCandidates401_3 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan401_3 : scan edge (search edge 3) 4 [2777] 2777 innerTail401_3 best = best := by
  rw [innerTail401_3, scan, if_neg (by decide), innerCandidates401_3_checked]
  change scan edge (search edge 3) 4 [2777] 2777 innerTail401_4
    (if 3 ≤ innerCandidates401_3.length then search edge 3 [2777, 5477] innerCandidates401_3 8254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan401_4

def innerTail401_2 : List Nat := 4973 :: innerTail401_3
def innerCandidates401_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates401_2_checked : candidates edge 4 2777 4973 innerTail401_3 best.bound = innerCandidates401_2 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan401_2 : scan edge (search edge 3) 4 [2777] 2777 innerTail401_2 best = best := by
  rw [innerTail401_2, scan, if_neg (by decide), innerCandidates401_2_checked]
  change scan edge (search edge 3) 4 [2777] 2777 innerTail401_3
    (if 3 ≤ innerCandidates401_2.length then search edge 3 [2777, 4973] innerCandidates401_2 7750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan401_3

def innerTail401_1 : List Nat := 4793 :: innerTail401_2
def innerCandidates401_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates401_1_checked : candidates edge 4 2777 4793 innerTail401_2 best.bound = innerCandidates401_1 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan401_1 : scan edge (search edge 3) 4 [2777] 2777 innerTail401_1 best = best := by
  rw [innerTail401_1, scan, if_neg (by decide), innerCandidates401_1_checked]
  change scan edge (search edge 3) 4 [2777] 2777 innerTail401_2
    (if 3 ≤ innerCandidates401_1.length then search edge 3 [2777, 4793] innerCandidates401_1 7570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan401_2

def innerTail401_0 : List Nat := 4289 :: innerTail401_1
def innerCandidates401_0 : List Nat := [6473, 8087]
set_option maxRecDepth 4096 in
theorem innerCandidates401_0_checked : candidates edge 4 2777 4289 innerTail401_1 best.bound = innerCandidates401_0 := by
  unfold candidates
  have row : edge 4289 = SearchData.row4289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan401_0 : scan edge (search edge 3) 4 [2777] 2777 innerTail401_0 best = best := by
  rw [innerTail401_0, scan, if_neg (by decide), innerCandidates401_0_checked]
  change scan edge (search edge 3) 4 [2777] 2777 innerTail401_1
    (if 3 ≤ innerCandidates401_0.length then search edge 3 [2777, 4289] innerCandidates401_0 7066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan401_1

theorem rootBranch401 : search edge 4 [2777] rootCandidates401 2777 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2777] 2777 innerTail401_0 best = best
  exact innerScan401_0

def rootCandidates400 : List Nat := [3109, 3187, 3733, 3889, 3967, 4099, 4513, 4783, 5059, 5107, 5119, 5689, 5737, 6361, 6679, 6967, 7459, 7477, 7537, 7561, 7621, 7723, 7933]
set_option maxRecDepth 4096 in
theorem rootCandidates400_checked : candidates edge 5 0 2767 tail401 best.bound = rootCandidates400 := by
  unfold candidates
  have row : edge 2767 = SearchData.row2767.contains := by funext q; rfl
  rw [row]
  decide

def innerTail400_23 : List Nat := []
theorem innerScan400_23 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_23 best = best := rfl

def innerTail400_22 : List Nat := 7933 :: innerTail400_23
def innerCandidates400_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_22_checked : candidates edge 4 2767 7933 innerTail400_23 best.bound = innerCandidates400_22 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_22 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_22 best = best := by
  rw [innerTail400_22, scan, if_neg (by decide), innerCandidates400_22_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_23
    (if 3 ≤ innerCandidates400_22.length then search edge 3 [2767, 7933] innerCandidates400_22 10700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_23

def innerTail400_21 : List Nat := 7723 :: innerTail400_22
def innerCandidates400_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_21_checked : candidates edge 4 2767 7723 innerTail400_22 best.bound = innerCandidates400_21 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_21 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_21 best = best := by
  rw [innerTail400_21, scan, if_neg (by decide), innerCandidates400_21_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_22
    (if 3 ≤ innerCandidates400_21.length then search edge 3 [2767, 7723] innerCandidates400_21 10490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_22

def innerTail400_20 : List Nat := 7621 :: innerTail400_21
def innerCandidates400_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_20_checked : candidates edge 4 2767 7621 innerTail400_21 best.bound = innerCandidates400_20 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_20 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_20 best = best := by
  rw [innerTail400_20, scan, if_neg (by decide), innerCandidates400_20_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_21
    (if 3 ≤ innerCandidates400_20.length then search edge 3 [2767, 7621] innerCandidates400_20 10388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_21

def innerTail400_19 : List Nat := 7561 :: innerTail400_20
def innerCandidates400_19 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates400_19_checked : candidates edge 4 2767 7561 innerTail400_20 best.bound = innerCandidates400_19 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_19 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_19 best = best := by
  rw [innerTail400_19, scan, if_neg (by decide), innerCandidates400_19_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_20
    (if 3 ≤ innerCandidates400_19.length then search edge 3 [2767, 7561] innerCandidates400_19 10328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_20

def innerTail400_18 : List Nat := 7537 :: innerTail400_19
def innerCandidates400_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_18_checked : candidates edge 4 2767 7537 innerTail400_19 best.bound = innerCandidates400_18 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_18 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_18 best = best := by
  rw [innerTail400_18, scan, if_neg (by decide), innerCandidates400_18_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_19
    (if 3 ≤ innerCandidates400_18.length then search edge 3 [2767, 7537] innerCandidates400_18 10304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_19

def innerTail400_17 : List Nat := 7477 :: innerTail400_18
def innerCandidates400_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_17_checked : candidates edge 4 2767 7477 innerTail400_18 best.bound = innerCandidates400_17 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_17 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_17 best = best := by
  rw [innerTail400_17, scan, if_neg (by decide), innerCandidates400_17_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_18
    (if 3 ≤ innerCandidates400_17.length then search edge 3 [2767, 7477] innerCandidates400_17 10244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_18

def innerTail400_16 : List Nat := 7459 :: innerTail400_17
def innerCandidates400_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_16_checked : candidates edge 4 2767 7459 innerTail400_17 best.bound = innerCandidates400_16 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_16 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_16 best = best := by
  rw [innerTail400_16, scan, if_neg (by decide), innerCandidates400_16_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_17
    (if 3 ≤ innerCandidates400_16.length then search edge 3 [2767, 7459] innerCandidates400_16 10226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_17

def innerTail400_15 : List Nat := 6967 :: innerTail400_16
def innerCandidates400_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_15_checked : candidates edge 4 2767 6967 innerTail400_16 best.bound = innerCandidates400_15 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_15 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_15 best = best := by
  rw [innerTail400_15, scan, if_neg (by decide), innerCandidates400_15_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_16
    (if 3 ≤ innerCandidates400_15.length then search edge 3 [2767, 6967] innerCandidates400_15 9734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_16

def innerTail400_14 : List Nat := 6679 :: innerTail400_15
def innerCandidates400_14 : List Nat := [6967]
set_option maxRecDepth 4096 in
theorem innerCandidates400_14_checked : candidates edge 4 2767 6679 innerTail400_15 best.bound = innerCandidates400_14 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_14 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_14 best = best := by
  rw [innerTail400_14, scan, if_neg (by decide), innerCandidates400_14_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_15
    (if 3 ≤ innerCandidates400_14.length then search edge 3 [2767, 6679] innerCandidates400_14 9446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_15

def innerTail400_13 : List Nat := 6361 :: innerTail400_14
def innerCandidates400_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_13_checked : candidates edge 4 2767 6361 innerTail400_14 best.bound = innerCandidates400_13 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_13 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_13 best = best := by
  rw [innerTail400_13, scan, if_neg (by decide), innerCandidates400_13_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_14
    (if 3 ≤ innerCandidates400_13.length then search edge 3 [2767, 6361] innerCandidates400_13 9128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_14

def innerTail400_12 : List Nat := 5737 :: innerTail400_13
def innerCandidates400_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_12_checked : candidates edge 4 2767 5737 innerTail400_13 best.bound = innerCandidates400_12 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_12 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_12 best = best := by
  rw [innerTail400_12, scan, if_neg (by decide), innerCandidates400_12_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_13
    (if 3 ≤ innerCandidates400_12.length then search edge 3 [2767, 5737] innerCandidates400_12 8504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_13

def innerTail400_11 : List Nat := 5689 :: innerTail400_12
def innerCandidates400_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_11_checked : candidates edge 4 2767 5689 innerTail400_12 best.bound = innerCandidates400_11 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_11 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_11 best = best := by
  rw [innerTail400_11, scan, if_neg (by decide), innerCandidates400_11_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_12
    (if 3 ≤ innerCandidates400_11.length then search edge 3 [2767, 5689] innerCandidates400_11 8456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_12

def innerTail400_10 : List Nat := 5119 :: innerTail400_11
def innerCandidates400_10 : List Nat := [7459, 7537, 7621]
set_option maxRecDepth 4096 in
theorem innerCandidates400_10_checked : candidates edge 4 2767 5119 innerTail400_11 best.bound = innerCandidates400_10 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild400_10 : search edge 3 [2767, 5119] innerCandidates400_10 7886 best = best := by decide

theorem innerScan400_10 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_10 best = best := by
  rw [innerTail400_10, scan, if_neg (by decide), innerCandidates400_10_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_11
    (if 3 ≤ innerCandidates400_10.length then search edge 3 [2767, 5119] innerCandidates400_10 7886 best else best) = best
  rw [if_pos (by decide), innerChild400_10]
  exact innerScan400_11

def innerTail400_9 : List Nat := 5107 :: innerTail400_10
def innerCandidates400_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_9_checked : candidates edge 4 2767 5107 innerTail400_10 best.bound = innerCandidates400_9 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_9 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_9 best = best := by
  rw [innerTail400_9, scan, if_neg (by decide), innerCandidates400_9_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_10
    (if 3 ≤ innerCandidates400_9.length then search edge 3 [2767, 5107] innerCandidates400_9 7874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_10

def innerTail400_8 : List Nat := 5059 :: innerTail400_9
def innerCandidates400_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_8_checked : candidates edge 4 2767 5059 innerTail400_9 best.bound = innerCandidates400_8 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_8 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_8 best = best := by
  rw [innerTail400_8, scan, if_neg (by decide), innerCandidates400_8_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_9
    (if 3 ≤ innerCandidates400_8.length then search edge 3 [2767, 5059] innerCandidates400_8 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_9

def innerTail400_7 : List Nat := 4783 :: innerTail400_8
def innerCandidates400_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_7_checked : candidates edge 4 2767 4783 innerTail400_8 best.bound = innerCandidates400_7 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_7 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_7 best = best := by
  rw [innerTail400_7, scan, if_neg (by decide), innerCandidates400_7_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_8
    (if 3 ≤ innerCandidates400_7.length then search edge 3 [2767, 4783] innerCandidates400_7 7550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_8

def innerTail400_6 : List Nat := 4513 :: innerTail400_7
def innerCandidates400_6 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates400_6_checked : candidates edge 4 2767 4513 innerTail400_7 best.bound = innerCandidates400_6 := by
  unfold candidates
  have row : edge 4513 = SearchData.row4513.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_6 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_6 best = best := by
  rw [innerTail400_6, scan, if_neg (by decide), innerCandidates400_6_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_7
    (if 3 ≤ innerCandidates400_6.length then search edge 3 [2767, 4513] innerCandidates400_6 7280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_7

def innerTail400_5 : List Nat := 4099 :: innerTail400_6
def innerCandidates400_5 : List Nat := [4783]
set_option maxRecDepth 4096 in
theorem innerCandidates400_5_checked : candidates edge 4 2767 4099 innerTail400_6 best.bound = innerCandidates400_5 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_5 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_5 best = best := by
  rw [innerTail400_5, scan, if_neg (by decide), innerCandidates400_5_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_6
    (if 3 ≤ innerCandidates400_5.length then search edge 3 [2767, 4099] innerCandidates400_5 6866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_6

def innerTail400_4 : List Nat := 3967 :: innerTail400_5
def innerCandidates400_4 : List Nat := [4783, 7933]
set_option maxRecDepth 4096 in
theorem innerCandidates400_4_checked : candidates edge 4 2767 3967 innerTail400_5 best.bound = innerCandidates400_4 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_4 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_4 best = best := by
  rw [innerTail400_4, scan, if_neg (by decide), innerCandidates400_4_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_5
    (if 3 ≤ innerCandidates400_4.length then search edge 3 [2767, 3967] innerCandidates400_4 6734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_5

def innerTail400_3 : List Nat := 3889 :: innerTail400_4
def innerCandidates400_3 : List Nat := [7477]
set_option maxRecDepth 4096 in
theorem innerCandidates400_3_checked : candidates edge 4 2767 3889 innerTail400_4 best.bound = innerCandidates400_3 := by
  unfold candidates
  have row : edge 3889 = SearchData.row3889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_3 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_3 best = best := by
  rw [innerTail400_3, scan, if_neg (by decide), innerCandidates400_3_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_4
    (if 3 ≤ innerCandidates400_3.length then search edge 3 [2767, 3889] innerCandidates400_3 6656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_4

def innerTail400_2 : List Nat := 3733 :: innerTail400_3
def innerCandidates400_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates400_2_checked : candidates edge 4 2767 3733 innerTail400_3 best.bound = innerCandidates400_2 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_2 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_2 best = best := by
  rw [innerTail400_2, scan, if_neg (by decide), innerCandidates400_2_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_3
    (if 3 ≤ innerCandidates400_2.length then search edge 3 [2767, 3733] innerCandidates400_2 6500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_3

def innerTail400_1 : List Nat := 3187 :: innerTail400_2
def innerCandidates400_1 : List Nat := [7477]
set_option maxRecDepth 4096 in
theorem innerCandidates400_1_checked : candidates edge 4 2767 3187 innerTail400_2 best.bound = innerCandidates400_1 := by
  unfold candidates
  have row : edge 3187 = SearchData.row3187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_1 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_1 best = best := by
  rw [innerTail400_1, scan, if_neg (by decide), innerCandidates400_1_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_2
    (if 3 ≤ innerCandidates400_1.length then search edge 3 [2767, 3187] innerCandidates400_1 5954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_2

def innerTail400_0 : List Nat := 3109 :: innerTail400_1
def innerCandidates400_0 : List Nat := [4099]
set_option maxRecDepth 4096 in
theorem innerCandidates400_0_checked : candidates edge 4 2767 3109 innerTail400_1 best.bound = innerCandidates400_0 := by
  unfold candidates
  have row : edge 3109 = SearchData.row3109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan400_0 : scan edge (search edge 3) 4 [2767] 2767 innerTail400_0 best = best := by
  rw [innerTail400_0, scan, if_neg (by decide), innerCandidates400_0_checked]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_1
    (if 3 ≤ innerCandidates400_0.length then search edge 3 [2767, 3109] innerCandidates400_0 5876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan400_1

theorem rootBranch400 : search edge 4 [2767] rootCandidates400 2767 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2767] 2767 innerTail400_0 best = best
  exact innerScan400_0

end ProjectEulerSolutions.PrimePairUpper.Search
