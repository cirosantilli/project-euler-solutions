import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates452 : List Nat := [3511, 3931, 4789, 5821, 6073, 6211, 6301, 6619, 7177]
set_option maxRecDepth 4096 in
theorem rootCandidates452_checked : candidates edge 5 0 3217 tail453 best.bound = rootCandidates452 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext q; rfl
  rw [row]
  decide

def innerTail452_9 : List Nat := []
theorem innerScan452_9 : scan edge (search edge 3) 4 [3217] 3217 innerTail452_9 best = best := rfl

def innerTail452_8 : List Nat := 7177 :: innerTail452_9
def innerCandidates452_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates452_8_checked : candidates edge 4 3217 7177 innerTail452_9 best.bound = innerCandidates452_8 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan452_8 : scan edge (search edge 3) 4 [3217] 3217 innerTail452_8 best = best := by
  rw [innerTail452_8, scan, if_neg (by decide), innerCandidates452_8_checked]
  change scan edge (search edge 3) 4 [3217] 3217 innerTail452_9
    (if 3 ≤ innerCandidates452_8.length then search edge 3 [3217, 7177] innerCandidates452_8 10394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan452_9

def innerTail452_7 : List Nat := 6619 :: innerTail452_8
def innerCandidates452_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates452_7_checked : candidates edge 4 3217 6619 innerTail452_8 best.bound = innerCandidates452_7 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan452_7 : scan edge (search edge 3) 4 [3217] 3217 innerTail452_7 best = best := by
  rw [innerTail452_7, scan, if_neg (by decide), innerCandidates452_7_checked]
  change scan edge (search edge 3) 4 [3217] 3217 innerTail452_8
    (if 3 ≤ innerCandidates452_7.length then search edge 3 [3217, 6619] innerCandidates452_7 9836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan452_8

def innerTail452_6 : List Nat := 6301 :: innerTail452_7
def innerCandidates452_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates452_6_checked : candidates edge 4 3217 6301 innerTail452_7 best.bound = innerCandidates452_6 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan452_6 : scan edge (search edge 3) 4 [3217] 3217 innerTail452_6 best = best := by
  rw [innerTail452_6, scan, if_neg (by decide), innerCandidates452_6_checked]
  change scan edge (search edge 3) 4 [3217] 3217 innerTail452_7
    (if 3 ≤ innerCandidates452_6.length then search edge 3 [3217, 6301] innerCandidates452_6 9518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan452_7

def innerTail452_5 : List Nat := 6211 :: innerTail452_6
def innerCandidates452_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates452_5_checked : candidates edge 4 3217 6211 innerTail452_6 best.bound = innerCandidates452_5 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan452_5 : scan edge (search edge 3) 4 [3217] 3217 innerTail452_5 best = best := by
  rw [innerTail452_5, scan, if_neg (by decide), innerCandidates452_5_checked]
  change scan edge (search edge 3) 4 [3217] 3217 innerTail452_6
    (if 3 ≤ innerCandidates452_5.length then search edge 3 [3217, 6211] innerCandidates452_5 9428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan452_6

def innerTail452_4 : List Nat := 6073 :: innerTail452_5
def innerCandidates452_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates452_4_checked : candidates edge 4 3217 6073 innerTail452_5 best.bound = innerCandidates452_4 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan452_4 : scan edge (search edge 3) 4 [3217] 3217 innerTail452_4 best = best := by
  rw [innerTail452_4, scan, if_neg (by decide), innerCandidates452_4_checked]
  change scan edge (search edge 3) 4 [3217] 3217 innerTail452_5
    (if 3 ≤ innerCandidates452_4.length then search edge 3 [3217, 6073] innerCandidates452_4 9290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan452_5

def innerTail452_3 : List Nat := 5821 :: innerTail452_4
def innerCandidates452_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates452_3_checked : candidates edge 4 3217 5821 innerTail452_4 best.bound = innerCandidates452_3 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan452_3 : scan edge (search edge 3) 4 [3217] 3217 innerTail452_3 best = best := by
  rw [innerTail452_3, scan, if_neg (by decide), innerCandidates452_3_checked]
  change scan edge (search edge 3) 4 [3217] 3217 innerTail452_4
    (if 3 ≤ innerCandidates452_3.length then search edge 3 [3217, 5821] innerCandidates452_3 9038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan452_4

def innerTail452_2 : List Nat := 4789 :: innerTail452_3
def innerCandidates452_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates452_2_checked : candidates edge 4 3217 4789 innerTail452_3 best.bound = innerCandidates452_2 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan452_2 : scan edge (search edge 3) 4 [3217] 3217 innerTail452_2 best = best := by
  rw [innerTail452_2, scan, if_neg (by decide), innerCandidates452_2_checked]
  change scan edge (search edge 3) 4 [3217] 3217 innerTail452_3
    (if 3 ≤ innerCandidates452_2.length then search edge 3 [3217, 4789] innerCandidates452_2 8006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan452_3

def innerTail452_1 : List Nat := 3931 :: innerTail452_2
def innerCandidates452_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates452_1_checked : candidates edge 4 3217 3931 innerTail452_2 best.bound = innerCandidates452_1 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan452_1 : scan edge (search edge 3) 4 [3217] 3217 innerTail452_1 best = best := by
  rw [innerTail452_1, scan, if_neg (by decide), innerCandidates452_1_checked]
  change scan edge (search edge 3) 4 [3217] 3217 innerTail452_2
    (if 3 ≤ innerCandidates452_1.length then search edge 3 [3217, 3931] innerCandidates452_1 7148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan452_2

def innerTail452_0 : List Nat := 3511 :: innerTail452_1
def innerCandidates452_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates452_0_checked : candidates edge 4 3217 3511 innerTail452_1 best.bound = innerCandidates452_0 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan452_0 : scan edge (search edge 3) 4 [3217] 3217 innerTail452_0 best = best := by
  rw [innerTail452_0, scan, if_neg (by decide), innerCandidates452_0_checked]
  change scan edge (search edge 3) 4 [3217] 3217 innerTail452_1
    (if 3 ≤ innerCandidates452_0.length then search edge 3 [3217, 3511] innerCandidates452_0 6728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan452_1

theorem rootBranch452 : search edge 4 [3217] rootCandidates452 3217 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3217] 3217 innerTail452_0 best = best
  exact innerScan452_0

def rootCandidates451 : List Nat := [3617, 3833, 4721, 5171, 5417, 5471, 5717, 6653, 6761, 7187, 7433, 7523, 7817]
set_option maxRecDepth 4096 in
theorem rootCandidates451_checked : candidates edge 5 0 3209 tail452 best.bound = rootCandidates451 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext q; rfl
  rw [row]
  decide

def innerTail451_13 : List Nat := []
theorem innerScan451_13 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_13 best = best := rfl

def innerTail451_12 : List Nat := 7817 :: innerTail451_13
def innerCandidates451_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates451_12_checked : candidates edge 4 3209 7817 innerTail451_13 best.bound = innerCandidates451_12 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_12 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_12 best = best := by
  rw [innerTail451_12, scan, if_neg (by decide), innerCandidates451_12_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_13
    (if 3 ≤ innerCandidates451_12.length then search edge 3 [3209, 7817] innerCandidates451_12 11026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_13

def innerTail451_11 : List Nat := 7523 :: innerTail451_12
def innerCandidates451_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates451_11_checked : candidates edge 4 3209 7523 innerTail451_12 best.bound = innerCandidates451_11 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_11 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_11 best = best := by
  rw [innerTail451_11, scan, if_neg (by decide), innerCandidates451_11_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_12
    (if 3 ≤ innerCandidates451_11.length then search edge 3 [3209, 7523] innerCandidates451_11 10732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_12

def innerTail451_10 : List Nat := 7433 :: innerTail451_11
def innerCandidates451_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates451_10_checked : candidates edge 4 3209 7433 innerTail451_11 best.bound = innerCandidates451_10 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_10 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_10 best = best := by
  rw [innerTail451_10, scan, if_neg (by decide), innerCandidates451_10_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_11
    (if 3 ≤ innerCandidates451_10.length then search edge 3 [3209, 7433] innerCandidates451_10 10642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_11

def innerTail451_9 : List Nat := 7187 :: innerTail451_10
def innerCandidates451_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates451_9_checked : candidates edge 4 3209 7187 innerTail451_10 best.bound = innerCandidates451_9 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_9 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_9 best = best := by
  rw [innerTail451_9, scan, if_neg (by decide), innerCandidates451_9_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_10
    (if 3 ≤ innerCandidates451_9.length then search edge 3 [3209, 7187] innerCandidates451_9 10396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_10

def innerTail451_8 : List Nat := 6761 :: innerTail451_9
def innerCandidates451_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates451_8_checked : candidates edge 4 3209 6761 innerTail451_9 best.bound = innerCandidates451_8 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_8 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_8 best = best := by
  rw [innerTail451_8, scan, if_neg (by decide), innerCandidates451_8_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_9
    (if 3 ≤ innerCandidates451_8.length then search edge 3 [3209, 6761] innerCandidates451_8 9970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_9

def innerTail451_7 : List Nat := 6653 :: innerTail451_8
def innerCandidates451_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates451_7_checked : candidates edge 4 3209 6653 innerTail451_8 best.bound = innerCandidates451_7 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_7 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_7 best = best := by
  rw [innerTail451_7, scan, if_neg (by decide), innerCandidates451_7_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_8
    (if 3 ≤ innerCandidates451_7.length then search edge 3 [3209, 6653] innerCandidates451_7 9862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_8

def innerTail451_6 : List Nat := 5717 :: innerTail451_7
def innerCandidates451_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates451_6_checked : candidates edge 4 3209 5717 innerTail451_7 best.bound = innerCandidates451_6 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_6 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_6 best = best := by
  rw [innerTail451_6, scan, if_neg (by decide), innerCandidates451_6_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_7
    (if 3 ≤ innerCandidates451_6.length then search edge 3 [3209, 5717] innerCandidates451_6 8926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_7

def innerTail451_5 : List Nat := 5471 :: innerTail451_6
def innerCandidates451_5 : List Nat := [6653]
set_option maxRecDepth 4096 in
theorem innerCandidates451_5_checked : candidates edge 4 3209 5471 innerTail451_6 best.bound = innerCandidates451_5 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_5 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_5 best = best := by
  rw [innerTail451_5, scan, if_neg (by decide), innerCandidates451_5_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_6
    (if 3 ≤ innerCandidates451_5.length then search edge 3 [3209, 5471] innerCandidates451_5 8680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_6

def innerTail451_4 : List Nat := 5417 :: innerTail451_5
def innerCandidates451_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates451_4_checked : candidates edge 4 3209 5417 innerTail451_5 best.bound = innerCandidates451_4 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_4 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_4 best = best := by
  rw [innerTail451_4, scan, if_neg (by decide), innerCandidates451_4_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_5
    (if 3 ≤ innerCandidates451_4.length then search edge 3 [3209, 5417] innerCandidates451_4 8626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_5

def innerTail451_3 : List Nat := 5171 :: innerTail451_4
def innerCandidates451_3 : List Nat := [7187]
set_option maxRecDepth 4096 in
theorem innerCandidates451_3_checked : candidates edge 4 3209 5171 innerTail451_4 best.bound = innerCandidates451_3 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_3 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_3 best = best := by
  rw [innerTail451_3, scan, if_neg (by decide), innerCandidates451_3_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_4
    (if 3 ≤ innerCandidates451_3.length then search edge 3 [3209, 5171] innerCandidates451_3 8380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_4

def innerTail451_2 : List Nat := 4721 :: innerTail451_3
def innerCandidates451_2 : List Nat := [5417]
set_option maxRecDepth 4096 in
theorem innerCandidates451_2_checked : candidates edge 4 3209 4721 innerTail451_3 best.bound = innerCandidates451_2 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_2 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_2 best = best := by
  rw [innerTail451_2, scan, if_neg (by decide), innerCandidates451_2_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_3
    (if 3 ≤ innerCandidates451_2.length then search edge 3 [3209, 4721] innerCandidates451_2 7930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_3

def innerTail451_1 : List Nat := 3833 :: innerTail451_2
def innerCandidates451_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates451_1_checked : candidates edge 4 3209 3833 innerTail451_2 best.bound = innerCandidates451_1 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_1 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_1 best = best := by
  rw [innerTail451_1, scan, if_neg (by decide), innerCandidates451_1_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_2
    (if 3 ≤ innerCandidates451_1.length then search edge 3 [3209, 3833] innerCandidates451_1 7042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_2

def innerTail451_0 : List Nat := 3617 :: innerTail451_1
def innerCandidates451_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates451_0_checked : candidates edge 4 3209 3617 innerTail451_1 best.bound = innerCandidates451_0 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan451_0 : scan edge (search edge 3) 4 [3209] 3209 innerTail451_0 best = best := by
  rw [innerTail451_0, scan, if_neg (by decide), innerCandidates451_0_checked]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_1
    (if 3 ≤ innerCandidates451_0.length then search edge 3 [3209, 3617] innerCandidates451_0 6826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan451_1

theorem rootBranch451 : search edge 4 [3209] rootCandidates451 3209 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3209] 3209 innerTail451_0 best = best
  exact innerScan451_0

def rootCandidates450 : List Nat := [3467, 4481, 5279, 5867, 6101, 6827, 7331, 7541, 7949, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates450_checked : candidates edge 5 0 3203 tail451 best.bound = rootCandidates450 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext q; rfl
  rw [row]
  decide

def innerTail450_10 : List Nat := []
theorem innerScan450_10 : scan edge (search edge 3) 4 [3203] 3203 innerTail450_10 best = best := rfl

def innerTail450_9 : List Nat := 8219 :: innerTail450_10
def innerCandidates450_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates450_9_checked : candidates edge 4 3203 8219 innerTail450_10 best.bound = innerCandidates450_9 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan450_9 : scan edge (search edge 3) 4 [3203] 3203 innerTail450_9 best = best := by
  rw [innerTail450_9, scan, if_neg (by decide), innerCandidates450_9_checked]
  change scan edge (search edge 3) 4 [3203] 3203 innerTail450_10
    (if 3 ≤ innerCandidates450_9.length then search edge 3 [3203, 8219] innerCandidates450_9 11422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan450_10

def innerTail450_8 : List Nat := 7949 :: innerTail450_9
def innerCandidates450_8 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates450_8_checked : candidates edge 4 3203 7949 innerTail450_9 best.bound = innerCandidates450_8 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan450_8 : scan edge (search edge 3) 4 [3203] 3203 innerTail450_8 best = best := by
  rw [innerTail450_8, scan, if_neg (by decide), innerCandidates450_8_checked]
  change scan edge (search edge 3) 4 [3203] 3203 innerTail450_9
    (if 3 ≤ innerCandidates450_8.length then search edge 3 [3203, 7949] innerCandidates450_8 11152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan450_9

def innerTail450_7 : List Nat := 7541 :: innerTail450_8
def innerCandidates450_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates450_7_checked : candidates edge 4 3203 7541 innerTail450_8 best.bound = innerCandidates450_7 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan450_7 : scan edge (search edge 3) 4 [3203] 3203 innerTail450_7 best = best := by
  rw [innerTail450_7, scan, if_neg (by decide), innerCandidates450_7_checked]
  change scan edge (search edge 3) 4 [3203] 3203 innerTail450_8
    (if 3 ≤ innerCandidates450_7.length then search edge 3 [3203, 7541] innerCandidates450_7 10744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan450_8

def innerTail450_6 : List Nat := 7331 :: innerTail450_7
def innerCandidates450_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates450_6_checked : candidates edge 4 3203 7331 innerTail450_7 best.bound = innerCandidates450_6 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan450_6 : scan edge (search edge 3) 4 [3203] 3203 innerTail450_6 best = best := by
  rw [innerTail450_6, scan, if_neg (by decide), innerCandidates450_6_checked]
  change scan edge (search edge 3) 4 [3203] 3203 innerTail450_7
    (if 3 ≤ innerCandidates450_6.length then search edge 3 [3203, 7331] innerCandidates450_6 10534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan450_7

def innerTail450_5 : List Nat := 6827 :: innerTail450_6
def innerCandidates450_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates450_5_checked : candidates edge 4 3203 6827 innerTail450_6 best.bound = innerCandidates450_5 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan450_5 : scan edge (search edge 3) 4 [3203] 3203 innerTail450_5 best = best := by
  rw [innerTail450_5, scan, if_neg (by decide), innerCandidates450_5_checked]
  change scan edge (search edge 3) 4 [3203] 3203 innerTail450_6
    (if 3 ≤ innerCandidates450_5.length then search edge 3 [3203, 6827] innerCandidates450_5 10030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan450_6

def innerTail450_4 : List Nat := 6101 :: innerTail450_5
def innerCandidates450_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates450_4_checked : candidates edge 4 3203 6101 innerTail450_5 best.bound = innerCandidates450_4 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan450_4 : scan edge (search edge 3) 4 [3203] 3203 innerTail450_4 best = best := by
  rw [innerTail450_4, scan, if_neg (by decide), innerCandidates450_4_checked]
  change scan edge (search edge 3) 4 [3203] 3203 innerTail450_5
    (if 3 ≤ innerCandidates450_4.length then search edge 3 [3203, 6101] innerCandidates450_4 9304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan450_5

def innerTail450_3 : List Nat := 5867 :: innerTail450_4
def innerCandidates450_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates450_3_checked : candidates edge 4 3203 5867 innerTail450_4 best.bound = innerCandidates450_3 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan450_3 : scan edge (search edge 3) 4 [3203] 3203 innerTail450_3 best = best := by
  rw [innerTail450_3, scan, if_neg (by decide), innerCandidates450_3_checked]
  change scan edge (search edge 3) 4 [3203] 3203 innerTail450_4
    (if 3 ≤ innerCandidates450_3.length then search edge 3 [3203, 5867] innerCandidates450_3 9070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan450_4

def innerTail450_2 : List Nat := 5279 :: innerTail450_3
def innerCandidates450_2 : List Nat := [6827]
set_option maxRecDepth 4096 in
theorem innerCandidates450_2_checked : candidates edge 4 3203 5279 innerTail450_3 best.bound = innerCandidates450_2 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan450_2 : scan edge (search edge 3) 4 [3203] 3203 innerTail450_2 best = best := by
  rw [innerTail450_2, scan, if_neg (by decide), innerCandidates450_2_checked]
  change scan edge (search edge 3) 4 [3203] 3203 innerTail450_3
    (if 3 ≤ innerCandidates450_2.length then search edge 3 [3203, 5279] innerCandidates450_2 8482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan450_3

def innerTail450_1 : List Nat := 4481 :: innerTail450_2
def innerCandidates450_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates450_1_checked : candidates edge 4 3203 4481 innerTail450_2 best.bound = innerCandidates450_1 := by
  unfold candidates
  have row : edge 4481 = SearchData.row4481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan450_1 : scan edge (search edge 3) 4 [3203] 3203 innerTail450_1 best = best := by
  rw [innerTail450_1, scan, if_neg (by decide), innerCandidates450_1_checked]
  change scan edge (search edge 3) 4 [3203] 3203 innerTail450_2
    (if 3 ≤ innerCandidates450_1.length then search edge 3 [3203, 4481] innerCandidates450_1 7684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan450_2

def innerTail450_0 : List Nat := 3467 :: innerTail450_1
def innerCandidates450_0 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates450_0_checked : candidates edge 4 3203 3467 innerTail450_1 best.bound = innerCandidates450_0 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan450_0 : scan edge (search edge 3) 4 [3203] 3203 innerTail450_0 best = best := by
  rw [innerTail450_0, scan, if_neg (by decide), innerCandidates450_0_checked]
  change scan edge (search edge 3) 4 [3203] 3203 innerTail450_1
    (if 3 ≤ innerCandidates450_0.length then search edge 3 [3203, 3467] innerCandidates450_0 6670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan450_1

theorem rootBranch450 : search edge 4 [3203] rootCandidates450 3203 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3203] 3203 innerTail450_0 best = best
  exact innerScan450_0

def rootCandidates449 : List Nat := [3221, 3821, 4397, 4583, 5531, 6779, 7043, 7451, 7547, 7703, 7883, 7937, 8069, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates449_checked : candidates edge 5 0 3191 tail450 best.bound = rootCandidates449 := by
  unfold candidates
  have row : edge 3191 = SearchData.row3191.contains := by funext q; rfl
  rw [row]
  decide

def innerTail449_14 : List Nat := []
theorem innerScan449_14 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_14 best = best := rfl

def innerTail449_13 : List Nat := 8111 :: innerTail449_14
def innerCandidates449_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_13_checked : candidates edge 4 3191 8111 innerTail449_14 best.bound = innerCandidates449_13 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_13 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_13 best = best := by
  rw [innerTail449_13, scan, if_neg (by decide), innerCandidates449_13_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_14
    (if 3 ≤ innerCandidates449_13.length then search edge 3 [3191, 8111] innerCandidates449_13 11302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_14

def innerTail449_12 : List Nat := 8069 :: innerTail449_13
def innerCandidates449_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_12_checked : candidates edge 4 3191 8069 innerTail449_13 best.bound = innerCandidates449_12 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_12 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_12 best = best := by
  rw [innerTail449_12, scan, if_neg (by decide), innerCandidates449_12_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_13
    (if 3 ≤ innerCandidates449_12.length then search edge 3 [3191, 8069] innerCandidates449_12 11260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_13

def innerTail449_11 : List Nat := 7937 :: innerTail449_12
def innerCandidates449_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_11_checked : candidates edge 4 3191 7937 innerTail449_12 best.bound = innerCandidates449_11 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_11 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_11 best = best := by
  rw [innerTail449_11, scan, if_neg (by decide), innerCandidates449_11_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_12
    (if 3 ≤ innerCandidates449_11.length then search edge 3 [3191, 7937] innerCandidates449_11 11128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_12

def innerTail449_10 : List Nat := 7883 :: innerTail449_11
def innerCandidates449_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_10_checked : candidates edge 4 3191 7883 innerTail449_11 best.bound = innerCandidates449_10 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_10 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_10 best = best := by
  rw [innerTail449_10, scan, if_neg (by decide), innerCandidates449_10_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_11
    (if 3 ≤ innerCandidates449_10.length then search edge 3 [3191, 7883] innerCandidates449_10 11074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_11

def innerTail449_9 : List Nat := 7703 :: innerTail449_10
def innerCandidates449_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_9_checked : candidates edge 4 3191 7703 innerTail449_10 best.bound = innerCandidates449_9 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_9 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_9 best = best := by
  rw [innerTail449_9, scan, if_neg (by decide), innerCandidates449_9_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_10
    (if 3 ≤ innerCandidates449_9.length then search edge 3 [3191, 7703] innerCandidates449_9 10894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_10

def innerTail449_8 : List Nat := 7547 :: innerTail449_9
def innerCandidates449_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_8_checked : candidates edge 4 3191 7547 innerTail449_9 best.bound = innerCandidates449_8 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_8 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_8 best = best := by
  rw [innerTail449_8, scan, if_neg (by decide), innerCandidates449_8_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_9
    (if 3 ≤ innerCandidates449_8.length then search edge 3 [3191, 7547] innerCandidates449_8 10738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_9

def innerTail449_7 : List Nat := 7451 :: innerTail449_8
def innerCandidates449_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_7_checked : candidates edge 4 3191 7451 innerTail449_8 best.bound = innerCandidates449_7 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_7 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_7 best = best := by
  rw [innerTail449_7, scan, if_neg (by decide), innerCandidates449_7_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_8
    (if 3 ≤ innerCandidates449_7.length then search edge 3 [3191, 7451] innerCandidates449_7 10642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_8

def innerTail449_6 : List Nat := 7043 :: innerTail449_7
def innerCandidates449_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_6_checked : candidates edge 4 3191 7043 innerTail449_7 best.bound = innerCandidates449_6 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_6 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_6 best = best := by
  rw [innerTail449_6, scan, if_neg (by decide), innerCandidates449_6_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_7
    (if 3 ≤ innerCandidates449_6.length then search edge 3 [3191, 7043] innerCandidates449_6 10234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_7

def innerTail449_5 : List Nat := 6779 :: innerTail449_6
def innerCandidates449_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_5_checked : candidates edge 4 3191 6779 innerTail449_6 best.bound = innerCandidates449_5 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_5 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_5 best = best := by
  rw [innerTail449_5, scan, if_neg (by decide), innerCandidates449_5_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_6
    (if 3 ≤ innerCandidates449_5.length then search edge 3 [3191, 6779] innerCandidates449_5 9970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_6

def innerTail449_4 : List Nat := 5531 :: innerTail449_5
def innerCandidates449_4 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem innerCandidates449_4_checked : candidates edge 4 3191 5531 innerTail449_5 best.bound = innerCandidates449_4 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_4 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_4 best = best := by
  rw [innerTail449_4, scan, if_neg (by decide), innerCandidates449_4_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_5
    (if 3 ≤ innerCandidates449_4.length then search edge 3 [3191, 5531] innerCandidates449_4 8722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_5

def innerTail449_3 : List Nat := 4583 :: innerTail449_4
def innerCandidates449_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_3_checked : candidates edge 4 3191 4583 innerTail449_4 best.bound = innerCandidates449_3 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_3 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_3 best = best := by
  rw [innerTail449_3, scan, if_neg (by decide), innerCandidates449_3_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_4
    (if 3 ≤ innerCandidates449_3.length then search edge 3 [3191, 4583] innerCandidates449_3 7774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_4

def innerTail449_2 : List Nat := 4397 :: innerTail449_3
def innerCandidates449_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_2_checked : candidates edge 4 3191 4397 innerTail449_3 best.bound = innerCandidates449_2 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_2 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_2 best = best := by
  rw [innerTail449_2, scan, if_neg (by decide), innerCandidates449_2_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_3
    (if 3 ≤ innerCandidates449_2.length then search edge 3 [3191, 4397] innerCandidates449_2 7588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_3

def innerTail449_1 : List Nat := 3821 :: innerTail449_2
def innerCandidates449_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_1_checked : candidates edge 4 3191 3821 innerTail449_2 best.bound = innerCandidates449_1 := by
  unfold candidates
  have row : edge 3821 = SearchData.row3821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_1 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_1 best = best := by
  rw [innerTail449_1, scan, if_neg (by decide), innerCandidates449_1_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_2
    (if 3 ≤ innerCandidates449_1.length then search edge 3 [3191, 3821] innerCandidates449_1 7012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_2

def innerTail449_0 : List Nat := 3221 :: innerTail449_1
def innerCandidates449_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates449_0_checked : candidates edge 4 3191 3221 innerTail449_1 best.bound = innerCandidates449_0 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan449_0 : scan edge (search edge 3) 4 [3191] 3191 innerTail449_0 best = best := by
  rw [innerTail449_0, scan, if_neg (by decide), innerCandidates449_0_checked]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_1
    (if 3 ≤ innerCandidates449_0.length then search edge 3 [3191, 3221] innerCandidates449_0 6412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan449_1

theorem rootBranch449 : search edge 4 [3191] rootCandidates449 3191 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3191] 3191 innerTail449_0 best = best
  exact innerScan449_0

def rootCandidates448 : List Nat := [3229, 3529, 4093, 4561, 4657, 4729, 4957, 5209, 5227, 5479, 6379, 6907, 7411, 7477, 8017, 8101, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates448_checked : candidates edge 5 0 3187 tail449 best.bound = rootCandidates448 := by
  unfold candidates
  have row : edge 3187 = SearchData.row3187.contains := by funext q; rfl
  rw [row]
  decide

def innerTail448_17 : List Nat := []
theorem innerScan448_17 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_17 best = best := rfl

def innerTail448_16 : List Nat := 8233 :: innerTail448_17
def innerCandidates448_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates448_16_checked : candidates edge 4 3187 8233 innerTail448_17 best.bound = innerCandidates448_16 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_16 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_16 best = best := by
  rw [innerTail448_16, scan, if_neg (by decide), innerCandidates448_16_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_17
    (if 3 ≤ innerCandidates448_16.length then search edge 3 [3187, 8233] innerCandidates448_16 11420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_17

def innerTail448_15 : List Nat := 8101 :: innerTail448_16
def innerCandidates448_15 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates448_15_checked : candidates edge 4 3187 8101 innerTail448_16 best.bound = innerCandidates448_15 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_15 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_15 best = best := by
  rw [innerTail448_15, scan, if_neg (by decide), innerCandidates448_15_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_16
    (if 3 ≤ innerCandidates448_15.length then search edge 3 [3187, 8101] innerCandidates448_15 11288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_16

def innerTail448_14 : List Nat := 8017 :: innerTail448_15
def innerCandidates448_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates448_14_checked : candidates edge 4 3187 8017 innerTail448_15 best.bound = innerCandidates448_14 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_14 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_14 best = best := by
  rw [innerTail448_14, scan, if_neg (by decide), innerCandidates448_14_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_15
    (if 3 ≤ innerCandidates448_14.length then search edge 3 [3187, 8017] innerCandidates448_14 11204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_15

def innerTail448_13 : List Nat := 7477 :: innerTail448_14
def innerCandidates448_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates448_13_checked : candidates edge 4 3187 7477 innerTail448_14 best.bound = innerCandidates448_13 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_13 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_13 best = best := by
  rw [innerTail448_13, scan, if_neg (by decide), innerCandidates448_13_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_14
    (if 3 ≤ innerCandidates448_13.length then search edge 3 [3187, 7477] innerCandidates448_13 10664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_14

def innerTail448_12 : List Nat := 7411 :: innerTail448_13
def innerCandidates448_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates448_12_checked : candidates edge 4 3187 7411 innerTail448_13 best.bound = innerCandidates448_12 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_12 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_12 best = best := by
  rw [innerTail448_12, scan, if_neg (by decide), innerCandidates448_12_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_13
    (if 3 ≤ innerCandidates448_12.length then search edge 3 [3187, 7411] innerCandidates448_12 10598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_13

def innerTail448_11 : List Nat := 6907 :: innerTail448_12
def innerCandidates448_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates448_11_checked : candidates edge 4 3187 6907 innerTail448_12 best.bound = innerCandidates448_11 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_11 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_11 best = best := by
  rw [innerTail448_11, scan, if_neg (by decide), innerCandidates448_11_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_12
    (if 3 ≤ innerCandidates448_11.length then search edge 3 [3187, 6907] innerCandidates448_11 10094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_12

def innerTail448_10 : List Nat := 6379 :: innerTail448_11
def innerCandidates448_10 : List Nat := [8017, 8233]
set_option maxRecDepth 4096 in
theorem innerCandidates448_10_checked : candidates edge 4 3187 6379 innerTail448_11 best.bound = innerCandidates448_10 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_10 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_10 best = best := by
  rw [innerTail448_10, scan, if_neg (by decide), innerCandidates448_10_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_11
    (if 3 ≤ innerCandidates448_10.length then search edge 3 [3187, 6379] innerCandidates448_10 9566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_11

def innerTail448_9 : List Nat := 5479 :: innerTail448_10
def innerCandidates448_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates448_9_checked : candidates edge 4 3187 5479 innerTail448_10 best.bound = innerCandidates448_9 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_9 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_9 best = best := by
  rw [innerTail448_9, scan, if_neg (by decide), innerCandidates448_9_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_10
    (if 3 ≤ innerCandidates448_9.length then search edge 3 [3187, 5479] innerCandidates448_9 8666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_10

def innerTail448_8 : List Nat := 5227 :: innerTail448_9
def innerCandidates448_8 : List Nat := [6379]
set_option maxRecDepth 4096 in
theorem innerCandidates448_8_checked : candidates edge 4 3187 5227 innerTail448_9 best.bound = innerCandidates448_8 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_8 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_8 best = best := by
  rw [innerTail448_8, scan, if_neg (by decide), innerCandidates448_8_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_9
    (if 3 ≤ innerCandidates448_8.length then search edge 3 [3187, 5227] innerCandidates448_8 8414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_9

def innerTail448_7 : List Nat := 5209 :: innerTail448_8
def innerCandidates448_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates448_7_checked : candidates edge 4 3187 5209 innerTail448_8 best.bound = innerCandidates448_7 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_7 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_7 best = best := by
  rw [innerTail448_7, scan, if_neg (by decide), innerCandidates448_7_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_8
    (if 3 ≤ innerCandidates448_7.length then search edge 3 [3187, 5209] innerCandidates448_7 8396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_8

def innerTail448_6 : List Nat := 4957 :: innerTail448_7
def innerCandidates448_6 : List Nat := [8017, 8233]
set_option maxRecDepth 4096 in
theorem innerCandidates448_6_checked : candidates edge 4 3187 4957 innerTail448_7 best.bound = innerCandidates448_6 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_6 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_6 best = best := by
  rw [innerTail448_6, scan, if_neg (by decide), innerCandidates448_6_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_7
    (if 3 ≤ innerCandidates448_6.length then search edge 3 [3187, 4957] innerCandidates448_6 8144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_7

def innerTail448_5 : List Nat := 4729 :: innerTail448_6
def innerCandidates448_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates448_5_checked : candidates edge 4 3187 4729 innerTail448_6 best.bound = innerCandidates448_5 := by
  unfold candidates
  have row : edge 4729 = SearchData.row4729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_5 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_5 best = best := by
  rw [innerTail448_5, scan, if_neg (by decide), innerCandidates448_5_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_6
    (if 3 ≤ innerCandidates448_5.length then search edge 3 [3187, 4729] innerCandidates448_5 7916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_6

def innerTail448_4 : List Nat := 4657 :: innerTail448_5
def innerCandidates448_4 : List Nat := [5227]
set_option maxRecDepth 4096 in
theorem innerCandidates448_4_checked : candidates edge 4 3187 4657 innerTail448_5 best.bound = innerCandidates448_4 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_4 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_4 best = best := by
  rw [innerTail448_4, scan, if_neg (by decide), innerCandidates448_4_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_5
    (if 3 ≤ innerCandidates448_4.length then search edge 3 [3187, 4657] innerCandidates448_4 7844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_5

def innerTail448_3 : List Nat := 4561 :: innerTail448_4
def innerCandidates448_3 : List Nat := [6379, 8017, 8101]
set_option maxRecDepth 4096 in
theorem innerCandidates448_3_checked : candidates edge 4 3187 4561 innerTail448_4 best.bound = innerCandidates448_3 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild448_3 : search edge 3 [3187, 4561] innerCandidates448_3 7748 best = best := by decide

theorem innerScan448_3 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_3 best = best := by
  rw [innerTail448_3, scan, if_neg (by decide), innerCandidates448_3_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_4
    (if 3 ≤ innerCandidates448_3.length then search edge 3 [3187, 4561] innerCandidates448_3 7748 best else best) = best
  rw [if_pos (by decide), innerChild448_3]
  exact innerScan448_4

def innerTail448_2 : List Nat := 4093 :: innerTail448_3
def innerCandidates448_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates448_2_checked : candidates edge 4 3187 4093 innerTail448_3 best.bound = innerCandidates448_2 := by
  unfold candidates
  have row : edge 4093 = SearchData.row4093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_2 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_2 best = best := by
  rw [innerTail448_2, scan, if_neg (by decide), innerCandidates448_2_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_3
    (if 3 ≤ innerCandidates448_2.length then search edge 3 [3187, 4093] innerCandidates448_2 7280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_3

def innerTail448_1 : List Nat := 3529 :: innerTail448_2
def innerCandidates448_1 : List Nat := [4657]
set_option maxRecDepth 4096 in
theorem innerCandidates448_1_checked : candidates edge 4 3187 3529 innerTail448_2 best.bound = innerCandidates448_1 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_1 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_1 best = best := by
  rw [innerTail448_1, scan, if_neg (by decide), innerCandidates448_1_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_2
    (if 3 ≤ innerCandidates448_1.length then search edge 3 [3187, 3529] innerCandidates448_1 6716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_2

def innerTail448_0 : List Nat := 3229 :: innerTail448_1
def innerCandidates448_0 : List Nat := [3529]
set_option maxRecDepth 4096 in
theorem innerCandidates448_0_checked : candidates edge 4 3187 3229 innerTail448_1 best.bound = innerCandidates448_0 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan448_0 : scan edge (search edge 3) 4 [3187] 3187 innerTail448_0 best = best := by
  rw [innerTail448_0, scan, if_neg (by decide), innerCandidates448_0_checked]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_1
    (if 3 ≤ innerCandidates448_0.length then search edge 3 [3187, 3229] innerCandidates448_0 6416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan448_1

theorem rootBranch448 : search edge 4 [3187] rootCandidates448 3187 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3187] 3187 innerTail448_0 best = best
  exact innerScan448_0

end ProjectEulerSolutions.PrimePairUpper.Search
