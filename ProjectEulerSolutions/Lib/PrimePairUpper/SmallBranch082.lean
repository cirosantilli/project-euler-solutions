import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates469 : List Nat := [3821, 4211, 4349, 4673, 4703, 4967, 5333, 5399, 6029, 6101, 6143, 6959, 7043, 7331, 7517, 7793, 8231, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates469_checked : candidates edge 5 0 3347 tail470 best.bound = rootCandidates469 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext q; rfl
  rw [row]
  decide

def innerTail469_18 : List Nat := []
theorem innerScan469_18 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_18 best = best := rfl

def innerTail469_17 : List Nat := 8273 :: innerTail469_18
def innerCandidates469_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_17_checked : candidates edge 4 3347 8273 innerTail469_18 best.bound = innerCandidates469_17 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_17 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_17 best = best := by
  rw [innerTail469_17, scan, if_neg (by decide), innerCandidates469_17_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_18
    (if 3 ≤ innerCandidates469_17.length then search edge 3 [3347, 8273] innerCandidates469_17 11620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_18

def innerTail469_16 : List Nat := 8231 :: innerTail469_17
def innerCandidates469_16 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates469_16_checked : candidates edge 4 3347 8231 innerTail469_17 best.bound = innerCandidates469_16 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_16 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_16 best = best := by
  rw [innerTail469_16, scan, if_neg (by decide), innerCandidates469_16_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_17
    (if 3 ≤ innerCandidates469_16.length then search edge 3 [3347, 8231] innerCandidates469_16 11578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_17

def innerTail469_15 : List Nat := 7793 :: innerTail469_16
def innerCandidates469_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_15_checked : candidates edge 4 3347 7793 innerTail469_16 best.bound = innerCandidates469_15 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_15 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_15 best = best := by
  rw [innerTail469_15, scan, if_neg (by decide), innerCandidates469_15_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_16
    (if 3 ≤ innerCandidates469_15.length then search edge 3 [3347, 7793] innerCandidates469_15 11140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_16

def innerTail469_14 : List Nat := 7517 :: innerTail469_15
def innerCandidates469_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_14_checked : candidates edge 4 3347 7517 innerTail469_15 best.bound = innerCandidates469_14 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_14 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_14 best = best := by
  rw [innerTail469_14, scan, if_neg (by decide), innerCandidates469_14_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_15
    (if 3 ≤ innerCandidates469_14.length then search edge 3 [3347, 7517] innerCandidates469_14 10864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_15

def innerTail469_13 : List Nat := 7331 :: innerTail469_14
def innerCandidates469_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_13_checked : candidates edge 4 3347 7331 innerTail469_14 best.bound = innerCandidates469_13 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_13 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_13 best = best := by
  rw [innerTail469_13, scan, if_neg (by decide), innerCandidates469_13_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_14
    (if 3 ≤ innerCandidates469_13.length then search edge 3 [3347, 7331] innerCandidates469_13 10678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_14

def innerTail469_12 : List Nat := 7043 :: innerTail469_13
def innerCandidates469_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_12_checked : candidates edge 4 3347 7043 innerTail469_13 best.bound = innerCandidates469_12 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_12 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_12 best = best := by
  rw [innerTail469_12, scan, if_neg (by decide), innerCandidates469_12_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_13
    (if 3 ≤ innerCandidates469_12.length then search edge 3 [3347, 7043] innerCandidates469_12 10390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_13

def innerTail469_11 : List Nat := 6959 :: innerTail469_12
def innerCandidates469_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_11_checked : candidates edge 4 3347 6959 innerTail469_12 best.bound = innerCandidates469_11 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_11 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_11 best = best := by
  rw [innerTail469_11, scan, if_neg (by decide), innerCandidates469_11_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_12
    (if 3 ≤ innerCandidates469_11.length then search edge 3 [3347, 6959] innerCandidates469_11 10306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_12

def innerTail469_10 : List Nat := 6143 :: innerTail469_11
def innerCandidates469_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_10_checked : candidates edge 4 3347 6143 innerTail469_11 best.bound = innerCandidates469_10 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_10 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_10 best = best := by
  rw [innerTail469_10, scan, if_neg (by decide), innerCandidates469_10_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_11
    (if 3 ≤ innerCandidates469_10.length then search edge 3 [3347, 6143] innerCandidates469_10 9490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_11

def innerTail469_9 : List Nat := 6101 :: innerTail469_10
def innerCandidates469_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_9_checked : candidates edge 4 3347 6101 innerTail469_10 best.bound = innerCandidates469_9 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_9 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_9 best = best := by
  rw [innerTail469_9, scan, if_neg (by decide), innerCandidates469_9_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_10
    (if 3 ≤ innerCandidates469_9.length then search edge 3 [3347, 6101] innerCandidates469_9 9448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_10

def innerTail469_8 : List Nat := 6029 :: innerTail469_9
def innerCandidates469_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_8_checked : candidates edge 4 3347 6029 innerTail469_9 best.bound = innerCandidates469_8 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_8 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_8 best = best := by
  rw [innerTail469_8, scan, if_neg (by decide), innerCandidates469_8_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_9
    (if 3 ≤ innerCandidates469_8.length then search edge 3 [3347, 6029] innerCandidates469_8 9376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_9

def innerTail469_7 : List Nat := 5399 :: innerTail469_8
def innerCandidates469_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_7_checked : candidates edge 4 3347 5399 innerTail469_8 best.bound = innerCandidates469_7 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_7 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_7 best = best := by
  rw [innerTail469_7, scan, if_neg (by decide), innerCandidates469_7_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_8
    (if 3 ≤ innerCandidates469_7.length then search edge 3 [3347, 5399] innerCandidates469_7 8746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_8

def innerTail469_6 : List Nat := 5333 :: innerTail469_7
def innerCandidates469_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_6_checked : candidates edge 4 3347 5333 innerTail469_7 best.bound = innerCandidates469_6 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_6 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_6 best = best := by
  rw [innerTail469_6, scan, if_neg (by decide), innerCandidates469_6_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_7
    (if 3 ≤ innerCandidates469_6.length then search edge 3 [3347, 5333] innerCandidates469_6 8680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_7

def innerTail469_5 : List Nat := 4967 :: innerTail469_6
def innerCandidates469_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_5_checked : candidates edge 4 3347 4967 innerTail469_6 best.bound = innerCandidates469_5 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_5 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_5 best = best := by
  rw [innerTail469_5, scan, if_neg (by decide), innerCandidates469_5_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_6
    (if 3 ≤ innerCandidates469_5.length then search edge 3 [3347, 4967] innerCandidates469_5 8314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_6

def innerTail469_4 : List Nat := 4703 :: innerTail469_5
def innerCandidates469_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_4_checked : candidates edge 4 3347 4703 innerTail469_5 best.bound = innerCandidates469_4 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_4 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_4 best = best := by
  rw [innerTail469_4, scan, if_neg (by decide), innerCandidates469_4_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_5
    (if 3 ≤ innerCandidates469_4.length then search edge 3 [3347, 4703] innerCandidates469_4 8050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_5

def innerTail469_3 : List Nat := 4673 :: innerTail469_4
def innerCandidates469_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_3_checked : candidates edge 4 3347 4673 innerTail469_4 best.bound = innerCandidates469_3 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_3 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_3 best = best := by
  rw [innerTail469_3, scan, if_neg (by decide), innerCandidates469_3_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_4
    (if 3 ≤ innerCandidates469_3.length then search edge 3 [3347, 4673] innerCandidates469_3 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_4

def innerTail469_2 : List Nat := 4349 :: innerTail469_3
def innerCandidates469_2 : List Nat := [4967, 7793]
set_option maxRecDepth 4096 in
theorem innerCandidates469_2_checked : candidates edge 4 3347 4349 innerTail469_3 best.bound = innerCandidates469_2 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_2 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_2 best = best := by
  rw [innerTail469_2, scan, if_neg (by decide), innerCandidates469_2_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_3
    (if 3 ≤ innerCandidates469_2.length then search edge 3 [3347, 4349] innerCandidates469_2 7696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_3

def innerTail469_1 : List Nat := 4211 :: innerTail469_2
def innerCandidates469_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates469_1_checked : candidates edge 4 3347 4211 innerTail469_2 best.bound = innerCandidates469_1 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_1 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_1 best = best := by
  rw [innerTail469_1, scan, if_neg (by decide), innerCandidates469_1_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_2
    (if 3 ≤ innerCandidates469_1.length then search edge 3 [3347, 4211] innerCandidates469_1 7558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_2

def innerTail469_0 : List Nat := 3821 :: innerTail469_1
def innerCandidates469_0 : List Nat := [4703]
set_option maxRecDepth 4096 in
theorem innerCandidates469_0_checked : candidates edge 4 3347 3821 innerTail469_1 best.bound = innerCandidates469_0 := by
  unfold candidates
  have row : edge 3821 = SearchData.row3821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan469_0 : scan edge (search edge 3) 4 [3347] 3347 innerTail469_0 best = best := by
  rw [innerTail469_0, scan, if_neg (by decide), innerCandidates469_0_checked]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_1
    (if 3 ≤ innerCandidates469_0.length then search edge 3 [3347, 3821] innerCandidates469_0 7168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan469_1

theorem rootBranch469 : search edge 4 [3347] rootCandidates469 3347 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3347] 3347 innerTail469_0 best = best
  exact innerScan469_0

def rootCandidates468 : List Nat := [3391, 3643, 3823, 4483, 4561, 4567, 4789, 5449, 5791, 6373, 6379, 6451, 6883, 6961, 7687]
set_option maxRecDepth 4096 in
theorem rootCandidates468_checked : candidates edge 5 0 3343 tail469 best.bound = rootCandidates468 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext q; rfl
  rw [row]
  decide

def innerTail468_15 : List Nat := []
theorem innerScan468_15 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_15 best = best := rfl

def innerTail468_14 : List Nat := 7687 :: innerTail468_15
def innerCandidates468_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates468_14_checked : candidates edge 4 3343 7687 innerTail468_15 best.bound = innerCandidates468_14 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_14 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_14 best = best := by
  rw [innerTail468_14, scan, if_neg (by decide), innerCandidates468_14_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_15
    (if 3 ≤ innerCandidates468_14.length then search edge 3 [3343, 7687] innerCandidates468_14 11030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_15

def innerTail468_13 : List Nat := 6961 :: innerTail468_14
def innerCandidates468_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates468_13_checked : candidates edge 4 3343 6961 innerTail468_14 best.bound = innerCandidates468_13 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_13 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_13 best = best := by
  rw [innerTail468_13, scan, if_neg (by decide), innerCandidates468_13_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_14
    (if 3 ≤ innerCandidates468_13.length then search edge 3 [3343, 6961] innerCandidates468_13 10304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_14

def innerTail468_12 : List Nat := 6883 :: innerTail468_13
def innerCandidates468_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates468_12_checked : candidates edge 4 3343 6883 innerTail468_13 best.bound = innerCandidates468_12 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_12 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_12 best = best := by
  rw [innerTail468_12, scan, if_neg (by decide), innerCandidates468_12_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_13
    (if 3 ≤ innerCandidates468_12.length then search edge 3 [3343, 6883] innerCandidates468_12 10226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_13

def innerTail468_11 : List Nat := 6451 :: innerTail468_12
def innerCandidates468_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates468_11_checked : candidates edge 4 3343 6451 innerTail468_12 best.bound = innerCandidates468_11 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_11 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_11 best = best := by
  rw [innerTail468_11, scan, if_neg (by decide), innerCandidates468_11_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_12
    (if 3 ≤ innerCandidates468_11.length then search edge 3 [3343, 6451] innerCandidates468_11 9794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_12

def innerTail468_10 : List Nat := 6379 :: innerTail468_11
def innerCandidates468_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates468_10_checked : candidates edge 4 3343 6379 innerTail468_11 best.bound = innerCandidates468_10 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_10 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_10 best = best := by
  rw [innerTail468_10, scan, if_neg (by decide), innerCandidates468_10_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_11
    (if 3 ≤ innerCandidates468_10.length then search edge 3 [3343, 6379] innerCandidates468_10 9722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_11

def innerTail468_9 : List Nat := 6373 :: innerTail468_10
def innerCandidates468_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates468_9_checked : candidates edge 4 3343 6373 innerTail468_10 best.bound = innerCandidates468_9 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_9 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_9 best = best := by
  rw [innerTail468_9, scan, if_neg (by decide), innerCandidates468_9_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_10
    (if 3 ≤ innerCandidates468_9.length then search edge 3 [3343, 6373] innerCandidates468_9 9716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_10

def innerTail468_8 : List Nat := 5791 :: innerTail468_9
def innerCandidates468_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates468_8_checked : candidates edge 4 3343 5791 innerTail468_9 best.bound = innerCandidates468_8 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_8 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_8 best = best := by
  rw [innerTail468_8, scan, if_neg (by decide), innerCandidates468_8_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_9
    (if 3 ≤ innerCandidates468_8.length then search edge 3 [3343, 5791] innerCandidates468_8 9134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_9

def innerTail468_7 : List Nat := 5449 :: innerTail468_8
def innerCandidates468_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates468_7_checked : candidates edge 4 3343 5449 innerTail468_8 best.bound = innerCandidates468_7 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_7 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_7 best = best := by
  rw [innerTail468_7, scan, if_neg (by decide), innerCandidates468_7_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_8
    (if 3 ≤ innerCandidates468_7.length then search edge 3 [3343, 5449] innerCandidates468_7 8792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_8

def innerTail468_6 : List Nat := 4789 :: innerTail468_7
def innerCandidates468_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates468_6_checked : candidates edge 4 3343 4789 innerTail468_7 best.bound = innerCandidates468_6 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_6 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_6 best = best := by
  rw [innerTail468_6, scan, if_neg (by decide), innerCandidates468_6_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_7
    (if 3 ≤ innerCandidates468_6.length then search edge 3 [3343, 4789] innerCandidates468_6 8132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_7

def innerTail468_5 : List Nat := 4567 :: innerTail468_6
def innerCandidates468_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates468_5_checked : candidates edge 4 3343 4567 innerTail468_6 best.bound = innerCandidates468_5 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_5 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_5 best = best := by
  rw [innerTail468_5, scan, if_neg (by decide), innerCandidates468_5_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_6
    (if 3 ≤ innerCandidates468_5.length then search edge 3 [3343, 4567] innerCandidates468_5 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_6

def innerTail468_4 : List Nat := 4561 :: innerTail468_5
def innerCandidates468_4 : List Nat := [6379]
set_option maxRecDepth 4096 in
theorem innerCandidates468_4_checked : candidates edge 4 3343 4561 innerTail468_5 best.bound = innerCandidates468_4 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_4 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_4 best = best := by
  rw [innerTail468_4, scan, if_neg (by decide), innerCandidates468_4_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_5
    (if 3 ≤ innerCandidates468_4.length then search edge 3 [3343, 4561] innerCandidates468_4 7904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_5

def innerTail468_3 : List Nat := 4483 :: innerTail468_4
def innerCandidates468_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates468_3_checked : candidates edge 4 3343 4483 innerTail468_4 best.bound = innerCandidates468_3 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_3 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_3 best = best := by
  rw [innerTail468_3, scan, if_neg (by decide), innerCandidates468_3_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_4
    (if 3 ≤ innerCandidates468_3.length then search edge 3 [3343, 4483] innerCandidates468_3 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_4

def innerTail468_2 : List Nat := 3823 :: innerTail468_3
def innerCandidates468_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates468_2_checked : candidates edge 4 3343 3823 innerTail468_3 best.bound = innerCandidates468_2 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_2 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_2 best = best := by
  rw [innerTail468_2, scan, if_neg (by decide), innerCandidates468_2_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_3
    (if 3 ≤ innerCandidates468_2.length then search edge 3 [3343, 3823] innerCandidates468_2 7166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_3

def innerTail468_1 : List Nat := 3643 :: innerTail468_2
def innerCandidates468_1 : List Nat := [4561, 5449]
set_option maxRecDepth 4096 in
theorem innerCandidates468_1_checked : candidates edge 4 3343 3643 innerTail468_2 best.bound = innerCandidates468_1 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_1 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_1 best = best := by
  rw [innerTail468_1, scan, if_neg (by decide), innerCandidates468_1_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_2
    (if 3 ≤ innerCandidates468_1.length then search edge 3 [3343, 3643] innerCandidates468_1 6986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_2

def innerTail468_0 : List Nat := 3391 :: innerTail468_1
def innerCandidates468_0 : List Nat := [3643]
set_option maxRecDepth 4096 in
theorem innerCandidates468_0_checked : candidates edge 4 3343 3391 innerTail468_1 best.bound = innerCandidates468_0 := by
  unfold candidates
  have row : edge 3391 = SearchData.row3391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan468_0 : scan edge (search edge 3) 4 [3343] 3343 innerTail468_0 best = best := by
  rw [innerTail468_0, scan, if_neg (by decide), innerCandidates468_0_checked]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_1
    (if 3 ≤ innerCandidates468_0.length then search edge 3 [3343, 3391] innerCandidates468_0 6734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan468_1

theorem rootBranch468 : search edge 4 [3343] rootCandidates468 3343 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3343] 3343 innerTail468_0 best = best
  exact innerScan468_0

def rootCandidates467 : List Nat := [3559, 4231, 4261, 4621, 4903, 5197, 5647, 5701, 7159, 7699, 7717, 7927, 7993, 8221]
set_option maxRecDepth 4096 in
theorem rootCandidates467_checked : candidates edge 5 0 3331 tail468 best.bound = rootCandidates467 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext q; rfl
  rw [row]
  decide

def innerTail467_14 : List Nat := []
theorem innerScan467_14 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_14 best = best := rfl

def innerTail467_13 : List Nat := 8221 :: innerTail467_14
def innerCandidates467_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates467_13_checked : candidates edge 4 3331 8221 innerTail467_14 best.bound = innerCandidates467_13 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_13 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_13 best = best := by
  rw [innerTail467_13, scan, if_neg (by decide), innerCandidates467_13_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_14
    (if 3 ≤ innerCandidates467_13.length then search edge 3 [3331, 8221] innerCandidates467_13 11552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_14

def innerTail467_12 : List Nat := 7993 :: innerTail467_13
def innerCandidates467_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates467_12_checked : candidates edge 4 3331 7993 innerTail467_13 best.bound = innerCandidates467_12 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_12 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_12 best = best := by
  rw [innerTail467_12, scan, if_neg (by decide), innerCandidates467_12_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_13
    (if 3 ≤ innerCandidates467_12.length then search edge 3 [3331, 7993] innerCandidates467_12 11324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_13

def innerTail467_11 : List Nat := 7927 :: innerTail467_12
def innerCandidates467_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates467_11_checked : candidates edge 4 3331 7927 innerTail467_12 best.bound = innerCandidates467_11 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_11 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_11 best = best := by
  rw [innerTail467_11, scan, if_neg (by decide), innerCandidates467_11_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_12
    (if 3 ≤ innerCandidates467_11.length then search edge 3 [3331, 7927] innerCandidates467_11 11258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_12

def innerTail467_10 : List Nat := 7717 :: innerTail467_11
def innerCandidates467_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates467_10_checked : candidates edge 4 3331 7717 innerTail467_11 best.bound = innerCandidates467_10 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_10 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_10 best = best := by
  rw [innerTail467_10, scan, if_neg (by decide), innerCandidates467_10_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_11
    (if 3 ≤ innerCandidates467_10.length then search edge 3 [3331, 7717] innerCandidates467_10 11048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_11

def innerTail467_9 : List Nat := 7699 :: innerTail467_10
def innerCandidates467_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates467_9_checked : candidates edge 4 3331 7699 innerTail467_10 best.bound = innerCandidates467_9 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_9 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_9 best = best := by
  rw [innerTail467_9, scan, if_neg (by decide), innerCandidates467_9_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_10
    (if 3 ≤ innerCandidates467_9.length then search edge 3 [3331, 7699] innerCandidates467_9 11030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_10

def innerTail467_8 : List Nat := 7159 :: innerTail467_9
def innerCandidates467_8 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates467_8_checked : candidates edge 4 3331 7159 innerTail467_9 best.bound = innerCandidates467_8 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_8 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_8 best = best := by
  rw [innerTail467_8, scan, if_neg (by decide), innerCandidates467_8_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_9
    (if 3 ≤ innerCandidates467_8.length then search edge 3 [3331, 7159] innerCandidates467_8 10490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_9

def innerTail467_7 : List Nat := 5701 :: innerTail467_8
def innerCandidates467_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates467_7_checked : candidates edge 4 3331 5701 innerTail467_8 best.bound = innerCandidates467_7 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_7 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_7 best = best := by
  rw [innerTail467_7, scan, if_neg (by decide), innerCandidates467_7_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_8
    (if 3 ≤ innerCandidates467_7.length then search edge 3 [3331, 5701] innerCandidates467_7 9032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_8

def innerTail467_6 : List Nat := 5647 :: innerTail467_7
def innerCandidates467_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates467_6_checked : candidates edge 4 3331 5647 innerTail467_7 best.bound = innerCandidates467_6 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_6 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_6 best = best := by
  rw [innerTail467_6, scan, if_neg (by decide), innerCandidates467_6_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_7
    (if 3 ≤ innerCandidates467_6.length then search edge 3 [3331, 5647] innerCandidates467_6 8978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_7

def innerTail467_5 : List Nat := 5197 :: innerTail467_6
def innerCandidates467_5 : List Nat := [5701, 8221]
set_option maxRecDepth 4096 in
theorem innerCandidates467_5_checked : candidates edge 4 3331 5197 innerTail467_6 best.bound = innerCandidates467_5 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_5 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_5 best = best := by
  rw [innerTail467_5, scan, if_neg (by decide), innerCandidates467_5_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_6
    (if 3 ≤ innerCandidates467_5.length then search edge 3 [3331, 5197] innerCandidates467_5 8528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_6

def innerTail467_4 : List Nat := 4903 :: innerTail467_5
def innerCandidates467_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates467_4_checked : candidates edge 4 3331 4903 innerTail467_5 best.bound = innerCandidates467_4 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_4 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_4 best = best := by
  rw [innerTail467_4, scan, if_neg (by decide), innerCandidates467_4_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_5
    (if 3 ≤ innerCandidates467_4.length then search edge 3 [3331, 4903] innerCandidates467_4 8234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_5

def innerTail467_3 : List Nat := 4621 :: innerTail467_4
def innerCandidates467_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates467_3_checked : candidates edge 4 3331 4621 innerTail467_4 best.bound = innerCandidates467_3 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_3 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_3 best = best := by
  rw [innerTail467_3, scan, if_neg (by decide), innerCandidates467_3_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_4
    (if 3 ≤ innerCandidates467_3.length then search edge 3 [3331, 4621] innerCandidates467_3 7952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_4

def innerTail467_2 : List Nat := 4261 :: innerTail467_3
def innerCandidates467_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates467_2_checked : candidates edge 4 3331 4261 innerTail467_3 best.bound = innerCandidates467_2 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_2 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_2 best = best := by
  rw [innerTail467_2, scan, if_neg (by decide), innerCandidates467_2_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_3
    (if 3 ≤ innerCandidates467_2.length then search edge 3 [3331, 4261] innerCandidates467_2 7592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_3

def innerTail467_1 : List Nat := 4231 :: innerTail467_2
def innerCandidates467_1 : List Nat := [5197]
set_option maxRecDepth 4096 in
theorem innerCandidates467_1_checked : candidates edge 4 3331 4231 innerTail467_2 best.bound = innerCandidates467_1 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_1 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_1 best = best := by
  rw [innerTail467_1, scan, if_neg (by decide), innerCandidates467_1_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_2
    (if 3 ≤ innerCandidates467_1.length then search edge 3 [3331, 4231] innerCandidates467_1 7562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_2

def innerTail467_0 : List Nat := 3559 :: innerTail467_1
def innerCandidates467_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates467_0_checked : candidates edge 4 3331 3559 innerTail467_1 best.bound = innerCandidates467_0 := by
  unfold candidates
  have row : edge 3559 = SearchData.row3559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan467_0 : scan edge (search edge 3) 4 [3331] 3331 innerTail467_0 best = best := by
  rw [innerTail467_0, scan, if_neg (by decide), innerCandidates467_0_checked]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_1
    (if 3 ≤ innerCandidates467_0.length then search edge 3 [3331, 3559] innerCandidates467_0 6890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan467_1

theorem rootBranch467 : search edge 4 [3331] rootCandidates467 3331 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3331] 3331 innerTail467_0 best = best
  exact innerScan467_0

def rootCandidates466 : List Nat := [3929, 4127, 4637, 4649, 4967, 5051, 5441, 6359, 7001, 7109, 7877]
set_option maxRecDepth 4096 in
theorem rootCandidates466_checked : candidates edge 5 0 3329 tail467 best.bound = rootCandidates466 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext q; rfl
  rw [row]
  decide

def innerTail466_11 : List Nat := []
theorem innerScan466_11 : scan edge (search edge 3) 4 [3329] 3329 innerTail466_11 best = best := rfl

def innerTail466_10 : List Nat := 7877 :: innerTail466_11
def innerCandidates466_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates466_10_checked : candidates edge 4 3329 7877 innerTail466_11 best.bound = innerCandidates466_10 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan466_10 : scan edge (search edge 3) 4 [3329] 3329 innerTail466_10 best = best := by
  rw [innerTail466_10, scan, if_neg (by decide), innerCandidates466_10_checked]
  change scan edge (search edge 3) 4 [3329] 3329 innerTail466_11
    (if 3 ≤ innerCandidates466_10.length then search edge 3 [3329, 7877] innerCandidates466_10 11206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan466_11

def innerTail466_9 : List Nat := 7109 :: innerTail466_10
def innerCandidates466_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates466_9_checked : candidates edge 4 3329 7109 innerTail466_10 best.bound = innerCandidates466_9 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan466_9 : scan edge (search edge 3) 4 [3329] 3329 innerTail466_9 best = best := by
  rw [innerTail466_9, scan, if_neg (by decide), innerCandidates466_9_checked]
  change scan edge (search edge 3) 4 [3329] 3329 innerTail466_10
    (if 3 ≤ innerCandidates466_9.length then search edge 3 [3329, 7109] innerCandidates466_9 10438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan466_10

def innerTail466_8 : List Nat := 7001 :: innerTail466_9
def innerCandidates466_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates466_8_checked : candidates edge 4 3329 7001 innerTail466_9 best.bound = innerCandidates466_8 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan466_8 : scan edge (search edge 3) 4 [3329] 3329 innerTail466_8 best = best := by
  rw [innerTail466_8, scan, if_neg (by decide), innerCandidates466_8_checked]
  change scan edge (search edge 3) 4 [3329] 3329 innerTail466_9
    (if 3 ≤ innerCandidates466_8.length then search edge 3 [3329, 7001] innerCandidates466_8 10330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan466_9

def innerTail466_7 : List Nat := 6359 :: innerTail466_8
def innerCandidates466_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates466_7_checked : candidates edge 4 3329 6359 innerTail466_8 best.bound = innerCandidates466_7 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan466_7 : scan edge (search edge 3) 4 [3329] 3329 innerTail466_7 best = best := by
  rw [innerTail466_7, scan, if_neg (by decide), innerCandidates466_7_checked]
  change scan edge (search edge 3) 4 [3329] 3329 innerTail466_8
    (if 3 ≤ innerCandidates466_7.length then search edge 3 [3329, 6359] innerCandidates466_7 9688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan466_8

def innerTail466_6 : List Nat := 5441 :: innerTail466_7
def innerCandidates466_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates466_6_checked : candidates edge 4 3329 5441 innerTail466_7 best.bound = innerCandidates466_6 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan466_6 : scan edge (search edge 3) 4 [3329] 3329 innerTail466_6 best = best := by
  rw [innerTail466_6, scan, if_neg (by decide), innerCandidates466_6_checked]
  change scan edge (search edge 3) 4 [3329] 3329 innerTail466_7
    (if 3 ≤ innerCandidates466_6.length then search edge 3 [3329, 5441] innerCandidates466_6 8770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan466_7

def innerTail466_5 : List Nat := 5051 :: innerTail466_6
def innerCandidates466_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates466_5_checked : candidates edge 4 3329 5051 innerTail466_6 best.bound = innerCandidates466_5 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan466_5 : scan edge (search edge 3) 4 [3329] 3329 innerTail466_5 best = best := by
  rw [innerTail466_5, scan, if_neg (by decide), innerCandidates466_5_checked]
  change scan edge (search edge 3) 4 [3329] 3329 innerTail466_6
    (if 3 ≤ innerCandidates466_5.length then search edge 3 [3329, 5051] innerCandidates466_5 8380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan466_6

def innerTail466_4 : List Nat := 4967 :: innerTail466_5
def innerCandidates466_4 : List Nat := [5441, 7877]
set_option maxRecDepth 4096 in
theorem innerCandidates466_4_checked : candidates edge 4 3329 4967 innerTail466_5 best.bound = innerCandidates466_4 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan466_4 : scan edge (search edge 3) 4 [3329] 3329 innerTail466_4 best = best := by
  rw [innerTail466_4, scan, if_neg (by decide), innerCandidates466_4_checked]
  change scan edge (search edge 3) 4 [3329] 3329 innerTail466_5
    (if 3 ≤ innerCandidates466_4.length then search edge 3 [3329, 4967] innerCandidates466_4 8296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan466_5

def innerTail466_3 : List Nat := 4649 :: innerTail466_4
def innerCandidates466_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates466_3_checked : candidates edge 4 3329 4649 innerTail466_4 best.bound = innerCandidates466_3 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan466_3 : scan edge (search edge 3) 4 [3329] 3329 innerTail466_3 best = best := by
  rw [innerTail466_3, scan, if_neg (by decide), innerCandidates466_3_checked]
  change scan edge (search edge 3) 4 [3329] 3329 innerTail466_4
    (if 3 ≤ innerCandidates466_3.length then search edge 3 [3329, 4649] innerCandidates466_3 7978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan466_4

def innerTail466_2 : List Nat := 4637 :: innerTail466_3
def innerCandidates466_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates466_2_checked : candidates edge 4 3329 4637 innerTail466_3 best.bound = innerCandidates466_2 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan466_2 : scan edge (search edge 3) 4 [3329] 3329 innerTail466_2 best = best := by
  rw [innerTail466_2, scan, if_neg (by decide), innerCandidates466_2_checked]
  change scan edge (search edge 3) 4 [3329] 3329 innerTail466_3
    (if 3 ≤ innerCandidates466_2.length then search edge 3 [3329, 4637] innerCandidates466_2 7966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan466_3

def innerTail466_1 : List Nat := 4127 :: innerTail466_2
def innerCandidates466_1 : List Nat := [4967]
set_option maxRecDepth 4096 in
theorem innerCandidates466_1_checked : candidates edge 4 3329 4127 innerTail466_2 best.bound = innerCandidates466_1 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan466_1 : scan edge (search edge 3) 4 [3329] 3329 innerTail466_1 best = best := by
  rw [innerTail466_1, scan, if_neg (by decide), innerCandidates466_1_checked]
  change scan edge (search edge 3) 4 [3329] 3329 innerTail466_2
    (if 3 ≤ innerCandidates466_1.length then search edge 3 [3329, 4127] innerCandidates466_1 7456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan466_2

def innerTail466_0 : List Nat := 3929 :: innerTail466_1
def innerCandidates466_0 : List Nat := [7109]
set_option maxRecDepth 4096 in
theorem innerCandidates466_0_checked : candidates edge 4 3329 3929 innerTail466_1 best.bound = innerCandidates466_0 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan466_0 : scan edge (search edge 3) 4 [3329] 3329 innerTail466_0 best = best := by
  rw [innerTail466_0, scan, if_neg (by decide), innerCandidates466_0_checked]
  change scan edge (search edge 3) 4 [3329] 3329 innerTail466_1
    (if 3 ≤ innerCandidates466_0.length then search edge 3 [3329, 3929] innerCandidates466_0 7258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan466_1

theorem rootBranch466 : search edge 4 [3329] rootCandidates466 3329 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3329] 3329 innerTail466_0 best = best
  exact innerScan466_0

end ProjectEulerSolutions.PrimePairUpper.Search
