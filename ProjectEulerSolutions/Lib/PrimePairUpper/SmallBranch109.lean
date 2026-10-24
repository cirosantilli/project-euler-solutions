import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates360 : List Nat := [3011, 3797, 4019, 4937, 5813, 5939, 6113, 6323, 7001, 7283, 7907, 8093, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates360_checked : candidates edge 5 0 2447 tail361 best.bound = rootCandidates360 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext q; rfl
  rw [row]
  decide

def innerTail360_13 : List Nat := []
theorem innerScan360_13 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_13 best = best := rfl

def innerTail360_12 : List Nat := 8123 :: innerTail360_13
def innerCandidates360_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates360_12_checked : candidates edge 4 2447 8123 innerTail360_13 best.bound = innerCandidates360_12 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_12 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_12 best = best := by
  rw [innerTail360_12, scan, if_neg (by decide), innerCandidates360_12_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_13
    (if 3 ≤ innerCandidates360_12.length then search edge 3 [2447, 8123] innerCandidates360_12 10570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_13

def innerTail360_11 : List Nat := 8093 :: innerTail360_12
def innerCandidates360_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates360_11_checked : candidates edge 4 2447 8093 innerTail360_12 best.bound = innerCandidates360_11 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_11 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_11 best = best := by
  rw [innerTail360_11, scan, if_neg (by decide), innerCandidates360_11_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_12
    (if 3 ≤ innerCandidates360_11.length then search edge 3 [2447, 8093] innerCandidates360_11 10540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_12

def innerTail360_10 : List Nat := 7907 :: innerTail360_11
def innerCandidates360_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates360_10_checked : candidates edge 4 2447 7907 innerTail360_11 best.bound = innerCandidates360_10 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_10 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_10 best = best := by
  rw [innerTail360_10, scan, if_neg (by decide), innerCandidates360_10_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_11
    (if 3 ≤ innerCandidates360_10.length then search edge 3 [2447, 7907] innerCandidates360_10 10354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_11

def innerTail360_9 : List Nat := 7283 :: innerTail360_10
def innerCandidates360_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates360_9_checked : candidates edge 4 2447 7283 innerTail360_10 best.bound = innerCandidates360_9 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_9 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_9 best = best := by
  rw [innerTail360_9, scan, if_neg (by decide), innerCandidates360_9_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_10
    (if 3 ≤ innerCandidates360_9.length then search edge 3 [2447, 7283] innerCandidates360_9 9730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_10

def innerTail360_8 : List Nat := 7001 :: innerTail360_9
def innerCandidates360_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates360_8_checked : candidates edge 4 2447 7001 innerTail360_9 best.bound = innerCandidates360_8 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_8 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_8 best = best := by
  rw [innerTail360_8, scan, if_neg (by decide), innerCandidates360_8_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_9
    (if 3 ≤ innerCandidates360_8.length then search edge 3 [2447, 7001] innerCandidates360_8 9448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_9

def innerTail360_7 : List Nat := 6323 :: innerTail360_8
def innerCandidates360_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates360_7_checked : candidates edge 4 2447 6323 innerTail360_8 best.bound = innerCandidates360_7 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_7 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_7 best = best := by
  rw [innerTail360_7, scan, if_neg (by decide), innerCandidates360_7_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_8
    (if 3 ≤ innerCandidates360_7.length then search edge 3 [2447, 6323] innerCandidates360_7 8770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_8

def innerTail360_6 : List Nat := 6113 :: innerTail360_7
def innerCandidates360_6 : List Nat := [7001]
set_option maxRecDepth 4096 in
theorem innerCandidates360_6_checked : candidates edge 4 2447 6113 innerTail360_7 best.bound = innerCandidates360_6 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_6 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_6 best = best := by
  rw [innerTail360_6, scan, if_neg (by decide), innerCandidates360_6_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_7
    (if 3 ≤ innerCandidates360_6.length then search edge 3 [2447, 6113] innerCandidates360_6 8560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_7

def innerTail360_5 : List Nat := 5939 :: innerTail360_6
def innerCandidates360_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates360_5_checked : candidates edge 4 2447 5939 innerTail360_6 best.bound = innerCandidates360_5 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_5 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_5 best = best := by
  rw [innerTail360_5, scan, if_neg (by decide), innerCandidates360_5_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_6
    (if 3 ≤ innerCandidates360_5.length then search edge 3 [2447, 5939] innerCandidates360_5 8386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_6

def innerTail360_4 : List Nat := 5813 :: innerTail360_5
def innerCandidates360_4 : List Nat := [7907]
set_option maxRecDepth 4096 in
theorem innerCandidates360_4_checked : candidates edge 4 2447 5813 innerTail360_5 best.bound = innerCandidates360_4 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_4 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_4 best = best := by
  rw [innerTail360_4, scan, if_neg (by decide), innerCandidates360_4_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_5
    (if 3 ≤ innerCandidates360_4.length then search edge 3 [2447, 5813] innerCandidates360_4 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_5

def innerTail360_3 : List Nat := 4937 :: innerTail360_4
def innerCandidates360_3 : List Nat := [7001]
set_option maxRecDepth 4096 in
theorem innerCandidates360_3_checked : candidates edge 4 2447 4937 innerTail360_4 best.bound = innerCandidates360_3 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_3 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_3 best = best := by
  rw [innerTail360_3, scan, if_neg (by decide), innerCandidates360_3_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_4
    (if 3 ≤ innerCandidates360_3.length then search edge 3 [2447, 4937] innerCandidates360_3 7384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_4

def innerTail360_2 : List Nat := 4019 :: innerTail360_3
def innerCandidates360_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates360_2_checked : candidates edge 4 2447 4019 innerTail360_3 best.bound = innerCandidates360_2 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_2 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_2 best = best := by
  rw [innerTail360_2, scan, if_neg (by decide), innerCandidates360_2_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_3
    (if 3 ≤ innerCandidates360_2.length then search edge 3 [2447, 4019] innerCandidates360_2 6466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_3

def innerTail360_1 : List Nat := 3797 :: innerTail360_2
def innerCandidates360_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates360_1_checked : candidates edge 4 2447 3797 innerTail360_2 best.bound = innerCandidates360_1 := by
  unfold candidates
  have row : edge 3797 = SearchData.row3797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_1 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_1 best = best := by
  rw [innerTail360_1, scan, if_neg (by decide), innerCandidates360_1_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_2
    (if 3 ≤ innerCandidates360_1.length then search edge 3 [2447, 3797] innerCandidates360_1 6244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_2

def innerTail360_0 : List Nat := 3011 :: innerTail360_1
def innerCandidates360_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates360_0_checked : candidates edge 4 2447 3011 innerTail360_1 best.bound = innerCandidates360_0 := by
  unfold candidates
  have row : edge 3011 = SearchData.row3011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan360_0 : scan edge (search edge 3) 4 [2447] 2447 innerTail360_0 best = best := by
  rw [innerTail360_0, scan, if_neg (by decide), innerCandidates360_0_checked]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_1
    (if 3 ≤ innerCandidates360_0.length then search edge 3 [2447, 3011] innerCandidates360_0 5458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan360_1

theorem rootBranch360 : search edge 4 [2447] rootCandidates360 2447 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2447] 2447 innerTail360_0 best = best
  exact innerScan360_0

def rootCandidates359 : List Nat := [2543, 3209, 3671, 3677, 4157, 4349, 4787, 5087, 5147, 5297, 5333, 5381, 6089, 6113, 6659, 7121, 7643, 7691, 7727, 8231]
set_option maxRecDepth 4096 in
theorem rootCandidates359_checked : candidates edge 5 0 2441 tail360 best.bound = rootCandidates359 := by
  unfold candidates
  have row : edge 2441 = PrimePairCache.smallRow2441.contains := by funext q; rfl
  rw [row]
  decide

def innerTail359_20 : List Nat := []
theorem innerScan359_20 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_20 best = best := rfl

def innerTail359_19 : List Nat := 8231 :: innerTail359_20
def innerCandidates359_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_19_checked : candidates edge 4 2441 8231 innerTail359_20 best.bound = innerCandidates359_19 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_19 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_19 best = best := by
  rw [innerTail359_19, scan, if_neg (by decide), innerCandidates359_19_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_20
    (if 3 ≤ innerCandidates359_19.length then search edge 3 [2441, 8231] innerCandidates359_19 10672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_20

def innerTail359_18 : List Nat := 7727 :: innerTail359_19
def innerCandidates359_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_18_checked : candidates edge 4 2441 7727 innerTail359_19 best.bound = innerCandidates359_18 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_18 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_18 best = best := by
  rw [innerTail359_18, scan, if_neg (by decide), innerCandidates359_18_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_19
    (if 3 ≤ innerCandidates359_18.length then search edge 3 [2441, 7727] innerCandidates359_18 10168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_19

def innerTail359_17 : List Nat := 7691 :: innerTail359_18
def innerCandidates359_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_17_checked : candidates edge 4 2441 7691 innerTail359_18 best.bound = innerCandidates359_17 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_17 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_17 best = best := by
  rw [innerTail359_17, scan, if_neg (by decide), innerCandidates359_17_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_18
    (if 3 ≤ innerCandidates359_17.length then search edge 3 [2441, 7691] innerCandidates359_17 10132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_18

def innerTail359_16 : List Nat := 7643 :: innerTail359_17
def innerCandidates359_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_16_checked : candidates edge 4 2441 7643 innerTail359_17 best.bound = innerCandidates359_16 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_16 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_16 best = best := by
  rw [innerTail359_16, scan, if_neg (by decide), innerCandidates359_16_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_17
    (if 3 ≤ innerCandidates359_16.length then search edge 3 [2441, 7643] innerCandidates359_16 10084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_17

def innerTail359_15 : List Nat := 7121 :: innerTail359_16
def innerCandidates359_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_15_checked : candidates edge 4 2441 7121 innerTail359_16 best.bound = innerCandidates359_15 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_15 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_15 best = best := by
  rw [innerTail359_15, scan, if_neg (by decide), innerCandidates359_15_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_16
    (if 3 ≤ innerCandidates359_15.length then search edge 3 [2441, 7121] innerCandidates359_15 9562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_16

def innerTail359_14 : List Nat := 6659 :: innerTail359_15
def innerCandidates359_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_14_checked : candidates edge 4 2441 6659 innerTail359_15 best.bound = innerCandidates359_14 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_14 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_14 best = best := by
  rw [innerTail359_14, scan, if_neg (by decide), innerCandidates359_14_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_15
    (if 3 ≤ innerCandidates359_14.length then search edge 3 [2441, 6659] innerCandidates359_14 9100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_15

def innerTail359_13 : List Nat := 6113 :: innerTail359_14
def innerCandidates359_13 : List Nat := [6659, 7691]
set_option maxRecDepth 4096 in
theorem innerCandidates359_13_checked : candidates edge 4 2441 6113 innerTail359_14 best.bound = innerCandidates359_13 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_13 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_13 best = best := by
  rw [innerTail359_13, scan, if_neg (by decide), innerCandidates359_13_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_14
    (if 3 ≤ innerCandidates359_13.length then search edge 3 [2441, 6113] innerCandidates359_13 8554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_14

def innerTail359_12 : List Nat := 6089 :: innerTail359_13
def innerCandidates359_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_12_checked : candidates edge 4 2441 6089 innerTail359_13 best.bound = innerCandidates359_12 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_12 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_12 best = best := by
  rw [innerTail359_12, scan, if_neg (by decide), innerCandidates359_12_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_13
    (if 3 ≤ innerCandidates359_12.length then search edge 3 [2441, 6089] innerCandidates359_12 8530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_13

def innerTail359_11 : List Nat := 5381 :: innerTail359_12
def innerCandidates359_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_11_checked : candidates edge 4 2441 5381 innerTail359_12 best.bound = innerCandidates359_11 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_11 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_11 best = best := by
  rw [innerTail359_11, scan, if_neg (by decide), innerCandidates359_11_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_12
    (if 3 ≤ innerCandidates359_11.length then search edge 3 [2441, 5381] innerCandidates359_11 7822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_12

def innerTail359_10 : List Nat := 5333 :: innerTail359_11
def innerCandidates359_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_10_checked : candidates edge 4 2441 5333 innerTail359_11 best.bound = innerCandidates359_10 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_10 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_10 best = best := by
  rw [innerTail359_10, scan, if_neg (by decide), innerCandidates359_10_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_11
    (if 3 ≤ innerCandidates359_10.length then search edge 3 [2441, 5333] innerCandidates359_10 7774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_11

def innerTail359_9 : List Nat := 5297 :: innerTail359_10
def innerCandidates359_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_9_checked : candidates edge 4 2441 5297 innerTail359_10 best.bound = innerCandidates359_9 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_9 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_9 best = best := by
  rw [innerTail359_9, scan, if_neg (by decide), innerCandidates359_9_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_10
    (if 3 ≤ innerCandidates359_9.length then search edge 3 [2441, 5297] innerCandidates359_9 7738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_10

def innerTail359_8 : List Nat := 5147 :: innerTail359_9
def innerCandidates359_8 : List Nat := [5297]
set_option maxRecDepth 4096 in
theorem innerCandidates359_8_checked : candidates edge 4 2441 5147 innerTail359_9 best.bound = innerCandidates359_8 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_8 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_8 best = best := by
  rw [innerTail359_8, scan, if_neg (by decide), innerCandidates359_8_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_9
    (if 3 ≤ innerCandidates359_8.length then search edge 3 [2441, 5147] innerCandidates359_8 7588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_9

def innerTail359_7 : List Nat := 5087 :: innerTail359_8
def innerCandidates359_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_7_checked : candidates edge 4 2441 5087 innerTail359_8 best.bound = innerCandidates359_7 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_7 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_7 best = best := by
  rw [innerTail359_7, scan, if_neg (by decide), innerCandidates359_7_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_8
    (if 3 ≤ innerCandidates359_7.length then search edge 3 [2441, 5087] innerCandidates359_7 7528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_8

def innerTail359_6 : List Nat := 4787 :: innerTail359_7
def innerCandidates359_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_6_checked : candidates edge 4 2441 4787 innerTail359_7 best.bound = innerCandidates359_6 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_6 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_6 best = best := by
  rw [innerTail359_6, scan, if_neg (by decide), innerCandidates359_6_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_7
    (if 3 ≤ innerCandidates359_6.length then search edge 3 [2441, 4787] innerCandidates359_6 7228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_7

def innerTail359_5 : List Nat := 4349 :: innerTail359_6
def innerCandidates359_5 : List Nat := [5297]
set_option maxRecDepth 4096 in
theorem innerCandidates359_5_checked : candidates edge 4 2441 4349 innerTail359_6 best.bound = innerCandidates359_5 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_5 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_5 best = best := by
  rw [innerTail359_5, scan, if_neg (by decide), innerCandidates359_5_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_6
    (if 3 ≤ innerCandidates359_5.length then search edge 3 [2441, 4349] innerCandidates359_5 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_6

def innerTail359_4 : List Nat := 4157 :: innerTail359_5
def innerCandidates359_4 : List Nat := [5333]
set_option maxRecDepth 4096 in
theorem innerCandidates359_4_checked : candidates edge 4 2441 4157 innerTail359_5 best.bound = innerCandidates359_4 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_4 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_4 best = best := by
  rw [innerTail359_4, scan, if_neg (by decide), innerCandidates359_4_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_5
    (if 3 ≤ innerCandidates359_4.length then search edge 3 [2441, 4157] innerCandidates359_4 6598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_5

def innerTail359_3 : List Nat := 3677 :: innerTail359_4
def innerCandidates359_3 : List Nat := [6659]
set_option maxRecDepth 4096 in
theorem innerCandidates359_3_checked : candidates edge 4 2441 3677 innerTail359_4 best.bound = innerCandidates359_3 := by
  unfold candidates
  have row : edge 3677 = SearchData.row3677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_3 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_3 best = best := by
  rw [innerTail359_3, scan, if_neg (by decide), innerCandidates359_3_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_4
    (if 3 ≤ innerCandidates359_3.length then search edge 3 [2441, 3677] innerCandidates359_3 6118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_4

def innerTail359_2 : List Nat := 3671 :: innerTail359_3
def innerCandidates359_2 : List Nat := [5381, 7691]
set_option maxRecDepth 4096 in
theorem innerCandidates359_2_checked : candidates edge 4 2441 3671 innerTail359_3 best.bound = innerCandidates359_2 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_2 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_2 best = best := by
  rw [innerTail359_2, scan, if_neg (by decide), innerCandidates359_2_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_3
    (if 3 ≤ innerCandidates359_2.length then search edge 3 [2441, 3671] innerCandidates359_2 6112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_3

def innerTail359_1 : List Nat := 3209 :: innerTail359_2
def innerCandidates359_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates359_1_checked : candidates edge 4 2441 3209 innerTail359_2 best.bound = innerCandidates359_1 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_1 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_1 best = best := by
  rw [innerTail359_1, scan, if_neg (by decide), innerCandidates359_1_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_2
    (if 3 ≤ innerCandidates359_1.length then search edge 3 [2441, 3209] innerCandidates359_1 5650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_2

def innerTail359_0 : List Nat := 2543 :: innerTail359_1
def innerCandidates359_0 : List Nat := [4349, 5147]
set_option maxRecDepth 4096 in
theorem innerCandidates359_0_checked : candidates edge 4 2441 2543 innerTail359_1 best.bound = innerCandidates359_0 := by
  unfold candidates
  have row : edge 2543 = PrimePairCache.smallRow2543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan359_0 : scan edge (search edge 3) 4 [2441] 2441 innerTail359_0 best = best := by
  rw [innerTail359_0, scan, if_neg (by decide), innerCandidates359_0_checked]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_1
    (if 3 ≤ innerCandidates359_0.length then search edge 3 [2441, 2543] innerCandidates359_0 4984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan359_1

theorem rootBranch359 : search edge 4 [2441] rootCandidates359 2441 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2441] 2441 innerTail359_0 best = best
  exact innerScan359_0

def rootCandidates358 : List Nat := [3301, 3361, 3541, 3889, 3931, 4111, 4621, 4969, 5077, 6073, 6301, 6337, 6607, 6967, 7027, 7039, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates358_checked : candidates edge 5 0 2437 tail359 best.bound = rootCandidates358 := by
  unfold candidates
  have row : edge 2437 = PrimePairCache.smallRow2437.contains := by funext q; rfl
  rw [row]
  decide

def innerTail358_17 : List Nat := []
theorem innerScan358_17 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_17 best = best := rfl

def innerTail358_16 : List Nat := 8353 :: innerTail358_17
def innerCandidates358_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_16_checked : candidates edge 4 2437 8353 innerTail358_17 best.bound = innerCandidates358_16 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_16 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_16 best = best := by
  rw [innerTail358_16, scan, if_neg (by decide), innerCandidates358_16_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_17
    (if 3 ≤ innerCandidates358_16.length then search edge 3 [2437, 8353] innerCandidates358_16 10790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_17

def innerTail358_15 : List Nat := 7039 :: innerTail358_16
def innerCandidates358_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_15_checked : candidates edge 4 2437 7039 innerTail358_16 best.bound = innerCandidates358_15 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_15 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_15 best = best := by
  rw [innerTail358_15, scan, if_neg (by decide), innerCandidates358_15_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_16
    (if 3 ≤ innerCandidates358_15.length then search edge 3 [2437, 7039] innerCandidates358_15 9476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_16

def innerTail358_14 : List Nat := 7027 :: innerTail358_15
def innerCandidates358_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_14_checked : candidates edge 4 2437 7027 innerTail358_15 best.bound = innerCandidates358_14 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_14 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_14 best = best := by
  rw [innerTail358_14, scan, if_neg (by decide), innerCandidates358_14_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_15
    (if 3 ≤ innerCandidates358_14.length then search edge 3 [2437, 7027] innerCandidates358_14 9464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_15

def innerTail358_13 : List Nat := 6967 :: innerTail358_14
def innerCandidates358_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_13_checked : candidates edge 4 2437 6967 innerTail358_14 best.bound = innerCandidates358_13 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_13 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_13 best = best := by
  rw [innerTail358_13, scan, if_neg (by decide), innerCandidates358_13_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_14
    (if 3 ≤ innerCandidates358_13.length then search edge 3 [2437, 6967] innerCandidates358_13 9404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_14

def innerTail358_12 : List Nat := 6607 :: innerTail358_13
def innerCandidates358_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_12_checked : candidates edge 4 2437 6607 innerTail358_13 best.bound = innerCandidates358_12 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_12 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_12 best = best := by
  rw [innerTail358_12, scan, if_neg (by decide), innerCandidates358_12_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_13
    (if 3 ≤ innerCandidates358_12.length then search edge 3 [2437, 6607] innerCandidates358_12 9044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_13

def innerTail358_11 : List Nat := 6337 :: innerTail358_12
def innerCandidates358_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_11_checked : candidates edge 4 2437 6337 innerTail358_12 best.bound = innerCandidates358_11 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_11 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_11 best = best := by
  rw [innerTail358_11, scan, if_neg (by decide), innerCandidates358_11_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_12
    (if 3 ≤ innerCandidates358_11.length then search edge 3 [2437, 6337] innerCandidates358_11 8774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_12

def innerTail358_10 : List Nat := 6301 :: innerTail358_11
def innerCandidates358_10 : List Nat := [6607, 7027, 7039]
set_option maxRecDepth 4096 in
theorem innerCandidates358_10_checked : candidates edge 4 2437 6301 innerTail358_11 best.bound = innerCandidates358_10 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild358_10 : search edge 3 [2437, 6301] innerCandidates358_10 8738 best = best := by decide

theorem innerScan358_10 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_10 best = best := by
  rw [innerTail358_10, scan, if_neg (by decide), innerCandidates358_10_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_11
    (if 3 ≤ innerCandidates358_10.length then search edge 3 [2437, 6301] innerCandidates358_10 8738 best else best) = best
  rw [if_pos (by decide), innerChild358_10]
  exact innerScan358_11

def innerTail358_9 : List Nat := 6073 :: innerTail358_10
def innerCandidates358_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_9_checked : candidates edge 4 2437 6073 innerTail358_10 best.bound = innerCandidates358_9 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_9 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_9 best = best := by
  rw [innerTail358_9, scan, if_neg (by decide), innerCandidates358_9_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_10
    (if 3 ≤ innerCandidates358_9.length then search edge 3 [2437, 6073] innerCandidates358_9 8510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_10

def innerTail358_8 : List Nat := 5077 :: innerTail358_9
def innerCandidates358_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_8_checked : candidates edge 4 2437 5077 innerTail358_9 best.bound = innerCandidates358_8 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_8 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_8 best = best := by
  rw [innerTail358_8, scan, if_neg (by decide), innerCandidates358_8_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_9
    (if 3 ≤ innerCandidates358_8.length then search edge 3 [2437, 5077] innerCandidates358_8 7514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_9

def innerTail358_7 : List Nat := 4969 :: innerTail358_8
def innerCandidates358_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_7_checked : candidates edge 4 2437 4969 innerTail358_8 best.bound = innerCandidates358_7 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_7 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_7 best = best := by
  rw [innerTail358_7, scan, if_neg (by decide), innerCandidates358_7_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_8
    (if 3 ≤ innerCandidates358_7.length then search edge 3 [2437, 4969] innerCandidates358_7 7406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_8

def innerTail358_6 : List Nat := 4621 :: innerTail358_7
def innerCandidates358_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_6_checked : candidates edge 4 2437 4621 innerTail358_7 best.bound = innerCandidates358_6 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_6 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_6 best = best := by
  rw [innerTail358_6, scan, if_neg (by decide), innerCandidates358_6_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_7
    (if 3 ≤ innerCandidates358_6.length then search edge 3 [2437, 4621] innerCandidates358_6 7058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_7

def innerTail358_5 : List Nat := 4111 :: innerTail358_6
def innerCandidates358_5 : List Nat := [6073]
set_option maxRecDepth 4096 in
theorem innerCandidates358_5_checked : candidates edge 4 2437 4111 innerTail358_6 best.bound = innerCandidates358_5 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_5 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_5 best = best := by
  rw [innerTail358_5, scan, if_neg (by decide), innerCandidates358_5_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_6
    (if 3 ≤ innerCandidates358_5.length then search edge 3 [2437, 4111] innerCandidates358_5 6548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_6

def innerTail358_4 : List Nat := 3931 :: innerTail358_5
def innerCandidates358_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_4_checked : candidates edge 4 2437 3931 innerTail358_5 best.bound = innerCandidates358_4 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_4 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_4 best = best := by
  rw [innerTail358_4, scan, if_neg (by decide), innerCandidates358_4_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_5
    (if 3 ≤ innerCandidates358_4.length then search edge 3 [2437, 3931] innerCandidates358_4 6368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_5

def innerTail358_3 : List Nat := 3889 :: innerTail358_4
def innerCandidates358_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_3_checked : candidates edge 4 2437 3889 innerTail358_4 best.bound = innerCandidates358_3 := by
  unfold candidates
  have row : edge 3889 = SearchData.row3889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_3 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_3 best = best := by
  rw [innerTail358_3, scan, if_neg (by decide), innerCandidates358_3_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_4
    (if 3 ≤ innerCandidates358_3.length then search edge 3 [2437, 3889] innerCandidates358_3 6326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_4

def innerTail358_2 : List Nat := 3541 :: innerTail358_3
def innerCandidates358_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_2_checked : candidates edge 4 2437 3541 innerTail358_3 best.bound = innerCandidates358_2 := by
  unfold candidates
  have row : edge 3541 = SearchData.row3541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_2 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_2 best = best := by
  rw [innerTail358_2, scan, if_neg (by decide), innerCandidates358_2_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_3
    (if 3 ≤ innerCandidates358_2.length then search edge 3 [2437, 3541] innerCandidates358_2 5978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_3

def innerTail358_1 : List Nat := 3361 :: innerTail358_2
def innerCandidates358_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates358_1_checked : candidates edge 4 2437 3361 innerTail358_2 best.bound = innerCandidates358_1 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_1 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_1 best = best := by
  rw [innerTail358_1, scan, if_neg (by decide), innerCandidates358_1_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_2
    (if 3 ≤ innerCandidates358_1.length then search edge 3 [2437, 3361] innerCandidates358_1 5798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_2

def innerTail358_0 : List Nat := 3301 :: innerTail358_1
def innerCandidates358_0 : List Nat := [6967]
set_option maxRecDepth 4096 in
theorem innerCandidates358_0_checked : candidates edge 4 2437 3301 innerTail358_1 best.bound = innerCandidates358_0 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan358_0 : scan edge (search edge 3) 4 [2437] 2437 innerTail358_0 best = best := by
  rw [innerTail358_0, scan, if_neg (by decide), innerCandidates358_0_checked]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_1
    (if 3 ≤ innerCandidates358_0.length then search edge 3 [2437, 3301] innerCandidates358_0 5738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan358_1

theorem rootBranch358 : search edge 4 [2437] rootCandidates358 2437 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2437] 2437 innerTail358_0 best = best
  exact innerScan358_0

def rootCandidates357 : List Nat := [3359, 3851, 3863, 4391, 4451, 4547, 5021, 5147, 5303, 5639, 5669, 7727, 7919, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates357_checked : candidates edge 5 0 2423 tail358 best.bound = rootCandidates357 := by
  unfold candidates
  have row : edge 2423 = PrimePairCache.smallRow2423.contains := by funext q; rfl
  rw [row]
  decide

def innerTail357_14 : List Nat := []
theorem innerScan357_14 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_14 best = best := rfl

def innerTail357_13 : List Nat := 8273 :: innerTail357_14
def innerCandidates357_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates357_13_checked : candidates edge 4 2423 8273 innerTail357_14 best.bound = innerCandidates357_13 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_13 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_13 best = best := by
  rw [innerTail357_13, scan, if_neg (by decide), innerCandidates357_13_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_14
    (if 3 ≤ innerCandidates357_13.length then search edge 3 [2423, 8273] innerCandidates357_13 10696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_14

def innerTail357_12 : List Nat := 7919 :: innerTail357_13
def innerCandidates357_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates357_12_checked : candidates edge 4 2423 7919 innerTail357_13 best.bound = innerCandidates357_12 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_12 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_12 best = best := by
  rw [innerTail357_12, scan, if_neg (by decide), innerCandidates357_12_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_13
    (if 3 ≤ innerCandidates357_12.length then search edge 3 [2423, 7919] innerCandidates357_12 10342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_13

def innerTail357_11 : List Nat := 7727 :: innerTail357_12
def innerCandidates357_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates357_11_checked : candidates edge 4 2423 7727 innerTail357_12 best.bound = innerCandidates357_11 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_11 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_11 best = best := by
  rw [innerTail357_11, scan, if_neg (by decide), innerCandidates357_11_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_12
    (if 3 ≤ innerCandidates357_11.length then search edge 3 [2423, 7727] innerCandidates357_11 10150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_12

def innerTail357_10 : List Nat := 5669 :: innerTail357_11
def innerCandidates357_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates357_10_checked : candidates edge 4 2423 5669 innerTail357_11 best.bound = innerCandidates357_10 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_10 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_10 best = best := by
  rw [innerTail357_10, scan, if_neg (by decide), innerCandidates357_10_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_11
    (if 3 ≤ innerCandidates357_10.length then search edge 3 [2423, 5669] innerCandidates357_10 8092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_11

def innerTail357_9 : List Nat := 5639 :: innerTail357_10
def innerCandidates357_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates357_9_checked : candidates edge 4 2423 5639 innerTail357_10 best.bound = innerCandidates357_9 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_9 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_9 best = best := by
  rw [innerTail357_9, scan, if_neg (by decide), innerCandidates357_9_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_10
    (if 3 ≤ innerCandidates357_9.length then search edge 3 [2423, 5639] innerCandidates357_9 8062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_10

def innerTail357_8 : List Nat := 5303 :: innerTail357_9
def innerCandidates357_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates357_8_checked : candidates edge 4 2423 5303 innerTail357_9 best.bound = innerCandidates357_8 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_8 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_8 best = best := by
  rw [innerTail357_8, scan, if_neg (by decide), innerCandidates357_8_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_9
    (if 3 ≤ innerCandidates357_8.length then search edge 3 [2423, 5303] innerCandidates357_8 7726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_9

def innerTail357_7 : List Nat := 5147 :: innerTail357_8
def innerCandidates357_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates357_7_checked : candidates edge 4 2423 5147 innerTail357_8 best.bound = innerCandidates357_7 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_7 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_7 best = best := by
  rw [innerTail357_7, scan, if_neg (by decide), innerCandidates357_7_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_8
    (if 3 ≤ innerCandidates357_7.length then search edge 3 [2423, 5147] innerCandidates357_7 7570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_8

def innerTail357_6 : List Nat := 5021 :: innerTail357_7
def innerCandidates357_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates357_6_checked : candidates edge 4 2423 5021 innerTail357_7 best.bound = innerCandidates357_6 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_6 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_6 best = best := by
  rw [innerTail357_6, scan, if_neg (by decide), innerCandidates357_6_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_7
    (if 3 ≤ innerCandidates357_6.length then search edge 3 [2423, 5021] innerCandidates357_6 7444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_7

def innerTail357_5 : List Nat := 4547 :: innerTail357_6
def innerCandidates357_5 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates357_5_checked : candidates edge 4 2423 4547 innerTail357_6 best.bound = innerCandidates357_5 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_5 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_5 best = best := by
  rw [innerTail357_5, scan, if_neg (by decide), innerCandidates357_5_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_6
    (if 3 ≤ innerCandidates357_5.length then search edge 3 [2423, 4547] innerCandidates357_5 6970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_6

def innerTail357_4 : List Nat := 4451 :: innerTail357_5
def innerCandidates357_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates357_4_checked : candidates edge 4 2423 4451 innerTail357_5 best.bound = innerCandidates357_4 := by
  unfold candidates
  have row : edge 4451 = SearchData.row4451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_4 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_4 best = best := by
  rw [innerTail357_4, scan, if_neg (by decide), innerCandidates357_4_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_5
    (if 3 ≤ innerCandidates357_4.length then search edge 3 [2423, 4451] innerCandidates357_4 6874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_5

def innerTail357_3 : List Nat := 4391 :: innerTail357_4
def innerCandidates357_3 : List Nat := [5639]
set_option maxRecDepth 4096 in
theorem innerCandidates357_3_checked : candidates edge 4 2423 4391 innerTail357_4 best.bound = innerCandidates357_3 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_3 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_3 best = best := by
  rw [innerTail357_3, scan, if_neg (by decide), innerCandidates357_3_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_4
    (if 3 ≤ innerCandidates357_3.length then search edge 3 [2423, 4391] innerCandidates357_3 6814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_4

def innerTail357_2 : List Nat := 3863 :: innerTail357_3
def innerCandidates357_2 : List Nat := [4451]
set_option maxRecDepth 4096 in
theorem innerCandidates357_2_checked : candidates edge 4 2423 3863 innerTail357_3 best.bound = innerCandidates357_2 := by
  unfold candidates
  have row : edge 3863 = SearchData.row3863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_2 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_2 best = best := by
  rw [innerTail357_2, scan, if_neg (by decide), innerCandidates357_2_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_3
    (if 3 ≤ innerCandidates357_2.length then search edge 3 [2423, 3863] innerCandidates357_2 6286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_3

def innerTail357_1 : List Nat := 3851 :: innerTail357_2
def innerCandidates357_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates357_1_checked : candidates edge 4 2423 3851 innerTail357_2 best.bound = innerCandidates357_1 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_1 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_1 best = best := by
  rw [innerTail357_1, scan, if_neg (by decide), innerCandidates357_1_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_2
    (if 3 ≤ innerCandidates357_1.length then search edge 3 [2423, 3851] innerCandidates357_1 6274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_2

def innerTail357_0 : List Nat := 3359 :: innerTail357_1
def innerCandidates357_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates357_0_checked : candidates edge 4 2423 3359 innerTail357_1 best.bound = innerCandidates357_0 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan357_0 : scan edge (search edge 3) 4 [2423] 2423 innerTail357_0 best = best := by
  rw [innerTail357_0, scan, if_neg (by decide), innerCandidates357_0_checked]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_1
    (if 3 ≤ innerCandidates357_0.length then search edge 3 [2423, 3359] innerCandidates357_0 5782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan357_1

theorem rootBranch357 : search edge 4 [2423] rootCandidates357 2423 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2423] 2423 innerTail357_0 best = best
  exact innerScan357_0

end ProjectEulerSolutions.PrimePairUpper.Search
