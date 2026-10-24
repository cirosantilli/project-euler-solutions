import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates204 : List Nat := [1297, 1399, 1459, 2179, 2389, 2803, 3463, 3559, 3691, 5059, 5179, 5437, 5479, 6037, 6271, 6607, 6781, 7069, 8101, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates204_checked : candidates edge 5 0 1279 tail205 best.bound = rootCandidates204 := by
  unfold candidates
  have row : edge 1279 = PrimePairCache.smallRow1279.contains := by funext q; rfl
  rw [row]
  decide

def innerTail204_20 : List Nat := []
theorem innerScan204_20 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_20 best = best := rfl

def innerTail204_19 : List Nat := 8293 :: innerTail204_20
def innerCandidates204_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_19_checked : candidates edge 4 1279 8293 innerTail204_20 best.bound = innerCandidates204_19 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_19 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_19 best = best := by
  rw [innerTail204_19, scan, if_neg (by decide), innerCandidates204_19_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_20
    (if 3 ≤ innerCandidates204_19.length then search edge 3 [1279, 8293] innerCandidates204_19 9572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_20

def innerTail204_18 : List Nat := 8101 :: innerTail204_19
def innerCandidates204_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_18_checked : candidates edge 4 1279 8101 innerTail204_19 best.bound = innerCandidates204_18 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_18 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_18 best = best := by
  rw [innerTail204_18, scan, if_neg (by decide), innerCandidates204_18_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_19
    (if 3 ≤ innerCandidates204_18.length then search edge 3 [1279, 8101] innerCandidates204_18 9380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_19

def innerTail204_17 : List Nat := 7069 :: innerTail204_18
def innerCandidates204_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_17_checked : candidates edge 4 1279 7069 innerTail204_18 best.bound = innerCandidates204_17 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_17 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_17 best = best := by
  rw [innerTail204_17, scan, if_neg (by decide), innerCandidates204_17_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_18
    (if 3 ≤ innerCandidates204_17.length then search edge 3 [1279, 7069] innerCandidates204_17 8348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_18

def innerTail204_16 : List Nat := 6781 :: innerTail204_17
def innerCandidates204_16 : List Nat := [7069]
set_option maxRecDepth 4096 in
theorem innerCandidates204_16_checked : candidates edge 4 1279 6781 innerTail204_17 best.bound = innerCandidates204_16 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_16 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_16 best = best := by
  rw [innerTail204_16, scan, if_neg (by decide), innerCandidates204_16_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_17
    (if 3 ≤ innerCandidates204_16.length then search edge 3 [1279, 6781] innerCandidates204_16 8060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_17

def innerTail204_15 : List Nat := 6607 :: innerTail204_16
def innerCandidates204_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_15_checked : candidates edge 4 1279 6607 innerTail204_16 best.bound = innerCandidates204_15 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_15 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_15 best = best := by
  rw [innerTail204_15, scan, if_neg (by decide), innerCandidates204_15_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_16
    (if 3 ≤ innerCandidates204_15.length then search edge 3 [1279, 6607] innerCandidates204_15 7886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_16

def innerTail204_14 : List Nat := 6271 :: innerTail204_15
def innerCandidates204_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_14_checked : candidates edge 4 1279 6271 innerTail204_15 best.bound = innerCandidates204_14 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_14 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_14 best = best := by
  rw [innerTail204_14, scan, if_neg (by decide), innerCandidates204_14_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_15
    (if 3 ≤ innerCandidates204_14.length then search edge 3 [1279, 6271] innerCandidates204_14 7550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_15

def innerTail204_13 : List Nat := 6037 :: innerTail204_14
def innerCandidates204_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_13_checked : candidates edge 4 1279 6037 innerTail204_14 best.bound = innerCandidates204_13 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_13 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_13 best = best := by
  rw [innerTail204_13, scan, if_neg (by decide), innerCandidates204_13_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_14
    (if 3 ≤ innerCandidates204_13.length then search edge 3 [1279, 6037] innerCandidates204_13 7316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_14

def innerTail204_12 : List Nat := 5479 :: innerTail204_13
def innerCandidates204_12 : List Nat := [6271, 6607]
set_option maxRecDepth 4096 in
theorem innerCandidates204_12_checked : candidates edge 4 1279 5479 innerTail204_13 best.bound = innerCandidates204_12 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_12 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_12 best = best := by
  rw [innerTail204_12, scan, if_neg (by decide), innerCandidates204_12_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_13
    (if 3 ≤ innerCandidates204_12.length then search edge 3 [1279, 5479] innerCandidates204_12 6758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_13

def innerTail204_11 : List Nat := 5437 :: innerTail204_12
def innerCandidates204_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_11_checked : candidates edge 4 1279 5437 innerTail204_12 best.bound = innerCandidates204_11 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_11 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_11 best = best := by
  rw [innerTail204_11, scan, if_neg (by decide), innerCandidates204_11_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_12
    (if 3 ≤ innerCandidates204_11.length then search edge 3 [1279, 5437] innerCandidates204_11 6716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_12

def innerTail204_10 : List Nat := 5179 :: innerTail204_11
def innerCandidates204_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_10_checked : candidates edge 4 1279 5179 innerTail204_11 best.bound = innerCandidates204_10 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_10 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_10 best = best := by
  rw [innerTail204_10, scan, if_neg (by decide), innerCandidates204_10_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_11
    (if 3 ≤ innerCandidates204_10.length then search edge 3 [1279, 5179] innerCandidates204_10 6458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_11

def innerTail204_9 : List Nat := 5059 :: innerTail204_10
def innerCandidates204_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_9_checked : candidates edge 4 1279 5059 innerTail204_10 best.bound = innerCandidates204_9 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_9 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_9 best = best := by
  rw [innerTail204_9, scan, if_neg (by decide), innerCandidates204_9_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_10
    (if 3 ≤ innerCandidates204_9.length then search edge 3 [1279, 5059] innerCandidates204_9 6338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_10

def innerTail204_8 : List Nat := 3691 :: innerTail204_9
def innerCandidates204_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_8_checked : candidates edge 4 1279 3691 innerTail204_9 best.bound = innerCandidates204_8 := by
  unfold candidates
  have row : edge 3691 = SearchData.row3691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_8 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_8 best = best := by
  rw [innerTail204_8, scan, if_neg (by decide), innerCandidates204_8_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_9
    (if 3 ≤ innerCandidates204_8.length then search edge 3 [1279, 3691] innerCandidates204_8 4970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_9

def innerTail204_7 : List Nat := 3559 :: innerTail204_8
def innerCandidates204_7 : List Nat := [6037, 6607]
set_option maxRecDepth 4096 in
theorem innerCandidates204_7_checked : candidates edge 4 1279 3559 innerTail204_8 best.bound = innerCandidates204_7 := by
  unfold candidates
  have row : edge 3559 = SearchData.row3559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_7 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_7 best = best := by
  rw [innerTail204_7, scan, if_neg (by decide), innerCandidates204_7_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_8
    (if 3 ≤ innerCandidates204_7.length then search edge 3 [1279, 3559] innerCandidates204_7 4838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_8

def innerTail204_6 : List Nat := 3463 :: innerTail204_7
def innerCandidates204_6 : List Nat := [5479, 6037]
set_option maxRecDepth 4096 in
theorem innerCandidates204_6_checked : candidates edge 4 1279 3463 innerTail204_7 best.bound = innerCandidates204_6 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_6 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_6 best = best := by
  rw [innerTail204_6, scan, if_neg (by decide), innerCandidates204_6_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_7
    (if 3 ≤ innerCandidates204_6.length then search edge 3 [1279, 3463] innerCandidates204_6 4742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_7

def innerTail204_5 : List Nat := 2803 :: innerTail204_6
def innerCandidates204_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_5_checked : candidates edge 4 1279 2803 innerTail204_6 best.bound = innerCandidates204_5 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_5 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_5 best = best := by
  rw [innerTail204_5, scan, if_neg (by decide), innerCandidates204_5_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_6
    (if 3 ≤ innerCandidates204_5.length then search edge 3 [1279, 2803] innerCandidates204_5 4082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_6

def innerTail204_4 : List Nat := 2389 :: innerTail204_5
def innerCandidates204_4 : List Nat := [5479]
set_option maxRecDepth 4096 in
theorem innerCandidates204_4_checked : candidates edge 4 1279 2389 innerTail204_5 best.bound = innerCandidates204_4 := by
  unfold candidates
  have row : edge 2389 = PrimePairCache.smallRow2389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_4 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_4 best = best := by
  rw [innerTail204_4, scan, if_neg (by decide), innerCandidates204_4_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_5
    (if 3 ≤ innerCandidates204_4.length then search edge 3 [1279, 2389] innerCandidates204_4 3668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_5

def innerTail204_3 : List Nat := 2179 :: innerTail204_4
def innerCandidates204_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_3_checked : candidates edge 4 1279 2179 innerTail204_4 best.bound = innerCandidates204_3 := by
  unfold candidates
  have row : edge 2179 = PrimePairCache.smallRow2179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_3 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_3 best = best := by
  rw [innerTail204_3, scan, if_neg (by decide), innerCandidates204_3_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_4
    (if 3 ≤ innerCandidates204_3.length then search edge 3 [1279, 2179] innerCandidates204_3 3458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_4

def innerTail204_2 : List Nat := 1459 :: innerTail204_3
def innerCandidates204_2 : List Nat := [5437, 6271]
set_option maxRecDepth 4096 in
theorem innerCandidates204_2_checked : candidates edge 4 1279 1459 innerTail204_3 best.bound = innerCandidates204_2 := by
  unfold candidates
  have row : edge 1459 = PrimePairCache.smallRow1459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_2 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_2 best = best := by
  rw [innerTail204_2, scan, if_neg (by decide), innerCandidates204_2_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_3
    (if 3 ≤ innerCandidates204_2.length then search edge 3 [1279, 1459] innerCandidates204_2 2738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_3

def innerTail204_1 : List Nat := 1399 :: innerTail204_2
def innerCandidates204_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates204_1_checked : candidates edge 4 1279 1399 innerTail204_2 best.bound = innerCandidates204_1 := by
  unfold candidates
  have row : edge 1399 = PrimePairCache.smallRow1399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_1 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_1 best = best := by
  rw [innerTail204_1, scan, if_neg (by decide), innerCandidates204_1_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_2
    (if 3 ≤ innerCandidates204_1.length then search edge 3 [1279, 1399] innerCandidates204_1 2678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_2

def innerTail204_0 : List Nat := 1297 :: innerTail204_1
def innerCandidates204_0 : List Nat := [6781]
set_option maxRecDepth 4096 in
theorem innerCandidates204_0_checked : candidates edge 4 1279 1297 innerTail204_1 best.bound = innerCandidates204_0 := by
  unfold candidates
  have row : edge 1297 = PrimePairCache.smallRow1297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan204_0 : scan edge (search edge 3) 4 [1279] 1279 innerTail204_0 best = best := by
  rw [innerTail204_0, scan, if_neg (by decide), innerCandidates204_0_checked]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_1
    (if 3 ≤ innerCandidates204_0.length then search edge 3 [1279, 1297] innerCandidates204_0 2576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan204_1

theorem rootBranch204 : search edge 4 [1279] rootCandidates204 1279 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1279] 1279 innerTail204_0 best = best
  exact innerScan204_0

def rootCandidates203 : List Nat := [1511, 1523, 3359, 4049, 4457, 4787, 4889, 4967, 5003, 6011, 6053, 6173, 6353, 6761, 7433, 7529, 7559, 7577, 7703, 8147, 8291, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates203_checked : candidates edge 5 0 1277 tail204 best.bound = rootCandidates203 := by
  unfold candidates
  have row : edge 1277 = PrimePairCache.smallRow1277.contains := by funext q; rfl
  rw [row]
  decide

def innerTail203_22 : List Nat := []
theorem innerScan203_22 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_22 best = best := rfl

def innerTail203_21 : List Nat := 8369 :: innerTail203_22
def innerCandidates203_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_21_checked : candidates edge 4 1277 8369 innerTail203_22 best.bound = innerCandidates203_21 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_21 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_21 best = best := by
  rw [innerTail203_21, scan, if_neg (by decide), innerCandidates203_21_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_22
    (if 3 ≤ innerCandidates203_21.length then search edge 3 [1277, 8369] innerCandidates203_21 9646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_22

def innerTail203_20 : List Nat := 8291 :: innerTail203_21
def innerCandidates203_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_20_checked : candidates edge 4 1277 8291 innerTail203_21 best.bound = innerCandidates203_20 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_20 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_20 best = best := by
  rw [innerTail203_20, scan, if_neg (by decide), innerCandidates203_20_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_21
    (if 3 ≤ innerCandidates203_20.length then search edge 3 [1277, 8291] innerCandidates203_20 9568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_21

def innerTail203_19 : List Nat := 8147 :: innerTail203_20
def innerCandidates203_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_19_checked : candidates edge 4 1277 8147 innerTail203_20 best.bound = innerCandidates203_19 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_19 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_19 best = best := by
  rw [innerTail203_19, scan, if_neg (by decide), innerCandidates203_19_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_20
    (if 3 ≤ innerCandidates203_19.length then search edge 3 [1277, 8147] innerCandidates203_19 9424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_20

def innerTail203_18 : List Nat := 7703 :: innerTail203_19
def innerCandidates203_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_18_checked : candidates edge 4 1277 7703 innerTail203_19 best.bound = innerCandidates203_18 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_18 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_18 best = best := by
  rw [innerTail203_18, scan, if_neg (by decide), innerCandidates203_18_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_19
    (if 3 ≤ innerCandidates203_18.length then search edge 3 [1277, 7703] innerCandidates203_18 8980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_19

def innerTail203_17 : List Nat := 7577 :: innerTail203_18
def innerCandidates203_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_17_checked : candidates edge 4 1277 7577 innerTail203_18 best.bound = innerCandidates203_17 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_17 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_17 best = best := by
  rw [innerTail203_17, scan, if_neg (by decide), innerCandidates203_17_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_18
    (if 3 ≤ innerCandidates203_17.length then search edge 3 [1277, 7577] innerCandidates203_17 8854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_18

def innerTail203_16 : List Nat := 7559 :: innerTail203_17
def innerCandidates203_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_16_checked : candidates edge 4 1277 7559 innerTail203_17 best.bound = innerCandidates203_16 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_16 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_16 best = best := by
  rw [innerTail203_16, scan, if_neg (by decide), innerCandidates203_16_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_17
    (if 3 ≤ innerCandidates203_16.length then search edge 3 [1277, 7559] innerCandidates203_16 8836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_17

def innerTail203_15 : List Nat := 7529 :: innerTail203_16
def innerCandidates203_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_15_checked : candidates edge 4 1277 7529 innerTail203_16 best.bound = innerCandidates203_15 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_15 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_15 best = best := by
  rw [innerTail203_15, scan, if_neg (by decide), innerCandidates203_15_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_16
    (if 3 ≤ innerCandidates203_15.length then search edge 3 [1277, 7529] innerCandidates203_15 8806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_16

def innerTail203_14 : List Nat := 7433 :: innerTail203_15
def innerCandidates203_14 : List Nat := [7559]
set_option maxRecDepth 4096 in
theorem innerCandidates203_14_checked : candidates edge 4 1277 7433 innerTail203_15 best.bound = innerCandidates203_14 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_14 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_14 best = best := by
  rw [innerTail203_14, scan, if_neg (by decide), innerCandidates203_14_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_15
    (if 3 ≤ innerCandidates203_14.length then search edge 3 [1277, 7433] innerCandidates203_14 8710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_15

def innerTail203_13 : List Nat := 6761 :: innerTail203_14
def innerCandidates203_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_13_checked : candidates edge 4 1277 6761 innerTail203_14 best.bound = innerCandidates203_13 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_13 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_13 best = best := by
  rw [innerTail203_13, scan, if_neg (by decide), innerCandidates203_13_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_14
    (if 3 ≤ innerCandidates203_13.length then search edge 3 [1277, 6761] innerCandidates203_13 8038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_14

def innerTail203_12 : List Nat := 6353 :: innerTail203_13
def innerCandidates203_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_12_checked : candidates edge 4 1277 6353 innerTail203_13 best.bound = innerCandidates203_12 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_12 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_12 best = best := by
  rw [innerTail203_12, scan, if_neg (by decide), innerCandidates203_12_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_13
    (if 3 ≤ innerCandidates203_12.length then search edge 3 [1277, 6353] innerCandidates203_12 7630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_13

def innerTail203_11 : List Nat := 6173 :: innerTail203_12
def innerCandidates203_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_11_checked : candidates edge 4 1277 6173 innerTail203_12 best.bound = innerCandidates203_11 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_11 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_11 best = best := by
  rw [innerTail203_11, scan, if_neg (by decide), innerCandidates203_11_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_12
    (if 3 ≤ innerCandidates203_11.length then search edge 3 [1277, 6173] innerCandidates203_11 7450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_12

def innerTail203_10 : List Nat := 6053 :: innerTail203_11
def innerCandidates203_10 : List Nat := [6173]
set_option maxRecDepth 4096 in
theorem innerCandidates203_10_checked : candidates edge 4 1277 6053 innerTail203_11 best.bound = innerCandidates203_10 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_10 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_10 best = best := by
  rw [innerTail203_10, scan, if_neg (by decide), innerCandidates203_10_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_11
    (if 3 ≤ innerCandidates203_10.length then search edge 3 [1277, 6053] innerCandidates203_10 7330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_11

def innerTail203_9 : List Nat := 6011 :: innerTail203_10
def innerCandidates203_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_9_checked : candidates edge 4 1277 6011 innerTail203_10 best.bound = innerCandidates203_9 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_9 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_9 best = best := by
  rw [innerTail203_9, scan, if_neg (by decide), innerCandidates203_9_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_10
    (if 3 ≤ innerCandidates203_9.length then search edge 3 [1277, 6011] innerCandidates203_9 7288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_10

def innerTail203_8 : List Nat := 5003 :: innerTail203_9
def innerCandidates203_8 : List Nat := [7703, 8147]
set_option maxRecDepth 4096 in
theorem innerCandidates203_8_checked : candidates edge 4 1277 5003 innerTail203_9 best.bound = innerCandidates203_8 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_8 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_8 best = best := by
  rw [innerTail203_8, scan, if_neg (by decide), innerCandidates203_8_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_9
    (if 3 ≤ innerCandidates203_8.length then search edge 3 [1277, 5003] innerCandidates203_8 6280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_9

def innerTail203_7 : List Nat := 4967 :: innerTail203_8
def innerCandidates203_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_7_checked : candidates edge 4 1277 4967 innerTail203_8 best.bound = innerCandidates203_7 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_7 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_7 best = best := by
  rw [innerTail203_7, scan, if_neg (by decide), innerCandidates203_7_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_8
    (if 3 ≤ innerCandidates203_7.length then search edge 3 [1277, 4967] innerCandidates203_7 6244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_8

def innerTail203_6 : List Nat := 4889 :: innerTail203_7
def innerCandidates203_6 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates203_6_checked : candidates edge 4 1277 4889 innerTail203_7 best.bound = innerCandidates203_6 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_6 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_6 best = best := by
  rw [innerTail203_6, scan, if_neg (by decide), innerCandidates203_6_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_7
    (if 3 ≤ innerCandidates203_6.length then search edge 3 [1277, 4889] innerCandidates203_6 6166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_7

def innerTail203_5 : List Nat := 4787 :: innerTail203_6
def innerCandidates203_5 : List Nat := [7433]
set_option maxRecDepth 4096 in
theorem innerCandidates203_5_checked : candidates edge 4 1277 4787 innerTail203_6 best.bound = innerCandidates203_5 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_5 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_5 best = best := by
  rw [innerTail203_5, scan, if_neg (by decide), innerCandidates203_5_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_6
    (if 3 ≤ innerCandidates203_5.length then search edge 3 [1277, 4787] innerCandidates203_5 6064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_6

def innerTail203_4 : List Nat := 4457 :: innerTail203_5
def innerCandidates203_4 : List Nat := [7433]
set_option maxRecDepth 4096 in
theorem innerCandidates203_4_checked : candidates edge 4 1277 4457 innerTail203_5 best.bound = innerCandidates203_4 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_4 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_4 best = best := by
  rw [innerTail203_4, scan, if_neg (by decide), innerCandidates203_4_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_5
    (if 3 ≤ innerCandidates203_4.length then search edge 3 [1277, 4457] innerCandidates203_4 5734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_5

def innerTail203_3 : List Nat := 4049 :: innerTail203_4
def innerCandidates203_3 : List Nat := [7703]
set_option maxRecDepth 4096 in
theorem innerCandidates203_3_checked : candidates edge 4 1277 4049 innerTail203_4 best.bound = innerCandidates203_3 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_3 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_3 best = best := by
  rw [innerTail203_3, scan, if_neg (by decide), innerCandidates203_3_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_4
    (if 3 ≤ innerCandidates203_3.length then search edge 3 [1277, 4049] innerCandidates203_3 5326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_4

def innerTail203_2 : List Nat := 3359 :: innerTail203_3
def innerCandidates203_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_2_checked : candidates edge 4 1277 3359 innerTail203_3 best.bound = innerCandidates203_2 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_2 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_2 best = best := by
  rw [innerTail203_2, scan, if_neg (by decide), innerCandidates203_2_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_3
    (if 3 ≤ innerCandidates203_2.length then search edge 3 [1277, 3359] innerCandidates203_2 4636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_3

def innerTail203_1 : List Nat := 1523 :: innerTail203_2
def innerCandidates203_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates203_1_checked : candidates edge 4 1277 1523 innerTail203_2 best.bound = innerCandidates203_1 := by
  unfold candidates
  have row : edge 1523 = PrimePairCache.smallRow1523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_1 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_1 best = best := by
  rw [innerTail203_1, scan, if_neg (by decide), innerCandidates203_1_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_2
    (if 3 ≤ innerCandidates203_1.length then search edge 3 [1277, 1523] innerCandidates203_1 2800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_2

def innerTail203_0 : List Nat := 1511 :: innerTail203_1
def innerCandidates203_0 : List Nat := [4889]
set_option maxRecDepth 4096 in
theorem innerCandidates203_0_checked : candidates edge 4 1277 1511 innerTail203_1 best.bound = innerCandidates203_0 := by
  unfold candidates
  have row : edge 1511 = PrimePairCache.smallRow1511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan203_0 : scan edge (search edge 3) 4 [1277] 1277 innerTail203_0 best = best := by
  rw [innerTail203_0, scan, if_neg (by decide), innerCandidates203_0_checked]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_1
    (if 3 ≤ innerCandidates203_0.length then search edge 3 [1277, 1511] innerCandidates203_0 2788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan203_1

theorem rootBranch203 : search edge 4 [1277] rootCandidates203 1277 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1277] 1277 innerTail203_0 best = best
  exact innerScan203_0

def rootCandidates202 : List Nat := [2213, 2273, 2297, 2357, 2609, 2633, 2819, 3989, 5039, 5507, 6599, 6959, 7283, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates202_checked : candidates edge 5 0 1259 tail203 best.bound = rootCandidates202 := by
  unfold candidates
  have row : edge 1259 = PrimePairCache.smallRow1259.contains := by funext q; rfl
  rw [row]
  decide

def innerTail202_14 : List Nat := []
theorem innerScan202_14 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_14 best = best := rfl

def innerTail202_13 : List Nat := 8291 :: innerTail202_14
def innerCandidates202_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates202_13_checked : candidates edge 4 1259 8291 innerTail202_14 best.bound = innerCandidates202_13 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_13 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_13 best = best := by
  rw [innerTail202_13, scan, if_neg (by decide), innerCandidates202_13_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_14
    (if 3 ≤ innerCandidates202_13.length then search edge 3 [1259, 8291] innerCandidates202_13 9550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_14

def innerTail202_12 : List Nat := 7283 :: innerTail202_13
def innerCandidates202_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates202_12_checked : candidates edge 4 1259 7283 innerTail202_13 best.bound = innerCandidates202_12 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_12 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_12 best = best := by
  rw [innerTail202_12, scan, if_neg (by decide), innerCandidates202_12_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_13
    (if 3 ≤ innerCandidates202_12.length then search edge 3 [1259, 7283] innerCandidates202_12 8542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_13

def innerTail202_11 : List Nat := 6959 :: innerTail202_12
def innerCandidates202_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates202_11_checked : candidates edge 4 1259 6959 innerTail202_12 best.bound = innerCandidates202_11 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_11 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_11 best = best := by
  rw [innerTail202_11, scan, if_neg (by decide), innerCandidates202_11_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_12
    (if 3 ≤ innerCandidates202_11.length then search edge 3 [1259, 6959] innerCandidates202_11 8218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_12

def innerTail202_10 : List Nat := 6599 :: innerTail202_11
def innerCandidates202_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates202_10_checked : candidates edge 4 1259 6599 innerTail202_11 best.bound = innerCandidates202_10 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_10 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_10 best = best := by
  rw [innerTail202_10, scan, if_neg (by decide), innerCandidates202_10_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_11
    (if 3 ≤ innerCandidates202_10.length then search edge 3 [1259, 6599] innerCandidates202_10 7858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_11

def innerTail202_9 : List Nat := 5507 :: innerTail202_10
def innerCandidates202_9 : List Nat := [8291]
set_option maxRecDepth 4096 in
theorem innerCandidates202_9_checked : candidates edge 4 1259 5507 innerTail202_10 best.bound = innerCandidates202_9 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_9 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_9 best = best := by
  rw [innerTail202_9, scan, if_neg (by decide), innerCandidates202_9_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_10
    (if 3 ≤ innerCandidates202_9.length then search edge 3 [1259, 5507] innerCandidates202_9 6766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_10

def innerTail202_8 : List Nat := 5039 :: innerTail202_9
def innerCandidates202_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates202_8_checked : candidates edge 4 1259 5039 innerTail202_9 best.bound = innerCandidates202_8 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_8 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_8 best = best := by
  rw [innerTail202_8, scan, if_neg (by decide), innerCandidates202_8_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_9
    (if 3 ≤ innerCandidates202_8.length then search edge 3 [1259, 5039] innerCandidates202_8 6298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_9

def innerTail202_7 : List Nat := 3989 :: innerTail202_8
def innerCandidates202_7 : List Nat := [8291]
set_option maxRecDepth 4096 in
theorem innerCandidates202_7_checked : candidates edge 4 1259 3989 innerTail202_8 best.bound = innerCandidates202_7 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_7 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_7 best = best := by
  rw [innerTail202_7, scan, if_neg (by decide), innerCandidates202_7_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_8
    (if 3 ≤ innerCandidates202_7.length then search edge 3 [1259, 3989] innerCandidates202_7 5248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_8

def innerTail202_6 : List Nat := 2819 :: innerTail202_7
def innerCandidates202_6 : List Nat := [5039]
set_option maxRecDepth 4096 in
theorem innerCandidates202_6_checked : candidates edge 4 1259 2819 innerTail202_7 best.bound = innerCandidates202_6 := by
  unfold candidates
  have row : edge 2819 = SearchData.row2819.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_6 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_6 best = best := by
  rw [innerTail202_6, scan, if_neg (by decide), innerCandidates202_6_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_7
    (if 3 ≤ innerCandidates202_6.length then search edge 3 [1259, 2819] innerCandidates202_6 4078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_7

def innerTail202_5 : List Nat := 2633 :: innerTail202_6
def innerCandidates202_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates202_5_checked : candidates edge 4 1259 2633 innerTail202_6 best.bound = innerCandidates202_5 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_5 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_5 best = best := by
  rw [innerTail202_5, scan, if_neg (by decide), innerCandidates202_5_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_6
    (if 3 ≤ innerCandidates202_5.length then search edge 3 [1259, 2633] innerCandidates202_5 3892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_6

def innerTail202_4 : List Nat := 2609 :: innerTail202_5
def innerCandidates202_4 : List Nat := [6599]
set_option maxRecDepth 4096 in
theorem innerCandidates202_4_checked : candidates edge 4 1259 2609 innerTail202_5 best.bound = innerCandidates202_4 := by
  unfold candidates
  have row : edge 2609 = PrimePairCache.smallRow2609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_4 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_4 best = best := by
  rw [innerTail202_4, scan, if_neg (by decide), innerCandidates202_4_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_5
    (if 3 ≤ innerCandidates202_4.length then search edge 3 [1259, 2609] innerCandidates202_4 3868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_5

def innerTail202_3 : List Nat := 2357 :: innerTail202_4
def innerCandidates202_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates202_3_checked : candidates edge 4 1259 2357 innerTail202_4 best.bound = innerCandidates202_3 := by
  unfold candidates
  have row : edge 2357 = PrimePairCache.smallRow2357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_3 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_3 best = best := by
  rw [innerTail202_3, scan, if_neg (by decide), innerCandidates202_3_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_4
    (if 3 ≤ innerCandidates202_3.length then search edge 3 [1259, 2357] innerCandidates202_3 3616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_4

def innerTail202_2 : List Nat := 2297 :: innerTail202_3
def innerCandidates202_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates202_2_checked : candidates edge 4 1259 2297 innerTail202_3 best.bound = innerCandidates202_2 := by
  unfold candidates
  have row : edge 2297 = PrimePairCache.smallRow2297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_2 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_2 best = best := by
  rw [innerTail202_2, scan, if_neg (by decide), innerCandidates202_2_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_3
    (if 3 ≤ innerCandidates202_2.length then search edge 3 [1259, 2297] innerCandidates202_2 3556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_3

def innerTail202_1 : List Nat := 2273 :: innerTail202_2
def innerCandidates202_1 : List Nat := [2609, 7283]
set_option maxRecDepth 4096 in
theorem innerCandidates202_1_checked : candidates edge 4 1259 2273 innerTail202_2 best.bound = innerCandidates202_1 := by
  unfold candidates
  have row : edge 2273 = PrimePairCache.smallRow2273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_1 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_1 best = best := by
  rw [innerTail202_1, scan, if_neg (by decide), innerCandidates202_1_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_2
    (if 3 ≤ innerCandidates202_1.length then search edge 3 [1259, 2273] innerCandidates202_1 3532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_2

def innerTail202_0 : List Nat := 2213 :: innerTail202_1
def innerCandidates202_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates202_0_checked : candidates edge 4 1259 2213 innerTail202_1 best.bound = innerCandidates202_0 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan202_0 : scan edge (search edge 3) 4 [1259] 1259 innerTail202_0 best = best := by
  rw [innerTail202_0, scan, if_neg (by decide), innerCandidates202_0_checked]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_1
    (if 3 ≤ innerCandidates202_0.length then search edge 3 [1259, 2213] innerCandidates202_0 3472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan202_1

theorem rootBranch202 : search edge 4 [1259] rootCandidates202 1259 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1259] 1259 innerTail202_0 best = best
  exact innerScan202_0

end ProjectEulerSolutions.PrimePairUpper.Search
