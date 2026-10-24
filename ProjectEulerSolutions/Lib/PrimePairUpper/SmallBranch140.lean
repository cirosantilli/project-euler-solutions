import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates255 : List Nat := [1753, 2917, 3217, 3463, 3637, 3769, 4591, 5011, 5179, 5821, 6277, 6961, 7159, 7507, 7741, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates255_checked : candidates edge 5 0 1627 tail256 best.bound = rootCandidates255 := by
  unfold candidates
  have row : edge 1627 = PrimePairCache.smallRow1627.contains := by funext q; rfl
  rw [row]
  decide

def innerTail255_16 : List Nat := []
theorem innerScan255_16 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_16 best = best := rfl

def innerTail255_15 : List Nat := 8287 :: innerTail255_16
def innerCandidates255_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates255_15_checked : candidates edge 4 1627 8287 innerTail255_16 best.bound = innerCandidates255_15 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_15 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_15 best = best := by
  rw [innerTail255_15, scan, if_neg (by decide), innerCandidates255_15_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_16
    (if 3 ≤ innerCandidates255_15.length then search edge 3 [1627, 8287] innerCandidates255_15 9914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_16

def innerTail255_14 : List Nat := 7741 :: innerTail255_15
def innerCandidates255_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates255_14_checked : candidates edge 4 1627 7741 innerTail255_15 best.bound = innerCandidates255_14 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_14 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_14 best = best := by
  rw [innerTail255_14, scan, if_neg (by decide), innerCandidates255_14_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_15
    (if 3 ≤ innerCandidates255_14.length then search edge 3 [1627, 7741] innerCandidates255_14 9368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_15

def innerTail255_13 : List Nat := 7507 :: innerTail255_14
def innerCandidates255_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates255_13_checked : candidates edge 4 1627 7507 innerTail255_14 best.bound = innerCandidates255_13 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_13 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_13 best = best := by
  rw [innerTail255_13, scan, if_neg (by decide), innerCandidates255_13_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_14
    (if 3 ≤ innerCandidates255_13.length then search edge 3 [1627, 7507] innerCandidates255_13 9134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_14

def innerTail255_12 : List Nat := 7159 :: innerTail255_13
def innerCandidates255_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates255_12_checked : candidates edge 4 1627 7159 innerTail255_13 best.bound = innerCandidates255_12 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_12 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_12 best = best := by
  rw [innerTail255_12, scan, if_neg (by decide), innerCandidates255_12_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_13
    (if 3 ≤ innerCandidates255_12.length then search edge 3 [1627, 7159] innerCandidates255_12 8786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_13

def innerTail255_11 : List Nat := 6961 :: innerTail255_12
def innerCandidates255_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates255_11_checked : candidates edge 4 1627 6961 innerTail255_12 best.bound = innerCandidates255_11 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_11 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_11 best = best := by
  rw [innerTail255_11, scan, if_neg (by decide), innerCandidates255_11_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_12
    (if 3 ≤ innerCandidates255_11.length then search edge 3 [1627, 6961] innerCandidates255_11 8588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_12

def innerTail255_10 : List Nat := 6277 :: innerTail255_11
def innerCandidates255_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates255_10_checked : candidates edge 4 1627 6277 innerTail255_11 best.bound = innerCandidates255_10 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_10 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_10 best = best := by
  rw [innerTail255_10, scan, if_neg (by decide), innerCandidates255_10_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_11
    (if 3 ≤ innerCandidates255_10.length then search edge 3 [1627, 6277] innerCandidates255_10 7904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_11

def innerTail255_9 : List Nat := 5821 :: innerTail255_10
def innerCandidates255_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates255_9_checked : candidates edge 4 1627 5821 innerTail255_10 best.bound = innerCandidates255_9 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_9 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_9 best = best := by
  rw [innerTail255_9, scan, if_neg (by decide), innerCandidates255_9_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_10
    (if 3 ≤ innerCandidates255_9.length then search edge 3 [1627, 5821] innerCandidates255_9 7448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_10

def innerTail255_8 : List Nat := 5179 :: innerTail255_9
def innerCandidates255_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates255_8_checked : candidates edge 4 1627 5179 innerTail255_9 best.bound = innerCandidates255_8 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_8 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_8 best = best := by
  rw [innerTail255_8, scan, if_neg (by decide), innerCandidates255_8_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_9
    (if 3 ≤ innerCandidates255_8.length then search edge 3 [1627, 5179] innerCandidates255_8 6806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_9

def innerTail255_7 : List Nat := 5011 :: innerTail255_8
def innerCandidates255_7 : List Nat := [6961]
set_option maxRecDepth 4096 in
theorem innerCandidates255_7_checked : candidates edge 4 1627 5011 innerTail255_8 best.bound = innerCandidates255_7 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_7 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_7 best = best := by
  rw [innerTail255_7, scan, if_neg (by decide), innerCandidates255_7_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_8
    (if 3 ≤ innerCandidates255_7.length then search edge 3 [1627, 5011] innerCandidates255_7 6638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_8

def innerTail255_6 : List Nat := 4591 :: innerTail255_7
def innerCandidates255_6 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates255_6_checked : candidates edge 4 1627 4591 innerTail255_7 best.bound = innerCandidates255_6 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_6 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_6 best = best := by
  rw [innerTail255_6, scan, if_neg (by decide), innerCandidates255_6_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_7
    (if 3 ≤ innerCandidates255_6.length then search edge 3 [1627, 4591] innerCandidates255_6 6218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_7

def innerTail255_5 : List Nat := 3769 :: innerTail255_6
def innerCandidates255_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates255_5_checked : candidates edge 4 1627 3769 innerTail255_6 best.bound = innerCandidates255_5 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_5 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_5 best = best := by
  rw [innerTail255_5, scan, if_neg (by decide), innerCandidates255_5_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_6
    (if 3 ≤ innerCandidates255_5.length then search edge 3 [1627, 3769] innerCandidates255_5 5396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_6

def innerTail255_4 : List Nat := 3637 :: innerTail255_5
def innerCandidates255_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates255_4_checked : candidates edge 4 1627 3637 innerTail255_5 best.bound = innerCandidates255_4 := by
  unfold candidates
  have row : edge 3637 = SearchData.row3637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_4 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_4 best = best := by
  rw [innerTail255_4, scan, if_neg (by decide), innerCandidates255_4_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_5
    (if 3 ≤ innerCandidates255_4.length then search edge 3 [1627, 3637] innerCandidates255_4 5264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_5

def innerTail255_3 : List Nat := 3463 :: innerTail255_4
def innerCandidates255_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates255_3_checked : candidates edge 4 1627 3463 innerTail255_4 best.bound = innerCandidates255_3 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_3 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_3 best = best := by
  rw [innerTail255_3, scan, if_neg (by decide), innerCandidates255_3_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_4
    (if 3 ≤ innerCandidates255_3.length then search edge 3 [1627, 3463] innerCandidates255_3 5090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_4

def innerTail255_2 : List Nat := 3217 :: innerTail255_3
def innerCandidates255_2 : List Nat := [5821]
set_option maxRecDepth 4096 in
theorem innerCandidates255_2_checked : candidates edge 4 1627 3217 innerTail255_3 best.bound = innerCandidates255_2 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_2 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_2 best = best := by
  rw [innerTail255_2, scan, if_neg (by decide), innerCandidates255_2_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_3
    (if 3 ≤ innerCandidates255_2.length then search edge 3 [1627, 3217] innerCandidates255_2 4844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_3

def innerTail255_1 : List Nat := 2917 :: innerTail255_2
def innerCandidates255_1 : List Nat := [5011]
set_option maxRecDepth 4096 in
theorem innerCandidates255_1_checked : candidates edge 4 1627 2917 innerTail255_2 best.bound = innerCandidates255_1 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_1 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_1 best = best := by
  rw [innerTail255_1, scan, if_neg (by decide), innerCandidates255_1_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_2
    (if 3 ≤ innerCandidates255_1.length then search edge 3 [1627, 2917] innerCandidates255_1 4544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_2

def innerTail255_0 : List Nat := 1753 :: innerTail255_1
def innerCandidates255_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates255_0_checked : candidates edge 4 1627 1753 innerTail255_1 best.bound = innerCandidates255_0 := by
  unfold candidates
  have row : edge 1753 = PrimePairCache.smallRow1753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan255_0 : scan edge (search edge 3) 4 [1627] 1627 innerTail255_0 best = best := by
  rw [innerTail255_0, scan, if_neg (by decide), innerCandidates255_0_checked]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_1
    (if 3 ≤ innerCandidates255_0.length then search edge 3 [1627, 1753] innerCandidates255_0 3380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan255_1

theorem rootBranch255 : search edge 4 [1627] rootCandidates255 1627 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1627] 1627 innerTail255_0 best = best
  exact innerScan255_0

def rootCandidates254 : List Nat := [1801, 2083, 2437, 2767, 2803, 2851, 3187, 3313, 3823, 3943, 4021, 4153, 4441, 5443, 5503, 5527, 6037, 6073, 6247, 7477, 7603, 7639, 7687, 7933]
set_option maxRecDepth 4096 in
theorem rootCandidates254_checked : candidates edge 5 0 1621 tail255 best.bound = rootCandidates254 := by
  unfold candidates
  have row : edge 1621 = PrimePairCache.smallRow1621.contains := by funext q; rfl
  rw [row]
  decide

def innerTail254_24 : List Nat := []
theorem innerScan254_24 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_24 best = best := rfl

def innerTail254_23 : List Nat := 7933 :: innerTail254_24
def innerCandidates254_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_23_checked : candidates edge 4 1621 7933 innerTail254_24 best.bound = innerCandidates254_23 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_23 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_23 best = best := by
  rw [innerTail254_23, scan, if_neg (by decide), innerCandidates254_23_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_24
    (if 3 ≤ innerCandidates254_23.length then search edge 3 [1621, 7933] innerCandidates254_23 9554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_24

def innerTail254_22 : List Nat := 7687 :: innerTail254_23
def innerCandidates254_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_22_checked : candidates edge 4 1621 7687 innerTail254_23 best.bound = innerCandidates254_22 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_22 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_22 best = best := by
  rw [innerTail254_22, scan, if_neg (by decide), innerCandidates254_22_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_23
    (if 3 ≤ innerCandidates254_22.length then search edge 3 [1621, 7687] innerCandidates254_22 9308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_23

def innerTail254_21 : List Nat := 7639 :: innerTail254_22
def innerCandidates254_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_21_checked : candidates edge 4 1621 7639 innerTail254_22 best.bound = innerCandidates254_21 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_21 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_21 best = best := by
  rw [innerTail254_21, scan, if_neg (by decide), innerCandidates254_21_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_22
    (if 3 ≤ innerCandidates254_21.length then search edge 3 [1621, 7639] innerCandidates254_21 9260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_22

def innerTail254_20 : List Nat := 7603 :: innerTail254_21
def innerCandidates254_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_20_checked : candidates edge 4 1621 7603 innerTail254_21 best.bound = innerCandidates254_20 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_20 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_20 best = best := by
  rw [innerTail254_20, scan, if_neg (by decide), innerCandidates254_20_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_21
    (if 3 ≤ innerCandidates254_20.length then search edge 3 [1621, 7603] innerCandidates254_20 9224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_21

def innerTail254_19 : List Nat := 7477 :: innerTail254_20
def innerCandidates254_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_19_checked : candidates edge 4 1621 7477 innerTail254_20 best.bound = innerCandidates254_19 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_19 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_19 best = best := by
  rw [innerTail254_19, scan, if_neg (by decide), innerCandidates254_19_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_20
    (if 3 ≤ innerCandidates254_19.length then search edge 3 [1621, 7477] innerCandidates254_19 9098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_20

def innerTail254_18 : List Nat := 6247 :: innerTail254_19
def innerCandidates254_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_18_checked : candidates edge 4 1621 6247 innerTail254_19 best.bound = innerCandidates254_18 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_18 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_18 best = best := by
  rw [innerTail254_18, scan, if_neg (by decide), innerCandidates254_18_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_19
    (if 3 ≤ innerCandidates254_18.length then search edge 3 [1621, 6247] innerCandidates254_18 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_19

def innerTail254_17 : List Nat := 6073 :: innerTail254_18
def innerCandidates254_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_17_checked : candidates edge 4 1621 6073 innerTail254_18 best.bound = innerCandidates254_17 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_17 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_17 best = best := by
  rw [innerTail254_17, scan, if_neg (by decide), innerCandidates254_17_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_18
    (if 3 ≤ innerCandidates254_17.length then search edge 3 [1621, 6073] innerCandidates254_17 7694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_18

def innerTail254_16 : List Nat := 6037 :: innerTail254_17
def innerCandidates254_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_16_checked : candidates edge 4 1621 6037 innerTail254_17 best.bound = innerCandidates254_16 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_16 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_16 best = best := by
  rw [innerTail254_16, scan, if_neg (by decide), innerCandidates254_16_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_17
    (if 3 ≤ innerCandidates254_16.length then search edge 3 [1621, 6037] innerCandidates254_16 7658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_17

def innerTail254_15 : List Nat := 5527 :: innerTail254_16
def innerCandidates254_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_15_checked : candidates edge 4 1621 5527 innerTail254_16 best.bound = innerCandidates254_15 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_15 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_15 best = best := by
  rw [innerTail254_15, scan, if_neg (by decide), innerCandidates254_15_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_16
    (if 3 ≤ innerCandidates254_15.length then search edge 3 [1621, 5527] innerCandidates254_15 7148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_16

def innerTail254_14 : List Nat := 5503 :: innerTail254_15
def innerCandidates254_14 : List Nat := [7603]
set_option maxRecDepth 4096 in
theorem innerCandidates254_14_checked : candidates edge 4 1621 5503 innerTail254_15 best.bound = innerCandidates254_14 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_14 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_14 best = best := by
  rw [innerTail254_14, scan, if_neg (by decide), innerCandidates254_14_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_15
    (if 3 ≤ innerCandidates254_14.length then search edge 3 [1621, 5503] innerCandidates254_14 7124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_15

def innerTail254_13 : List Nat := 5443 :: innerTail254_14
def innerCandidates254_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_13_checked : candidates edge 4 1621 5443 innerTail254_14 best.bound = innerCandidates254_13 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_13 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_13 best = best := by
  rw [innerTail254_13, scan, if_neg (by decide), innerCandidates254_13_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_14
    (if 3 ≤ innerCandidates254_13.length then search edge 3 [1621, 5443] innerCandidates254_13 7064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_14

def innerTail254_12 : List Nat := 4441 :: innerTail254_13
def innerCandidates254_12 : List Nat := [6247]
set_option maxRecDepth 4096 in
theorem innerCandidates254_12_checked : candidates edge 4 1621 4441 innerTail254_13 best.bound = innerCandidates254_12 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_12 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_12 best = best := by
  rw [innerTail254_12, scan, if_neg (by decide), innerCandidates254_12_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_13
    (if 3 ≤ innerCandidates254_12.length then search edge 3 [1621, 4441] innerCandidates254_12 6062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_13

def innerTail254_11 : List Nat := 4153 :: innerTail254_12
def innerCandidates254_11 : List Nat := [5503]
set_option maxRecDepth 4096 in
theorem innerCandidates254_11_checked : candidates edge 4 1621 4153 innerTail254_12 best.bound = innerCandidates254_11 := by
  unfold candidates
  have row : edge 4153 = SearchData.row4153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_11 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_11 best = best := by
  rw [innerTail254_11, scan, if_neg (by decide), innerCandidates254_11_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_12
    (if 3 ≤ innerCandidates254_11.length then search edge 3 [1621, 4153] innerCandidates254_11 5774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_12

def innerTail254_10 : List Nat := 4021 :: innerTail254_11
def innerCandidates254_10 : List Nat := [4441]
set_option maxRecDepth 4096 in
theorem innerCandidates254_10_checked : candidates edge 4 1621 4021 innerTail254_11 best.bound = innerCandidates254_10 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_10 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_10 best = best := by
  rw [innerTail254_10, scan, if_neg (by decide), innerCandidates254_10_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_11
    (if 3 ≤ innerCandidates254_10.length then search edge 3 [1621, 4021] innerCandidates254_10 5642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_11

def innerTail254_9 : List Nat := 3943 :: innerTail254_10
def innerCandidates254_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_9_checked : candidates edge 4 1621 3943 innerTail254_10 best.bound = innerCandidates254_9 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_9 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_9 best = best := by
  rw [innerTail254_9, scan, if_neg (by decide), innerCandidates254_9_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_10
    (if 3 ≤ innerCandidates254_9.length then search edge 3 [1621, 3943] innerCandidates254_9 5564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_10

def innerTail254_8 : List Nat := 3823 :: innerTail254_9
def innerCandidates254_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_8_checked : candidates edge 4 1621 3823 innerTail254_9 best.bound = innerCandidates254_8 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_8 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_8 best = best := by
  rw [innerTail254_8, scan, if_neg (by decide), innerCandidates254_8_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_9
    (if 3 ≤ innerCandidates254_8.length then search edge 3 [1621, 3823] innerCandidates254_8 5444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_9

def innerTail254_7 : List Nat := 3313 :: innerTail254_8
def innerCandidates254_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_7_checked : candidates edge 4 1621 3313 innerTail254_8 best.bound = innerCandidates254_7 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_7 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_7 best = best := by
  rw [innerTail254_7, scan, if_neg (by decide), innerCandidates254_7_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_8
    (if 3 ≤ innerCandidates254_7.length then search edge 3 [1621, 3313] innerCandidates254_7 4934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_8

def innerTail254_6 : List Nat := 3187 :: innerTail254_7
def innerCandidates254_6 : List Nat := [7477]
set_option maxRecDepth 4096 in
theorem innerCandidates254_6_checked : candidates edge 4 1621 3187 innerTail254_7 best.bound = innerCandidates254_6 := by
  unfold candidates
  have row : edge 3187 = SearchData.row3187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_6 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_6 best = best := by
  rw [innerTail254_6, scan, if_neg (by decide), innerCandidates254_6_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_7
    (if 3 ≤ innerCandidates254_6.length then search edge 3 [1621, 3187] innerCandidates254_6 4808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_7

def innerTail254_5 : List Nat := 2851 :: innerTail254_6
def innerCandidates254_5 : List Nat := [3187, 3313]
set_option maxRecDepth 4096 in
theorem innerCandidates254_5_checked : candidates edge 4 1621 2851 innerTail254_6 best.bound = innerCandidates254_5 := by
  unfold candidates
  have row : edge 2851 = SearchData.row2851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_5 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_5 best = best := by
  rw [innerTail254_5, scan, if_neg (by decide), innerCandidates254_5_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_6
    (if 3 ≤ innerCandidates254_5.length then search edge 3 [1621, 2851] innerCandidates254_5 4472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_6

def innerTail254_4 : List Nat := 2803 :: innerTail254_5
def innerCandidates254_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates254_4_checked : candidates edge 4 1621 2803 innerTail254_5 best.bound = innerCandidates254_4 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_4 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_4 best = best := by
  rw [innerTail254_4, scan, if_neg (by decide), innerCandidates254_4_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_5
    (if 3 ≤ innerCandidates254_4.length then search edge 3 [1621, 2803] innerCandidates254_4 4424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_5

def innerTail254_3 : List Nat := 2767 :: innerTail254_4
def innerCandidates254_3 : List Nat := [3187, 7477, 7933]
set_option maxRecDepth 4096 in
theorem innerCandidates254_3_checked : candidates edge 4 1621 2767 innerTail254_4 best.bound = innerCandidates254_3 := by
  unfold candidates
  have row : edge 2767 = SearchData.row2767.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild254_3 : search edge 3 [1621, 2767] innerCandidates254_3 4388 best = best := by decide

theorem innerScan254_3 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_3 best = best := by
  rw [innerTail254_3, scan, if_neg (by decide), innerCandidates254_3_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_4
    (if 3 ≤ innerCandidates254_3.length then search edge 3 [1621, 2767] innerCandidates254_3 4388 best else best) = best
  rw [if_pos (by decide), innerChild254_3]
  exact innerScan254_4

def innerTail254_2 : List Nat := 2437 :: innerTail254_3
def innerCandidates254_2 : List Nat := [6073]
set_option maxRecDepth 4096 in
theorem innerCandidates254_2_checked : candidates edge 4 1621 2437 innerTail254_3 best.bound = innerCandidates254_2 := by
  unfold candidates
  have row : edge 2437 = PrimePairCache.smallRow2437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_2 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_2 best = best := by
  rw [innerTail254_2, scan, if_neg (by decide), innerCandidates254_2_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_3
    (if 3 ≤ innerCandidates254_2.length then search edge 3 [1621, 2437] innerCandidates254_2 4058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_3

def innerTail254_1 : List Nat := 2083 :: innerTail254_2
def innerCandidates254_1 : List Nat := [5443]
set_option maxRecDepth 4096 in
theorem innerCandidates254_1_checked : candidates edge 4 1621 2083 innerTail254_2 best.bound = innerCandidates254_1 := by
  unfold candidates
  have row : edge 2083 = PrimePairCache.smallRow2083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_1 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_1 best = best := by
  rw [innerTail254_1, scan, if_neg (by decide), innerCandidates254_1_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_2
    (if 3 ≤ innerCandidates254_1.length then search edge 3 [1621, 2083] innerCandidates254_1 3704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_2

def innerTail254_0 : List Nat := 1801 :: innerTail254_1
def innerCandidates254_0 : List Nat := [3823]
set_option maxRecDepth 4096 in
theorem innerCandidates254_0_checked : candidates edge 4 1621 1801 innerTail254_1 best.bound = innerCandidates254_0 := by
  unfold candidates
  have row : edge 1801 = PrimePairCache.smallRow1801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan254_0 : scan edge (search edge 3) 4 [1621] 1621 innerTail254_0 best = best := by
  rw [innerTail254_0, scan, if_neg (by decide), innerCandidates254_0_checked]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_1
    (if 3 ≤ innerCandidates254_0.length then search edge 3 [1621, 1801] innerCandidates254_0 3422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan254_1

theorem rootBranch254 : search edge 4 [1621] rootCandidates254 1621 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1621] 1621 innerTail254_0 best = best
  exact innerScan254_0

def rootCandidates253 : List Nat := [2003, 2213, 2339, 2531, 2777, 3119, 3257, 3407, 3677, 4127, 4547, 4817, 4943, 5279, 5573, 5801, 5903, 5987, 6143, 6197, 6701, 7331, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates253_checked : candidates edge 5 0 1619 tail254 best.bound = rootCandidates253 := by
  unfold candidates
  have row : edge 1619 = PrimePairCache.smallRow1619.contains := by funext q; rfl
  rw [row]
  decide

def innerTail253_23 : List Nat := []
theorem innerScan253_23 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_23 best = best := rfl

def innerTail253_22 : List Nat := 7949 :: innerTail253_23
def innerCandidates253_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_22_checked : candidates edge 4 1619 7949 innerTail253_23 best.bound = innerCandidates253_22 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_22 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_22 best = best := by
  rw [innerTail253_22, scan, if_neg (by decide), innerCandidates253_22_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_23
    (if 3 ≤ innerCandidates253_22.length then search edge 3 [1619, 7949] innerCandidates253_22 9568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_23

def innerTail253_21 : List Nat := 7331 :: innerTail253_22
def innerCandidates253_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_21_checked : candidates edge 4 1619 7331 innerTail253_22 best.bound = innerCandidates253_21 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_21 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_21 best = best := by
  rw [innerTail253_21, scan, if_neg (by decide), innerCandidates253_21_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_22
    (if 3 ≤ innerCandidates253_21.length then search edge 3 [1619, 7331] innerCandidates253_21 8950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_22

def innerTail253_20 : List Nat := 6701 :: innerTail253_21
def innerCandidates253_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_20_checked : candidates edge 4 1619 6701 innerTail253_21 best.bound = innerCandidates253_20 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_20 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_20 best = best := by
  rw [innerTail253_20, scan, if_neg (by decide), innerCandidates253_20_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_21
    (if 3 ≤ innerCandidates253_20.length then search edge 3 [1619, 6701] innerCandidates253_20 8320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_21

def innerTail253_19 : List Nat := 6197 :: innerTail253_20
def innerCandidates253_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_19_checked : candidates edge 4 1619 6197 innerTail253_20 best.bound = innerCandidates253_19 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_19 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_19 best = best := by
  rw [innerTail253_19, scan, if_neg (by decide), innerCandidates253_19_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_20
    (if 3 ≤ innerCandidates253_19.length then search edge 3 [1619, 6197] innerCandidates253_19 7816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_20

def innerTail253_18 : List Nat := 6143 :: innerTail253_19
def innerCandidates253_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_18_checked : candidates edge 4 1619 6143 innerTail253_19 best.bound = innerCandidates253_18 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_18 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_18 best = best := by
  rw [innerTail253_18, scan, if_neg (by decide), innerCandidates253_18_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_19
    (if 3 ≤ innerCandidates253_18.length then search edge 3 [1619, 6143] innerCandidates253_18 7762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_19

def innerTail253_17 : List Nat := 5987 :: innerTail253_18
def innerCandidates253_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_17_checked : candidates edge 4 1619 5987 innerTail253_18 best.bound = innerCandidates253_17 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_17 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_17 best = best := by
  rw [innerTail253_17, scan, if_neg (by decide), innerCandidates253_17_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_18
    (if 3 ≤ innerCandidates253_17.length then search edge 3 [1619, 5987] innerCandidates253_17 7606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_18

def innerTail253_16 : List Nat := 5903 :: innerTail253_17
def innerCandidates253_16 : List Nat := [6143]
set_option maxRecDepth 4096 in
theorem innerCandidates253_16_checked : candidates edge 4 1619 5903 innerTail253_17 best.bound = innerCandidates253_16 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_16 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_16 best = best := by
  rw [innerTail253_16, scan, if_neg (by decide), innerCandidates253_16_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_17
    (if 3 ≤ innerCandidates253_16.length then search edge 3 [1619, 5903] innerCandidates253_16 7522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_17

def innerTail253_15 : List Nat := 5801 :: innerTail253_16
def innerCandidates253_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_15_checked : candidates edge 4 1619 5801 innerTail253_16 best.bound = innerCandidates253_15 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_15 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_15 best = best := by
  rw [innerTail253_15, scan, if_neg (by decide), innerCandidates253_15_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_16
    (if 3 ≤ innerCandidates253_15.length then search edge 3 [1619, 5801] innerCandidates253_15 7420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_16

def innerTail253_14 : List Nat := 5573 :: innerTail253_15
def innerCandidates253_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_14_checked : candidates edge 4 1619 5573 innerTail253_15 best.bound = innerCandidates253_14 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_14 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_14 best = best := by
  rw [innerTail253_14, scan, if_neg (by decide), innerCandidates253_14_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_15
    (if 3 ≤ innerCandidates253_14.length then search edge 3 [1619, 5573] innerCandidates253_14 7192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_15

def innerTail253_13 : List Nat := 5279 :: innerTail253_14
def innerCandidates253_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_13_checked : candidates edge 4 1619 5279 innerTail253_14 best.bound = innerCandidates253_13 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_13 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_13 best = best := by
  rw [innerTail253_13, scan, if_neg (by decide), innerCandidates253_13_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_14
    (if 3 ≤ innerCandidates253_13.length then search edge 3 [1619, 5279] innerCandidates253_13 6898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_14

def innerTail253_12 : List Nat := 4943 :: innerTail253_13
def innerCandidates253_12 : List Nat := [6701]
set_option maxRecDepth 4096 in
theorem innerCandidates253_12_checked : candidates edge 4 1619 4943 innerTail253_13 best.bound = innerCandidates253_12 := by
  unfold candidates
  have row : edge 4943 = SearchData.row4943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_12 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_12 best = best := by
  rw [innerTail253_12, scan, if_neg (by decide), innerCandidates253_12_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_13
    (if 3 ≤ innerCandidates253_12.length then search edge 3 [1619, 4943] innerCandidates253_12 6562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_13

def innerTail253_11 : List Nat := 4817 :: innerTail253_12
def innerCandidates253_11 : List Nat := [5903, 6197]
set_option maxRecDepth 4096 in
theorem innerCandidates253_11_checked : candidates edge 4 1619 4817 innerTail253_12 best.bound = innerCandidates253_11 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_11 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_11 best = best := by
  rw [innerTail253_11, scan, if_neg (by decide), innerCandidates253_11_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_12
    (if 3 ≤ innerCandidates253_11.length then search edge 3 [1619, 4817] innerCandidates253_11 6436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_12

def innerTail253_10 : List Nat := 4547 :: innerTail253_11
def innerCandidates253_10 : List Nat := [4943]
set_option maxRecDepth 4096 in
theorem innerCandidates253_10_checked : candidates edge 4 1619 4547 innerTail253_11 best.bound = innerCandidates253_10 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_10 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_10 best = best := by
  rw [innerTail253_10, scan, if_neg (by decide), innerCandidates253_10_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_11
    (if 3 ≤ innerCandidates253_10.length then search edge 3 [1619, 4547] innerCandidates253_10 6166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_11

def innerTail253_9 : List Nat := 4127 :: innerTail253_10
def innerCandidates253_9 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates253_9_checked : candidates edge 4 1619 4127 innerTail253_10 best.bound = innerCandidates253_9 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_9 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_9 best = best := by
  rw [innerTail253_9, scan, if_neg (by decide), innerCandidates253_9_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_10
    (if 3 ≤ innerCandidates253_9.length then search edge 3 [1619, 4127] innerCandidates253_9 5746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_10

def innerTail253_8 : List Nat := 3677 :: innerTail253_9
def innerCandidates253_8 : List Nat := [5903]
set_option maxRecDepth 4096 in
theorem innerCandidates253_8_checked : candidates edge 4 1619 3677 innerTail253_9 best.bound = innerCandidates253_8 := by
  unfold candidates
  have row : edge 3677 = SearchData.row3677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_8 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_8 best = best := by
  rw [innerTail253_8, scan, if_neg (by decide), innerCandidates253_8_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_9
    (if 3 ≤ innerCandidates253_8.length then search edge 3 [1619, 3677] innerCandidates253_8 5296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_9

def innerTail253_7 : List Nat := 3407 :: innerTail253_8
def innerCandidates253_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_7_checked : candidates edge 4 1619 3407 innerTail253_8 best.bound = innerCandidates253_7 := by
  unfold candidates
  have row : edge 3407 = SearchData.row3407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_7 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_7 best = best := by
  rw [innerTail253_7, scan, if_neg (by decide), innerCandidates253_7_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_8
    (if 3 ≤ innerCandidates253_7.length then search edge 3 [1619, 3407] innerCandidates253_7 5026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_8

def innerTail253_6 : List Nat := 3257 :: innerTail253_7
def innerCandidates253_6 : List Nat := [3407, 4547, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates253_6_checked : candidates edge 4 1619 3257 innerTail253_7 best.bound = innerCandidates253_6 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild253_6 : search edge 3 [1619, 3257] innerCandidates253_6 4876 best = best := by decide

theorem innerScan253_6 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_6 best = best := by
  rw [innerTail253_6, scan, if_neg (by decide), innerCandidates253_6_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_7
    (if 3 ≤ innerCandidates253_6.length then search edge 3 [1619, 3257] innerCandidates253_6 4876 best else best) = best
  rw [if_pos (by decide), innerChild253_6]
  exact innerScan253_7

def innerTail253_5 : List Nat := 3119 :: innerTail253_6
def innerCandidates253_5 : List Nat := [3677]
set_option maxRecDepth 4096 in
theorem innerCandidates253_5_checked : candidates edge 4 1619 3119 innerTail253_6 best.bound = innerCandidates253_5 := by
  unfold candidates
  have row : edge 3119 = SearchData.row3119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_5 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_5 best = best := by
  rw [innerTail253_5, scan, if_neg (by decide), innerCandidates253_5_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_6
    (if 3 ≤ innerCandidates253_5.length then search edge 3 [1619, 3119] innerCandidates253_5 4738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_6

def innerTail253_4 : List Nat := 2777 :: innerTail253_5
def innerCandidates253_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_4_checked : candidates edge 4 1619 2777 innerTail253_5 best.bound = innerCandidates253_4 := by
  unfold candidates
  have row : edge 2777 = SearchData.row2777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_4 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_4 best = best := by
  rw [innerTail253_4, scan, if_neg (by decide), innerCandidates253_4_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_5
    (if 3 ≤ innerCandidates253_4.length then search edge 3 [1619, 2777] innerCandidates253_4 4396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_5

def innerTail253_3 : List Nat := 2531 :: innerTail253_4
def innerCandidates253_3 : List Nat := [6701]
set_option maxRecDepth 4096 in
theorem innerCandidates253_3_checked : candidates edge 4 1619 2531 innerTail253_4 best.bound = innerCandidates253_3 := by
  unfold candidates
  have row : edge 2531 = PrimePairCache.smallRow2531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_3 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_3 best = best := by
  rw [innerTail253_3, scan, if_neg (by decide), innerCandidates253_3_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_4
    (if 3 ≤ innerCandidates253_3.length then search edge 3 [1619, 2531] innerCandidates253_3 4150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_4

def innerTail253_2 : List Nat := 2339 :: innerTail253_3
def innerCandidates253_2 : List Nat := [4127]
set_option maxRecDepth 4096 in
theorem innerCandidates253_2_checked : candidates edge 4 1619 2339 innerTail253_3 best.bound = innerCandidates253_2 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_2 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_2 best = best := by
  rw [innerTail253_2, scan, if_neg (by decide), innerCandidates253_2_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_3
    (if 3 ≤ innerCandidates253_2.length then search edge 3 [1619, 2339] innerCandidates253_2 3958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_3

def innerTail253_1 : List Nat := 2213 :: innerTail253_2
def innerCandidates253_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_1_checked : candidates edge 4 1619 2213 innerTail253_2 best.bound = innerCandidates253_1 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_1 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_1 best = best := by
  rw [innerTail253_1, scan, if_neg (by decide), innerCandidates253_1_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_2
    (if 3 ≤ innerCandidates253_1.length then search edge 3 [1619, 2213] innerCandidates253_1 3832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_2

def innerTail253_0 : List Nat := 2003 :: innerTail253_1
def innerCandidates253_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates253_0_checked : candidates edge 4 1619 2003 innerTail253_1 best.bound = innerCandidates253_0 := by
  unfold candidates
  have row : edge 2003 = PrimePairCache.smallRow2003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan253_0 : scan edge (search edge 3) 4 [1619] 1619 innerTail253_0 best = best := by
  rw [innerTail253_0, scan, if_neg (by decide), innerCandidates253_0_checked]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_1
    (if 3 ≤ innerCandidates253_0.length then search edge 3 [1619, 2003] innerCandidates253_0 3622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan253_1

theorem rootBranch253 : search edge 4 [1619] rootCandidates253 1619 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1619] 1619 innerTail253_0 best = best
  exact innerScan253_0

end ProjectEulerSolutions.PrimePairUpper.Search
