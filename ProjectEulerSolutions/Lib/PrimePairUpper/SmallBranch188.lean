import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates129 : List Nat := [881, 947, 1097, 1481, 1493, 1871, 1889, 2099, 2273, 2309, 2753, 3359, 3527, 4001, 4283, 5081, 5591, 6113, 6323, 8117]
set_option maxRecDepth 4096 in
theorem rootCandidates129_checked : candidates edge 5 0 743 tail130 best.bound = rootCandidates129 := by
  unfold candidates
  have row : edge 743 = PrimePairCache.smallRow743.contains := by funext q; rfl
  rw [row]
  decide

def innerTail129_20 : List Nat := []
theorem innerScan129_20 : scan edge (search edge 3) 4 [743] 743 innerTail129_20 best = best := rfl

def innerTail129_19 : List Nat := 8117 :: innerTail129_20
def innerCandidates129_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_19_checked : candidates edge 4 743 8117 innerTail129_20 best.bound = innerCandidates129_19 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_19 : scan edge (search edge 3) 4 [743] 743 innerTail129_19 best = best := by
  rw [innerTail129_19, scan, if_neg (by decide), innerCandidates129_19_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_20
    (if 3 ≤ innerCandidates129_19.length then search edge 3 [743, 8117] innerCandidates129_19 8860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_20

def innerTail129_18 : List Nat := 6323 :: innerTail129_19
def innerCandidates129_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_18_checked : candidates edge 4 743 6323 innerTail129_19 best.bound = innerCandidates129_18 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_18 : scan edge (search edge 3) 4 [743] 743 innerTail129_18 best = best := by
  rw [innerTail129_18, scan, if_neg (by decide), innerCandidates129_18_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_19
    (if 3 ≤ innerCandidates129_18.length then search edge 3 [743, 6323] innerCandidates129_18 7066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_19

def innerTail129_17 : List Nat := 6113 :: innerTail129_18
def innerCandidates129_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_17_checked : candidates edge 4 743 6113 innerTail129_18 best.bound = innerCandidates129_17 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_17 : scan edge (search edge 3) 4 [743] 743 innerTail129_17 best = best := by
  rw [innerTail129_17, scan, if_neg (by decide), innerCandidates129_17_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_18
    (if 3 ≤ innerCandidates129_17.length then search edge 3 [743, 6113] innerCandidates129_17 6856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_18

def innerTail129_16 : List Nat := 5591 :: innerTail129_17
def innerCandidates129_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_16_checked : candidates edge 4 743 5591 innerTail129_17 best.bound = innerCandidates129_16 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_16 : scan edge (search edge 3) 4 [743] 743 innerTail129_16 best = best := by
  rw [innerTail129_16, scan, if_neg (by decide), innerCandidates129_16_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_17
    (if 3 ≤ innerCandidates129_16.length then search edge 3 [743, 5591] innerCandidates129_16 6334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_17

def innerTail129_15 : List Nat := 5081 :: innerTail129_16
def innerCandidates129_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_15_checked : candidates edge 4 743 5081 innerTail129_16 best.bound = innerCandidates129_15 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_15 : scan edge (search edge 3) 4 [743] 743 innerTail129_15 best = best := by
  rw [innerTail129_15, scan, if_neg (by decide), innerCandidates129_15_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_16
    (if 3 ≤ innerCandidates129_15.length then search edge 3 [743, 5081] innerCandidates129_15 5824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_16

def innerTail129_14 : List Nat := 4283 :: innerTail129_15
def innerCandidates129_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_14_checked : candidates edge 4 743 4283 innerTail129_15 best.bound = innerCandidates129_14 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_14 : scan edge (search edge 3) 4 [743] 743 innerTail129_14 best = best := by
  rw [innerTail129_14, scan, if_neg (by decide), innerCandidates129_14_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_15
    (if 3 ≤ innerCandidates129_14.length then search edge 3 [743, 4283] innerCandidates129_14 5026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_15

def innerTail129_13 : List Nat := 4001 :: innerTail129_14
def innerCandidates129_13 : List Nat := [6323]
set_option maxRecDepth 4096 in
theorem innerCandidates129_13_checked : candidates edge 4 743 4001 innerTail129_14 best.bound = innerCandidates129_13 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_13 : scan edge (search edge 3) 4 [743] 743 innerTail129_13 best = best := by
  rw [innerTail129_13, scan, if_neg (by decide), innerCandidates129_13_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_14
    (if 3 ≤ innerCandidates129_13.length then search edge 3 [743, 4001] innerCandidates129_13 4744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_14

def innerTail129_12 : List Nat := 3527 :: innerTail129_13
def innerCandidates129_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_12_checked : candidates edge 4 743 3527 innerTail129_13 best.bound = innerCandidates129_12 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_12 : scan edge (search edge 3) 4 [743] 743 innerTail129_12 best = best := by
  rw [innerTail129_12, scan, if_neg (by decide), innerCandidates129_12_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_13
    (if 3 ≤ innerCandidates129_12.length then search edge 3 [743, 3527] innerCandidates129_12 4270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_13

def innerTail129_11 : List Nat := 3359 :: innerTail129_12
def innerCandidates129_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_11_checked : candidates edge 4 743 3359 innerTail129_12 best.bound = innerCandidates129_11 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_11 : scan edge (search edge 3) 4 [743] 743 innerTail129_11 best = best := by
  rw [innerTail129_11, scan, if_neg (by decide), innerCandidates129_11_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_12
    (if 3 ≤ innerCandidates129_11.length then search edge 3 [743, 3359] innerCandidates129_11 4102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_12

def innerTail129_10 : List Nat := 2753 :: innerTail129_11
def innerCandidates129_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_10_checked : candidates edge 4 743 2753 innerTail129_11 best.bound = innerCandidates129_10 := by
  unfold candidates
  have row : edge 2753 = SearchData.row2753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_10 : scan edge (search edge 3) 4 [743] 743 innerTail129_10 best = best := by
  rw [innerTail129_10, scan, if_neg (by decide), innerCandidates129_10_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_11
    (if 3 ≤ innerCandidates129_10.length then search edge 3 [743, 2753] innerCandidates129_10 3496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_11

def innerTail129_9 : List Nat := 2309 :: innerTail129_10
def innerCandidates129_9 : List Nat := [5081]
set_option maxRecDepth 4096 in
theorem innerCandidates129_9_checked : candidates edge 4 743 2309 innerTail129_10 best.bound = innerCandidates129_9 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_9 : scan edge (search edge 3) 4 [743] 743 innerTail129_9 best = best := by
  rw [innerTail129_9, scan, if_neg (by decide), innerCandidates129_9_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_10
    (if 3 ≤ innerCandidates129_9.length then search edge 3 [743, 2309] innerCandidates129_9 3052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_10

def innerTail129_8 : List Nat := 2273 :: innerTail129_9
def innerCandidates129_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_8_checked : candidates edge 4 743 2273 innerTail129_9 best.bound = innerCandidates129_8 := by
  unfold candidates
  have row : edge 2273 = PrimePairCache.smallRow2273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_8 : scan edge (search edge 3) 4 [743] 743 innerTail129_8 best = best := by
  rw [innerTail129_8, scan, if_neg (by decide), innerCandidates129_8_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_9
    (if 3 ≤ innerCandidates129_8.length then search edge 3 [743, 2273] innerCandidates129_8 3016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_9

def innerTail129_7 : List Nat := 2099 :: innerTail129_8
def innerCandidates129_7 : List Nat := [3527]
set_option maxRecDepth 4096 in
theorem innerCandidates129_7_checked : candidates edge 4 743 2099 innerTail129_8 best.bound = innerCandidates129_7 := by
  unfold candidates
  have row : edge 2099 = PrimePairCache.smallRow2099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_7 : scan edge (search edge 3) 4 [743] 743 innerTail129_7 best = best := by
  rw [innerTail129_7, scan, if_neg (by decide), innerCandidates129_7_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_8
    (if 3 ≤ innerCandidates129_7.length then search edge 3 [743, 2099] innerCandidates129_7 2842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_8

def innerTail129_6 : List Nat := 1889 :: innerTail129_7
def innerCandidates129_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_6_checked : candidates edge 4 743 1889 innerTail129_7 best.bound = innerCandidates129_6 := by
  unfold candidates
  have row : edge 1889 = PrimePairCache.smallRow1889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_6 : scan edge (search edge 3) 4 [743] 743 innerTail129_6 best = best := by
  rw [innerTail129_6, scan, if_neg (by decide), innerCandidates129_6_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_7
    (if 3 ≤ innerCandidates129_6.length then search edge 3 [743, 1889] innerCandidates129_6 2632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_7

def innerTail129_5 : List Nat := 1871 :: innerTail129_6
def innerCandidates129_5 : List Nat := [6323]
set_option maxRecDepth 4096 in
theorem innerCandidates129_5_checked : candidates edge 4 743 1871 innerTail129_6 best.bound = innerCandidates129_5 := by
  unfold candidates
  have row : edge 1871 = PrimePairCache.smallRow1871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_5 : scan edge (search edge 3) 4 [743] 743 innerTail129_5 best = best := by
  rw [innerTail129_5, scan, if_neg (by decide), innerCandidates129_5_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_6
    (if 3 ≤ innerCandidates129_5.length then search edge 3 [743, 1871] innerCandidates129_5 2614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_6

def innerTail129_4 : List Nat := 1493 :: innerTail129_5
def innerCandidates129_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_4_checked : candidates edge 4 743 1493 innerTail129_5 best.bound = innerCandidates129_4 := by
  unfold candidates
  have row : edge 1493 = PrimePairCache.smallRow1493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_4 : scan edge (search edge 3) 4 [743] 743 innerTail129_4 best = best := by
  rw [innerTail129_4, scan, if_neg (by decide), innerCandidates129_4_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_5
    (if 3 ≤ innerCandidates129_4.length then search edge 3 [743, 1493] innerCandidates129_4 2236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_5

def innerTail129_3 : List Nat := 1481 :: innerTail129_4
def innerCandidates129_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates129_3_checked : candidates edge 4 743 1481 innerTail129_4 best.bound = innerCandidates129_3 := by
  unfold candidates
  have row : edge 1481 = PrimePairCache.smallRow1481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_3 : scan edge (search edge 3) 4 [743] 743 innerTail129_3 best = best := by
  rw [innerTail129_3, scan, if_neg (by decide), innerCandidates129_3_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_4
    (if 3 ≤ innerCandidates129_3.length then search edge 3 [743, 1481] innerCandidates129_3 2224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_4

def innerTail129_2 : List Nat := 1097 :: innerTail129_3
def innerCandidates129_2 : List Nat := [4283, 6113, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates129_2_checked : candidates edge 4 743 1097 innerTail129_3 best.bound = innerCandidates129_2 := by
  unfold candidates
  have row : edge 1097 = PrimePairCache.smallRow1097.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild129_2 : search edge 3 [743, 1097] innerCandidates129_2 1840 best = best := by decide

theorem innerScan129_2 : scan edge (search edge 3) 4 [743] 743 innerTail129_2 best = best := by
  rw [innerTail129_2, scan, if_neg (by decide), innerCandidates129_2_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_3
    (if 3 ≤ innerCandidates129_2.length then search edge 3 [743, 1097] innerCandidates129_2 1840 best else best) = best
  rw [if_pos (by decide), innerChild129_2]
  exact innerScan129_3

def innerTail129_1 : List Nat := 947 :: innerTail129_2
def innerCandidates129_1 : List Nat := [1097]
set_option maxRecDepth 4096 in
theorem innerCandidates129_1_checked : candidates edge 4 743 947 innerTail129_2 best.bound = innerCandidates129_1 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_1 : scan edge (search edge 3) 4 [743] 743 innerTail129_1 best = best := by
  rw [innerTail129_1, scan, if_neg (by decide), innerCandidates129_1_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_2
    (if 3 ≤ innerCandidates129_1.length then search edge 3 [743, 947] innerCandidates129_1 1690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_2

def innerTail129_0 : List Nat := 881 :: innerTail129_1
def innerCandidates129_0 : List Nat := [5591]
set_option maxRecDepth 4096 in
theorem innerCandidates129_0_checked : candidates edge 4 743 881 innerTail129_1 best.bound = innerCandidates129_0 := by
  unfold candidates
  have row : edge 881 = PrimePairCache.smallRow881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan129_0 : scan edge (search edge 3) 4 [743] 743 innerTail129_0 best = best := by
  rw [innerTail129_0, scan, if_neg (by decide), innerCandidates129_0_checked]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_1
    (if 3 ≤ innerCandidates129_0.length then search edge 3 [743, 881] innerCandidates129_0 1624 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan129_1

theorem rootBranch129 : search edge 4 [743] rootCandidates129 743 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [743] 743 innerTail129_0 best = best
  exact innerScan129_0

def rootCandidates128 : List Nat := [751, 853, 967, 1093, 1123, 1381, 3121, 3187, 3613, 3733, 4357, 4549, 4759, 4909, 6007, 6673, 6793, 6841, 7219, 7393, 7699]
set_option maxRecDepth 4096 in
theorem rootCandidates128_checked : candidates edge 5 0 739 tail129 best.bound = rootCandidates128 := by
  unfold candidates
  have row : edge 739 = PrimePairCache.smallRow739.contains := by funext q; rfl
  rw [row]
  decide

def innerTail128_21 : List Nat := []
theorem innerScan128_21 : scan edge (search edge 3) 4 [739] 739 innerTail128_21 best = best := rfl

def innerTail128_20 : List Nat := 7699 :: innerTail128_21
def innerCandidates128_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_20_checked : candidates edge 4 739 7699 innerTail128_21 best.bound = innerCandidates128_20 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_20 : scan edge (search edge 3) 4 [739] 739 innerTail128_20 best = best := by
  rw [innerTail128_20, scan, if_neg (by decide), innerCandidates128_20_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_21
    (if 3 ≤ innerCandidates128_20.length then search edge 3 [739, 7699] innerCandidates128_20 8438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_21

def innerTail128_19 : List Nat := 7393 :: innerTail128_20
def innerCandidates128_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_19_checked : candidates edge 4 739 7393 innerTail128_20 best.bound = innerCandidates128_19 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_19 : scan edge (search edge 3) 4 [739] 739 innerTail128_19 best = best := by
  rw [innerTail128_19, scan, if_neg (by decide), innerCandidates128_19_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_20
    (if 3 ≤ innerCandidates128_19.length then search edge 3 [739, 7393] innerCandidates128_19 8132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_20

def innerTail128_18 : List Nat := 7219 :: innerTail128_19
def innerCandidates128_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_18_checked : candidates edge 4 739 7219 innerTail128_19 best.bound = innerCandidates128_18 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_18 : scan edge (search edge 3) 4 [739] 739 innerTail128_18 best = best := by
  rw [innerTail128_18, scan, if_neg (by decide), innerCandidates128_18_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_19
    (if 3 ≤ innerCandidates128_18.length then search edge 3 [739, 7219] innerCandidates128_18 7958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_19

def innerTail128_17 : List Nat := 6841 :: innerTail128_18
def innerCandidates128_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_17_checked : candidates edge 4 739 6841 innerTail128_18 best.bound = innerCandidates128_17 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_17 : scan edge (search edge 3) 4 [739] 739 innerTail128_17 best = best := by
  rw [innerTail128_17, scan, if_neg (by decide), innerCandidates128_17_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_18
    (if 3 ≤ innerCandidates128_17.length then search edge 3 [739, 6841] innerCandidates128_17 7580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_18

def innerTail128_16 : List Nat := 6793 :: innerTail128_17
def innerCandidates128_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_16_checked : candidates edge 4 739 6793 innerTail128_17 best.bound = innerCandidates128_16 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_16 : scan edge (search edge 3) 4 [739] 739 innerTail128_16 best = best := by
  rw [innerTail128_16, scan, if_neg (by decide), innerCandidates128_16_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_17
    (if 3 ≤ innerCandidates128_16.length then search edge 3 [739, 6793] innerCandidates128_16 7532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_17

def innerTail128_15 : List Nat := 6673 :: innerTail128_16
def innerCandidates128_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_15_checked : candidates edge 4 739 6673 innerTail128_16 best.bound = innerCandidates128_15 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_15 : scan edge (search edge 3) 4 [739] 739 innerTail128_15 best = best := by
  rw [innerTail128_15, scan, if_neg (by decide), innerCandidates128_15_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_16
    (if 3 ≤ innerCandidates128_15.length then search edge 3 [739, 6673] innerCandidates128_15 7412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_16

def innerTail128_14 : List Nat := 6007 :: innerTail128_15
def innerCandidates128_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_14_checked : candidates edge 4 739 6007 innerTail128_15 best.bound = innerCandidates128_14 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_14 : scan edge (search edge 3) 4 [739] 739 innerTail128_14 best = best := by
  rw [innerTail128_14, scan, if_neg (by decide), innerCandidates128_14_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_15
    (if 3 ≤ innerCandidates128_14.length then search edge 3 [739, 6007] innerCandidates128_14 6746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_15

def innerTail128_13 : List Nat := 4909 :: innerTail128_14
def innerCandidates128_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_13_checked : candidates edge 4 739 4909 innerTail128_14 best.bound = innerCandidates128_13 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_13 : scan edge (search edge 3) 4 [739] 739 innerTail128_13 best = best := by
  rw [innerTail128_13, scan, if_neg (by decide), innerCandidates128_13_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_14
    (if 3 ≤ innerCandidates128_13.length then search edge 3 [739, 4909] innerCandidates128_13 5648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_14

def innerTail128_12 : List Nat := 4759 :: innerTail128_13
def innerCandidates128_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_12_checked : candidates edge 4 739 4759 innerTail128_13 best.bound = innerCandidates128_12 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_12 : scan edge (search edge 3) 4 [739] 739 innerTail128_12 best = best := by
  rw [innerTail128_12, scan, if_neg (by decide), innerCandidates128_12_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_13
    (if 3 ≤ innerCandidates128_12.length then search edge 3 [739, 4759] innerCandidates128_12 5498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_13

def innerTail128_11 : List Nat := 4549 :: innerTail128_12
def innerCandidates128_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_11_checked : candidates edge 4 739 4549 innerTail128_12 best.bound = innerCandidates128_11 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_11 : scan edge (search edge 3) 4 [739] 739 innerTail128_11 best = best := by
  rw [innerTail128_11, scan, if_neg (by decide), innerCandidates128_11_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_12
    (if 3 ≤ innerCandidates128_11.length then search edge 3 [739, 4549] innerCandidates128_11 5288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_12

def innerTail128_10 : List Nat := 4357 :: innerTail128_11
def innerCandidates128_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_10_checked : candidates edge 4 739 4357 innerTail128_11 best.bound = innerCandidates128_10 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_10 : scan edge (search edge 3) 4 [739] 739 innerTail128_10 best = best := by
  rw [innerTail128_10, scan, if_neg (by decide), innerCandidates128_10_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_11
    (if 3 ≤ innerCandidates128_10.length then search edge 3 [739, 4357] innerCandidates128_10 5096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_11

def innerTail128_9 : List Nat := 3733 :: innerTail128_10
def innerCandidates128_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_9_checked : candidates edge 4 739 3733 innerTail128_10 best.bound = innerCandidates128_9 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_9 : scan edge (search edge 3) 4 [739] 739 innerTail128_9 best = best := by
  rw [innerTail128_9, scan, if_neg (by decide), innerCandidates128_9_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_10
    (if 3 ≤ innerCandidates128_9.length then search edge 3 [739, 3733] innerCandidates128_9 4472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_10

def innerTail128_8 : List Nat := 3613 :: innerTail128_9
def innerCandidates128_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_8_checked : candidates edge 4 739 3613 innerTail128_9 best.bound = innerCandidates128_8 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_8 : scan edge (search edge 3) 4 [739] 739 innerTail128_8 best = best := by
  rw [innerTail128_8, scan, if_neg (by decide), innerCandidates128_8_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_9
    (if 3 ≤ innerCandidates128_8.length then search edge 3 [739, 3613] innerCandidates128_8 4352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_9

def innerTail128_7 : List Nat := 3187 :: innerTail128_8
def innerCandidates128_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_7_checked : candidates edge 4 739 3187 innerTail128_8 best.bound = innerCandidates128_7 := by
  unfold candidates
  have row : edge 3187 = SearchData.row3187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_7 : scan edge (search edge 3) 4 [739] 739 innerTail128_7 best = best := by
  rw [innerTail128_7, scan, if_neg (by decide), innerCandidates128_7_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_8
    (if 3 ≤ innerCandidates128_7.length then search edge 3 [739, 3187] innerCandidates128_7 3926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_8

def innerTail128_6 : List Nat := 3121 :: innerTail128_7
def innerCandidates128_6 : List Nat := [4357, 7393]
set_option maxRecDepth 4096 in
theorem innerCandidates128_6_checked : candidates edge 4 739 3121 innerTail128_7 best.bound = innerCandidates128_6 := by
  unfold candidates
  have row : edge 3121 = SearchData.row3121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_6 : scan edge (search edge 3) 4 [739] 739 innerTail128_6 best = best := by
  rw [innerTail128_6, scan, if_neg (by decide), innerCandidates128_6_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_7
    (if 3 ≤ innerCandidates128_6.length then search edge 3 [739, 3121] innerCandidates128_6 3860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_7

def innerTail128_5 : List Nat := 1381 :: innerTail128_6
def innerCandidates128_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_5_checked : candidates edge 4 739 1381 innerTail128_6 best.bound = innerCandidates128_5 := by
  unfold candidates
  have row : edge 1381 = PrimePairCache.smallRow1381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_5 : scan edge (search edge 3) 4 [739] 739 innerTail128_5 best = best := by
  rw [innerTail128_5, scan, if_neg (by decide), innerCandidates128_5_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_6
    (if 3 ≤ innerCandidates128_5.length then search edge 3 [739, 1381] innerCandidates128_5 2120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_6

def innerTail128_4 : List Nat := 1123 :: innerTail128_5
def innerCandidates128_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_4_checked : candidates edge 4 739 1123 innerTail128_5 best.bound = innerCandidates128_4 := by
  unfold candidates
  have row : edge 1123 = PrimePairCache.smallRow1123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_4 : scan edge (search edge 3) 4 [739] 739 innerTail128_4 best = best := by
  rw [innerTail128_4, scan, if_neg (by decide), innerCandidates128_4_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_5
    (if 3 ≤ innerCandidates128_4.length then search edge 3 [739, 1123] innerCandidates128_4 1862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_5

def innerTail128_3 : List Nat := 1093 :: innerTail128_4
def innerCandidates128_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_3_checked : candidates edge 4 739 1093 innerTail128_4 best.bound = innerCandidates128_3 := by
  unfold candidates
  have row : edge 1093 = PrimePairCache.smallRow1093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_3 : scan edge (search edge 3) 4 [739] 739 innerTail128_3 best = best := by
  rw [innerTail128_3, scan, if_neg (by decide), innerCandidates128_3_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_4
    (if 3 ≤ innerCandidates128_3.length then search edge 3 [739, 1093] innerCandidates128_3 1832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_4

def innerTail128_2 : List Nat := 967 :: innerTail128_3
def innerCandidates128_2 : List Nat := [1381]
set_option maxRecDepth 4096 in
theorem innerCandidates128_2_checked : candidates edge 4 739 967 innerTail128_3 best.bound = innerCandidates128_2 := by
  unfold candidates
  have row : edge 967 = PrimePairCache.smallRow967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_2 : scan edge (search edge 3) 4 [739] 739 innerTail128_2 best = best := by
  rw [innerTail128_2, scan, if_neg (by decide), innerCandidates128_2_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_3
    (if 3 ≤ innerCandidates128_2.length then search edge 3 [739, 967] innerCandidates128_2 1706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_3

def innerTail128_1 : List Nat := 853 :: innerTail128_2
def innerCandidates128_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_1_checked : candidates edge 4 739 853 innerTail128_2 best.bound = innerCandidates128_1 := by
  unfold candidates
  have row : edge 853 = PrimePairCache.smallRow853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_1 : scan edge (search edge 3) 4 [739] 739 innerTail128_1 best = best := by
  rw [innerTail128_1, scan, if_neg (by decide), innerCandidates128_1_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_2
    (if 3 ≤ innerCandidates128_1.length then search edge 3 [739, 853] innerCandidates128_1 1592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_2

def innerTail128_0 : List Nat := 751 :: innerTail128_1
def innerCandidates128_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates128_0_checked : candidates edge 4 739 751 innerTail128_1 best.bound = innerCandidates128_0 := by
  unfold candidates
  have row : edge 751 = PrimePairCache.smallRow751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan128_0 : scan edge (search edge 3) 4 [739] 739 innerTail128_0 best = best := by
  rw [innerTail128_0, scan, if_neg (by decide), innerCandidates128_0_checked]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_1
    (if 3 ≤ innerCandidates128_0.length then search edge 3 [739, 751] innerCandidates128_0 1490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan128_1

theorem rootBranch128 : search edge 4 [739] rootCandidates128 739 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [739] 739 innerTail128_0 best = best
  exact innerScan128_0

def rootCandidates127 : List Nat := [829, 883, 991, 1039, 1249, 1999, 2113, 2551, 3331, 3709, 4027, 4357, 4441, 4549, 4903, 5701, 5743, 5851, 7717, 7723, 7879, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates127_checked : candidates edge 5 0 733 tail128 best.bound = rootCandidates127 := by
  unfold candidates
  have row : edge 733 = PrimePairCache.smallRow733.contains := by funext q; rfl
  rw [row]
  decide

def innerTail127_22 : List Nat := []
theorem innerScan127_22 : scan edge (search edge 3) 4 [733] 733 innerTail127_22 best = best := rfl

def innerTail127_21 : List Nat := 8263 :: innerTail127_22
def innerCandidates127_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_21_checked : candidates edge 4 733 8263 innerTail127_22 best.bound = innerCandidates127_21 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_21 : scan edge (search edge 3) 4 [733] 733 innerTail127_21 best = best := by
  rw [innerTail127_21, scan, if_neg (by decide), innerCandidates127_21_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_22
    (if 3 ≤ innerCandidates127_21.length then search edge 3 [733, 8263] innerCandidates127_21 8996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_22

def innerTail127_20 : List Nat := 7879 :: innerTail127_21
def innerCandidates127_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_20_checked : candidates edge 4 733 7879 innerTail127_21 best.bound = innerCandidates127_20 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_20 : scan edge (search edge 3) 4 [733] 733 innerTail127_20 best = best := by
  rw [innerTail127_20, scan, if_neg (by decide), innerCandidates127_20_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_21
    (if 3 ≤ innerCandidates127_20.length then search edge 3 [733, 7879] innerCandidates127_20 8612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_21

def innerTail127_19 : List Nat := 7723 :: innerTail127_20
def innerCandidates127_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_19_checked : candidates edge 4 733 7723 innerTail127_20 best.bound = innerCandidates127_19 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_19 : scan edge (search edge 3) 4 [733] 733 innerTail127_19 best = best := by
  rw [innerTail127_19, scan, if_neg (by decide), innerCandidates127_19_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_20
    (if 3 ≤ innerCandidates127_19.length then search edge 3 [733, 7723] innerCandidates127_19 8456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_20

def innerTail127_18 : List Nat := 7717 :: innerTail127_19
def innerCandidates127_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_18_checked : candidates edge 4 733 7717 innerTail127_19 best.bound = innerCandidates127_18 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_18 : scan edge (search edge 3) 4 [733] 733 innerTail127_18 best = best := by
  rw [innerTail127_18, scan, if_neg (by decide), innerCandidates127_18_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_19
    (if 3 ≤ innerCandidates127_18.length then search edge 3 [733, 7717] innerCandidates127_18 8450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_19

def innerTail127_17 : List Nat := 5851 :: innerTail127_18
def innerCandidates127_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_17_checked : candidates edge 4 733 5851 innerTail127_18 best.bound = innerCandidates127_17 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_17 : scan edge (search edge 3) 4 [733] 733 innerTail127_17 best = best := by
  rw [innerTail127_17, scan, if_neg (by decide), innerCandidates127_17_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_18
    (if 3 ≤ innerCandidates127_17.length then search edge 3 [733, 5851] innerCandidates127_17 6584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_18

def innerTail127_16 : List Nat := 5743 :: innerTail127_17
def innerCandidates127_16 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates127_16_checked : candidates edge 4 733 5743 innerTail127_17 best.bound = innerCandidates127_16 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_16 : scan edge (search edge 3) 4 [733] 733 innerTail127_16 best = best := by
  rw [innerTail127_16, scan, if_neg (by decide), innerCandidates127_16_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_17
    (if 3 ≤ innerCandidates127_16.length then search edge 3 [733, 5743] innerCandidates127_16 6476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_17

def innerTail127_15 : List Nat := 5701 :: innerTail127_16
def innerCandidates127_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_15_checked : candidates edge 4 733 5701 innerTail127_16 best.bound = innerCandidates127_15 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_15 : scan edge (search edge 3) 4 [733] 733 innerTail127_15 best = best := by
  rw [innerTail127_15, scan, if_neg (by decide), innerCandidates127_15_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_16
    (if 3 ≤ innerCandidates127_15.length then search edge 3 [733, 5701] innerCandidates127_15 6434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_16

def innerTail127_14 : List Nat := 4903 :: innerTail127_15
def innerCandidates127_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_14_checked : candidates edge 4 733 4903 innerTail127_15 best.bound = innerCandidates127_14 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_14 : scan edge (search edge 3) 4 [733] 733 innerTail127_14 best = best := by
  rw [innerTail127_14, scan, if_neg (by decide), innerCandidates127_14_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_15
    (if 3 ≤ innerCandidates127_14.length then search edge 3 [733, 4903] innerCandidates127_14 5636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_15

def innerTail127_13 : List Nat := 4549 :: innerTail127_14
def innerCandidates127_13 : List Nat := [5851, 7717]
set_option maxRecDepth 4096 in
theorem innerCandidates127_13_checked : candidates edge 4 733 4549 innerTail127_14 best.bound = innerCandidates127_13 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_13 : scan edge (search edge 3) 4 [733] 733 innerTail127_13 best = best := by
  rw [innerTail127_13, scan, if_neg (by decide), innerCandidates127_13_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_14
    (if 3 ≤ innerCandidates127_13.length then search edge 3 [733, 4549] innerCandidates127_13 5282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_14

def innerTail127_12 : List Nat := 4441 :: innerTail127_13
def innerCandidates127_12 : List Nat := [5743]
set_option maxRecDepth 4096 in
theorem innerCandidates127_12_checked : candidates edge 4 733 4441 innerTail127_13 best.bound = innerCandidates127_12 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_12 : scan edge (search edge 3) 4 [733] 733 innerTail127_12 best = best := by
  rw [innerTail127_12, scan, if_neg (by decide), innerCandidates127_12_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_13
    (if 3 ≤ innerCandidates127_12.length then search edge 3 [733, 4441] innerCandidates127_12 5174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_13

def innerTail127_11 : List Nat := 4357 :: innerTail127_12
def innerCandidates127_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_11_checked : candidates edge 4 733 4357 innerTail127_12 best.bound = innerCandidates127_11 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_11 : scan edge (search edge 3) 4 [733] 733 innerTail127_11 best = best := by
  rw [innerTail127_11, scan, if_neg (by decide), innerCandidates127_11_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_12
    (if 3 ≤ innerCandidates127_11.length then search edge 3 [733, 4357] innerCandidates127_11 5090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_12

def innerTail127_10 : List Nat := 4027 :: innerTail127_11
def innerCandidates127_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_10_checked : candidates edge 4 733 4027 innerTail127_11 best.bound = innerCandidates127_10 := by
  unfold candidates
  have row : edge 4027 = SearchData.row4027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_10 : scan edge (search edge 3) 4 [733] 733 innerTail127_10 best = best := by
  rw [innerTail127_10, scan, if_neg (by decide), innerCandidates127_10_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_11
    (if 3 ≤ innerCandidates127_10.length then search edge 3 [733, 4027] innerCandidates127_10 4760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_11

def innerTail127_9 : List Nat := 3709 :: innerTail127_10
def innerCandidates127_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_9_checked : candidates edge 4 733 3709 innerTail127_10 best.bound = innerCandidates127_9 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_9 : scan edge (search edge 3) 4 [733] 733 innerTail127_9 best = best := by
  rw [innerTail127_9, scan, if_neg (by decide), innerCandidates127_9_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_10
    (if 3 ≤ innerCandidates127_9.length then search edge 3 [733, 3709] innerCandidates127_9 4442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_10

def innerTail127_8 : List Nat := 3331 :: innerTail127_9
def innerCandidates127_8 : List Nat := [4903, 5701, 7717]
set_option maxRecDepth 4096 in
theorem innerCandidates127_8_checked : candidates edge 4 733 3331 innerTail127_9 best.bound = innerCandidates127_8 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild127_8 : search edge 3 [733, 3331] innerCandidates127_8 4064 best = best := by decide

theorem innerScan127_8 : scan edge (search edge 3) 4 [733] 733 innerTail127_8 best = best := by
  rw [innerTail127_8, scan, if_neg (by decide), innerCandidates127_8_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_9
    (if 3 ≤ innerCandidates127_8.length then search edge 3 [733, 3331] innerCandidates127_8 4064 best else best) = best
  rw [if_pos (by decide), innerChild127_8]
  exact innerScan127_9

def innerTail127_7 : List Nat := 2551 :: innerTail127_8
def innerCandidates127_7 : List Nat := [3331]
set_option maxRecDepth 4096 in
theorem innerCandidates127_7_checked : candidates edge 4 733 2551 innerTail127_8 best.bound = innerCandidates127_7 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_7 : scan edge (search edge 3) 4 [733] 733 innerTail127_7 best = best := by
  rw [innerTail127_7, scan, if_neg (by decide), innerCandidates127_7_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_8
    (if 3 ≤ innerCandidates127_7.length then search edge 3 [733, 2551] innerCandidates127_7 3284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_8

def innerTail127_6 : List Nat := 2113 :: innerTail127_7
def innerCandidates127_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_6_checked : candidates edge 4 733 2113 innerTail127_7 best.bound = innerCandidates127_6 := by
  unfold candidates
  have row : edge 2113 = PrimePairCache.smallRow2113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_6 : scan edge (search edge 3) 4 [733] 733 innerTail127_6 best = best := by
  rw [innerTail127_6, scan, if_neg (by decide), innerCandidates127_6_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_7
    (if 3 ≤ innerCandidates127_6.length then search edge 3 [733, 2113] innerCandidates127_6 2846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_7

def innerTail127_5 : List Nat := 1999 :: innerTail127_6
def innerCandidates127_5 : List Nat := [7717, 8263]
set_option maxRecDepth 4096 in
theorem innerCandidates127_5_checked : candidates edge 4 733 1999 innerTail127_6 best.bound = innerCandidates127_5 := by
  unfold candidates
  have row : edge 1999 = PrimePairCache.smallRow1999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_5 : scan edge (search edge 3) 4 [733] 733 innerTail127_5 best = best := by
  rw [innerTail127_5, scan, if_neg (by decide), innerCandidates127_5_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_6
    (if 3 ≤ innerCandidates127_5.length then search edge 3 [733, 1999] innerCandidates127_5 2732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_6

def innerTail127_4 : List Nat := 1249 :: innerTail127_5
def innerCandidates127_4 : List Nat := [2551, 4441, 5851]
set_option maxRecDepth 4096 in
theorem innerCandidates127_4_checked : candidates edge 4 733 1249 innerTail127_5 best.bound = innerCandidates127_4 := by
  unfold candidates
  have row : edge 1249 = PrimePairCache.smallRow1249.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild127_4 : search edge 3 [733, 1249] innerCandidates127_4 1982 best = best := by decide

theorem innerScan127_4 : scan edge (search edge 3) 4 [733] 733 innerTail127_4 best = best := by
  rw [innerTail127_4, scan, if_neg (by decide), innerCandidates127_4_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_5
    (if 3 ≤ innerCandidates127_4.length then search edge 3 [733, 1249] innerCandidates127_4 1982 best else best) = best
  rw [if_pos (by decide), innerChild127_4]
  exact innerScan127_5

def innerTail127_3 : List Nat := 1039 :: innerTail127_4
def innerCandidates127_3 : List Nat := [1999, 5851]
set_option maxRecDepth 4096 in
theorem innerCandidates127_3_checked : candidates edge 4 733 1039 innerTail127_4 best.bound = innerCandidates127_3 := by
  unfold candidates
  have row : edge 1039 = PrimePairCache.smallRow1039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_3 : scan edge (search edge 3) 4 [733] 733 innerTail127_3 best = best := by
  rw [innerTail127_3, scan, if_neg (by decide), innerCandidates127_3_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_4
    (if 3 ≤ innerCandidates127_3.length then search edge 3 [733, 1039] innerCandidates127_3 1772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_4

def innerTail127_2 : List Nat := 991 :: innerTail127_3
def innerCandidates127_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_2_checked : candidates edge 4 733 991 innerTail127_3 best.bound = innerCandidates127_2 := by
  unfold candidates
  have row : edge 991 = PrimePairCache.smallRow991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_2 : scan edge (search edge 3) 4 [733] 733 innerTail127_2 best = best := by
  rw [innerTail127_2, scan, if_neg (by decide), innerCandidates127_2_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_3
    (if 3 ≤ innerCandidates127_2.length then search edge 3 [733, 991] innerCandidates127_2 1724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_3

def innerTail127_1 : List Nat := 883 :: innerTail127_2
def innerCandidates127_1 : List Nat := [991]
set_option maxRecDepth 4096 in
theorem innerCandidates127_1_checked : candidates edge 4 733 883 innerTail127_2 best.bound = innerCandidates127_1 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_1 : scan edge (search edge 3) 4 [733] 733 innerTail127_1 best = best := by
  rw [innerTail127_1, scan, if_neg (by decide), innerCandidates127_1_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_2
    (if 3 ≤ innerCandidates127_1.length then search edge 3 [733, 883] innerCandidates127_1 1616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_2

def innerTail127_0 : List Nat := 829 :: innerTail127_1
def innerCandidates127_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates127_0_checked : candidates edge 4 733 829 innerTail127_1 best.bound = innerCandidates127_0 := by
  unfold candidates
  have row : edge 829 = PrimePairCache.smallRow829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan127_0 : scan edge (search edge 3) 4 [733] 733 innerTail127_0 best = best := by
  rw [innerTail127_0, scan, if_neg (by decide), innerCandidates127_0_checked]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_1
    (if 3 ≤ innerCandidates127_0.length then search edge 3 [733, 829] innerCandidates127_0 1562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan127_1

theorem rootBranch127 : search edge 4 [733] rootCandidates127 733 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [733] 733 innerTail127_0 best = best
  exact innerScan127_0

end ProjectEulerSolutions.PrimePairUpper.Search
