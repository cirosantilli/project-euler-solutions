import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates137 : List Nat := [821, 827, 839, 929, 983, 1091, 1361, 1427, 1553, 1583, 2141, 2243, 2927, 2963, 3011, 3209, 3947, 4271, 4679, 5237, 5393, 5531, 6719]
set_option maxRecDepth 4096 in
theorem rootCandidates137_checked : candidates edge 5 0 809 tail138 best.bound = rootCandidates137 := by
  unfold candidates
  have row : edge 809 = PrimePairCache.smallRow809.contains := by funext q; rfl
  rw [row]
  decide

def innerTail137_23 : List Nat := []
theorem innerScan137_23 : scan edge (search edge 3) 4 [809] 809 innerTail137_23 best = best := rfl

def innerTail137_22 : List Nat := 6719 :: innerTail137_23
def innerCandidates137_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates137_22_checked : candidates edge 4 809 6719 innerTail137_23 best.bound = innerCandidates137_22 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_22 : scan edge (search edge 3) 4 [809] 809 innerTail137_22 best = best := by
  rw [innerTail137_22, scan, if_neg (by decide), innerCandidates137_22_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_23
    (if 3 ≤ innerCandidates137_22.length then search edge 3 [809, 6719] innerCandidates137_22 7528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_23

def innerTail137_21 : List Nat := 5531 :: innerTail137_22
def innerCandidates137_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates137_21_checked : candidates edge 4 809 5531 innerTail137_22 best.bound = innerCandidates137_21 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_21 : scan edge (search edge 3) 4 [809] 809 innerTail137_21 best = best := by
  rw [innerTail137_21, scan, if_neg (by decide), innerCandidates137_21_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_22
    (if 3 ≤ innerCandidates137_21.length then search edge 3 [809, 5531] innerCandidates137_21 6340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_22

def innerTail137_20 : List Nat := 5393 :: innerTail137_21
def innerCandidates137_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates137_20_checked : candidates edge 4 809 5393 innerTail137_21 best.bound = innerCandidates137_20 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_20 : scan edge (search edge 3) 4 [809] 809 innerTail137_20 best = best := by
  rw [innerTail137_20, scan, if_neg (by decide), innerCandidates137_20_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_21
    (if 3 ≤ innerCandidates137_20.length then search edge 3 [809, 5393] innerCandidates137_20 6202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_21

def innerTail137_19 : List Nat := 5237 :: innerTail137_20
def innerCandidates137_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates137_19_checked : candidates edge 4 809 5237 innerTail137_20 best.bound = innerCandidates137_19 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_19 : scan edge (search edge 3) 4 [809] 809 innerTail137_19 best = best := by
  rw [innerTail137_19, scan, if_neg (by decide), innerCandidates137_19_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_20
    (if 3 ≤ innerCandidates137_19.length then search edge 3 [809, 5237] innerCandidates137_19 6046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_20

def innerTail137_18 : List Nat := 4679 :: innerTail137_19
def innerCandidates137_18 : List Nat := [5393]
set_option maxRecDepth 4096 in
theorem innerCandidates137_18_checked : candidates edge 4 809 4679 innerTail137_19 best.bound = innerCandidates137_18 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_18 : scan edge (search edge 3) 4 [809] 809 innerTail137_18 best = best := by
  rw [innerTail137_18, scan, if_neg (by decide), innerCandidates137_18_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_19
    (if 3 ≤ innerCandidates137_18.length then search edge 3 [809, 4679] innerCandidates137_18 5488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_19

def innerTail137_17 : List Nat := 4271 :: innerTail137_18
def innerCandidates137_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates137_17_checked : candidates edge 4 809 4271 innerTail137_18 best.bound = innerCandidates137_17 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_17 : scan edge (search edge 3) 4 [809] 809 innerTail137_17 best = best := by
  rw [innerTail137_17, scan, if_neg (by decide), innerCandidates137_17_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_18
    (if 3 ≤ innerCandidates137_17.length then search edge 3 [809, 4271] innerCandidates137_17 5080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_18

def innerTail137_16 : List Nat := 3947 :: innerTail137_17
def innerCandidates137_16 : List Nat := [4679, 5393]
set_option maxRecDepth 4096 in
theorem innerCandidates137_16_checked : candidates edge 4 809 3947 innerTail137_17 best.bound = innerCandidates137_16 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_16 : scan edge (search edge 3) 4 [809] 809 innerTail137_16 best = best := by
  rw [innerTail137_16, scan, if_neg (by decide), innerCandidates137_16_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_17
    (if 3 ≤ innerCandidates137_16.length then search edge 3 [809, 3947] innerCandidates137_16 4756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_17

def innerTail137_15 : List Nat := 3209 :: innerTail137_16
def innerCandidates137_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates137_15_checked : candidates edge 4 809 3209 innerTail137_16 best.bound = innerCandidates137_15 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_15 : scan edge (search edge 3) 4 [809] 809 innerTail137_15 best = best := by
  rw [innerTail137_15, scan, if_neg (by decide), innerCandidates137_15_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_16
    (if 3 ≤ innerCandidates137_15.length then search edge 3 [809, 3209] innerCandidates137_15 4018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_16

def innerTail137_14 : List Nat := 3011 :: innerTail137_15
def innerCandidates137_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates137_14_checked : candidates edge 4 809 3011 innerTail137_15 best.bound = innerCandidates137_14 := by
  unfold candidates
  have row : edge 3011 = SearchData.row3011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_14 : scan edge (search edge 3) 4 [809] 809 innerTail137_14 best = best := by
  rw [innerTail137_14, scan, if_neg (by decide), innerCandidates137_14_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_15
    (if 3 ≤ innerCandidates137_14.length then search edge 3 [809, 3011] innerCandidates137_14 3820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_15

def innerTail137_13 : List Nat := 2963 :: innerTail137_14
def innerCandidates137_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates137_13_checked : candidates edge 4 809 2963 innerTail137_14 best.bound = innerCandidates137_13 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_13 : scan edge (search edge 3) 4 [809] 809 innerTail137_13 best = best := by
  rw [innerTail137_13, scan, if_neg (by decide), innerCandidates137_13_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_14
    (if 3 ≤ innerCandidates137_13.length then search edge 3 [809, 2963] innerCandidates137_13 3772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_14

def innerTail137_12 : List Nat := 2927 :: innerTail137_13
def innerCandidates137_12 : List Nat := [5531]
set_option maxRecDepth 4096 in
theorem innerCandidates137_12_checked : candidates edge 4 809 2927 innerTail137_13 best.bound = innerCandidates137_12 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_12 : scan edge (search edge 3) 4 [809] 809 innerTail137_12 best = best := by
  rw [innerTail137_12, scan, if_neg (by decide), innerCandidates137_12_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_13
    (if 3 ≤ innerCandidates137_12.length then search edge 3 [809, 2927] innerCandidates137_12 3736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_13

def innerTail137_11 : List Nat := 2243 :: innerTail137_12
def innerCandidates137_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates137_11_checked : candidates edge 4 809 2243 innerTail137_12 best.bound = innerCandidates137_11 := by
  unfold candidates
  have row : edge 2243 = PrimePairCache.smallRow2243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_11 : scan edge (search edge 3) 4 [809] 809 innerTail137_11 best = best := by
  rw [innerTail137_11, scan, if_neg (by decide), innerCandidates137_11_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_12
    (if 3 ≤ innerCandidates137_11.length then search edge 3 [809, 2243] innerCandidates137_11 3052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_12

def innerTail137_10 : List Nat := 2141 :: innerTail137_11
def innerCandidates137_10 : List Nat := [3947]
set_option maxRecDepth 4096 in
theorem innerCandidates137_10_checked : candidates edge 4 809 2141 innerTail137_11 best.bound = innerCandidates137_10 := by
  unfold candidates
  have row : edge 2141 = PrimePairCache.smallRow2141.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_10 : scan edge (search edge 3) 4 [809] 809 innerTail137_10 best = best := by
  rw [innerTail137_10, scan, if_neg (by decide), innerCandidates137_10_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_11
    (if 3 ≤ innerCandidates137_10.length then search edge 3 [809, 2141] innerCandidates137_10 2950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_11

def innerTail137_9 : List Nat := 1583 :: innerTail137_10
def innerCandidates137_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates137_9_checked : candidates edge 4 809 1583 innerTail137_10 best.bound = innerCandidates137_9 := by
  unfold candidates
  have row : edge 1583 = PrimePairCache.smallRow1583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_9 : scan edge (search edge 3) 4 [809] 809 innerTail137_9 best = best := by
  rw [innerTail137_9, scan, if_neg (by decide), innerCandidates137_9_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_10
    (if 3 ≤ innerCandidates137_9.length then search edge 3 [809, 1583] innerCandidates137_9 2392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_10

def innerTail137_8 : List Nat := 1553 :: innerTail137_9
def innerCandidates137_8 : List Nat := [1583]
set_option maxRecDepth 4096 in
theorem innerCandidates137_8_checked : candidates edge 4 809 1553 innerTail137_9 best.bound = innerCandidates137_8 := by
  unfold candidates
  have row : edge 1553 = PrimePairCache.smallRow1553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_8 : scan edge (search edge 3) 4 [809] 809 innerTail137_8 best = best := by
  rw [innerTail137_8, scan, if_neg (by decide), innerCandidates137_8_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_9
    (if 3 ≤ innerCandidates137_8.length then search edge 3 [809, 1553] innerCandidates137_8 2362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_9

def innerTail137_7 : List Nat := 1427 :: innerTail137_8
def innerCandidates137_7 : List Nat := [3011]
set_option maxRecDepth 4096 in
theorem innerCandidates137_7_checked : candidates edge 4 809 1427 innerTail137_8 best.bound = innerCandidates137_7 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_7 : scan edge (search edge 3) 4 [809] 809 innerTail137_7 best = best := by
  rw [innerTail137_7, scan, if_neg (by decide), innerCandidates137_7_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_8
    (if 3 ≤ innerCandidates137_7.length then search edge 3 [809, 1427] innerCandidates137_7 2236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_8

def innerTail137_6 : List Nat := 1361 :: innerTail137_7
def innerCandidates137_6 : List Nat := [2141, 3947]
set_option maxRecDepth 4096 in
theorem innerCandidates137_6_checked : candidates edge 4 809 1361 innerTail137_7 best.bound = innerCandidates137_6 := by
  unfold candidates
  have row : edge 1361 = PrimePairCache.smallRow1361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_6 : scan edge (search edge 3) 4 [809] 809 innerTail137_6 best = best := by
  rw [innerTail137_6, scan, if_neg (by decide), innerCandidates137_6_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_7
    (if 3 ≤ innerCandidates137_6.length then search edge 3 [809, 1361] innerCandidates137_6 2170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_7

def innerTail137_5 : List Nat := 1091 :: innerTail137_6
def innerCandidates137_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates137_5_checked : candidates edge 4 809 1091 innerTail137_6 best.bound = innerCandidates137_5 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_5 : scan edge (search edge 3) 4 [809] 809 innerTail137_5 best = best := by
  rw [innerTail137_5, scan, if_neg (by decide), innerCandidates137_5_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_6
    (if 3 ≤ innerCandidates137_5.length then search edge 3 [809, 1091] innerCandidates137_5 1900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_6

def innerTail137_4 : List Nat := 983 :: innerTail137_5
def innerCandidates137_4 : List Nat := [1091, 6719]
set_option maxRecDepth 4096 in
theorem innerCandidates137_4_checked : candidates edge 4 809 983 innerTail137_5 best.bound = innerCandidates137_4 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_4 : scan edge (search edge 3) 4 [809] 809 innerTail137_4 best = best := by
  rw [innerTail137_4, scan, if_neg (by decide), innerCandidates137_4_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_5
    (if 3 ≤ innerCandidates137_4.length then search edge 3 [809, 983] innerCandidates137_4 1792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_5

def innerTail137_3 : List Nat := 929 :: innerTail137_4
def innerCandidates137_3 : List Nat := [983, 3209]
set_option maxRecDepth 4096 in
theorem innerCandidates137_3_checked : candidates edge 4 809 929 innerTail137_4 best.bound = innerCandidates137_3 := by
  unfold candidates
  have row : edge 929 = PrimePairCache.smallRow929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_3 : scan edge (search edge 3) 4 [809] 809 innerTail137_3 best = best := by
  rw [innerTail137_3, scan, if_neg (by decide), innerCandidates137_3_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_4
    (if 3 ≤ innerCandidates137_3.length then search edge 3 [809, 929] innerCandidates137_3 1738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_4

def innerTail137_2 : List Nat := 839 :: innerTail137_3
def innerCandidates137_2 : List Nat := [4679, 5237]
set_option maxRecDepth 4096 in
theorem innerCandidates137_2_checked : candidates edge 4 809 839 innerTail137_3 best.bound = innerCandidates137_2 := by
  unfold candidates
  have row : edge 839 = PrimePairCache.smallRow839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_2 : scan edge (search edge 3) 4 [809] 809 innerTail137_2 best = best := by
  rw [innerTail137_2, scan, if_neg (by decide), innerCandidates137_2_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_3
    (if 3 ≤ innerCandidates137_2.length then search edge 3 [809, 839] innerCandidates137_2 1648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_3

def innerTail137_1 : List Nat := 827 :: innerTail137_2
def innerCandidates137_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates137_1_checked : candidates edge 4 809 827 innerTail137_2 best.bound = innerCandidates137_1 := by
  unfold candidates
  have row : edge 827 = PrimePairCache.smallRow827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_1 : scan edge (search edge 3) 4 [809] 809 innerTail137_1 best = best := by
  rw [innerTail137_1, scan, if_neg (by decide), innerCandidates137_1_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_2
    (if 3 ≤ innerCandidates137_1.length then search edge 3 [809, 827] innerCandidates137_1 1636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_2

def innerTail137_0 : List Nat := 821 :: innerTail137_1
def innerCandidates137_0 : List Nat := [827]
set_option maxRecDepth 4096 in
theorem innerCandidates137_0_checked : candidates edge 4 809 821 innerTail137_1 best.bound = innerCandidates137_0 := by
  unfold candidates
  have row : edge 821 = PrimePairCache.smallRow821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan137_0 : scan edge (search edge 3) 4 [809] 809 innerTail137_0 best = best := by
  rw [innerTail137_0, scan, if_neg (by decide), innerCandidates137_0_checked]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_1
    (if 3 ≤ innerCandidates137_0.length then search edge 3 [809, 821] innerCandidates137_0 1630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan137_1

theorem rootBranch137 : search edge 4 [809] rootCandidates137 809 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [809] 809 innerTail137_0 best = best
  exact innerScan137_0

def rootCandidates136 : List Nat := [2297, 2339, 2411, 2417, 2621, 2969, 3209, 3299, 3461, 3659, 3929, 4259, 4481, 4937, 5147, 5189, 5519, 5531, 6389, 6833, 6911, 7457, 7649, 7841]
set_option maxRecDepth 4096 in
theorem rootCandidates136_checked : candidates edge 5 0 797 tail137 best.bound = rootCandidates136 := by
  unfold candidates
  have row : edge 797 = PrimePairCache.smallRow797.contains := by funext q; rfl
  rw [row]
  decide

def innerTail136_24 : List Nat := []
theorem innerScan136_24 : scan edge (search edge 3) 4 [797] 797 innerTail136_24 best = best := rfl

def innerTail136_23 : List Nat := 7841 :: innerTail136_24
def innerCandidates136_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_23_checked : candidates edge 4 797 7841 innerTail136_24 best.bound = innerCandidates136_23 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_23 : scan edge (search edge 3) 4 [797] 797 innerTail136_23 best = best := by
  rw [innerTail136_23, scan, if_neg (by decide), innerCandidates136_23_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_24
    (if 3 ≤ innerCandidates136_23.length then search edge 3 [797, 7841] innerCandidates136_23 8638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_24

def innerTail136_22 : List Nat := 7649 :: innerTail136_23
def innerCandidates136_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_22_checked : candidates edge 4 797 7649 innerTail136_23 best.bound = innerCandidates136_22 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_22 : scan edge (search edge 3) 4 [797] 797 innerTail136_22 best = best := by
  rw [innerTail136_22, scan, if_neg (by decide), innerCandidates136_22_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_23
    (if 3 ≤ innerCandidates136_22.length then search edge 3 [797, 7649] innerCandidates136_22 8446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_23

def innerTail136_21 : List Nat := 7457 :: innerTail136_22
def innerCandidates136_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_21_checked : candidates edge 4 797 7457 innerTail136_22 best.bound = innerCandidates136_21 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_21 : scan edge (search edge 3) 4 [797] 797 innerTail136_21 best = best := by
  rw [innerTail136_21, scan, if_neg (by decide), innerCandidates136_21_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_22
    (if 3 ≤ innerCandidates136_21.length then search edge 3 [797, 7457] innerCandidates136_21 8254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_22

def innerTail136_20 : List Nat := 6911 :: innerTail136_21
def innerCandidates136_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_20_checked : candidates edge 4 797 6911 innerTail136_21 best.bound = innerCandidates136_20 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_20 : scan edge (search edge 3) 4 [797] 797 innerTail136_20 best = best := by
  rw [innerTail136_20, scan, if_neg (by decide), innerCandidates136_20_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_21
    (if 3 ≤ innerCandidates136_20.length then search edge 3 [797, 6911] innerCandidates136_20 7708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_21

def innerTail136_19 : List Nat := 6833 :: innerTail136_20
def innerCandidates136_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_19_checked : candidates edge 4 797 6833 innerTail136_20 best.bound = innerCandidates136_19 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_19 : scan edge (search edge 3) 4 [797] 797 innerTail136_19 best = best := by
  rw [innerTail136_19, scan, if_neg (by decide), innerCandidates136_19_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_20
    (if 3 ≤ innerCandidates136_19.length then search edge 3 [797, 6833] innerCandidates136_19 7630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_20

def innerTail136_18 : List Nat := 6389 :: innerTail136_19
def innerCandidates136_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_18_checked : candidates edge 4 797 6389 innerTail136_19 best.bound = innerCandidates136_18 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_18 : scan edge (search edge 3) 4 [797] 797 innerTail136_18 best = best := by
  rw [innerTail136_18, scan, if_neg (by decide), innerCandidates136_18_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_19
    (if 3 ≤ innerCandidates136_18.length then search edge 3 [797, 6389] innerCandidates136_18 7186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_19

def innerTail136_17 : List Nat := 5531 :: innerTail136_18
def innerCandidates136_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_17_checked : candidates edge 4 797 5531 innerTail136_18 best.bound = innerCandidates136_17 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_17 : scan edge (search edge 3) 4 [797] 797 innerTail136_17 best = best := by
  rw [innerTail136_17, scan, if_neg (by decide), innerCandidates136_17_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_18
    (if 3 ≤ innerCandidates136_17.length then search edge 3 [797, 5531] innerCandidates136_17 6328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_18

def innerTail136_16 : List Nat := 5519 :: innerTail136_17
def innerCandidates136_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_16_checked : candidates edge 4 797 5519 innerTail136_17 best.bound = innerCandidates136_16 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_16 : scan edge (search edge 3) 4 [797] 797 innerTail136_16 best = best := by
  rw [innerTail136_16, scan, if_neg (by decide), innerCandidates136_16_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_17
    (if 3 ≤ innerCandidates136_16.length then search edge 3 [797, 5519] innerCandidates136_16 6316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_17

def innerTail136_15 : List Nat := 5189 :: innerTail136_16
def innerCandidates136_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_15_checked : candidates edge 4 797 5189 innerTail136_16 best.bound = innerCandidates136_15 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_15 : scan edge (search edge 3) 4 [797] 797 innerTail136_15 best = best := by
  rw [innerTail136_15, scan, if_neg (by decide), innerCandidates136_15_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_16
    (if 3 ≤ innerCandidates136_15.length then search edge 3 [797, 5189] innerCandidates136_15 5986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_16

def innerTail136_14 : List Nat := 5147 :: innerTail136_15
def innerCandidates136_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_14_checked : candidates edge 4 797 5147 innerTail136_15 best.bound = innerCandidates136_14 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_14 : scan edge (search edge 3) 4 [797] 797 innerTail136_14 best = best := by
  rw [innerTail136_14, scan, if_neg (by decide), innerCandidates136_14_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_15
    (if 3 ≤ innerCandidates136_14.length then search edge 3 [797, 5147] innerCandidates136_14 5944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_15

def innerTail136_13 : List Nat := 4937 :: innerTail136_14
def innerCandidates136_13 : List Nat := [7649, 7841]
set_option maxRecDepth 4096 in
theorem innerCandidates136_13_checked : candidates edge 4 797 4937 innerTail136_14 best.bound = innerCandidates136_13 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_13 : scan edge (search edge 3) 4 [797] 797 innerTail136_13 best = best := by
  rw [innerTail136_13, scan, if_neg (by decide), innerCandidates136_13_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_14
    (if 3 ≤ innerCandidates136_13.length then search edge 3 [797, 4937] innerCandidates136_13 5734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_14

def innerTail136_12 : List Nat := 4481 :: innerTail136_13
def innerCandidates136_12 : List Nat := [5189]
set_option maxRecDepth 4096 in
theorem innerCandidates136_12_checked : candidates edge 4 797 4481 innerTail136_13 best.bound = innerCandidates136_12 := by
  unfold candidates
  have row : edge 4481 = SearchData.row4481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_12 : scan edge (search edge 3) 4 [797] 797 innerTail136_12 best = best := by
  rw [innerTail136_12, scan, if_neg (by decide), innerCandidates136_12_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_13
    (if 3 ≤ innerCandidates136_12.length then search edge 3 [797, 4481] innerCandidates136_12 5278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_13

def innerTail136_11 : List Nat := 4259 :: innerTail136_12
def innerCandidates136_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_11_checked : candidates edge 4 797 4259 innerTail136_12 best.bound = innerCandidates136_11 := by
  unfold candidates
  have row : edge 4259 = SearchData.row4259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_11 : scan edge (search edge 3) 4 [797] 797 innerTail136_11 best = best := by
  rw [innerTail136_11, scan, if_neg (by decide), innerCandidates136_11_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_12
    (if 3 ≤ innerCandidates136_11.length then search edge 3 [797, 4259] innerCandidates136_11 5056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_12

def innerTail136_10 : List Nat := 3929 :: innerTail136_11
def innerCandidates136_10 : List Nat := [7457]
set_option maxRecDepth 4096 in
theorem innerCandidates136_10_checked : candidates edge 4 797 3929 innerTail136_11 best.bound = innerCandidates136_10 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_10 : scan edge (search edge 3) 4 [797] 797 innerTail136_10 best = best := by
  rw [innerTail136_10, scan, if_neg (by decide), innerCandidates136_10_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_11
    (if 3 ≤ innerCandidates136_10.length then search edge 3 [797, 3929] innerCandidates136_10 4726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_11

def innerTail136_9 : List Nat := 3659 :: innerTail136_10
def innerCandidates136_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_9_checked : candidates edge 4 797 3659 innerTail136_10 best.bound = innerCandidates136_9 := by
  unfold candidates
  have row : edge 3659 = SearchData.row3659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_9 : scan edge (search edge 3) 4 [797] 797 innerTail136_9 best = best := by
  rw [innerTail136_9, scan, if_neg (by decide), innerCandidates136_9_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_10
    (if 3 ≤ innerCandidates136_9.length then search edge 3 [797, 3659] innerCandidates136_9 4456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_10

def innerTail136_8 : List Nat := 3461 :: innerTail136_9
def innerCandidates136_8 : List Nat := [3659]
set_option maxRecDepth 4096 in
theorem innerCandidates136_8_checked : candidates edge 4 797 3461 innerTail136_9 best.bound = innerCandidates136_8 := by
  unfold candidates
  have row : edge 3461 = SearchData.row3461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_8 : scan edge (search edge 3) 4 [797] 797 innerTail136_8 best = best := by
  rw [innerTail136_8, scan, if_neg (by decide), innerCandidates136_8_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_9
    (if 3 ≤ innerCandidates136_8.length then search edge 3 [797, 3461] innerCandidates136_8 4258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_9

def innerTail136_7 : List Nat := 3299 :: innerTail136_8
def innerCandidates136_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_7_checked : candidates edge 4 797 3299 innerTail136_8 best.bound = innerCandidates136_7 := by
  unfold candidates
  have row : edge 3299 = SearchData.row3299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_7 : scan edge (search edge 3) 4 [797] 797 innerTail136_7 best = best := by
  rw [innerTail136_7, scan, if_neg (by decide), innerCandidates136_7_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_8
    (if 3 ≤ innerCandidates136_7.length then search edge 3 [797, 3299] innerCandidates136_7 4096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_8

def innerTail136_6 : List Nat := 3209 :: innerTail136_7
def innerCandidates136_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_6_checked : candidates edge 4 797 3209 innerTail136_7 best.bound = innerCandidates136_6 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_6 : scan edge (search edge 3) 4 [797] 797 innerTail136_6 best = best := by
  rw [innerTail136_6, scan, if_neg (by decide), innerCandidates136_6_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_7
    (if 3 ≤ innerCandidates136_6.length then search edge 3 [797, 3209] innerCandidates136_6 4006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_7

def innerTail136_5 : List Nat := 2969 :: innerTail136_6
def innerCandidates136_5 : List Nat := [7841]
set_option maxRecDepth 4096 in
theorem innerCandidates136_5_checked : candidates edge 4 797 2969 innerTail136_6 best.bound = innerCandidates136_5 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_5 : scan edge (search edge 3) 4 [797] 797 innerTail136_5 best = best := by
  rw [innerTail136_5, scan, if_neg (by decide), innerCandidates136_5_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_6
    (if 3 ≤ innerCandidates136_5.length then search edge 3 [797, 2969] innerCandidates136_5 3766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_6

def innerTail136_4 : List Nat := 2621 :: innerTail136_5
def innerCandidates136_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_4_checked : candidates edge 4 797 2621 innerTail136_5 best.bound = innerCandidates136_4 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_4 : scan edge (search edge 3) 4 [797] 797 innerTail136_4 best = best := by
  rw [innerTail136_4, scan, if_neg (by decide), innerCandidates136_4_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_5
    (if 3 ≤ innerCandidates136_4.length then search edge 3 [797, 2621] innerCandidates136_4 3418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_5

def innerTail136_3 : List Nat := 2417 :: innerTail136_4
def innerCandidates136_3 : List Nat := [6389]
set_option maxRecDepth 4096 in
theorem innerCandidates136_3_checked : candidates edge 4 797 2417 innerTail136_4 best.bound = innerCandidates136_3 := by
  unfold candidates
  have row : edge 2417 = PrimePairCache.smallRow2417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_3 : scan edge (search edge 3) 4 [797] 797 innerTail136_3 best = best := by
  rw [innerTail136_3, scan, if_neg (by decide), innerCandidates136_3_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_4
    (if 3 ≤ innerCandidates136_3.length then search edge 3 [797, 2417] innerCandidates136_3 3214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_4

def innerTail136_2 : List Nat := 2411 :: innerTail136_3
def innerCandidates136_2 : List Nat := [5519]
set_option maxRecDepth 4096 in
theorem innerCandidates136_2_checked : candidates edge 4 797 2411 innerTail136_3 best.bound = innerCandidates136_2 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_2 : scan edge (search edge 3) 4 [797] 797 innerTail136_2 best = best := by
  rw [innerTail136_2, scan, if_neg (by decide), innerCandidates136_2_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_3
    (if 3 ≤ innerCandidates136_2.length then search edge 3 [797, 2411] innerCandidates136_2 3208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_3

def innerTail136_1 : List Nat := 2339 :: innerTail136_2
def innerCandidates136_1 : List Nat := [7841]
set_option maxRecDepth 4096 in
theorem innerCandidates136_1_checked : candidates edge 4 797 2339 innerTail136_2 best.bound = innerCandidates136_1 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_1 : scan edge (search edge 3) 4 [797] 797 innerTail136_1 best = best := by
  rw [innerTail136_1, scan, if_neg (by decide), innerCandidates136_1_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_2
    (if 3 ≤ innerCandidates136_1.length then search edge 3 [797, 2339] innerCandidates136_1 3136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_2

def innerTail136_0 : List Nat := 2297 :: innerTail136_1
def innerCandidates136_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates136_0_checked : candidates edge 4 797 2297 innerTail136_1 best.bound = innerCandidates136_0 := by
  unfold candidates
  have row : edge 2297 = PrimePairCache.smallRow2297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan136_0 : scan edge (search edge 3) 4 [797] 797 innerTail136_0 best = best := by
  rw [innerTail136_0, scan, if_neg (by decide), innerCandidates136_0_checked]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_1
    (if 3 ≤ innerCandidates136_0.length then search edge 3 [797, 2297] innerCandidates136_0 3094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan136_1

theorem rootBranch136 : search edge 4 [797] rootCandidates136 797 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [797] 797 innerTail136_0 best = best
  exact innerScan136_0

end ProjectEulerSolutions.PrimePairUpper.Search
