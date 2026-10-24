import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates110 : List Nat := [647, 887, 983, 1091, 1301, 1427, 2063, 2657, 2741, 2909, 3593, 4001, 4253, 4391, 5261, 5333, 5693, 6011, 6323, 7229, 7541, 8087, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates110_checked : candidates edge 5 0 617 tail111 best.bound = rootCandidates110 := by
  unfold candidates
  have row : edge 617 = PrimePairCache.smallRow617.contains := by funext q; rfl
  rw [row]
  decide

def innerTail110_23 : List Nat := []
theorem innerScan110_23 : scan edge (search edge 3) 4 [617] 617 innerTail110_23 best = best := rfl

def innerTail110_22 : List Nat := 8243 :: innerTail110_23
def innerCandidates110_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_22_checked : candidates edge 4 617 8243 innerTail110_23 best.bound = innerCandidates110_22 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_22 : scan edge (search edge 3) 4 [617] 617 innerTail110_22 best = best := by
  rw [innerTail110_22, scan, if_neg (by decide), innerCandidates110_22_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_23
    (if 3 ≤ innerCandidates110_22.length then search edge 3 [617, 8243] innerCandidates110_22 8860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_23

def innerTail110_21 : List Nat := 8087 :: innerTail110_22
def innerCandidates110_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_21_checked : candidates edge 4 617 8087 innerTail110_22 best.bound = innerCandidates110_21 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_21 : scan edge (search edge 3) 4 [617] 617 innerTail110_21 best = best := by
  rw [innerTail110_21, scan, if_neg (by decide), innerCandidates110_21_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_22
    (if 3 ≤ innerCandidates110_21.length then search edge 3 [617, 8087] innerCandidates110_21 8704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_22

def innerTail110_20 : List Nat := 7541 :: innerTail110_21
def innerCandidates110_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_20_checked : candidates edge 4 617 7541 innerTail110_21 best.bound = innerCandidates110_20 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_20 : scan edge (search edge 3) 4 [617] 617 innerTail110_20 best = best := by
  rw [innerTail110_20, scan, if_neg (by decide), innerCandidates110_20_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_21
    (if 3 ≤ innerCandidates110_20.length then search edge 3 [617, 7541] innerCandidates110_20 8158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_21

def innerTail110_19 : List Nat := 7229 :: innerTail110_20
def innerCandidates110_19 : List Nat := [7541]
set_option maxRecDepth 4096 in
theorem innerCandidates110_19_checked : candidates edge 4 617 7229 innerTail110_20 best.bound = innerCandidates110_19 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_19 : scan edge (search edge 3) 4 [617] 617 innerTail110_19 best = best := by
  rw [innerTail110_19, scan, if_neg (by decide), innerCandidates110_19_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_20
    (if 3 ≤ innerCandidates110_19.length then search edge 3 [617, 7229] innerCandidates110_19 7846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_20

def innerTail110_18 : List Nat := 6323 :: innerTail110_19
def innerCandidates110_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_18_checked : candidates edge 4 617 6323 innerTail110_19 best.bound = innerCandidates110_18 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_18 : scan edge (search edge 3) 4 [617] 617 innerTail110_18 best = best := by
  rw [innerTail110_18, scan, if_neg (by decide), innerCandidates110_18_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_19
    (if 3 ≤ innerCandidates110_18.length then search edge 3 [617, 6323] innerCandidates110_18 6940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_19

def innerTail110_17 : List Nat := 6011 :: innerTail110_18
def innerCandidates110_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_17_checked : candidates edge 4 617 6011 innerTail110_18 best.bound = innerCandidates110_17 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_17 : scan edge (search edge 3) 4 [617] 617 innerTail110_17 best = best := by
  rw [innerTail110_17, scan, if_neg (by decide), innerCandidates110_17_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_18
    (if 3 ≤ innerCandidates110_17.length then search edge 3 [617, 6011] innerCandidates110_17 6628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_18

def innerTail110_16 : List Nat := 5693 :: innerTail110_17
def innerCandidates110_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_16_checked : candidates edge 4 617 5693 innerTail110_17 best.bound = innerCandidates110_16 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_16 : scan edge (search edge 3) 4 [617] 617 innerTail110_16 best = best := by
  rw [innerTail110_16, scan, if_neg (by decide), innerCandidates110_16_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_17
    (if 3 ≤ innerCandidates110_16.length then search edge 3 [617, 5693] innerCandidates110_16 6310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_17

def innerTail110_15 : List Nat := 5333 :: innerTail110_16
def innerCandidates110_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_15_checked : candidates edge 4 617 5333 innerTail110_16 best.bound = innerCandidates110_15 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_15 : scan edge (search edge 3) 4 [617] 617 innerTail110_15 best = best := by
  rw [innerTail110_15, scan, if_neg (by decide), innerCandidates110_15_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_16
    (if 3 ≤ innerCandidates110_15.length then search edge 3 [617, 5333] innerCandidates110_15 5950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_16

def innerTail110_14 : List Nat := 5261 :: innerTail110_15
def innerCandidates110_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_14_checked : candidates edge 4 617 5261 innerTail110_15 best.bound = innerCandidates110_14 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_14 : scan edge (search edge 3) 4 [617] 617 innerTail110_14 best = best := by
  rw [innerTail110_14, scan, if_neg (by decide), innerCandidates110_14_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_15
    (if 3 ≤ innerCandidates110_14.length then search edge 3 [617, 5261] innerCandidates110_14 5878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_15

def innerTail110_13 : List Nat := 4391 :: innerTail110_14
def innerCandidates110_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_13_checked : candidates edge 4 617 4391 innerTail110_14 best.bound = innerCandidates110_13 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_13 : scan edge (search edge 3) 4 [617] 617 innerTail110_13 best = best := by
  rw [innerTail110_13, scan, if_neg (by decide), innerCandidates110_13_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_14
    (if 3 ≤ innerCandidates110_13.length then search edge 3 [617, 4391] innerCandidates110_13 5008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_14

def innerTail110_12 : List Nat := 4253 :: innerTail110_13
def innerCandidates110_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_12_checked : candidates edge 4 617 4253 innerTail110_13 best.bound = innerCandidates110_12 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_12 : scan edge (search edge 3) 4 [617] 617 innerTail110_12 best = best := by
  rw [innerTail110_12, scan, if_neg (by decide), innerCandidates110_12_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_13
    (if 3 ≤ innerCandidates110_12.length then search edge 3 [617, 4253] innerCandidates110_12 4870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_13

def innerTail110_11 : List Nat := 4001 :: innerTail110_12
def innerCandidates110_11 : List Nat := [4253, 4391, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates110_11_checked : candidates edge 4 617 4001 innerTail110_12 best.bound = innerCandidates110_11 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild110_11 : search edge 3 [617, 4001] innerCandidates110_11 4618 best = best := by decide

theorem innerScan110_11 : scan edge (search edge 3) 4 [617] 617 innerTail110_11 best = best := by
  rw [innerTail110_11, scan, if_neg (by decide), innerCandidates110_11_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_12
    (if 3 ≤ innerCandidates110_11.length then search edge 3 [617, 4001] innerCandidates110_11 4618 best else best) = best
  rw [if_pos (by decide), innerChild110_11]
  exact innerScan110_12

def innerTail110_10 : List Nat := 3593 :: innerTail110_11
def innerCandidates110_10 : List Nat := [4001, 5693]
set_option maxRecDepth 4096 in
theorem innerCandidates110_10_checked : candidates edge 4 617 3593 innerTail110_11 best.bound = innerCandidates110_10 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_10 : scan edge (search edge 3) 4 [617] 617 innerTail110_10 best = best := by
  rw [innerTail110_10, scan, if_neg (by decide), innerCandidates110_10_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_11
    (if 3 ≤ innerCandidates110_10.length then search edge 3 [617, 3593] innerCandidates110_10 4210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_11

def innerTail110_9 : List Nat := 2909 :: innerTail110_10
def innerCandidates110_9 : List Nat := [4253, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates110_9_checked : candidates edge 4 617 2909 innerTail110_10 best.bound = innerCandidates110_9 := by
  unfold candidates
  have row : edge 2909 = SearchData.row2909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_9 : scan edge (search edge 3) 4 [617] 617 innerTail110_9 best = best := by
  rw [innerTail110_9, scan, if_neg (by decide), innerCandidates110_9_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_10
    (if 3 ≤ innerCandidates110_9.length then search edge 3 [617, 2909] innerCandidates110_9 3526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_10

def innerTail110_8 : List Nat := 2741 :: innerTail110_9
def innerCandidates110_8 : List Nat := [4253]
set_option maxRecDepth 4096 in
theorem innerCandidates110_8_checked : candidates edge 4 617 2741 innerTail110_9 best.bound = innerCandidates110_8 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_8 : scan edge (search edge 3) 4 [617] 617 innerTail110_8 best = best := by
  rw [innerTail110_8, scan, if_neg (by decide), innerCandidates110_8_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_9
    (if 3 ≤ innerCandidates110_8.length then search edge 3 [617, 2741] innerCandidates110_8 3358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_9

def innerTail110_7 : List Nat := 2657 :: innerTail110_8
def innerCandidates110_7 : List Nat := [4391, 8243]
set_option maxRecDepth 4096 in
theorem innerCandidates110_7_checked : candidates edge 4 617 2657 innerTail110_8 best.bound = innerCandidates110_7 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_7 : scan edge (search edge 3) 4 [617] 617 innerTail110_7 best = best := by
  rw [innerTail110_7, scan, if_neg (by decide), innerCandidates110_7_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_8
    (if 3 ≤ innerCandidates110_7.length then search edge 3 [617, 2657] innerCandidates110_7 3274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_8

def innerTail110_6 : List Nat := 2063 :: innerTail110_7
def innerCandidates110_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_6_checked : candidates edge 4 617 2063 innerTail110_7 best.bound = innerCandidates110_6 := by
  unfold candidates
  have row : edge 2063 = PrimePairCache.smallRow2063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_6 : scan edge (search edge 3) 4 [617] 617 innerTail110_6 best = best := by
  rw [innerTail110_6, scan, if_neg (by decide), innerCandidates110_6_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_7
    (if 3 ≤ innerCandidates110_6.length then search edge 3 [617, 2063] innerCandidates110_6 2680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_7

def innerTail110_5 : List Nat := 1427 :: innerTail110_6
def innerCandidates110_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_5_checked : candidates edge 4 617 1427 innerTail110_6 best.bound = innerCandidates110_5 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_5 : scan edge (search edge 3) 4 [617] 617 innerTail110_5 best = best := by
  rw [innerTail110_5, scan, if_neg (by decide), innerCandidates110_5_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_6
    (if 3 ≤ innerCandidates110_5.length then search edge 3 [617, 1427] innerCandidates110_5 2044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_6

def innerTail110_4 : List Nat := 1301 :: innerTail110_5
def innerCandidates110_4 : List Nat := [6011]
set_option maxRecDepth 4096 in
theorem innerCandidates110_4_checked : candidates edge 4 617 1301 innerTail110_5 best.bound = innerCandidates110_4 := by
  unfold candidates
  have row : edge 1301 = PrimePairCache.smallRow1301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_4 : scan edge (search edge 3) 4 [617] 617 innerTail110_4 best = best := by
  rw [innerTail110_4, scan, if_neg (by decide), innerCandidates110_4_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_5
    (if 3 ≤ innerCandidates110_4.length then search edge 3 [617, 1301] innerCandidates110_4 1918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_5

def innerTail110_3 : List Nat := 1091 :: innerTail110_4
def innerCandidates110_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_3_checked : candidates edge 4 617 1091 innerTail110_4 best.bound = innerCandidates110_3 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_3 : scan edge (search edge 3) 4 [617] 617 innerTail110_3 best = best := by
  rw [innerTail110_3, scan, if_neg (by decide), innerCandidates110_3_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_4
    (if 3 ≤ innerCandidates110_3.length then search edge 3 [617, 1091] innerCandidates110_3 1708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_4

def innerTail110_2 : List Nat := 983 :: innerTail110_3
def innerCandidates110_2 : List Nat := [1091]
set_option maxRecDepth 4096 in
theorem innerCandidates110_2_checked : candidates edge 4 617 983 innerTail110_3 best.bound = innerCandidates110_2 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_2 : scan edge (search edge 3) 4 [617] 617 innerTail110_2 best = best := by
  rw [innerTail110_2, scan, if_neg (by decide), innerCandidates110_2_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_3
    (if 3 ≤ innerCandidates110_2.length then search edge 3 [617, 983] innerCandidates110_2 1600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_3

def innerTail110_1 : List Nat := 887 :: innerTail110_2
def innerCandidates110_1 : List Nat := [1091, 1427, 2741, 8087]
set_option maxRecDepth 4096 in
theorem innerCandidates110_1_checked : candidates edge 4 617 887 innerTail110_2 best.bound = innerCandidates110_1 := by
  unfold candidates
  have row : edge 887 = PrimePairCache.smallRow887.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild110_1 : search edge 3 [617, 887] innerCandidates110_1 1504 best = best := by decide

theorem innerScan110_1 : scan edge (search edge 3) 4 [617] 617 innerTail110_1 best = best := by
  rw [innerTail110_1, scan, if_neg (by decide), innerCandidates110_1_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_2
    (if 3 ≤ innerCandidates110_1.length then search edge 3 [617, 887] innerCandidates110_1 1504 best else best) = best
  rw [if_pos (by decide), innerChild110_1]
  exact innerScan110_2

def innerTail110_0 : List Nat := 647 :: innerTail110_1
def innerCandidates110_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates110_0_checked : candidates edge 4 617 647 innerTail110_1 best.bound = innerCandidates110_0 := by
  unfold candidates
  have row : edge 647 = PrimePairCache.smallRow647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan110_0 : scan edge (search edge 3) 4 [617] 617 innerTail110_0 best = best := by
  rw [innerTail110_0, scan, if_neg (by decide), innerCandidates110_0_checked]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_1
    (if 3 ≤ innerCandidates110_0.length then search edge 3 [617, 647] innerCandidates110_0 1264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan110_1

theorem rootBranch110 : search edge 4 [617] rootCandidates110 617 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [617] 617 innerTail110_0 best = best
  exact innerScan110_0

def rootCandidates109 : List Nat := [661, 673, 829, 883, 1303, 1753, 2089, 2851, 3037, 3643, 3769, 3943, 4021, 4129, 4153, 4339, 4363, 4801, 5581, 5743, 6379, 6547, 6673, 6679, 6733, 6967, 7069, 7177, 8191]
set_option maxRecDepth 4096 in
theorem rootCandidates109_checked : candidates edge 5 0 613 tail110 best.bound = rootCandidates109 := by
  unfold candidates
  have row : edge 613 = PrimePairCache.smallRow613.contains := by funext q; rfl
  rw [row]
  decide

def innerTail109_29 : List Nat := []
theorem innerScan109_29 : scan edge (search edge 3) 4 [613] 613 innerTail109_29 best = best := rfl

def innerTail109_28 : List Nat := 8191 :: innerTail109_29
def innerCandidates109_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_28_checked : candidates edge 4 613 8191 innerTail109_29 best.bound = innerCandidates109_28 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_28 : scan edge (search edge 3) 4 [613] 613 innerTail109_28 best = best := by
  rw [innerTail109_28, scan, if_neg (by decide), innerCandidates109_28_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_29
    (if 3 ≤ innerCandidates109_28.length then search edge 3 [613, 8191] innerCandidates109_28 8804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_29

def innerTail109_27 : List Nat := 7177 :: innerTail109_28
def innerCandidates109_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_27_checked : candidates edge 4 613 7177 innerTail109_28 best.bound = innerCandidates109_27 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_27 : scan edge (search edge 3) 4 [613] 613 innerTail109_27 best = best := by
  rw [innerTail109_27, scan, if_neg (by decide), innerCandidates109_27_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_28
    (if 3 ≤ innerCandidates109_27.length then search edge 3 [613, 7177] innerCandidates109_27 7790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_28

def innerTail109_26 : List Nat := 7069 :: innerTail109_27
def innerCandidates109_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_26_checked : candidates edge 4 613 7069 innerTail109_27 best.bound = innerCandidates109_26 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_26 : scan edge (search edge 3) 4 [613] 613 innerTail109_26 best = best := by
  rw [innerTail109_26, scan, if_neg (by decide), innerCandidates109_26_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_27
    (if 3 ≤ innerCandidates109_26.length then search edge 3 [613, 7069] innerCandidates109_26 7682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_27

def innerTail109_25 : List Nat := 6967 :: innerTail109_26
def innerCandidates109_25 : List Nat := [7177]
set_option maxRecDepth 4096 in
theorem innerCandidates109_25_checked : candidates edge 4 613 6967 innerTail109_26 best.bound = innerCandidates109_25 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_25 : scan edge (search edge 3) 4 [613] 613 innerTail109_25 best = best := by
  rw [innerTail109_25, scan, if_neg (by decide), innerCandidates109_25_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_26
    (if 3 ≤ innerCandidates109_25.length then search edge 3 [613, 6967] innerCandidates109_25 7580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_26

def innerTail109_24 : List Nat := 6733 :: innerTail109_25
def innerCandidates109_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_24_checked : candidates edge 4 613 6733 innerTail109_25 best.bound = innerCandidates109_24 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_24 : scan edge (search edge 3) 4 [613] 613 innerTail109_24 best = best := by
  rw [innerTail109_24, scan, if_neg (by decide), innerCandidates109_24_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_25
    (if 3 ≤ innerCandidates109_24.length then search edge 3 [613, 6733] innerCandidates109_24 7346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_25

def innerTail109_23 : List Nat := 6679 :: innerTail109_24
def innerCandidates109_23 : List Nat := [6967]
set_option maxRecDepth 4096 in
theorem innerCandidates109_23_checked : candidates edge 4 613 6679 innerTail109_24 best.bound = innerCandidates109_23 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_23 : scan edge (search edge 3) 4 [613] 613 innerTail109_23 best = best := by
  rw [innerTail109_23, scan, if_neg (by decide), innerCandidates109_23_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_24
    (if 3 ≤ innerCandidates109_23.length then search edge 3 [613, 6679] innerCandidates109_23 7292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_24

def innerTail109_22 : List Nat := 6673 :: innerTail109_23
def innerCandidates109_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_22_checked : candidates edge 4 613 6673 innerTail109_23 best.bound = innerCandidates109_22 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_22 : scan edge (search edge 3) 4 [613] 613 innerTail109_22 best = best := by
  rw [innerTail109_22, scan, if_neg (by decide), innerCandidates109_22_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_23
    (if 3 ≤ innerCandidates109_22.length then search edge 3 [613, 6673] innerCandidates109_22 7286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_23

def innerTail109_21 : List Nat := 6547 :: innerTail109_22
def innerCandidates109_21 : List Nat := [6967]
set_option maxRecDepth 4096 in
theorem innerCandidates109_21_checked : candidates edge 4 613 6547 innerTail109_22 best.bound = innerCandidates109_21 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_21 : scan edge (search edge 3) 4 [613] 613 innerTail109_21 best = best := by
  rw [innerTail109_21, scan, if_neg (by decide), innerCandidates109_21_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_22
    (if 3 ≤ innerCandidates109_21.length then search edge 3 [613, 6547] innerCandidates109_21 7160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_22

def innerTail109_20 : List Nat := 6379 :: innerTail109_21
def innerCandidates109_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_20_checked : candidates edge 4 613 6379 innerTail109_21 best.bound = innerCandidates109_20 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_20 : scan edge (search edge 3) 4 [613] 613 innerTail109_20 best = best := by
  rw [innerTail109_20, scan, if_neg (by decide), innerCandidates109_20_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_21
    (if 3 ≤ innerCandidates109_20.length then search edge 3 [613, 6379] innerCandidates109_20 6992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_21

def innerTail109_19 : List Nat := 5743 :: innerTail109_20
def innerCandidates109_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_19_checked : candidates edge 4 613 5743 innerTail109_20 best.bound = innerCandidates109_19 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_19 : scan edge (search edge 3) 4 [613] 613 innerTail109_19 best = best := by
  rw [innerTail109_19, scan, if_neg (by decide), innerCandidates109_19_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_20
    (if 3 ≤ innerCandidates109_19.length then search edge 3 [613, 5743] innerCandidates109_19 6356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_20

def innerTail109_18 : List Nat := 5581 :: innerTail109_19
def innerCandidates109_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_18_checked : candidates edge 4 613 5581 innerTail109_19 best.bound = innerCandidates109_18 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_18 : scan edge (search edge 3) 4 [613] 613 innerTail109_18 best = best := by
  rw [innerTail109_18, scan, if_neg (by decide), innerCandidates109_18_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_19
    (if 3 ≤ innerCandidates109_18.length then search edge 3 [613, 5581] innerCandidates109_18 6194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_19

def innerTail109_17 : List Nat := 4801 :: innerTail109_18
def innerCandidates109_17 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates109_17_checked : candidates edge 4 613 4801 innerTail109_18 best.bound = innerCandidates109_17 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_17 : scan edge (search edge 3) 4 [613] 613 innerTail109_17 best = best := by
  rw [innerTail109_17, scan, if_neg (by decide), innerCandidates109_17_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_18
    (if 3 ≤ innerCandidates109_17.length then search edge 3 [613, 4801] innerCandidates109_17 5414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_18

def innerTail109_16 : List Nat := 4363 :: innerTail109_17
def innerCandidates109_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_16_checked : candidates edge 4 613 4363 innerTail109_17 best.bound = innerCandidates109_16 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_16 : scan edge (search edge 3) 4 [613] 613 innerTail109_16 best = best := by
  rw [innerTail109_16, scan, if_neg (by decide), innerCandidates109_16_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_17
    (if 3 ≤ innerCandidates109_16.length then search edge 3 [613, 4363] innerCandidates109_16 4976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_17

def innerTail109_15 : List Nat := 4339 :: innerTail109_16
def innerCandidates109_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_15_checked : candidates edge 4 613 4339 innerTail109_16 best.bound = innerCandidates109_15 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_15 : scan edge (search edge 3) 4 [613] 613 innerTail109_15 best = best := by
  rw [innerTail109_15, scan, if_neg (by decide), innerCandidates109_15_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_16
    (if 3 ≤ innerCandidates109_15.length then search edge 3 [613, 4339] innerCandidates109_15 4952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_16

def innerTail109_14 : List Nat := 4153 :: innerTail109_15
def innerCandidates109_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_14_checked : candidates edge 4 613 4153 innerTail109_15 best.bound = innerCandidates109_14 := by
  unfold candidates
  have row : edge 4153 = SearchData.row4153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_14 : scan edge (search edge 3) 4 [613] 613 innerTail109_14 best = best := by
  rw [innerTail109_14, scan, if_neg (by decide), innerCandidates109_14_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_15
    (if 3 ≤ innerCandidates109_14.length then search edge 3 [613, 4153] innerCandidates109_14 4766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_15

def innerTail109_13 : List Nat := 4129 :: innerTail109_14
def innerCandidates109_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_13_checked : candidates edge 4 613 4129 innerTail109_14 best.bound = innerCandidates109_13 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_13 : scan edge (search edge 3) 4 [613] 613 innerTail109_13 best = best := by
  rw [innerTail109_13, scan, if_neg (by decide), innerCandidates109_13_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_14
    (if 3 ≤ innerCandidates109_13.length then search edge 3 [613, 4129] innerCandidates109_13 4742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_14

def innerTail109_12 : List Nat := 4021 :: innerTail109_13
def innerCandidates109_12 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates109_12_checked : candidates edge 4 613 4021 innerTail109_13 best.bound = innerCandidates109_12 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_12 : scan edge (search edge 3) 4 [613] 613 innerTail109_12 best = best := by
  rw [innerTail109_12, scan, if_neg (by decide), innerCandidates109_12_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_13
    (if 3 ≤ innerCandidates109_12.length then search edge 3 [613, 4021] innerCandidates109_12 4634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_13

def innerTail109_11 : List Nat := 3943 :: innerTail109_12
def innerCandidates109_11 : List Nat := [5581]
set_option maxRecDepth 4096 in
theorem innerCandidates109_11_checked : candidates edge 4 613 3943 innerTail109_12 best.bound = innerCandidates109_11 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_11 : scan edge (search edge 3) 4 [613] 613 innerTail109_11 best = best := by
  rw [innerTail109_11, scan, if_neg (by decide), innerCandidates109_11_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_12
    (if 3 ≤ innerCandidates109_11.length then search edge 3 [613, 3943] innerCandidates109_11 4556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_12

def innerTail109_10 : List Nat := 3769 :: innerTail109_11
def innerCandidates109_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_10_checked : candidates edge 4 613 3769 innerTail109_11 best.bound = innerCandidates109_10 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_10 : scan edge (search edge 3) 4 [613] 613 innerTail109_10 best = best := by
  rw [innerTail109_10, scan, if_neg (by decide), innerCandidates109_10_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_11
    (if 3 ≤ innerCandidates109_10.length then search edge 3 [613, 3769] innerCandidates109_10 4382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_11

def innerTail109_9 : List Nat := 3643 :: innerTail109_10
def innerCandidates109_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_9_checked : candidates edge 4 613 3643 innerTail109_10 best.bound = innerCandidates109_9 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_9 : scan edge (search edge 3) 4 [613] 613 innerTail109_9 best = best := by
  rw [innerTail109_9, scan, if_neg (by decide), innerCandidates109_9_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_10
    (if 3 ≤ innerCandidates109_9.length then search edge 3 [613, 3643] innerCandidates109_9 4256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_10

def innerTail109_8 : List Nat := 3037 :: innerTail109_9
def innerCandidates109_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_8_checked : candidates edge 4 613 3037 innerTail109_9 best.bound = innerCandidates109_8 := by
  unfold candidates
  have row : edge 3037 = SearchData.row3037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_8 : scan edge (search edge 3) 4 [613] 613 innerTail109_8 best = best := by
  rw [innerTail109_8, scan, if_neg (by decide), innerCandidates109_8_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_9
    (if 3 ≤ innerCandidates109_8.length then search edge 3 [613, 3037] innerCandidates109_8 3650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_9

def innerTail109_7 : List Nat := 2851 :: innerTail109_8
def innerCandidates109_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_7_checked : candidates edge 4 613 2851 innerTail109_8 best.bound = innerCandidates109_7 := by
  unfold candidates
  have row : edge 2851 = SearchData.row2851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_7 : scan edge (search edge 3) 4 [613] 613 innerTail109_7 best = best := by
  rw [innerTail109_7, scan, if_neg (by decide), innerCandidates109_7_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_8
    (if 3 ≤ innerCandidates109_7.length then search edge 3 [613, 2851] innerCandidates109_7 3464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_8

def innerTail109_6 : List Nat := 2089 :: innerTail109_7
def innerCandidates109_6 : List Nat := [3643]
set_option maxRecDepth 4096 in
theorem innerCandidates109_6_checked : candidates edge 4 613 2089 innerTail109_7 best.bound = innerCandidates109_6 := by
  unfold candidates
  have row : edge 2089 = PrimePairCache.smallRow2089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_6 : scan edge (search edge 3) 4 [613] 613 innerTail109_6 best = best := by
  rw [innerTail109_6, scan, if_neg (by decide), innerCandidates109_6_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_7
    (if 3 ≤ innerCandidates109_6.length then search edge 3 [613, 2089] innerCandidates109_6 2702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_7

def innerTail109_5 : List Nat := 1753 :: innerTail109_6
def innerCandidates109_5 : List Nat := [3643]
set_option maxRecDepth 4096 in
theorem innerCandidates109_5_checked : candidates edge 4 613 1753 innerTail109_6 best.bound = innerCandidates109_5 := by
  unfold candidates
  have row : edge 1753 = PrimePairCache.smallRow1753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_5 : scan edge (search edge 3) 4 [613] 613 innerTail109_5 best = best := by
  rw [innerTail109_5, scan, if_neg (by decide), innerCandidates109_5_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_6
    (if 3 ≤ innerCandidates109_5.length then search edge 3 [613, 1753] innerCandidates109_5 2366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_6

def innerTail109_4 : List Nat := 1303 :: innerTail109_5
def innerCandidates109_4 : List Nat := [4363]
set_option maxRecDepth 4096 in
theorem innerCandidates109_4_checked : candidates edge 4 613 1303 innerTail109_5 best.bound = innerCandidates109_4 := by
  unfold candidates
  have row : edge 1303 = PrimePairCache.smallRow1303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_4 : scan edge (search edge 3) 4 [613] 613 innerTail109_4 best = best := by
  rw [innerTail109_4, scan, if_neg (by decide), innerCandidates109_4_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_5
    (if 3 ≤ innerCandidates109_4.length then search edge 3 [613, 1303] innerCandidates109_4 1916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_5

def innerTail109_3 : List Nat := 883 :: innerTail109_4
def innerCandidates109_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_3_checked : candidates edge 4 613 883 innerTail109_4 best.bound = innerCandidates109_3 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_3 : scan edge (search edge 3) 4 [613] 613 innerTail109_3 best = best := by
  rw [innerTail109_3, scan, if_neg (by decide), innerCandidates109_3_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_4
    (if 3 ≤ innerCandidates109_3.length then search edge 3 [613, 883] innerCandidates109_3 1496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_4

def innerTail109_2 : List Nat := 829 :: innerTail109_3
def innerCandidates109_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates109_2_checked : candidates edge 4 613 829 innerTail109_3 best.bound = innerCandidates109_2 := by
  unfold candidates
  have row : edge 829 = PrimePairCache.smallRow829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_2 : scan edge (search edge 3) 4 [613] 613 innerTail109_2 best = best := by
  rw [innerTail109_2, scan, if_neg (by decide), innerCandidates109_2_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_3
    (if 3 ≤ innerCandidates109_2.length then search edge 3 [613, 829] innerCandidates109_2 1442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_3

def innerTail109_1 : List Nat := 673 :: innerTail109_2
def innerCandidates109_1 : List Nat := [7177]
set_option maxRecDepth 4096 in
theorem innerCandidates109_1_checked : candidates edge 4 613 673 innerTail109_2 best.bound = innerCandidates109_1 := by
  unfold candidates
  have row : edge 673 = PrimePairCache.smallRow673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_1 : scan edge (search edge 3) 4 [613] 613 innerTail109_1 best = best := by
  rw [innerTail109_1, scan, if_neg (by decide), innerCandidates109_1_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_2
    (if 3 ≤ innerCandidates109_1.length then search edge 3 [613, 673] innerCandidates109_1 1286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_2

def innerTail109_0 : List Nat := 661 :: innerTail109_1
def innerCandidates109_0 : List Nat := [883]
set_option maxRecDepth 4096 in
theorem innerCandidates109_0_checked : candidates edge 4 613 661 innerTail109_1 best.bound = innerCandidates109_0 := by
  unfold candidates
  have row : edge 661 = PrimePairCache.smallRow661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan109_0 : scan edge (search edge 3) 4 [613] 613 innerTail109_0 best = best := by
  rw [innerTail109_0, scan, if_neg (by decide), innerCandidates109_0_checked]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_1
    (if 3 ≤ innerCandidates109_0.length then search edge 3 [613, 661] innerCandidates109_0 1274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan109_1

theorem rootBranch109 : search edge 4 [613] rootCandidates109 613 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [613] 613 innerTail109_0 best = best
  exact innerScan109_0

end ProjectEulerSolutions.PrimePairUpper.Search
