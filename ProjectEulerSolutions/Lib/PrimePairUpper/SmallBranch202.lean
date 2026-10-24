import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates98 : List Nat := [577, 643, 661, 709, 769, 787, 823, 853, 1237, 1663, 1699, 2473, 3001, 3229, 3271, 4519, 4549, 5443, 5557, 6073, 6451, 6673, 7237, 7867, 8311]
set_option maxRecDepth 4096 in
theorem rootCandidates98_checked : candidates edge 5 0 547 tail99 best.bound = rootCandidates98 := by
  unfold candidates
  have row : edge 547 = PrimePairCache.smallRow547.contains := by funext q; rfl
  rw [row]
  decide

def innerTail98_25 : List Nat := []
theorem innerScan98_25 : scan edge (search edge 3) 4 [547] 547 innerTail98_25 best = best := rfl

def innerTail98_24 : List Nat := 8311 :: innerTail98_25
def innerCandidates98_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_24_checked : candidates edge 4 547 8311 innerTail98_25 best.bound = innerCandidates98_24 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_24 : scan edge (search edge 3) 4 [547] 547 innerTail98_24 best = best := by
  rw [innerTail98_24, scan, if_neg (by decide), innerCandidates98_24_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_25
    (if 3 ≤ innerCandidates98_24.length then search edge 3 [547, 8311] innerCandidates98_24 8858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_25

def innerTail98_23 : List Nat := 7867 :: innerTail98_24
def innerCandidates98_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_23_checked : candidates edge 4 547 7867 innerTail98_24 best.bound = innerCandidates98_23 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_23 : scan edge (search edge 3) 4 [547] 547 innerTail98_23 best = best := by
  rw [innerTail98_23, scan, if_neg (by decide), innerCandidates98_23_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_24
    (if 3 ≤ innerCandidates98_23.length then search edge 3 [547, 7867] innerCandidates98_23 8414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_24

def innerTail98_22 : List Nat := 7237 :: innerTail98_23
def innerCandidates98_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_22_checked : candidates edge 4 547 7237 innerTail98_23 best.bound = innerCandidates98_22 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_22 : scan edge (search edge 3) 4 [547] 547 innerTail98_22 best = best := by
  rw [innerTail98_22, scan, if_neg (by decide), innerCandidates98_22_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_23
    (if 3 ≤ innerCandidates98_22.length then search edge 3 [547, 7237] innerCandidates98_22 7784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_23

def innerTail98_21 : List Nat := 6673 :: innerTail98_22
def innerCandidates98_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_21_checked : candidates edge 4 547 6673 innerTail98_22 best.bound = innerCandidates98_21 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_21 : scan edge (search edge 3) 4 [547] 547 innerTail98_21 best = best := by
  rw [innerTail98_21, scan, if_neg (by decide), innerCandidates98_21_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_22
    (if 3 ≤ innerCandidates98_21.length then search edge 3 [547, 6673] innerCandidates98_21 7220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_22

def innerTail98_20 : List Nat := 6451 :: innerTail98_21
def innerCandidates98_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_20_checked : candidates edge 4 547 6451 innerTail98_21 best.bound = innerCandidates98_20 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_20 : scan edge (search edge 3) 4 [547] 547 innerTail98_20 best = best := by
  rw [innerTail98_20, scan, if_neg (by decide), innerCandidates98_20_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_21
    (if 3 ≤ innerCandidates98_20.length then search edge 3 [547, 6451] innerCandidates98_20 6998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_21

def innerTail98_19 : List Nat := 6073 :: innerTail98_20
def innerCandidates98_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_19_checked : candidates edge 4 547 6073 innerTail98_20 best.bound = innerCandidates98_19 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_19 : scan edge (search edge 3) 4 [547] 547 innerTail98_19 best = best := by
  rw [innerTail98_19, scan, if_neg (by decide), innerCandidates98_19_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_20
    (if 3 ≤ innerCandidates98_19.length then search edge 3 [547, 6073] innerCandidates98_19 6620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_20

def innerTail98_18 : List Nat := 5557 :: innerTail98_19
def innerCandidates98_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_18_checked : candidates edge 4 547 5557 innerTail98_19 best.bound = innerCandidates98_18 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_18 : scan edge (search edge 3) 4 [547] 547 innerTail98_18 best = best := by
  rw [innerTail98_18, scan, if_neg (by decide), innerCandidates98_18_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_19
    (if 3 ≤ innerCandidates98_18.length then search edge 3 [547, 5557] innerCandidates98_18 6104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_19

def innerTail98_17 : List Nat := 5443 :: innerTail98_18
def innerCandidates98_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_17_checked : candidates edge 4 547 5443 innerTail98_18 best.bound = innerCandidates98_17 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_17 : scan edge (search edge 3) 4 [547] 547 innerTail98_17 best = best := by
  rw [innerTail98_17, scan, if_neg (by decide), innerCandidates98_17_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_18
    (if 3 ≤ innerCandidates98_17.length then search edge 3 [547, 5443] innerCandidates98_17 5990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_18

def innerTail98_16 : List Nat := 4549 :: innerTail98_17
def innerCandidates98_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_16_checked : candidates edge 4 547 4549 innerTail98_17 best.bound = innerCandidates98_16 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_16 : scan edge (search edge 3) 4 [547] 547 innerTail98_16 best = best := by
  rw [innerTail98_16, scan, if_neg (by decide), innerCandidates98_16_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_17
    (if 3 ≤ innerCandidates98_16.length then search edge 3 [547, 4549] innerCandidates98_16 5096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_17

def innerTail98_15 : List Nat := 4519 :: innerTail98_16
def innerCandidates98_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_15_checked : candidates edge 4 547 4519 innerTail98_16 best.bound = innerCandidates98_15 := by
  unfold candidates
  have row : edge 4519 = SearchData.row4519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_15 : scan edge (search edge 3) 4 [547] 547 innerTail98_15 best = best := by
  rw [innerTail98_15, scan, if_neg (by decide), innerCandidates98_15_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_16
    (if 3 ≤ innerCandidates98_15.length then search edge 3 [547, 4519] innerCandidates98_15 5066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_16

def innerTail98_14 : List Nat := 3271 :: innerTail98_15
def innerCandidates98_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_14_checked : candidates edge 4 547 3271 innerTail98_15 best.bound = innerCandidates98_14 := by
  unfold candidates
  have row : edge 3271 = SearchData.row3271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_14 : scan edge (search edge 3) 4 [547] 547 innerTail98_14 best = best := by
  rw [innerTail98_14, scan, if_neg (by decide), innerCandidates98_14_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_15
    (if 3 ≤ innerCandidates98_14.length then search edge 3 [547, 3271] innerCandidates98_14 3818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_15

def innerTail98_13 : List Nat := 3229 :: innerTail98_14
def innerCandidates98_13 : List Nat := [6673]
set_option maxRecDepth 4096 in
theorem innerCandidates98_13_checked : candidates edge 4 547 3229 innerTail98_14 best.bound = innerCandidates98_13 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_13 : scan edge (search edge 3) 4 [547] 547 innerTail98_13 best = best := by
  rw [innerTail98_13, scan, if_neg (by decide), innerCandidates98_13_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_14
    (if 3 ≤ innerCandidates98_13.length then search edge 3 [547, 3229] innerCandidates98_13 3776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_14

def innerTail98_12 : List Nat := 3001 :: innerTail98_13
def innerCandidates98_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_12_checked : candidates edge 4 547 3001 innerTail98_13 best.bound = innerCandidates98_12 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_12 : scan edge (search edge 3) 4 [547] 547 innerTail98_12 best = best := by
  rw [innerTail98_12, scan, if_neg (by decide), innerCandidates98_12_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_13
    (if 3 ≤ innerCandidates98_12.length then search edge 3 [547, 3001] innerCandidates98_12 3548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_13

def innerTail98_11 : List Nat := 2473 :: innerTail98_12
def innerCandidates98_11 : List Nat := [3001]
set_option maxRecDepth 4096 in
theorem innerCandidates98_11_checked : candidates edge 4 547 2473 innerTail98_12 best.bound = innerCandidates98_11 := by
  unfold candidates
  have row : edge 2473 = PrimePairCache.smallRow2473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_11 : scan edge (search edge 3) 4 [547] 547 innerTail98_11 best = best := by
  rw [innerTail98_11, scan, if_neg (by decide), innerCandidates98_11_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_12
    (if 3 ≤ innerCandidates98_11.length then search edge 3 [547, 2473] innerCandidates98_11 3020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_12

def innerTail98_10 : List Nat := 1699 :: innerTail98_11
def innerCandidates98_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_10_checked : candidates edge 4 547 1699 innerTail98_11 best.bound = innerCandidates98_10 := by
  unfold candidates
  have row : edge 1699 = PrimePairCache.smallRow1699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_10 : scan edge (search edge 3) 4 [547] 547 innerTail98_10 best = best := by
  rw [innerTail98_10, scan, if_neg (by decide), innerCandidates98_10_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_11
    (if 3 ≤ innerCandidates98_10.length then search edge 3 [547, 1699] innerCandidates98_10 2246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_11

def innerTail98_9 : List Nat := 1663 :: innerTail98_10
def innerCandidates98_9 : List Nat := [3229]
set_option maxRecDepth 4096 in
theorem innerCandidates98_9_checked : candidates edge 4 547 1663 innerTail98_10 best.bound = innerCandidates98_9 := by
  unfold candidates
  have row : edge 1663 = PrimePairCache.smallRow1663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_9 : scan edge (search edge 3) 4 [547] 547 innerTail98_9 best = best := by
  rw [innerTail98_9, scan, if_neg (by decide), innerCandidates98_9_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_10
    (if 3 ≤ innerCandidates98_9.length then search edge 3 [547, 1663] innerCandidates98_9 2210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_10

def innerTail98_8 : List Nat := 1237 :: innerTail98_9
def innerCandidates98_8 : List Nat := [5557, 6673]
set_option maxRecDepth 4096 in
theorem innerCandidates98_8_checked : candidates edge 4 547 1237 innerTail98_9 best.bound = innerCandidates98_8 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_8 : scan edge (search edge 3) 4 [547] 547 innerTail98_8 best = best := by
  rw [innerTail98_8, scan, if_neg (by decide), innerCandidates98_8_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_9
    (if 3 ≤ innerCandidates98_8.length then search edge 3 [547, 1237] innerCandidates98_8 1784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_9

def innerTail98_7 : List Nat := 853 :: innerTail98_8
def innerCandidates98_7 : List Nat := [3001]
set_option maxRecDepth 4096 in
theorem innerCandidates98_7_checked : candidates edge 4 547 853 innerTail98_8 best.bound = innerCandidates98_7 := by
  unfold candidates
  have row : edge 853 = PrimePairCache.smallRow853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_7 : scan edge (search edge 3) 4 [547] 547 innerTail98_7 best = best := by
  rw [innerTail98_7, scan, if_neg (by decide), innerCandidates98_7_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_8
    (if 3 ≤ innerCandidates98_7.length then search edge 3 [547, 853] innerCandidates98_7 1400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_8

def innerTail98_6 : List Nat := 823 :: innerTail98_7
def innerCandidates98_6 : List Nat := [7237]
set_option maxRecDepth 4096 in
theorem innerCandidates98_6_checked : candidates edge 4 547 823 innerTail98_7 best.bound = innerCandidates98_6 := by
  unfold candidates
  have row : edge 823 = PrimePairCache.smallRow823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_6 : scan edge (search edge 3) 4 [547] 547 innerTail98_6 best = best := by
  rw [innerTail98_6, scan, if_neg (by decide), innerCandidates98_6_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_7
    (if 3 ≤ innerCandidates98_6.length then search edge 3 [547, 823] innerCandidates98_6 1370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_7

def innerTail98_5 : List Nat := 787 :: innerTail98_6
def innerCandidates98_5 : List Nat := [823]
set_option maxRecDepth 4096 in
theorem innerCandidates98_5_checked : candidates edge 4 547 787 innerTail98_6 best.bound = innerCandidates98_5 := by
  unfold candidates
  have row : edge 787 = PrimePairCache.smallRow787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_5 : scan edge (search edge 3) 4 [547] 547 innerTail98_5 best = best := by
  rw [innerTail98_5, scan, if_neg (by decide), innerCandidates98_5_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_6
    (if 3 ≤ innerCandidates98_5.length then search edge 3 [547, 787] innerCandidates98_5 1334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_6

def innerTail98_4 : List Nat := 769 :: innerTail98_5
def innerCandidates98_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_4_checked : candidates edge 4 547 769 innerTail98_5 best.bound = innerCandidates98_4 := by
  unfold candidates
  have row : edge 769 = PrimePairCache.smallRow769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_4 : scan edge (search edge 3) 4 [547] 547 innerTail98_4 best = best := by
  rw [innerTail98_4, scan, if_neg (by decide), innerCandidates98_4_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_5
    (if 3 ≤ innerCandidates98_4.length then search edge 3 [547, 769] innerCandidates98_4 1316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_5

def innerTail98_3 : List Nat := 709 :: innerTail98_4
def innerCandidates98_3 : List Nat := [823, 1663, 4519]
set_option maxRecDepth 4096 in
theorem innerCandidates98_3_checked : candidates edge 4 547 709 innerTail98_4 best.bound = innerCandidates98_3 := by
  unfold candidates
  have row : edge 709 = PrimePairCache.smallRow709.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild98_3 : search edge 3 [547, 709] innerCandidates98_3 1256 best = best := by decide

theorem innerScan98_3 : scan edge (search edge 3) 4 [547] 547 innerTail98_3 best = best := by
  rw [innerTail98_3, scan, if_neg (by decide), innerCandidates98_3_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_4
    (if 3 ≤ innerCandidates98_3.length then search edge 3 [547, 709] innerCandidates98_3 1256 best else best) = best
  rw [if_pos (by decide), innerChild98_3]
  exact innerScan98_4

def innerTail98_2 : List Nat := 661 :: innerTail98_3
def innerCandidates98_2 : List Nat := [769, 1237]
set_option maxRecDepth 4096 in
theorem innerCandidates98_2_checked : candidates edge 4 547 661 innerTail98_3 best.bound = innerCandidates98_2 := by
  unfold candidates
  have row : edge 661 = PrimePairCache.smallRow661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_2 : scan edge (search edge 3) 4 [547] 547 innerTail98_2 best = best := by
  rw [innerTail98_2, scan, if_neg (by decide), innerCandidates98_2_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_3
    (if 3 ≤ innerCandidates98_2.length then search edge 3 [547, 661] innerCandidates98_2 1208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_3

def innerTail98_1 : List Nat := 643 :: innerTail98_2
def innerCandidates98_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_1_checked : candidates edge 4 547 643 innerTail98_2 best.bound = innerCandidates98_1 := by
  unfold candidates
  have row : edge 643 = PrimePairCache.smallRow643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_1 : scan edge (search edge 3) 4 [547] 547 innerTail98_1 best = best := by
  rw [innerTail98_1, scan, if_neg (by decide), innerCandidates98_1_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_2
    (if 3 ≤ innerCandidates98_1.length then search edge 3 [547, 643] innerCandidates98_1 1190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_2

def innerTail98_0 : List Nat := 577 :: innerTail98_1
def innerCandidates98_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates98_0_checked : candidates edge 4 547 577 innerTail98_1 best.bound = innerCandidates98_0 := by
  unfold candidates
  have row : edge 577 = PrimePairCache.smallRow577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan98_0 : scan edge (search edge 3) 4 [547] 547 innerTail98_0 best = best := by
  rw [innerTail98_0, scan, if_neg (by decide), innerCandidates98_0_checked]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_1
    (if 3 ≤ innerCandidates98_0.length then search edge 3 [547, 577] innerCandidates98_0 1124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan98_1

theorem rootBranch98 : search edge 4 [547] rootCandidates98 547 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [547] 547 innerTail98_0 best = best
  exact innerScan98_0

def rootCandidates97 : List Nat := [571, 661, 727, 991, 1033, 1093, 1621, 2467, 2791, 2887, 3301, 3631, 4159, 4231, 4261, 5101, 5479, 6709, 6991, 7369, 8167, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates97_checked : candidates edge 5 0 541 tail98 best.bound = rootCandidates97 := by
  unfold candidates
  have row : edge 541 = PrimePairCache.smallRow541.contains := by funext q; rfl
  rw [row]
  decide

def innerTail97_22 : List Nat := []
theorem innerScan97_22 : scan edge (search edge 3) 4 [541] 541 innerTail97_22 best = best := rfl

def innerTail97_21 : List Nat := 8233 :: innerTail97_22
def innerCandidates97_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_21_checked : candidates edge 4 541 8233 innerTail97_22 best.bound = innerCandidates97_21 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_21 : scan edge (search edge 3) 4 [541] 541 innerTail97_21 best = best := by
  rw [innerTail97_21, scan, if_neg (by decide), innerCandidates97_21_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_22
    (if 3 ≤ innerCandidates97_21.length then search edge 3 [541, 8233] innerCandidates97_21 8774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_22

def innerTail97_20 : List Nat := 8167 :: innerTail97_21
def innerCandidates97_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_20_checked : candidates edge 4 541 8167 innerTail97_21 best.bound = innerCandidates97_20 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_20 : scan edge (search edge 3) 4 [541] 541 innerTail97_20 best = best := by
  rw [innerTail97_20, scan, if_neg (by decide), innerCandidates97_20_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_21
    (if 3 ≤ innerCandidates97_20.length then search edge 3 [541, 8167] innerCandidates97_20 8708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_21

def innerTail97_19 : List Nat := 7369 :: innerTail97_20
def innerCandidates97_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_19_checked : candidates edge 4 541 7369 innerTail97_20 best.bound = innerCandidates97_19 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_19 : scan edge (search edge 3) 4 [541] 541 innerTail97_19 best = best := by
  rw [innerTail97_19, scan, if_neg (by decide), innerCandidates97_19_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_20
    (if 3 ≤ innerCandidates97_19.length then search edge 3 [541, 7369] innerCandidates97_19 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_20

def innerTail97_18 : List Nat := 6991 :: innerTail97_19
def innerCandidates97_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_18_checked : candidates edge 4 541 6991 innerTail97_19 best.bound = innerCandidates97_18 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_18 : scan edge (search edge 3) 4 [541] 541 innerTail97_18 best = best := by
  rw [innerTail97_18, scan, if_neg (by decide), innerCandidates97_18_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_19
    (if 3 ≤ innerCandidates97_18.length then search edge 3 [541, 6991] innerCandidates97_18 7532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_19

def innerTail97_17 : List Nat := 6709 :: innerTail97_18
def innerCandidates97_17 : List Nat := [7369]
set_option maxRecDepth 4096 in
theorem innerCandidates97_17_checked : candidates edge 4 541 6709 innerTail97_18 best.bound = innerCandidates97_17 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_17 : scan edge (search edge 3) 4 [541] 541 innerTail97_17 best = best := by
  rw [innerTail97_17, scan, if_neg (by decide), innerCandidates97_17_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_18
    (if 3 ≤ innerCandidates97_17.length then search edge 3 [541, 6709] innerCandidates97_17 7250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_18

def innerTail97_16 : List Nat := 5479 :: innerTail97_17
def innerCandidates97_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_16_checked : candidates edge 4 541 5479 innerTail97_17 best.bound = innerCandidates97_16 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_16 : scan edge (search edge 3) 4 [541] 541 innerTail97_16 best = best := by
  rw [innerTail97_16, scan, if_neg (by decide), innerCandidates97_16_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_17
    (if 3 ≤ innerCandidates97_16.length then search edge 3 [541, 5479] innerCandidates97_16 6020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_17

def innerTail97_15 : List Nat := 5101 :: innerTail97_16
def innerCandidates97_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_15_checked : candidates edge 4 541 5101 innerTail97_16 best.bound = innerCandidates97_15 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_15 : scan edge (search edge 3) 4 [541] 541 innerTail97_15 best = best := by
  rw [innerTail97_15, scan, if_neg (by decide), innerCandidates97_15_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_16
    (if 3 ≤ innerCandidates97_15.length then search edge 3 [541, 5101] innerCandidates97_15 5642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_16

def innerTail97_14 : List Nat := 4261 :: innerTail97_15
def innerCandidates97_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_14_checked : candidates edge 4 541 4261 innerTail97_15 best.bound = innerCandidates97_14 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_14 : scan edge (search edge 3) 4 [541] 541 innerTail97_14 best = best := by
  rw [innerTail97_14, scan, if_neg (by decide), innerCandidates97_14_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_15
    (if 3 ≤ innerCandidates97_14.length then search edge 3 [541, 4261] innerCandidates97_14 4802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_15

def innerTail97_13 : List Nat := 4231 :: innerTail97_14
def innerCandidates97_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_13_checked : candidates edge 4 541 4231 innerTail97_14 best.bound = innerCandidates97_13 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_13 : scan edge (search edge 3) 4 [541] 541 innerTail97_13 best = best := by
  rw [innerTail97_13, scan, if_neg (by decide), innerCandidates97_13_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_14
    (if 3 ≤ innerCandidates97_13.length then search edge 3 [541, 4231] innerCandidates97_13 4772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_14

def innerTail97_12 : List Nat := 4159 :: innerTail97_13
def innerCandidates97_12 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates97_12_checked : candidates edge 4 541 4159 innerTail97_13 best.bound = innerCandidates97_12 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_12 : scan edge (search edge 3) 4 [541] 541 innerTail97_12 best = best := by
  rw [innerTail97_12, scan, if_neg (by decide), innerCandidates97_12_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_13
    (if 3 ≤ innerCandidates97_12.length then search edge 3 [541, 4159] innerCandidates97_12 4700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_13

def innerTail97_11 : List Nat := 3631 :: innerTail97_12
def innerCandidates97_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_11_checked : candidates edge 4 541 3631 innerTail97_12 best.bound = innerCandidates97_11 := by
  unfold candidates
  have row : edge 3631 = SearchData.row3631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_11 : scan edge (search edge 3) 4 [541] 541 innerTail97_11 best = best := by
  rw [innerTail97_11, scan, if_neg (by decide), innerCandidates97_11_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_12
    (if 3 ≤ innerCandidates97_11.length then search edge 3 [541, 3631] innerCandidates97_11 4172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_12

def innerTail97_10 : List Nat := 3301 :: innerTail97_11
def innerCandidates97_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_10_checked : candidates edge 4 541 3301 innerTail97_11 best.bound = innerCandidates97_10 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_10 : scan edge (search edge 3) 4 [541] 541 innerTail97_10 best = best := by
  rw [innerTail97_10, scan, if_neg (by decide), innerCandidates97_10_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_11
    (if 3 ≤ innerCandidates97_10.length then search edge 3 [541, 3301] innerCandidates97_10 3842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_11

def innerTail97_9 : List Nat := 2887 :: innerTail97_10
def innerCandidates97_9 : List Nat := [4231, 6709]
set_option maxRecDepth 4096 in
theorem innerCandidates97_9_checked : candidates edge 4 541 2887 innerTail97_10 best.bound = innerCandidates97_9 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_9 : scan edge (search edge 3) 4 [541] 541 innerTail97_9 best = best := by
  rw [innerTail97_9, scan, if_neg (by decide), innerCandidates97_9_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_10
    (if 3 ≤ innerCandidates97_9.length then search edge 3 [541, 2887] innerCandidates97_9 3428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_10

def innerTail97_8 : List Nat := 2791 :: innerTail97_9
def innerCandidates97_8 : List Nat := [7369]
set_option maxRecDepth 4096 in
theorem innerCandidates97_8_checked : candidates edge 4 541 2791 innerTail97_9 best.bound = innerCandidates97_8 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_8 : scan edge (search edge 3) 4 [541] 541 innerTail97_8 best = best := by
  rw [innerTail97_8, scan, if_neg (by decide), innerCandidates97_8_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_9
    (if 3 ≤ innerCandidates97_8.length then search edge 3 [541, 2791] innerCandidates97_8 3332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_9

def innerTail97_7 : List Nat := 2467 :: innerTail97_8
def innerCandidates97_7 : List Nat := [8167]
set_option maxRecDepth 4096 in
theorem innerCandidates97_7_checked : candidates edge 4 541 2467 innerTail97_8 best.bound = innerCandidates97_7 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_7 : scan edge (search edge 3) 4 [541] 541 innerTail97_7 best = best := by
  rw [innerTail97_7, scan, if_neg (by decide), innerCandidates97_7_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_8
    (if 3 ≤ innerCandidates97_7.length then search edge 3 [541, 2467] innerCandidates97_7 3008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_8

def innerTail97_6 : List Nat := 1621 :: innerTail97_7
def innerCandidates97_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_6_checked : candidates edge 4 541 1621 innerTail97_7 best.bound = innerCandidates97_6 := by
  unfold candidates
  have row : edge 1621 = PrimePairCache.smallRow1621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_6 : scan edge (search edge 3) 4 [541] 541 innerTail97_6 best = best := by
  rw [innerTail97_6, scan, if_neg (by decide), innerCandidates97_6_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_7
    (if 3 ≤ innerCandidates97_6.length then search edge 3 [541, 1621] innerCandidates97_6 2162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_7

def innerTail97_5 : List Nat := 1093 :: innerTail97_6
def innerCandidates97_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_5_checked : candidates edge 4 541 1093 innerTail97_6 best.bound = innerCandidates97_5 := by
  unfold candidates
  have row : edge 1093 = PrimePairCache.smallRow1093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_5 : scan edge (search edge 3) 4 [541] 541 innerTail97_5 best = best := by
  rw [innerTail97_5, scan, if_neg (by decide), innerCandidates97_5_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_6
    (if 3 ≤ innerCandidates97_5.length then search edge 3 [541, 1093] innerCandidates97_5 1634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_6

def innerTail97_4 : List Nat := 1033 :: innerTail97_5
def innerCandidates97_4 : List Nat := [4231]
set_option maxRecDepth 4096 in
theorem innerCandidates97_4_checked : candidates edge 4 541 1033 innerTail97_5 best.bound = innerCandidates97_4 := by
  unfold candidates
  have row : edge 1033 = PrimePairCache.smallRow1033.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_4 : scan edge (search edge 3) 4 [541] 541 innerTail97_4 best = best := by
  rw [innerTail97_4, scan, if_neg (by decide), innerCandidates97_4_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_5
    (if 3 ≤ innerCandidates97_4.length then search edge 3 [541, 1033] innerCandidates97_4 1574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_5

def innerTail97_3 : List Nat := 991 :: innerTail97_4
def innerCandidates97_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_3_checked : candidates edge 4 541 991 innerTail97_4 best.bound = innerCandidates97_3 := by
  unfold candidates
  have row : edge 991 = PrimePairCache.smallRow991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_3 : scan edge (search edge 3) 4 [541] 541 innerTail97_3 best = best := by
  rw [innerTail97_3, scan, if_neg (by decide), innerCandidates97_3_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_4
    (if 3 ≤ innerCandidates97_3.length then search edge 3 [541, 991] innerCandidates97_3 1532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_4

def innerTail97_2 : List Nat := 727 :: innerTail97_3
def innerCandidates97_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates97_2_checked : candidates edge 4 541 727 innerTail97_3 best.bound = innerCandidates97_2 := by
  unfold candidates
  have row : edge 727 = PrimePairCache.smallRow727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_2 : scan edge (search edge 3) 4 [541] 541 innerTail97_2 best = best := by
  rw [innerTail97_2, scan, if_neg (by decide), innerCandidates97_2_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_3
    (if 3 ≤ innerCandidates97_2.length then search edge 3 [541, 727] innerCandidates97_2 1268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_3

def innerTail97_1 : List Nat := 661 :: innerTail97_2
def innerCandidates97_1 : List Nat := [2791]
set_option maxRecDepth 4096 in
theorem innerCandidates97_1_checked : candidates edge 4 541 661 innerTail97_2 best.bound = innerCandidates97_1 := by
  unfold candidates
  have row : edge 661 = PrimePairCache.smallRow661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_1 : scan edge (search edge 3) 4 [541] 541 innerTail97_1 best = best := by
  rw [innerTail97_1, scan, if_neg (by decide), innerCandidates97_1_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_2
    (if 3 ≤ innerCandidates97_1.length then search edge 3 [541, 661] innerCandidates97_1 1202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_2

def innerTail97_0 : List Nat := 571 :: innerTail97_1
def innerCandidates97_0 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates97_0_checked : candidates edge 4 541 571 innerTail97_1 best.bound = innerCandidates97_0 := by
  unfold candidates
  have row : edge 571 = PrimePairCache.smallRow571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan97_0 : scan edge (search edge 3) 4 [541] 541 innerTail97_0 best = best := by
  rw [innerTail97_0, scan, if_neg (by decide), innerCandidates97_0_checked]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_1
    (if 3 ≤ innerCandidates97_0.length then search edge 3 [541, 571] innerCandidates97_0 1112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan97_1

theorem rootBranch97 : search edge 4 [541] rootCandidates97 541 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [541] 541 innerTail97_0 best = best
  exact innerScan97_0

end ProjectEulerSolutions.PrimePairUpper.Search
