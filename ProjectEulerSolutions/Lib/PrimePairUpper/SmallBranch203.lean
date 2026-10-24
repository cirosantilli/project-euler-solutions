import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates96 : List Nat := [541, 577, 1129, 1297, 1663, 1987, 2089, 2113, 2671, 2677, 3307, 3469, 3499, 3919, 4591, 4909, 4987, 5233, 5851, 6367, 6967, 7237, 7753]
set_option maxRecDepth 4096 in
theorem rootCandidates96_checked : candidates edge 5 0 523 tail97 best.bound = rootCandidates96 := by
  unfold candidates
  have row : edge 523 = PrimePairCache.smallRow523.contains := by funext q; rfl
  rw [row]
  decide

def innerTail96_23 : List Nat := []
theorem innerScan96_23 : scan edge (search edge 3) 4 [523] 523 innerTail96_23 best = best := rfl

def innerTail96_22 : List Nat := 7753 :: innerTail96_23
def innerCandidates96_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_22_checked : candidates edge 4 523 7753 innerTail96_23 best.bound = innerCandidates96_22 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_22 : scan edge (search edge 3) 4 [523] 523 innerTail96_22 best = best := by
  rw [innerTail96_22, scan, if_neg (by decide), innerCandidates96_22_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_23
    (if 3 ≤ innerCandidates96_22.length then search edge 3 [523, 7753] innerCandidates96_22 8276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_23

def innerTail96_21 : List Nat := 7237 :: innerTail96_22
def innerCandidates96_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_21_checked : candidates edge 4 523 7237 innerTail96_22 best.bound = innerCandidates96_21 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_21 : scan edge (search edge 3) 4 [523] 523 innerTail96_21 best = best := by
  rw [innerTail96_21, scan, if_neg (by decide), innerCandidates96_21_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_22
    (if 3 ≤ innerCandidates96_21.length then search edge 3 [523, 7237] innerCandidates96_21 7760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_22

def innerTail96_20 : List Nat := 6967 :: innerTail96_21
def innerCandidates96_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_20_checked : candidates edge 4 523 6967 innerTail96_21 best.bound = innerCandidates96_20 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_20 : scan edge (search edge 3) 4 [523] 523 innerTail96_20 best = best := by
  rw [innerTail96_20, scan, if_neg (by decide), innerCandidates96_20_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_21
    (if 3 ≤ innerCandidates96_20.length then search edge 3 [523, 6967] innerCandidates96_20 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_21

def innerTail96_19 : List Nat := 6367 :: innerTail96_20
def innerCandidates96_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_19_checked : candidates edge 4 523 6367 innerTail96_20 best.bound = innerCandidates96_19 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_19 : scan edge (search edge 3) 4 [523] 523 innerTail96_19 best = best := by
  rw [innerTail96_19, scan, if_neg (by decide), innerCandidates96_19_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_20
    (if 3 ≤ innerCandidates96_19.length then search edge 3 [523, 6367] innerCandidates96_19 6890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_20

def innerTail96_18 : List Nat := 5851 :: innerTail96_19
def innerCandidates96_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_18_checked : candidates edge 4 523 5851 innerTail96_19 best.bound = innerCandidates96_18 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_18 : scan edge (search edge 3) 4 [523] 523 innerTail96_18 best = best := by
  rw [innerTail96_18, scan, if_neg (by decide), innerCandidates96_18_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_19
    (if 3 ≤ innerCandidates96_18.length then search edge 3 [523, 5851] innerCandidates96_18 6374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_19

def innerTail96_17 : List Nat := 5233 :: innerTail96_18
def innerCandidates96_17 : List Nat := [6367]
set_option maxRecDepth 4096 in
theorem innerCandidates96_17_checked : candidates edge 4 523 5233 innerTail96_18 best.bound = innerCandidates96_17 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_17 : scan edge (search edge 3) 4 [523] 523 innerTail96_17 best = best := by
  rw [innerTail96_17, scan, if_neg (by decide), innerCandidates96_17_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_18
    (if 3 ≤ innerCandidates96_17.length then search edge 3 [523, 5233] innerCandidates96_17 5756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_18

def innerTail96_16 : List Nat := 4987 :: innerTail96_17
def innerCandidates96_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_16_checked : candidates edge 4 523 4987 innerTail96_17 best.bound = innerCandidates96_16 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_16 : scan edge (search edge 3) 4 [523] 523 innerTail96_16 best = best := by
  rw [innerTail96_16, scan, if_neg (by decide), innerCandidates96_16_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_17
    (if 3 ≤ innerCandidates96_16.length then search edge 3 [523, 4987] innerCandidates96_16 5510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_17

def innerTail96_15 : List Nat := 4909 :: innerTail96_16
def innerCandidates96_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_15_checked : candidates edge 4 523 4909 innerTail96_16 best.bound = innerCandidates96_15 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_15 : scan edge (search edge 3) 4 [523] 523 innerTail96_15 best = best := by
  rw [innerTail96_15, scan, if_neg (by decide), innerCandidates96_15_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_16
    (if 3 ≤ innerCandidates96_15.length then search edge 3 [523, 4909] innerCandidates96_15 5432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_16

def innerTail96_14 : List Nat := 4591 :: innerTail96_15
def innerCandidates96_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_14_checked : candidates edge 4 523 4591 innerTail96_15 best.bound = innerCandidates96_14 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_14 : scan edge (search edge 3) 4 [523] 523 innerTail96_14 best = best := by
  rw [innerTail96_14, scan, if_neg (by decide), innerCandidates96_14_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_15
    (if 3 ≤ innerCandidates96_14.length then search edge 3 [523, 4591] innerCandidates96_14 5114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_15

def innerTail96_13 : List Nat := 3919 :: innerTail96_14
def innerCandidates96_13 : List Nat := [4591]
set_option maxRecDepth 4096 in
theorem innerCandidates96_13_checked : candidates edge 4 523 3919 innerTail96_14 best.bound = innerCandidates96_13 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_13 : scan edge (search edge 3) 4 [523] 523 innerTail96_13 best = best := by
  rw [innerTail96_13, scan, if_neg (by decide), innerCandidates96_13_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_14
    (if 3 ≤ innerCandidates96_13.length then search edge 3 [523, 3919] innerCandidates96_13 4442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_14

def innerTail96_12 : List Nat := 3499 :: innerTail96_13
def innerCandidates96_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_12_checked : candidates edge 4 523 3499 innerTail96_13 best.bound = innerCandidates96_12 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_12 : scan edge (search edge 3) 4 [523] 523 innerTail96_12 best = best := by
  rw [innerTail96_12, scan, if_neg (by decide), innerCandidates96_12_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_13
    (if 3 ≤ innerCandidates96_12.length then search edge 3 [523, 3499] innerCandidates96_12 4022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_13

def innerTail96_11 : List Nat := 3469 :: innerTail96_12
def innerCandidates96_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_11_checked : candidates edge 4 523 3469 innerTail96_12 best.bound = innerCandidates96_11 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_11 : scan edge (search edge 3) 4 [523] 523 innerTail96_11 best = best := by
  rw [innerTail96_11, scan, if_neg (by decide), innerCandidates96_11_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_12
    (if 3 ≤ innerCandidates96_11.length then search edge 3 [523, 3469] innerCandidates96_11 3992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_12

def innerTail96_10 : List Nat := 3307 :: innerTail96_11
def innerCandidates96_10 : List Nat := [6367]
set_option maxRecDepth 4096 in
theorem innerCandidates96_10_checked : candidates edge 4 523 3307 innerTail96_11 best.bound = innerCandidates96_10 := by
  unfold candidates
  have row : edge 3307 = SearchData.row3307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_10 : scan edge (search edge 3) 4 [523] 523 innerTail96_10 best = best := by
  rw [innerTail96_10, scan, if_neg (by decide), innerCandidates96_10_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_11
    (if 3 ≤ innerCandidates96_10.length then search edge 3 [523, 3307] innerCandidates96_10 3830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_11

def innerTail96_9 : List Nat := 2677 :: innerTail96_10
def innerCandidates96_9 : List Nat := [3469, 5851]
set_option maxRecDepth 4096 in
theorem innerCandidates96_9_checked : candidates edge 4 523 2677 innerTail96_10 best.bound = innerCandidates96_9 := by
  unfold candidates
  have row : edge 2677 = SearchData.row2677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_9 : scan edge (search edge 3) 4 [523] 523 innerTail96_9 best = best := by
  rw [innerTail96_9, scan, if_neg (by decide), innerCandidates96_9_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_10
    (if 3 ≤ innerCandidates96_9.length then search edge 3 [523, 2677] innerCandidates96_9 3200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_10

def innerTail96_8 : List Nat := 2671 :: innerTail96_9
def innerCandidates96_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_8_checked : candidates edge 4 523 2671 innerTail96_9 best.bound = innerCandidates96_8 := by
  unfold candidates
  have row : edge 2671 = SearchData.row2671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_8 : scan edge (search edge 3) 4 [523] 523 innerTail96_8 best = best := by
  rw [innerTail96_8, scan, if_neg (by decide), innerCandidates96_8_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_9
    (if 3 ≤ innerCandidates96_8.length then search edge 3 [523, 2671] innerCandidates96_8 3194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_9

def innerTail96_7 : List Nat := 2113 :: innerTail96_8
def innerCandidates96_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_7_checked : candidates edge 4 523 2113 innerTail96_8 best.bound = innerCandidates96_7 := by
  unfold candidates
  have row : edge 2113 = PrimePairCache.smallRow2113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_7 : scan edge (search edge 3) 4 [523] 523 innerTail96_7 best = best := by
  rw [innerTail96_7, scan, if_neg (by decide), innerCandidates96_7_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_8
    (if 3 ≤ innerCandidates96_7.length then search edge 3 [523, 2113] innerCandidates96_7 2636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_8

def innerTail96_6 : List Nat := 2089 :: innerTail96_7
def innerCandidates96_6 : List Nat := [7237]
set_option maxRecDepth 4096 in
theorem innerCandidates96_6_checked : candidates edge 4 523 2089 innerTail96_7 best.bound = innerCandidates96_6 := by
  unfold candidates
  have row : edge 2089 = PrimePairCache.smallRow2089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_6 : scan edge (search edge 3) 4 [523] 523 innerTail96_6 best = best := by
  rw [innerTail96_6, scan, if_neg (by decide), innerCandidates96_6_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_7
    (if 3 ≤ innerCandidates96_6.length then search edge 3 [523, 2089] innerCandidates96_6 2612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_7

def innerTail96_5 : List Nat := 1987 :: innerTail96_6
def innerCandidates96_5 : List Nat := [3499, 7237]
set_option maxRecDepth 4096 in
theorem innerCandidates96_5_checked : candidates edge 4 523 1987 innerTail96_6 best.bound = innerCandidates96_5 := by
  unfold candidates
  have row : edge 1987 = PrimePairCache.smallRow1987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_5 : scan edge (search edge 3) 4 [523] 523 innerTail96_5 best = best := by
  rw [innerTail96_5, scan, if_neg (by decide), innerCandidates96_5_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_6
    (if 3 ≤ innerCandidates96_5.length then search edge 3 [523, 1987] innerCandidates96_5 2510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_6

def innerTail96_4 : List Nat := 1663 :: innerTail96_5
def innerCandidates96_4 : List Nat := [4987]
set_option maxRecDepth 4096 in
theorem innerCandidates96_4_checked : candidates edge 4 523 1663 innerTail96_5 best.bound = innerCandidates96_4 := by
  unfold candidates
  have row : edge 1663 = PrimePairCache.smallRow1663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_4 : scan edge (search edge 3) 4 [523] 523 innerTail96_4 best = best := by
  rw [innerTail96_4, scan, if_neg (by decide), innerCandidates96_4_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_5
    (if 3 ≤ innerCandidates96_4.length then search edge 3 [523, 1663] innerCandidates96_4 2186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_5

def innerTail96_3 : List Nat := 1297 :: innerTail96_4
def innerCandidates96_3 : List Nat := [2671]
set_option maxRecDepth 4096 in
theorem innerCandidates96_3_checked : candidates edge 4 523 1297 innerTail96_4 best.bound = innerCandidates96_3 := by
  unfold candidates
  have row : edge 1297 = PrimePairCache.smallRow1297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_3 : scan edge (search edge 3) 4 [523] 523 innerTail96_3 best = best := by
  rw [innerTail96_3, scan, if_neg (by decide), innerCandidates96_3_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_4
    (if 3 ≤ innerCandidates96_3.length then search edge 3 [523, 1297] innerCandidates96_3 1820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_4

def innerTail96_2 : List Nat := 1129 :: innerTail96_3
def innerCandidates96_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_2_checked : candidates edge 4 523 1129 innerTail96_3 best.bound = innerCandidates96_2 := by
  unfold candidates
  have row : edge 1129 = PrimePairCache.smallRow1129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_2 : scan edge (search edge 3) 4 [523] 523 innerTail96_2 best = best := by
  rw [innerTail96_2, scan, if_neg (by decide), innerCandidates96_2_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_3
    (if 3 ≤ innerCandidates96_2.length then search edge 3 [523, 1129] innerCandidates96_2 1652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_3

def innerTail96_1 : List Nat := 577 :: innerTail96_2
def innerCandidates96_1 : List Nat := [3919]
set_option maxRecDepth 4096 in
theorem innerCandidates96_1_checked : candidates edge 4 523 577 innerTail96_2 best.bound = innerCandidates96_1 := by
  unfold candidates
  have row : edge 577 = PrimePairCache.smallRow577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_1 : scan edge (search edge 3) 4 [523] 523 innerTail96_1 best = best := by
  rw [innerTail96_1, scan, if_neg (by decide), innerCandidates96_1_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_2
    (if 3 ≤ innerCandidates96_1.length then search edge 3 [523, 577] innerCandidates96_1 1100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_2

def innerTail96_0 : List Nat := 541 :: innerTail96_1
def innerCandidates96_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates96_0_checked : candidates edge 4 523 541 innerTail96_1 best.bound = innerCandidates96_0 := by
  unfold candidates
  have row : edge 541 = PrimePairCache.smallRow541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan96_0 : scan edge (search edge 3) 4 [523] 523 innerTail96_0 best = best := by
  rw [innerTail96_0, scan, if_neg (by decide), innerCandidates96_0_checked]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_1
    (if 3 ≤ innerCandidates96_0.length then search edge 3 [523, 541] innerCandidates96_0 1064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan96_1

theorem rootBranch96 : search edge 4 [523] rootCandidates96 523 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [523] 523 innerTail96_0 best = best
  exact innerScan96_0

def rootCandidates95 : List Nat := [557, 641, 659, 809, 1091, 1229, 1373, 1523, 1637, 1721, 2381, 2447, 2477, 2777, 3137, 3389, 3539, 4337, 4691, 5381, 5417, 6053, 6269, 6311, 6353, 6521, 7607, 7673, 7907]
set_option maxRecDepth 4096 in
theorem rootCandidates95_checked : candidates edge 5 0 521 tail96 best.bound = rootCandidates95 := by
  unfold candidates
  have row : edge 521 = PrimePairCache.smallRow521.contains := by funext q; rfl
  rw [row]
  decide

def innerTail95_29 : List Nat := []
theorem innerScan95_29 : scan edge (search edge 3) 4 [521] 521 innerTail95_29 best = best := rfl

def innerTail95_28 : List Nat := 7907 :: innerTail95_29
def innerCandidates95_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_28_checked : candidates edge 4 521 7907 innerTail95_29 best.bound = innerCandidates95_28 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_28 : scan edge (search edge 3) 4 [521] 521 innerTail95_28 best = best := by
  rw [innerTail95_28, scan, if_neg (by decide), innerCandidates95_28_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_29
    (if 3 ≤ innerCandidates95_28.length then search edge 3 [521, 7907] innerCandidates95_28 8428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_29

def innerTail95_27 : List Nat := 7673 :: innerTail95_28
def innerCandidates95_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_27_checked : candidates edge 4 521 7673 innerTail95_28 best.bound = innerCandidates95_27 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_27 : scan edge (search edge 3) 4 [521] 521 innerTail95_27 best = best := by
  rw [innerTail95_27, scan, if_neg (by decide), innerCandidates95_27_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_28
    (if 3 ≤ innerCandidates95_27.length then search edge 3 [521, 7673] innerCandidates95_27 8194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_28

def innerTail95_26 : List Nat := 7607 :: innerTail95_27
def innerCandidates95_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_26_checked : candidates edge 4 521 7607 innerTail95_27 best.bound = innerCandidates95_26 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_26 : scan edge (search edge 3) 4 [521] 521 innerTail95_26 best = best := by
  rw [innerTail95_26, scan, if_neg (by decide), innerCandidates95_26_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_27
    (if 3 ≤ innerCandidates95_26.length then search edge 3 [521, 7607] innerCandidates95_26 8128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_27

def innerTail95_25 : List Nat := 6521 :: innerTail95_26
def innerCandidates95_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_25_checked : candidates edge 4 521 6521 innerTail95_26 best.bound = innerCandidates95_25 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_25 : scan edge (search edge 3) 4 [521] 521 innerTail95_25 best = best := by
  rw [innerTail95_25, scan, if_neg (by decide), innerCandidates95_25_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_26
    (if 3 ≤ innerCandidates95_25.length then search edge 3 [521, 6521] innerCandidates95_25 7042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_26

def innerTail95_24 : List Nat := 6353 :: innerTail95_25
def innerCandidates95_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_24_checked : candidates edge 4 521 6353 innerTail95_25 best.bound = innerCandidates95_24 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_24 : scan edge (search edge 3) 4 [521] 521 innerTail95_24 best = best := by
  rw [innerTail95_24, scan, if_neg (by decide), innerCandidates95_24_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_25
    (if 3 ≤ innerCandidates95_24.length then search edge 3 [521, 6353] innerCandidates95_24 6874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_25

def innerTail95_23 : List Nat := 6311 :: innerTail95_24
def innerCandidates95_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_23_checked : candidates edge 4 521 6311 innerTail95_24 best.bound = innerCandidates95_23 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_23 : scan edge (search edge 3) 4 [521] 521 innerTail95_23 best = best := by
  rw [innerTail95_23, scan, if_neg (by decide), innerCandidates95_23_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_24
    (if 3 ≤ innerCandidates95_23.length then search edge 3 [521, 6311] innerCandidates95_23 6832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_24

def innerTail95_22 : List Nat := 6269 :: innerTail95_23
def innerCandidates95_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_22_checked : candidates edge 4 521 6269 innerTail95_23 best.bound = innerCandidates95_22 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_22 : scan edge (search edge 3) 4 [521] 521 innerTail95_22 best = best := by
  rw [innerTail95_22, scan, if_neg (by decide), innerCandidates95_22_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_23
    (if 3 ≤ innerCandidates95_22.length then search edge 3 [521, 6269] innerCandidates95_22 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_23

def innerTail95_21 : List Nat := 6053 :: innerTail95_22
def innerCandidates95_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_21_checked : candidates edge 4 521 6053 innerTail95_22 best.bound = innerCandidates95_21 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_21 : scan edge (search edge 3) 4 [521] 521 innerTail95_21 best = best := by
  rw [innerTail95_21, scan, if_neg (by decide), innerCandidates95_21_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_22
    (if 3 ≤ innerCandidates95_21.length then search edge 3 [521, 6053] innerCandidates95_21 6574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_22

def innerTail95_20 : List Nat := 5417 :: innerTail95_21
def innerCandidates95_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_20_checked : candidates edge 4 521 5417 innerTail95_21 best.bound = innerCandidates95_20 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_20 : scan edge (search edge 3) 4 [521] 521 innerTail95_20 best = best := by
  rw [innerTail95_20, scan, if_neg (by decide), innerCandidates95_20_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_21
    (if 3 ≤ innerCandidates95_20.length then search edge 3 [521, 5417] innerCandidates95_20 5938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_21

def innerTail95_19 : List Nat := 5381 :: innerTail95_20
def innerCandidates95_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_19_checked : candidates edge 4 521 5381 innerTail95_20 best.bound = innerCandidates95_19 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_19 : scan edge (search edge 3) 4 [521] 521 innerTail95_19 best = best := by
  rw [innerTail95_19, scan, if_neg (by decide), innerCandidates95_19_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_20
    (if 3 ≤ innerCandidates95_19.length then search edge 3 [521, 5381] innerCandidates95_19 5902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_20

def innerTail95_18 : List Nat := 4691 :: innerTail95_19
def innerCandidates95_18 : List Nat := [6521]
set_option maxRecDepth 4096 in
theorem innerCandidates95_18_checked : candidates edge 4 521 4691 innerTail95_19 best.bound = innerCandidates95_18 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_18 : scan edge (search edge 3) 4 [521] 521 innerTail95_18 best = best := by
  rw [innerTail95_18, scan, if_neg (by decide), innerCandidates95_18_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_19
    (if 3 ≤ innerCandidates95_18.length then search edge 3 [521, 4691] innerCandidates95_18 5212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_19

def innerTail95_17 : List Nat := 4337 :: innerTail95_18
def innerCandidates95_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_17_checked : candidates edge 4 521 4337 innerTail95_18 best.bound = innerCandidates95_17 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_17 : scan edge (search edge 3) 4 [521] 521 innerTail95_17 best = best := by
  rw [innerTail95_17, scan, if_neg (by decide), innerCandidates95_17_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_18
    (if 3 ≤ innerCandidates95_17.length then search edge 3 [521, 4337] innerCandidates95_17 4858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_18

def innerTail95_16 : List Nat := 3539 :: innerTail95_17
def innerCandidates95_16 : List Nat := [6521]
set_option maxRecDepth 4096 in
theorem innerCandidates95_16_checked : candidates edge 4 521 3539 innerTail95_17 best.bound = innerCandidates95_16 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_16 : scan edge (search edge 3) 4 [521] 521 innerTail95_16 best = best := by
  rw [innerTail95_16, scan, if_neg (by decide), innerCandidates95_16_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_17
    (if 3 ≤ innerCandidates95_16.length then search edge 3 [521, 3539] innerCandidates95_16 4060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_17

def innerTail95_15 : List Nat := 3389 :: innerTail95_16
def innerCandidates95_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_15_checked : candidates edge 4 521 3389 innerTail95_16 best.bound = innerCandidates95_15 := by
  unfold candidates
  have row : edge 3389 = SearchData.row3389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_15 : scan edge (search edge 3) 4 [521] 521 innerTail95_15 best = best := by
  rw [innerTail95_15, scan, if_neg (by decide), innerCandidates95_15_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_16
    (if 3 ≤ innerCandidates95_15.length then search edge 3 [521, 3389] innerCandidates95_15 3910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_16

def innerTail95_14 : List Nat := 3137 :: innerTail95_15
def innerCandidates95_14 : List Nat := [6311]
set_option maxRecDepth 4096 in
theorem innerCandidates95_14_checked : candidates edge 4 521 3137 innerTail95_15 best.bound = innerCandidates95_14 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_14 : scan edge (search edge 3) 4 [521] 521 innerTail95_14 best = best := by
  rw [innerTail95_14, scan, if_neg (by decide), innerCandidates95_14_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_15
    (if 3 ≤ innerCandidates95_14.length then search edge 3 [521, 3137] innerCandidates95_14 3658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_15

def innerTail95_13 : List Nat := 2777 :: innerTail95_14
def innerCandidates95_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_13_checked : candidates edge 4 521 2777 innerTail95_14 best.bound = innerCandidates95_13 := by
  unfold candidates
  have row : edge 2777 = SearchData.row2777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_13 : scan edge (search edge 3) 4 [521] 521 innerTail95_13 best = best := by
  rw [innerTail95_13, scan, if_neg (by decide), innerCandidates95_13_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_14
    (if 3 ≤ innerCandidates95_13.length then search edge 3 [521, 2777] innerCandidates95_13 3298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_14

def innerTail95_12 : List Nat := 2477 :: innerTail95_13
def innerCandidates95_12 : List Nat := [5381]
set_option maxRecDepth 4096 in
theorem innerCandidates95_12_checked : candidates edge 4 521 2477 innerTail95_13 best.bound = innerCandidates95_12 := by
  unfold candidates
  have row : edge 2477 = PrimePairCache.smallRow2477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_12 : scan edge (search edge 3) 4 [521] 521 innerTail95_12 best = best := by
  rw [innerTail95_12, scan, if_neg (by decide), innerCandidates95_12_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_13
    (if 3 ≤ innerCandidates95_12.length then search edge 3 [521, 2477] innerCandidates95_12 2998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_13

def innerTail95_11 : List Nat := 2447 :: innerTail95_12
def innerCandidates95_11 : List Nat := [7907]
set_option maxRecDepth 4096 in
theorem innerCandidates95_11_checked : candidates edge 4 521 2447 innerTail95_12 best.bound = innerCandidates95_11 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_11 : scan edge (search edge 3) 4 [521] 521 innerTail95_11 best = best := by
  rw [innerTail95_11, scan, if_neg (by decide), innerCandidates95_11_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_12
    (if 3 ≤ innerCandidates95_11.length then search edge 3 [521, 2447] innerCandidates95_11 2968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_12

def innerTail95_10 : List Nat := 2381 :: innerTail95_11
def innerCandidates95_10 : List Nat := [2447, 3539]
set_option maxRecDepth 4096 in
theorem innerCandidates95_10_checked : candidates edge 4 521 2381 innerTail95_11 best.bound = innerCandidates95_10 := by
  unfold candidates
  have row : edge 2381 = PrimePairCache.smallRow2381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_10 : scan edge (search edge 3) 4 [521] 521 innerTail95_10 best = best := by
  rw [innerTail95_10, scan, if_neg (by decide), innerCandidates95_10_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_11
    (if 3 ≤ innerCandidates95_10.length then search edge 3 [521, 2381] innerCandidates95_10 2902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_11

def innerTail95_9 : List Nat := 1721 :: innerTail95_10
def innerCandidates95_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_9_checked : candidates edge 4 521 1721 innerTail95_10 best.bound = innerCandidates95_9 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_9 : scan edge (search edge 3) 4 [521] 521 innerTail95_9 best = best := by
  rw [innerTail95_9, scan, if_neg (by decide), innerCandidates95_9_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_10
    (if 3 ≤ innerCandidates95_9.length then search edge 3 [521, 1721] innerCandidates95_9 2242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_10

def innerTail95_8 : List Nat := 1637 :: innerTail95_9
def innerCandidates95_8 : List Nat := [6053, 7607]
set_option maxRecDepth 4096 in
theorem innerCandidates95_8_checked : candidates edge 4 521 1637 innerTail95_9 best.bound = innerCandidates95_8 := by
  unfold candidates
  have row : edge 1637 = PrimePairCache.smallRow1637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_8 : scan edge (search edge 3) 4 [521] 521 innerTail95_8 best = best := by
  rw [innerTail95_8, scan, if_neg (by decide), innerCandidates95_8_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_9
    (if 3 ≤ innerCandidates95_8.length then search edge 3 [521, 1637] innerCandidates95_8 2158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_9

def innerTail95_7 : List Nat := 1523 :: innerTail95_8
def innerCandidates95_7 : List Nat := [2447, 6311, 7907]
set_option maxRecDepth 4096 in
theorem innerCandidates95_7_checked : candidates edge 4 521 1523 innerTail95_8 best.bound = innerCandidates95_7 := by
  unfold candidates
  have row : edge 1523 = PrimePairCache.smallRow1523.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild95_7 : search edge 3 [521, 1523] innerCandidates95_7 2044 best = best := by decide

theorem innerScan95_7 : scan edge (search edge 3) 4 [521] 521 innerTail95_7 best = best := by
  rw [innerTail95_7, scan, if_neg (by decide), innerCandidates95_7_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_8
    (if 3 ≤ innerCandidates95_7.length then search edge 3 [521, 1523] innerCandidates95_7 2044 best else best) = best
  rw [if_pos (by decide), innerChild95_7]
  exact innerScan95_8

def innerTail95_6 : List Nat := 1373 :: innerTail95_7
def innerCandidates95_6 : List Nat := [1637, 3539, 7907]
set_option maxRecDepth 4096 in
theorem innerCandidates95_6_checked : candidates edge 4 521 1373 innerTail95_7 best.bound = innerCandidates95_6 := by
  unfold candidates
  have row : edge 1373 = PrimePairCache.smallRow1373.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild95_6 : search edge 3 [521, 1373] innerCandidates95_6 1894 best = best := by decide

theorem innerScan95_6 : scan edge (search edge 3) 4 [521] 521 innerTail95_6 best = best := by
  rw [innerTail95_6, scan, if_neg (by decide), innerCandidates95_6_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_7
    (if 3 ≤ innerCandidates95_6.length then search edge 3 [521, 1373] innerCandidates95_6 1894 best else best) = best
  rw [if_pos (by decide), innerChild95_6]
  exact innerScan95_7

def innerTail95_5 : List Nat := 1229 :: innerTail95_6
def innerCandidates95_5 : List Nat := [3389, 6353, 7907]
set_option maxRecDepth 4096 in
theorem innerCandidates95_5_checked : candidates edge 4 521 1229 innerTail95_6 best.bound = innerCandidates95_5 := by
  unfold candidates
  have row : edge 1229 = PrimePairCache.smallRow1229.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild95_5 : search edge 3 [521, 1229] innerCandidates95_5 1750 best = best := by decide

theorem innerScan95_5 : scan edge (search edge 3) 4 [521] 521 innerTail95_5 best = best := by
  rw [innerTail95_5, scan, if_neg (by decide), innerCandidates95_5_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_6
    (if 3 ≤ innerCandidates95_5.length then search edge 3 [521, 1229] innerCandidates95_5 1750 best else best) = best
  rw [if_pos (by decide), innerChild95_5]
  exact innerScan95_6

def innerTail95_4 : List Nat := 1091 :: innerTail95_5
def innerCandidates95_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_4_checked : candidates edge 4 521 1091 innerTail95_5 best.bound = innerCandidates95_4 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_4 : scan edge (search edge 3) 4 [521] 521 innerTail95_4 best = best := by
  rw [innerTail95_4, scan, if_neg (by decide), innerCandidates95_4_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_5
    (if 3 ≤ innerCandidates95_4.length then search edge 3 [521, 1091] innerCandidates95_4 1612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_5

def innerTail95_3 : List Nat := 809 :: innerTail95_4
def innerCandidates95_3 : List Nat := [1091]
set_option maxRecDepth 4096 in
theorem innerCandidates95_3_checked : candidates edge 4 521 809 innerTail95_4 best.bound = innerCandidates95_3 := by
  unfold candidates
  have row : edge 809 = PrimePairCache.smallRow809.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_3 : scan edge (search edge 3) 4 [521] 521 innerTail95_3 best = best := by
  rw [innerTail95_3, scan, if_neg (by decide), innerCandidates95_3_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_4
    (if 3 ≤ innerCandidates95_3.length then search edge 3 [521, 809] innerCandidates95_3 1330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_4

def innerTail95_2 : List Nat := 659 :: innerTail95_3
def innerCandidates95_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_2_checked : candidates edge 4 521 659 innerTail95_3 best.bound = innerCandidates95_2 := by
  unfold candidates
  have row : edge 659 = PrimePairCache.smallRow659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_2 : scan edge (search edge 3) 4 [521] 521 innerTail95_2 best = best := by
  rw [innerTail95_2, scan, if_neg (by decide), innerCandidates95_2_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_3
    (if 3 ≤ innerCandidates95_2.length then search edge 3 [521, 659] innerCandidates95_2 1180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_3

def innerTail95_1 : List Nat := 641 :: innerTail95_2
def innerCandidates95_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates95_1_checked : candidates edge 4 521 641 innerTail95_2 best.bound = innerCandidates95_1 := by
  unfold candidates
  have row : edge 641 = PrimePairCache.smallRow641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_1 : scan edge (search edge 3) 4 [521] 521 innerTail95_1 best = best := by
  rw [innerTail95_1, scan, if_neg (by decide), innerCandidates95_1_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_2
    (if 3 ≤ innerCandidates95_1.length then search edge 3 [521, 641] innerCandidates95_1 1162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_2

def innerTail95_0 : List Nat := 557 :: innerTail95_1
def innerCandidates95_0 : List Nat := [1721, 3389]
set_option maxRecDepth 4096 in
theorem innerCandidates95_0_checked : candidates edge 4 521 557 innerTail95_1 best.bound = innerCandidates95_0 := by
  unfold candidates
  have row : edge 557 = PrimePairCache.smallRow557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan95_0 : scan edge (search edge 3) 4 [521] 521 innerTail95_0 best = best := by
  rw [innerTail95_0, scan, if_neg (by decide), innerCandidates95_0_checked]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_1
    (if 3 ≤ innerCandidates95_0.length then search edge 3 [521, 557] innerCandidates95_0 1078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan95_1

theorem rootBranch95 : search edge 4 [521] rootCandidates95 521 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [521] 521 innerTail95_0 best = best
  exact innerScan95_0

end ProjectEulerSolutions.PrimePairUpper.Search
