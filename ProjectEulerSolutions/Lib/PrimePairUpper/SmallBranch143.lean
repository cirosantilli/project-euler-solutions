import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates247 : List Nat := [1619, 1637, 1877, 1973, 2447, 3347, 3557, 3929, 4001, 4229, 4283, 4451, 4799, 5009, 5333, 5861, 5879, 6113, 6143, 6287, 6551, 6659, 6803, 6827, 6869, 7043, 7079, 7643, 7949, 8171, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates247_checked : candidates edge 5 0 1583 tail248 best.bound = rootCandidates247 := by
  unfold candidates
  have row : edge 1583 = PrimePairCache.smallRow1583.contains := by funext q; rfl
  rw [row]
  decide

def innerTail247_31 : List Nat := []
theorem innerScan247_31 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_31 best = best := rfl

def innerTail247_30 : List Nat := 8369 :: innerTail247_31
def innerCandidates247_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_30_checked : candidates edge 4 1583 8369 innerTail247_31 best.bound = innerCandidates247_30 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_30 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_30 best = best := by
  rw [innerTail247_30, scan, if_neg (by decide), innerCandidates247_30_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_31
    (if 3 ≤ innerCandidates247_30.length then search edge 3 [1583, 8369] innerCandidates247_30 9952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_31

def innerTail247_29 : List Nat := 8171 :: innerTail247_30
def innerCandidates247_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_29_checked : candidates edge 4 1583 8171 innerTail247_30 best.bound = innerCandidates247_29 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_29 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_29 best = best := by
  rw [innerTail247_29, scan, if_neg (by decide), innerCandidates247_29_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_30
    (if 3 ≤ innerCandidates247_29.length then search edge 3 [1583, 8171] innerCandidates247_29 9754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_30

def innerTail247_28 : List Nat := 7949 :: innerTail247_29
def innerCandidates247_28 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates247_28_checked : candidates edge 4 1583 7949 innerTail247_29 best.bound = innerCandidates247_28 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_28 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_28 best = best := by
  rw [innerTail247_28, scan, if_neg (by decide), innerCandidates247_28_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_29
    (if 3 ≤ innerCandidates247_28.length then search edge 3 [1583, 7949] innerCandidates247_28 9532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_29

def innerTail247_27 : List Nat := 7643 :: innerTail247_28
def innerCandidates247_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_27_checked : candidates edge 4 1583 7643 innerTail247_28 best.bound = innerCandidates247_27 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_27 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_27 best = best := by
  rw [innerTail247_27, scan, if_neg (by decide), innerCandidates247_27_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_28
    (if 3 ≤ innerCandidates247_27.length then search edge 3 [1583, 7643] innerCandidates247_27 9226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_28

def innerTail247_26 : List Nat := 7079 :: innerTail247_27
def innerCandidates247_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_26_checked : candidates edge 4 1583 7079 innerTail247_27 best.bound = innerCandidates247_26 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_26 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_26 best = best := by
  rw [innerTail247_26, scan, if_neg (by decide), innerCandidates247_26_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_27
    (if 3 ≤ innerCandidates247_26.length then search edge 3 [1583, 7079] innerCandidates247_26 8662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_27

def innerTail247_25 : List Nat := 7043 :: innerTail247_26
def innerCandidates247_25 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates247_25_checked : candidates edge 4 1583 7043 innerTail247_26 best.bound = innerCandidates247_25 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_25 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_25 best = best := by
  rw [innerTail247_25, scan, if_neg (by decide), innerCandidates247_25_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_26
    (if 3 ≤ innerCandidates247_25.length then search edge 3 [1583, 7043] innerCandidates247_25 8626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_26

def innerTail247_24 : List Nat := 6869 :: innerTail247_25
def innerCandidates247_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_24_checked : candidates edge 4 1583 6869 innerTail247_25 best.bound = innerCandidates247_24 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_24 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_24 best = best := by
  rw [innerTail247_24, scan, if_neg (by decide), innerCandidates247_24_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_25
    (if 3 ≤ innerCandidates247_24.length then search edge 3 [1583, 6869] innerCandidates247_24 8452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_25

def innerTail247_23 : List Nat := 6827 :: innerTail247_24
def innerCandidates247_23 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates247_23_checked : candidates edge 4 1583 6827 innerTail247_24 best.bound = innerCandidates247_23 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_23 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_23 best = best := by
  rw [innerTail247_23, scan, if_neg (by decide), innerCandidates247_23_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_24
    (if 3 ≤ innerCandidates247_23.length then search edge 3 [1583, 6827] innerCandidates247_23 8410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_24

def innerTail247_22 : List Nat := 6803 :: innerTail247_23
def innerCandidates247_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_22_checked : candidates edge 4 1583 6803 innerTail247_23 best.bound = innerCandidates247_22 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_22 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_22 best = best := by
  rw [innerTail247_22, scan, if_neg (by decide), innerCandidates247_22_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_23
    (if 3 ≤ innerCandidates247_22.length then search edge 3 [1583, 6803] innerCandidates247_22 8386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_23

def innerTail247_21 : List Nat := 6659 :: innerTail247_22
def innerCandidates247_21 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates247_21_checked : candidates edge 4 1583 6659 innerTail247_22 best.bound = innerCandidates247_21 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_21 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_21 best = best := by
  rw [innerTail247_21, scan, if_neg (by decide), innerCandidates247_21_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_22
    (if 3 ≤ innerCandidates247_21.length then search edge 3 [1583, 6659] innerCandidates247_21 8242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_22

def innerTail247_20 : List Nat := 6551 :: innerTail247_21
def innerCandidates247_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_20_checked : candidates edge 4 1583 6551 innerTail247_21 best.bound = innerCandidates247_20 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_20 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_20 best = best := by
  rw [innerTail247_20, scan, if_neg (by decide), innerCandidates247_20_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_21
    (if 3 ≤ innerCandidates247_20.length then search edge 3 [1583, 6551] innerCandidates247_20 8134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_21

def innerTail247_19 : List Nat := 6287 :: innerTail247_20
def innerCandidates247_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_19_checked : candidates edge 4 1583 6287 innerTail247_20 best.bound = innerCandidates247_19 := by
  unfold candidates
  have row : edge 6287 = SearchData.row6287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_19 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_19 best = best := by
  rw [innerTail247_19, scan, if_neg (by decide), innerCandidates247_19_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_20
    (if 3 ≤ innerCandidates247_19.length then search edge 3 [1583, 6287] innerCandidates247_19 7870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_20

def innerTail247_18 : List Nat := 6143 :: innerTail247_19
def innerCandidates247_18 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates247_18_checked : candidates edge 4 1583 6143 innerTail247_19 best.bound = innerCandidates247_18 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_18 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_18 best = best := by
  rw [innerTail247_18, scan, if_neg (by decide), innerCandidates247_18_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_19
    (if 3 ≤ innerCandidates247_18.length then search edge 3 [1583, 6143] innerCandidates247_18 7726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_19

def innerTail247_17 : List Nat := 6113 :: innerTail247_18
def innerCandidates247_17 : List Nat := [6659]
set_option maxRecDepth 4096 in
theorem innerCandidates247_17_checked : candidates edge 4 1583 6113 innerTail247_18 best.bound = innerCandidates247_17 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_17 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_17 best = best := by
  rw [innerTail247_17, scan, if_neg (by decide), innerCandidates247_17_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_18
    (if 3 ≤ innerCandidates247_17.length then search edge 3 [1583, 6113] innerCandidates247_17 7696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_18

def innerTail247_16 : List Nat := 5879 :: innerTail247_17
def innerCandidates247_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_16_checked : candidates edge 4 1583 5879 innerTail247_17 best.bound = innerCandidates247_16 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_16 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_16 best = best := by
  rw [innerTail247_16, scan, if_neg (by decide), innerCandidates247_16_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_17
    (if 3 ≤ innerCandidates247_16.length then search edge 3 [1583, 5879] innerCandidates247_16 7462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_17

def innerTail247_15 : List Nat := 5861 :: innerTail247_16
def innerCandidates247_15 : List Nat := [6143, 6827, 7043, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates247_15_checked : candidates edge 4 1583 5861 innerTail247_16 best.bound = innerCandidates247_15 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild247_15 : search edge 3 [1583, 5861] innerCandidates247_15 7444 best = best := by decide

theorem innerScan247_15 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_15 best = best := by
  rw [innerTail247_15, scan, if_neg (by decide), innerCandidates247_15_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_16
    (if 3 ≤ innerCandidates247_15.length then search edge 3 [1583, 5861] innerCandidates247_15 7444 best else best) = best
  rw [if_pos (by decide), innerChild247_15]
  exact innerScan247_16

def innerTail247_14 : List Nat := 5333 :: innerTail247_15
def innerCandidates247_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_14_checked : candidates edge 4 1583 5333 innerTail247_15 best.bound = innerCandidates247_14 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_14 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_14 best = best := by
  rw [innerTail247_14, scan, if_neg (by decide), innerCandidates247_14_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_15
    (if 3 ≤ innerCandidates247_14.length then search edge 3 [1583, 5333] innerCandidates247_14 6916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_15

def innerTail247_13 : List Nat := 5009 :: innerTail247_14
def innerCandidates247_13 : List Nat := [5861]
set_option maxRecDepth 4096 in
theorem innerCandidates247_13_checked : candidates edge 4 1583 5009 innerTail247_14 best.bound = innerCandidates247_13 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_13 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_13 best = best := by
  rw [innerTail247_13, scan, if_neg (by decide), innerCandidates247_13_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_14
    (if 3 ≤ innerCandidates247_13.length then search edge 3 [1583, 5009] innerCandidates247_13 6592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_14

def innerTail247_12 : List Nat := 4799 :: innerTail247_13
def innerCandidates247_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_12_checked : candidates edge 4 1583 4799 innerTail247_13 best.bound = innerCandidates247_12 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_12 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_12 best = best := by
  rw [innerTail247_12, scan, if_neg (by decide), innerCandidates247_12_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_13
    (if 3 ≤ innerCandidates247_12.length then search edge 3 [1583, 4799] innerCandidates247_12 6382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_13

def innerTail247_11 : List Nat := 4451 :: innerTail247_12
def innerCandidates247_11 : List Nat := [6803]
set_option maxRecDepth 4096 in
theorem innerCandidates247_11_checked : candidates edge 4 1583 4451 innerTail247_12 best.bound = innerCandidates247_11 := by
  unfold candidates
  have row : edge 4451 = SearchData.row4451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_11 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_11 best = best := by
  rw [innerTail247_11, scan, if_neg (by decide), innerCandidates247_11_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_12
    (if 3 ≤ innerCandidates247_11.length then search edge 3 [1583, 4451] innerCandidates247_11 6034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_12

def innerTail247_10 : List Nat := 4283 :: innerTail247_11
def innerCandidates247_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_10_checked : candidates edge 4 1583 4283 innerTail247_11 best.bound = innerCandidates247_10 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_10 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_10 best = best := by
  rw [innerTail247_10, scan, if_neg (by decide), innerCandidates247_10_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_11
    (if 3 ≤ innerCandidates247_10.length then search edge 3 [1583, 4283] innerCandidates247_10 5866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_11

def innerTail247_9 : List Nat := 4229 :: innerTail247_10
def innerCandidates247_9 : List Nat := [7643]
set_option maxRecDepth 4096 in
theorem innerCandidates247_9_checked : candidates edge 4 1583 4229 innerTail247_10 best.bound = innerCandidates247_9 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_9 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_9 best = best := by
  rw [innerTail247_9, scan, if_neg (by decide), innerCandidates247_9_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_10
    (if 3 ≤ innerCandidates247_9.length then search edge 3 [1583, 4229] innerCandidates247_9 5812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_10

def innerTail247_8 : List Nat := 4001 :: innerTail247_9
def innerCandidates247_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_8_checked : candidates edge 4 1583 4001 innerTail247_9 best.bound = innerCandidates247_8 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_8 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_8 best = best := by
  rw [innerTail247_8, scan, if_neg (by decide), innerCandidates247_8_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_9
    (if 3 ≤ innerCandidates247_8.length then search edge 3 [1583, 4001] innerCandidates247_8 5584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_9

def innerTail247_7 : List Nat := 3929 :: innerTail247_8
def innerCandidates247_7 : List Nat := [6143, 7043]
set_option maxRecDepth 4096 in
theorem innerCandidates247_7_checked : candidates edge 4 1583 3929 innerTail247_8 best.bound = innerCandidates247_7 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_7 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_7 best = best := by
  rw [innerTail247_7, scan, if_neg (by decide), innerCandidates247_7_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_8
    (if 3 ≤ innerCandidates247_7.length then search edge 3 [1583, 3929] innerCandidates247_7 5512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_8

def innerTail247_6 : List Nat := 3557 :: innerTail247_7
def innerCandidates247_6 : List Nat := [6551]
set_option maxRecDepth 4096 in
theorem innerCandidates247_6_checked : candidates edge 4 1583 3557 innerTail247_7 best.bound = innerCandidates247_6 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_6 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_6 best = best := by
  rw [innerTail247_6, scan, if_neg (by decide), innerCandidates247_6_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_7
    (if 3 ≤ innerCandidates247_6.length then search edge 3 [1583, 3557] innerCandidates247_6 5140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_7

def innerTail247_5 : List Nat := 3347 :: innerTail247_6
def innerCandidates247_5 : List Nat := [5333, 6143, 7043]
set_option maxRecDepth 4096 in
theorem innerCandidates247_5_checked : candidates edge 4 1583 3347 innerTail247_6 best.bound = innerCandidates247_5 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild247_5 : search edge 3 [1583, 3347] innerCandidates247_5 4930 best = best := by decide

theorem innerScan247_5 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_5 best = best := by
  rw [innerTail247_5, scan, if_neg (by decide), innerCandidates247_5_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_6
    (if 3 ≤ innerCandidates247_5.length then search edge 3 [1583, 3347] innerCandidates247_5 4930 best else best) = best
  rw [if_pos (by decide), innerChild247_5]
  exact innerScan247_6

def innerTail247_4 : List Nat := 2447 :: innerTail247_5
def innerCandidates247_4 : List Nat := [6113]
set_option maxRecDepth 4096 in
theorem innerCandidates247_4_checked : candidates edge 4 1583 2447 innerTail247_5 best.bound = innerCandidates247_4 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_4 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_4 best = best := by
  rw [innerTail247_4, scan, if_neg (by decide), innerCandidates247_4_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_5
    (if 3 ≤ innerCandidates247_4.length then search edge 3 [1583, 2447] innerCandidates247_4 4030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_5

def innerTail247_3 : List Nat := 1973 :: innerTail247_4
def innerCandidates247_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_3_checked : candidates edge 4 1583 1973 innerTail247_4 best.bound = innerCandidates247_3 := by
  unfold candidates
  have row : edge 1973 = PrimePairCache.smallRow1973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_3 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_3 best = best := by
  rw [innerTail247_3, scan, if_neg (by decide), innerCandidates247_3_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_4
    (if 3 ≤ innerCandidates247_3.length then search edge 3 [1583, 1973] innerCandidates247_3 3556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_4

def innerTail247_2 : List Nat := 1877 :: innerTail247_3
def innerCandidates247_2 : List Nat := [2447, 6113, 6287]
set_option maxRecDepth 4096 in
theorem innerCandidates247_2_checked : candidates edge 4 1583 1877 innerTail247_3 best.bound = innerCandidates247_2 := by
  unfold candidates
  have row : edge 1877 = PrimePairCache.smallRow1877.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild247_2 : search edge 3 [1583, 1877] innerCandidates247_2 3460 best = best := by decide

theorem innerScan247_2 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_2 best = best := by
  rw [innerTail247_2, scan, if_neg (by decide), innerCandidates247_2_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_3
    (if 3 ≤ innerCandidates247_2.length then search edge 3 [1583, 1877] innerCandidates247_2 3460 best else best) = best
  rw [if_pos (by decide), innerChild247_2]
  exact innerScan247_3

def innerTail247_1 : List Nat := 1637 :: innerTail247_2
def innerCandidates247_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates247_1_checked : candidates edge 4 1583 1637 innerTail247_2 best.bound = innerCandidates247_1 := by
  unfold candidates
  have row : edge 1637 = PrimePairCache.smallRow1637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_1 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_1 best = best := by
  rw [innerTail247_1, scan, if_neg (by decide), innerCandidates247_1_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_2
    (if 3 ≤ innerCandidates247_1.length then search edge 3 [1583, 1637] innerCandidates247_1 3220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_2

def innerTail247_0 : List Nat := 1619 :: innerTail247_1
def innerCandidates247_0 : List Nat := [6143, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates247_0_checked : candidates edge 4 1583 1619 innerTail247_1 best.bound = innerCandidates247_0 := by
  unfold candidates
  have row : edge 1619 = PrimePairCache.smallRow1619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan247_0 : scan edge (search edge 3) 4 [1583] 1583 innerTail247_0 best = best := by
  rw [innerTail247_0, scan, if_neg (by decide), innerCandidates247_0_checked]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_1
    (if 3 ≤ innerCandidates247_0.length then search edge 3 [1583, 1619] innerCandidates247_0 3202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan247_1

theorem rootBranch247 : search edge 4 [1583] rootCandidates247 1583 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1583] 1583 innerTail247_0 best = best
  exact innerScan247_0

def rootCandidates246 : List Nat := [1627, 1669, 3301, 3907, 4783, 6073, 6301, 7477, 7723, 8059]
set_option maxRecDepth 4096 in
theorem rootCandidates246_checked : candidates edge 5 0 1579 tail247 best.bound = rootCandidates246 := by
  unfold candidates
  have row : edge 1579 = PrimePairCache.smallRow1579.contains := by funext q; rfl
  rw [row]
  decide

def innerTail246_10 : List Nat := []
theorem innerScan246_10 : scan edge (search edge 3) 4 [1579] 1579 innerTail246_10 best = best := rfl

def innerTail246_9 : List Nat := 8059 :: innerTail246_10
def innerCandidates246_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates246_9_checked : candidates edge 4 1579 8059 innerTail246_10 best.bound = innerCandidates246_9 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan246_9 : scan edge (search edge 3) 4 [1579] 1579 innerTail246_9 best = best := by
  rw [innerTail246_9, scan, if_neg (by decide), innerCandidates246_9_checked]
  change scan edge (search edge 3) 4 [1579] 1579 innerTail246_10
    (if 3 ≤ innerCandidates246_9.length then search edge 3 [1579, 8059] innerCandidates246_9 9638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan246_10

def innerTail246_8 : List Nat := 7723 :: innerTail246_9
def innerCandidates246_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates246_8_checked : candidates edge 4 1579 7723 innerTail246_9 best.bound = innerCandidates246_8 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan246_8 : scan edge (search edge 3) 4 [1579] 1579 innerTail246_8 best = best := by
  rw [innerTail246_8, scan, if_neg (by decide), innerCandidates246_8_checked]
  change scan edge (search edge 3) 4 [1579] 1579 innerTail246_9
    (if 3 ≤ innerCandidates246_8.length then search edge 3 [1579, 7723] innerCandidates246_8 9302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan246_9

def innerTail246_7 : List Nat := 7477 :: innerTail246_8
def innerCandidates246_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates246_7_checked : candidates edge 4 1579 7477 innerTail246_8 best.bound = innerCandidates246_7 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan246_7 : scan edge (search edge 3) 4 [1579] 1579 innerTail246_7 best = best := by
  rw [innerTail246_7, scan, if_neg (by decide), innerCandidates246_7_checked]
  change scan edge (search edge 3) 4 [1579] 1579 innerTail246_8
    (if 3 ≤ innerCandidates246_7.length then search edge 3 [1579, 7477] innerCandidates246_7 9056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan246_8

def innerTail246_6 : List Nat := 6301 :: innerTail246_7
def innerCandidates246_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates246_6_checked : candidates edge 4 1579 6301 innerTail246_7 best.bound = innerCandidates246_6 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan246_6 : scan edge (search edge 3) 4 [1579] 1579 innerTail246_6 best = best := by
  rw [innerTail246_6, scan, if_neg (by decide), innerCandidates246_6_checked]
  change scan edge (search edge 3) 4 [1579] 1579 innerTail246_7
    (if 3 ≤ innerCandidates246_6.length then search edge 3 [1579, 6301] innerCandidates246_6 7880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan246_7

def innerTail246_5 : List Nat := 6073 :: innerTail246_6
def innerCandidates246_5 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates246_5_checked : candidates edge 4 1579 6073 innerTail246_6 best.bound = innerCandidates246_5 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan246_5 : scan edge (search edge 3) 4 [1579] 1579 innerTail246_5 best = best := by
  rw [innerTail246_5, scan, if_neg (by decide), innerCandidates246_5_checked]
  change scan edge (search edge 3) 4 [1579] 1579 innerTail246_6
    (if 3 ≤ innerCandidates246_5.length then search edge 3 [1579, 6073] innerCandidates246_5 7652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan246_6

def innerTail246_4 : List Nat := 4783 :: innerTail246_5
def innerCandidates246_4 : List Nat := [6073]
set_option maxRecDepth 4096 in
theorem innerCandidates246_4_checked : candidates edge 4 1579 4783 innerTail246_5 best.bound = innerCandidates246_4 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan246_4 : scan edge (search edge 3) 4 [1579] 1579 innerTail246_4 best = best := by
  rw [innerTail246_4, scan, if_neg (by decide), innerCandidates246_4_checked]
  change scan edge (search edge 3) 4 [1579] 1579 innerTail246_5
    (if 3 ≤ innerCandidates246_4.length then search edge 3 [1579, 4783] innerCandidates246_4 6362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan246_5

def innerTail246_3 : List Nat := 3907 :: innerTail246_4
def innerCandidates246_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates246_3_checked : candidates edge 4 1579 3907 innerTail246_4 best.bound = innerCandidates246_3 := by
  unfold candidates
  have row : edge 3907 = SearchData.row3907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan246_3 : scan edge (search edge 3) 4 [1579] 1579 innerTail246_3 best = best := by
  rw [innerTail246_3, scan, if_neg (by decide), innerCandidates246_3_checked]
  change scan edge (search edge 3) 4 [1579] 1579 innerTail246_4
    (if 3 ≤ innerCandidates246_3.length then search edge 3 [1579, 3907] innerCandidates246_3 5486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan246_4

def innerTail246_2 : List Nat := 3301 :: innerTail246_3
def innerCandidates246_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates246_2_checked : candidates edge 4 1579 3301 innerTail246_3 best.bound = innerCandidates246_2 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan246_2 : scan edge (search edge 3) 4 [1579] 1579 innerTail246_2 best = best := by
  rw [innerTail246_2, scan, if_neg (by decide), innerCandidates246_2_checked]
  change scan edge (search edge 3) 4 [1579] 1579 innerTail246_3
    (if 3 ≤ innerCandidates246_2.length then search edge 3 [1579, 3301] innerCandidates246_2 4880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan246_3

def innerTail246_1 : List Nat := 1669 :: innerTail246_2
def innerCandidates246_1 : List Nat := [3301]
set_option maxRecDepth 4096 in
theorem innerCandidates246_1_checked : candidates edge 4 1579 1669 innerTail246_2 best.bound = innerCandidates246_1 := by
  unfold candidates
  have row : edge 1669 = PrimePairCache.smallRow1669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan246_1 : scan edge (search edge 3) 4 [1579] 1579 innerTail246_1 best = best := by
  rw [innerTail246_1, scan, if_neg (by decide), innerCandidates246_1_checked]
  change scan edge (search edge 3) 4 [1579] 1579 innerTail246_2
    (if 3 ≤ innerCandidates246_1.length then search edge 3 [1579, 1669] innerCandidates246_1 3248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan246_2

def innerTail246_0 : List Nat := 1627 :: innerTail246_1
def innerCandidates246_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates246_0_checked : candidates edge 4 1579 1627 innerTail246_1 best.bound = innerCandidates246_0 := by
  unfold candidates
  have row : edge 1627 = PrimePairCache.smallRow1627.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan246_0 : scan edge (search edge 3) 4 [1579] 1579 innerTail246_0 best = best := by
  rw [innerTail246_0, scan, if_neg (by decide), innerCandidates246_0_checked]
  change scan edge (search edge 3) 4 [1579] 1579 innerTail246_1
    (if 3 ≤ innerCandidates246_0.length then search edge 3 [1579, 1627] innerCandidates246_0 3206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan246_1

theorem rootBranch246 : search edge 4 [1579] rootCandidates246 1579 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1579] 1579 innerTail246_0 best = best
  exact innerScan246_0

def rootCandidates245 : List Nat := [1607, 1907, 2399, 2819, 4133, 4157, 4547, 5333, 5879, 5981, 6563, 6983, 7529, 7907]
set_option maxRecDepth 4096 in
theorem rootCandidates245_checked : candidates edge 5 0 1571 tail246 best.bound = rootCandidates245 := by
  unfold candidates
  have row : edge 1571 = PrimePairCache.smallRow1571.contains := by funext q; rfl
  rw [row]
  decide

def innerTail245_14 : List Nat := []
theorem innerScan245_14 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_14 best = best := rfl

def innerTail245_13 : List Nat := 7907 :: innerTail245_14
def innerCandidates245_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates245_13_checked : candidates edge 4 1571 7907 innerTail245_14 best.bound = innerCandidates245_13 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_13 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_13 best = best := by
  rw [innerTail245_13, scan, if_neg (by decide), innerCandidates245_13_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_14
    (if 3 ≤ innerCandidates245_13.length then search edge 3 [1571, 7907] innerCandidates245_13 9478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_14

def innerTail245_12 : List Nat := 7529 :: innerTail245_13
def innerCandidates245_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates245_12_checked : candidates edge 4 1571 7529 innerTail245_13 best.bound = innerCandidates245_12 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_12 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_12 best = best := by
  rw [innerTail245_12, scan, if_neg (by decide), innerCandidates245_12_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_13
    (if 3 ≤ innerCandidates245_12.length then search edge 3 [1571, 7529] innerCandidates245_12 9100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_13

def innerTail245_11 : List Nat := 6983 :: innerTail245_12
def innerCandidates245_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates245_11_checked : candidates edge 4 1571 6983 innerTail245_12 best.bound = innerCandidates245_11 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_11 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_11 best = best := by
  rw [innerTail245_11, scan, if_neg (by decide), innerCandidates245_11_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_12
    (if 3 ≤ innerCandidates245_11.length then search edge 3 [1571, 6983] innerCandidates245_11 8554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_12

def innerTail245_10 : List Nat := 6563 :: innerTail245_11
def innerCandidates245_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates245_10_checked : candidates edge 4 1571 6563 innerTail245_11 best.bound = innerCandidates245_10 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_10 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_10 best = best := by
  rw [innerTail245_10, scan, if_neg (by decide), innerCandidates245_10_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_11
    (if 3 ≤ innerCandidates245_10.length then search edge 3 [1571, 6563] innerCandidates245_10 8134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_11

def innerTail245_9 : List Nat := 5981 :: innerTail245_10
def innerCandidates245_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates245_9_checked : candidates edge 4 1571 5981 innerTail245_10 best.bound = innerCandidates245_9 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_9 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_9 best = best := by
  rw [innerTail245_9, scan, if_neg (by decide), innerCandidates245_9_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_10
    (if 3 ≤ innerCandidates245_9.length then search edge 3 [1571, 5981] innerCandidates245_9 7552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_10

def innerTail245_8 : List Nat := 5879 :: innerTail245_9
def innerCandidates245_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates245_8_checked : candidates edge 4 1571 5879 innerTail245_9 best.bound = innerCandidates245_8 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_8 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_8 best = best := by
  rw [innerTail245_8, scan, if_neg (by decide), innerCandidates245_8_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_9
    (if 3 ≤ innerCandidates245_8.length then search edge 3 [1571, 5879] innerCandidates245_8 7450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_9

def innerTail245_7 : List Nat := 5333 :: innerTail245_8
def innerCandidates245_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates245_7_checked : candidates edge 4 1571 5333 innerTail245_8 best.bound = innerCandidates245_7 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_7 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_7 best = best := by
  rw [innerTail245_7, scan, if_neg (by decide), innerCandidates245_7_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_8
    (if 3 ≤ innerCandidates245_7.length then search edge 3 [1571, 5333] innerCandidates245_7 6904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_8

def innerTail245_6 : List Nat := 4547 :: innerTail245_7
def innerCandidates245_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates245_6_checked : candidates edge 4 1571 4547 innerTail245_7 best.bound = innerCandidates245_6 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_6 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_6 best = best := by
  rw [innerTail245_6, scan, if_neg (by decide), innerCandidates245_6_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_7
    (if 3 ≤ innerCandidates245_6.length then search edge 3 [1571, 4547] innerCandidates245_6 6118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_7

def innerTail245_5 : List Nat := 4157 :: innerTail245_6
def innerCandidates245_5 : List Nat := [5333]
set_option maxRecDepth 4096 in
theorem innerCandidates245_5_checked : candidates edge 4 1571 4157 innerTail245_6 best.bound = innerCandidates245_5 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_5 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_5 best = best := by
  rw [innerTail245_5, scan, if_neg (by decide), innerCandidates245_5_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_6
    (if 3 ≤ innerCandidates245_5.length then search edge 3 [1571, 4157] innerCandidates245_5 5728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_6

def innerTail245_4 : List Nat := 4133 :: innerTail245_5
def innerCandidates245_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates245_4_checked : candidates edge 4 1571 4133 innerTail245_5 best.bound = innerCandidates245_4 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_4 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_4 best = best := by
  rw [innerTail245_4, scan, if_neg (by decide), innerCandidates245_4_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_5
    (if 3 ≤ innerCandidates245_4.length then search edge 3 [1571, 4133] innerCandidates245_4 5704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_5

def innerTail245_3 : List Nat := 2819 :: innerTail245_4
def innerCandidates245_3 : List Nat := [5879]
set_option maxRecDepth 4096 in
theorem innerCandidates245_3_checked : candidates edge 4 1571 2819 innerTail245_4 best.bound = innerCandidates245_3 := by
  unfold candidates
  have row : edge 2819 = SearchData.row2819.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_3 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_3 best = best := by
  rw [innerTail245_3, scan, if_neg (by decide), innerCandidates245_3_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_4
    (if 3 ≤ innerCandidates245_3.length then search edge 3 [1571, 2819] innerCandidates245_3 4390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_4

def innerTail245_2 : List Nat := 2399 :: innerTail245_3
def innerCandidates245_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates245_2_checked : candidates edge 4 1571 2399 innerTail245_3 best.bound = innerCandidates245_2 := by
  unfold candidates
  have row : edge 2399 = PrimePairCache.smallRow2399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_2 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_2 best = best := by
  rw [innerTail245_2, scan, if_neg (by decide), innerCandidates245_2_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_3
    (if 3 ≤ innerCandidates245_2.length then search edge 3 [1571, 2399] innerCandidates245_2 3970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_3

def innerTail245_1 : List Nat := 1907 :: innerTail245_2
def innerCandidates245_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates245_1_checked : candidates edge 4 1571 1907 innerTail245_2 best.bound = innerCandidates245_1 := by
  unfold candidates
  have row : edge 1907 = PrimePairCache.smallRow1907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_1 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_1 best = best := by
  rw [innerTail245_1, scan, if_neg (by decide), innerCandidates245_1_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_2
    (if 3 ≤ innerCandidates245_1.length then search edge 3 [1571, 1907] innerCandidates245_1 3478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_2

def innerTail245_0 : List Nat := 1607 :: innerTail245_1
def innerCandidates245_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates245_0_checked : candidates edge 4 1571 1607 innerTail245_1 best.bound = innerCandidates245_0 := by
  unfold candidates
  have row : edge 1607 = PrimePairCache.smallRow1607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan245_0 : scan edge (search edge 3) 4 [1571] 1571 innerTail245_0 best = best := by
  rw [innerTail245_0, scan, if_neg (by decide), innerCandidates245_0_checked]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_1
    (if 3 ≤ innerCandidates245_0.length then search edge 3 [1571, 1607] innerCandidates245_0 3178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan245_1

theorem rootBranch245 : search edge 4 [1571] rootCandidates245 1571 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1571] 1571 innerTail245_0 best = best
  exact innerScan245_0

end ProjectEulerSolutions.PrimePairUpper.Search
