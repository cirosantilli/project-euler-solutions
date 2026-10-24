import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates31 : List Nat := [367, 457, 547, 619, 661, 709, 787, 907, 967, 1051, 1447, 1549, 1597, 2803, 3229, 3253, 3313, 3697, 3739, 4177, 5059, 5167, 5323, 5743, 5869, 6679, 6841, 6949, 7057, 7177, 7219, 7933, 8011, 8053, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates31_checked : candidates edge 5 0 139 tail32 best.bound = rootCandidates31 := by
  unfold candidates
  have row : edge 139 = PrimePairCache.smallRow139.contains := by funext q; rfl
  rw [row]
  decide

def innerTail31_35 : List Nat := []
theorem innerScan31_35 : scan edge (search edge 3) 4 [139] 139 innerTail31_35 best = best := rfl

def innerTail31_34 : List Nat := 8269 :: innerTail31_35
def innerCandidates31_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_34_checked : candidates edge 4 139 8269 innerTail31_35 best.bound = innerCandidates31_34 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_34 : scan edge (search edge 3) 4 [139] 139 innerTail31_34 best = best := by
  rw [innerTail31_34, scan, if_neg (by decide), innerCandidates31_34_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_35
    (if 3 ≤ innerCandidates31_34.length then search edge 3 [139, 8269] innerCandidates31_34 8408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_35

def innerTail31_33 : List Nat := 8053 :: innerTail31_34
def innerCandidates31_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_33_checked : candidates edge 4 139 8053 innerTail31_34 best.bound = innerCandidates31_33 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_33 : scan edge (search edge 3) 4 [139] 139 innerTail31_33 best = best := by
  rw [innerTail31_33, scan, if_neg (by decide), innerCandidates31_33_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_34
    (if 3 ≤ innerCandidates31_33.length then search edge 3 [139, 8053] innerCandidates31_33 8192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_34

def innerTail31_32 : List Nat := 8011 :: innerTail31_33
def innerCandidates31_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_32_checked : candidates edge 4 139 8011 innerTail31_33 best.bound = innerCandidates31_32 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_32 : scan edge (search edge 3) 4 [139] 139 innerTail31_32 best = best := by
  rw [innerTail31_32, scan, if_neg (by decide), innerCandidates31_32_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_33
    (if 3 ≤ innerCandidates31_32.length then search edge 3 [139, 8011] innerCandidates31_32 8150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_33

def innerTail31_31 : List Nat := 7933 :: innerTail31_32
def innerCandidates31_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_31_checked : candidates edge 4 139 7933 innerTail31_32 best.bound = innerCandidates31_31 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_31 : scan edge (search edge 3) 4 [139] 139 innerTail31_31 best = best := by
  rw [innerTail31_31, scan, if_neg (by decide), innerCandidates31_31_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_32
    (if 3 ≤ innerCandidates31_31.length then search edge 3 [139, 7933] innerCandidates31_31 8072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_32

def innerTail31_30 : List Nat := 7219 :: innerTail31_31
def innerCandidates31_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_30_checked : candidates edge 4 139 7219 innerTail31_31 best.bound = innerCandidates31_30 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_30 : scan edge (search edge 3) 4 [139] 139 innerTail31_30 best = best := by
  rw [innerTail31_30, scan, if_neg (by decide), innerCandidates31_30_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_31
    (if 3 ≤ innerCandidates31_30.length then search edge 3 [139, 7219] innerCandidates31_30 7358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_31

def innerTail31_29 : List Nat := 7177 :: innerTail31_30
def innerCandidates31_29 : List Nat := [7933]
set_option maxRecDepth 4096 in
theorem innerCandidates31_29_checked : candidates edge 4 139 7177 innerTail31_30 best.bound = innerCandidates31_29 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_29 : scan edge (search edge 3) 4 [139] 139 innerTail31_29 best = best := by
  rw [innerTail31_29, scan, if_neg (by decide), innerCandidates31_29_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_30
    (if 3 ≤ innerCandidates31_29.length then search edge 3 [139, 7177] innerCandidates31_29 7316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_30

def innerTail31_28 : List Nat := 7057 :: innerTail31_29
def innerCandidates31_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_28_checked : candidates edge 4 139 7057 innerTail31_29 best.bound = innerCandidates31_28 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_28 : scan edge (search edge 3) 4 [139] 139 innerTail31_28 best = best := by
  rw [innerTail31_28, scan, if_neg (by decide), innerCandidates31_28_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_29
    (if 3 ≤ innerCandidates31_28.length then search edge 3 [139, 7057] innerCandidates31_28 7196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_29

def innerTail31_27 : List Nat := 6949 :: innerTail31_28
def innerCandidates31_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_27_checked : candidates edge 4 139 6949 innerTail31_28 best.bound = innerCandidates31_27 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_27 : scan edge (search edge 3) 4 [139] 139 innerTail31_27 best = best := by
  rw [innerTail31_27, scan, if_neg (by decide), innerCandidates31_27_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_28
    (if 3 ≤ innerCandidates31_27.length then search edge 3 [139, 6949] innerCandidates31_27 7088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_28

def innerTail31_26 : List Nat := 6841 :: innerTail31_27
def innerCandidates31_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_26_checked : candidates edge 4 139 6841 innerTail31_27 best.bound = innerCandidates31_26 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_26 : scan edge (search edge 3) 4 [139] 139 innerTail31_26 best = best := by
  rw [innerTail31_26, scan, if_neg (by decide), innerCandidates31_26_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_27
    (if 3 ≤ innerCandidates31_26.length then search edge 3 [139, 6841] innerCandidates31_26 6980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_27

def innerTail31_25 : List Nat := 6679 :: innerTail31_26
def innerCandidates31_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_25_checked : candidates edge 4 139 6679 innerTail31_26 best.bound = innerCandidates31_25 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_25 : scan edge (search edge 3) 4 [139] 139 innerTail31_25 best = best := by
  rw [innerTail31_25, scan, if_neg (by decide), innerCandidates31_25_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_26
    (if 3 ≤ innerCandidates31_25.length then search edge 3 [139, 6679] innerCandidates31_25 6818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_26

def innerTail31_24 : List Nat := 5869 :: innerTail31_25
def innerCandidates31_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_24_checked : candidates edge 4 139 5869 innerTail31_25 best.bound = innerCandidates31_24 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_24 : scan edge (search edge 3) 4 [139] 139 innerTail31_24 best = best := by
  rw [innerTail31_24, scan, if_neg (by decide), innerCandidates31_24_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_25
    (if 3 ≤ innerCandidates31_24.length then search edge 3 [139, 5869] innerCandidates31_24 6008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_25

def innerTail31_23 : List Nat := 5743 :: innerTail31_24
def innerCandidates31_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_23_checked : candidates edge 4 139 5743 innerTail31_24 best.bound = innerCandidates31_23 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_23 : scan edge (search edge 3) 4 [139] 139 innerTail31_23 best = best := by
  rw [innerTail31_23, scan, if_neg (by decide), innerCandidates31_23_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_24
    (if 3 ≤ innerCandidates31_23.length then search edge 3 [139, 5743] innerCandidates31_23 5882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_24

def innerTail31_22 : List Nat := 5323 :: innerTail31_23
def innerCandidates31_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_22_checked : candidates edge 4 139 5323 innerTail31_23 best.bound = innerCandidates31_22 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_22 : scan edge (search edge 3) 4 [139] 139 innerTail31_22 best = best := by
  rw [innerTail31_22, scan, if_neg (by decide), innerCandidates31_22_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_23
    (if 3 ≤ innerCandidates31_22.length then search edge 3 [139, 5323] innerCandidates31_22 5462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_23

def innerTail31_21 : List Nat := 5167 :: innerTail31_22
def innerCandidates31_21 : List Nat := [7177]
set_option maxRecDepth 4096 in
theorem innerCandidates31_21_checked : candidates edge 4 139 5167 innerTail31_22 best.bound = innerCandidates31_21 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_21 : scan edge (search edge 3) 4 [139] 139 innerTail31_21 best = best := by
  rw [innerTail31_21, scan, if_neg (by decide), innerCandidates31_21_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_22
    (if 3 ≤ innerCandidates31_21.length then search edge 3 [139, 5167] innerCandidates31_21 5306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_22

def innerTail31_20 : List Nat := 5059 :: innerTail31_21
def innerCandidates31_20 : List Nat := [5323, 8269]
set_option maxRecDepth 4096 in
theorem innerCandidates31_20_checked : candidates edge 4 139 5059 innerTail31_21 best.bound = innerCandidates31_20 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_20 : scan edge (search edge 3) 4 [139] 139 innerTail31_20 best = best := by
  rw [innerTail31_20, scan, if_neg (by decide), innerCandidates31_20_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_21
    (if 3 ≤ innerCandidates31_20.length then search edge 3 [139, 5059] innerCandidates31_20 5198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_21

def innerTail31_19 : List Nat := 4177 :: innerTail31_20
def innerCandidates31_19 : List Nat := [5323, 8053]
set_option maxRecDepth 4096 in
theorem innerCandidates31_19_checked : candidates edge 4 139 4177 innerTail31_20 best.bound = innerCandidates31_19 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_19 : scan edge (search edge 3) 4 [139] 139 innerTail31_19 best = best := by
  rw [innerTail31_19, scan, if_neg (by decide), innerCandidates31_19_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_20
    (if 3 ≤ innerCandidates31_19.length then search edge 3 [139, 4177] innerCandidates31_19 4316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_20

def innerTail31_18 : List Nat := 3739 :: innerTail31_19
def innerCandidates31_18 : List Nat := [6679]
set_option maxRecDepth 4096 in
theorem innerCandidates31_18_checked : candidates edge 4 139 3739 innerTail31_19 best.bound = innerCandidates31_18 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_18 : scan edge (search edge 3) 4 [139] 139 innerTail31_18 best = best := by
  rw [innerTail31_18, scan, if_neg (by decide), innerCandidates31_18_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_19
    (if 3 ≤ innerCandidates31_18.length then search edge 3 [139, 3739] innerCandidates31_18 3878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_19

def innerTail31_17 : List Nat := 3697 :: innerTail31_18
def innerCandidates31_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_17_checked : candidates edge 4 139 3697 innerTail31_18 best.bound = innerCandidates31_17 := by
  unfold candidates
  have row : edge 3697 = SearchData.row3697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_17 : scan edge (search edge 3) 4 [139] 139 innerTail31_17 best = best := by
  rw [innerTail31_17, scan, if_neg (by decide), innerCandidates31_17_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_18
    (if 3 ≤ innerCandidates31_17.length then search edge 3 [139, 3697] innerCandidates31_17 3836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_18

def innerTail31_16 : List Nat := 3313 :: innerTail31_17
def innerCandidates31_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_16_checked : candidates edge 4 139 3313 innerTail31_17 best.bound = innerCandidates31_16 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_16 : scan edge (search edge 3) 4 [139] 139 innerTail31_16 best = best := by
  rw [innerTail31_16, scan, if_neg (by decide), innerCandidates31_16_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_17
    (if 3 ≤ innerCandidates31_16.length then search edge 3 [139, 3313] innerCandidates31_16 3452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_17

def innerTail31_15 : List Nat := 3253 :: innerTail31_16
def innerCandidates31_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_15_checked : candidates edge 4 139 3253 innerTail31_16 best.bound = innerCandidates31_15 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_15 : scan edge (search edge 3) 4 [139] 139 innerTail31_15 best = best := by
  rw [innerTail31_15, scan, if_neg (by decide), innerCandidates31_15_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_16
    (if 3 ≤ innerCandidates31_15.length then search edge 3 [139, 3253] innerCandidates31_15 3392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_16

def innerTail31_14 : List Nat := 3229 :: innerTail31_15
def innerCandidates31_14 : List Nat := [6841]
set_option maxRecDepth 4096 in
theorem innerCandidates31_14_checked : candidates edge 4 139 3229 innerTail31_15 best.bound = innerCandidates31_14 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_14 : scan edge (search edge 3) 4 [139] 139 innerTail31_14 best = best := by
  rw [innerTail31_14, scan, if_neg (by decide), innerCandidates31_14_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_15
    (if 3 ≤ innerCandidates31_14.length then search edge 3 [139, 3229] innerCandidates31_14 3368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_15

def innerTail31_13 : List Nat := 2803 :: innerTail31_14
def innerCandidates31_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates31_13_checked : candidates edge 4 139 2803 innerTail31_14 best.bound = innerCandidates31_13 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_13 : scan edge (search edge 3) 4 [139] 139 innerTail31_13 best = best := by
  rw [innerTail31_13, scan, if_neg (by decide), innerCandidates31_13_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_14
    (if 3 ≤ innerCandidates31_13.length then search edge 3 [139, 2803] innerCandidates31_13 2942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_14

def innerTail31_12 : List Nat := 1597 :: innerTail31_13
def innerCandidates31_12 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates31_12_checked : candidates edge 4 139 1597 innerTail31_13 best.bound = innerCandidates31_12 := by
  unfold candidates
  have row : edge 1597 = PrimePairCache.smallRow1597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_12 : scan edge (search edge 3) 4 [139] 139 innerTail31_12 best = best := by
  rw [innerTail31_12, scan, if_neg (by decide), innerCandidates31_12_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_13
    (if 3 ≤ innerCandidates31_12.length then search edge 3 [139, 1597] innerCandidates31_12 1736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_13

def innerTail31_11 : List Nat := 1549 :: innerTail31_12
def innerCandidates31_11 : List Nat := [3253, 5059]
set_option maxRecDepth 4096 in
theorem innerCandidates31_11_checked : candidates edge 4 139 1549 innerTail31_12 best.bound = innerCandidates31_11 := by
  unfold candidates
  have row : edge 1549 = PrimePairCache.smallRow1549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_11 : scan edge (search edge 3) 4 [139] 139 innerTail31_11 best = best := by
  rw [innerTail31_11, scan, if_neg (by decide), innerCandidates31_11_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_12
    (if 3 ≤ innerCandidates31_11.length then search edge 3 [139, 1549] innerCandidates31_11 1688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_12

def innerTail31_10 : List Nat := 1447 :: innerTail31_11
def innerCandidates31_10 : List Nat := [1549, 2803, 3229, 3313, 6949]
set_option maxRecDepth 4096 in
theorem innerCandidates31_10_checked : candidates edge 4 139 1447 innerTail31_11 best.bound = innerCandidates31_10 := by
  unfold candidates
  have row : edge 1447 = PrimePairCache.smallRow1447.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild31_10 : search edge 3 [139, 1447] innerCandidates31_10 1586 best = best := by decide

theorem innerScan31_10 : scan edge (search edge 3) 4 [139] 139 innerTail31_10 best = best := by
  rw [innerTail31_10, scan, if_neg (by decide), innerCandidates31_10_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_11
    (if 3 ≤ innerCandidates31_10.length then search edge 3 [139, 1447] innerCandidates31_10 1586 best else best) = best
  rw [if_pos (by decide), innerChild31_10]
  exact innerScan31_11

def innerTail31_9 : List Nat := 1051 :: innerTail31_10
def innerCandidates31_9 : List Nat := [3739]
set_option maxRecDepth 4096 in
theorem innerCandidates31_9_checked : candidates edge 4 139 1051 innerTail31_10 best.bound = innerCandidates31_9 := by
  unfold candidates
  have row : edge 1051 = PrimePairCache.smallRow1051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_9 : scan edge (search edge 3) 4 [139] 139 innerTail31_9 best = best := by
  rw [innerTail31_9, scan, if_neg (by decide), innerCandidates31_9_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_10
    (if 3 ≤ innerCandidates31_9.length then search edge 3 [139, 1051] innerCandidates31_9 1190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_10

def innerTail31_8 : List Nat := 967 :: innerTail31_9
def innerCandidates31_8 : List Nat := [6949, 8269]
set_option maxRecDepth 4096 in
theorem innerCandidates31_8_checked : candidates edge 4 139 967 innerTail31_9 best.bound = innerCandidates31_8 := by
  unfold candidates
  have row : edge 967 = PrimePairCache.smallRow967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_8 : scan edge (search edge 3) 4 [139] 139 innerTail31_8 best = best := by
  rw [innerTail31_8, scan, if_neg (by decide), innerCandidates31_8_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_9
    (if 3 ≤ innerCandidates31_8.length then search edge 3 [139, 967] innerCandidates31_8 1106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_9

def innerTail31_7 : List Nat := 907 :: innerTail31_8
def innerCandidates31_7 : List Nat := [6841]
set_option maxRecDepth 4096 in
theorem innerCandidates31_7_checked : candidates edge 4 139 907 innerTail31_8 best.bound = innerCandidates31_7 := by
  unfold candidates
  have row : edge 907 = PrimePairCache.smallRow907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_7 : scan edge (search edge 3) 4 [139] 139 innerTail31_7 best = best := by
  rw [innerTail31_7, scan, if_neg (by decide), innerCandidates31_7_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_8
    (if 3 ≤ innerCandidates31_7.length then search edge 3 [139, 907] innerCandidates31_7 1046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_8

def innerTail31_6 : List Nat := 787 :: innerTail31_7
def innerCandidates31_6 : List Nat := [4177]
set_option maxRecDepth 4096 in
theorem innerCandidates31_6_checked : candidates edge 4 139 787 innerTail31_7 best.bound = innerCandidates31_6 := by
  unfold candidates
  have row : edge 787 = PrimePairCache.smallRow787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_6 : scan edge (search edge 3) 4 [139] 139 innerTail31_6 best = best := by
  rw [innerTail31_6, scan, if_neg (by decide), innerCandidates31_6_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_7
    (if 3 ≤ innerCandidates31_6.length then search edge 3 [139, 787] innerCandidates31_6 926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_7

def innerTail31_5 : List Nat := 709 :: innerTail31_6
def innerCandidates31_5 : List Nat := [967, 4177]
set_option maxRecDepth 4096 in
theorem innerCandidates31_5_checked : candidates edge 4 139 709 innerTail31_6 best.bound = innerCandidates31_5 := by
  unfold candidates
  have row : edge 709 = PrimePairCache.smallRow709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_5 : scan edge (search edge 3) 4 [139] 139 innerTail31_5 best = best := by
  rw [innerTail31_5, scan, if_neg (by decide), innerCandidates31_5_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_6
    (if 3 ≤ innerCandidates31_5.length then search edge 3 [139, 709] innerCandidates31_5 848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_6

def innerTail31_4 : List Nat := 661 :: innerTail31_5
def innerCandidates31_4 : List Nat := [1447]
set_option maxRecDepth 4096 in
theorem innerCandidates31_4_checked : candidates edge 4 139 661 innerTail31_5 best.bound = innerCandidates31_4 := by
  unfold candidates
  have row : edge 661 = PrimePairCache.smallRow661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_4 : scan edge (search edge 3) 4 [139] 139 innerTail31_4 best = best := by
  rw [innerTail31_4, scan, if_neg (by decide), innerCandidates31_4_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_5
    (if 3 ≤ innerCandidates31_4.length then search edge 3 [139, 661] innerCandidates31_4 800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_5

def innerTail31_3 : List Nat := 619 :: innerTail31_4
def innerCandidates31_3 : List Nat := [1051, 1597]
set_option maxRecDepth 4096 in
theorem innerCandidates31_3_checked : candidates edge 4 139 619 innerTail31_4 best.bound = innerCandidates31_3 := by
  unfold candidates
  have row : edge 619 = PrimePairCache.smallRow619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_3 : scan edge (search edge 3) 4 [139] 139 innerTail31_3 best = best := by
  rw [innerTail31_3, scan, if_neg (by decide), innerCandidates31_3_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_4
    (if 3 ≤ innerCandidates31_3.length then search edge 3 [139, 619] innerCandidates31_3 758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_4

def innerTail31_2 : List Nat := 547 :: innerTail31_3
def innerCandidates31_2 : List Nat := [661, 709, 787, 3229]
set_option maxRecDepth 4096 in
theorem innerCandidates31_2_checked : candidates edge 4 139 547 innerTail31_3 best.bound = innerCandidates31_2 := by
  unfold candidates
  have row : edge 547 = PrimePairCache.smallRow547.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild31_2 : search edge 3 [139, 547] innerCandidates31_2 686 best = best := by decide

theorem innerScan31_2 : scan edge (search edge 3) 4 [139] 139 innerTail31_2 best = best := by
  rw [innerTail31_2, scan, if_neg (by decide), innerCandidates31_2_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_3
    (if 3 ≤ innerCandidates31_2.length then search edge 3 [139, 547] innerCandidates31_2 686 best else best) = best
  rw [if_pos (by decide), innerChild31_2]
  exact innerScan31_3

def innerTail31_1 : List Nat := 457 :: innerTail31_2
def innerCandidates31_1 : List Nat := [1597]
set_option maxRecDepth 4096 in
theorem innerCandidates31_1_checked : candidates edge 4 139 457 innerTail31_2 best.bound = innerCandidates31_1 := by
  unfold candidates
  have row : edge 457 = PrimePairCache.smallRow457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_1 : scan edge (search edge 3) 4 [139] 139 innerTail31_1 best = best := by
  rw [innerTail31_1, scan, if_neg (by decide), innerCandidates31_1_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_2
    (if 3 ≤ innerCandidates31_1.length then search edge 3 [139, 457] innerCandidates31_1 596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_2

def innerTail31_0 : List Nat := 367 :: innerTail31_1
def innerCandidates31_0 : List Nat := [457, 5059]
set_option maxRecDepth 4096 in
theorem innerCandidates31_0_checked : candidates edge 4 139 367 innerTail31_1 best.bound = innerCandidates31_0 := by
  unfold candidates
  have row : edge 367 = PrimePairCache.smallRow367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan31_0 : scan edge (search edge 3) 4 [139] 139 innerTail31_0 best = best := by
  rw [innerTail31_0, scan, if_neg (by decide), innerCandidates31_0_checked]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_1
    (if 3 ≤ innerCandidates31_0.length then search edge 3 [139, 367] innerCandidates31_0 506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan31_1

theorem rootBranch31 : search edge 4 [139] rootCandidates31 139 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [139] 139 innerTail31_0 best = best
  exact innerScan31_0

end ProjectEulerSolutions.PrimePairUpper.Search
