import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates122 : List Nat := [709, 727, 907, 919, 1249, 1291, 1423, 1693, 2131, 2593, 3121, 3229, 3433, 3469, 3541, 3607, 3943, 4603, 4657, 4729, 4861, 4951, 5197, 5431, 5743, 5827, 5851, 5869, 6199, 6397, 6451, 6673, 7177]
set_option maxRecDepth 4096 in
theorem rootCandidates122_checked : candidates edge 5 0 691 tail123 best.bound = rootCandidates122 := by
  unfold candidates
  have row : edge 691 = PrimePairCache.smallRow691.contains := by funext q; rfl
  rw [row]
  decide

def innerTail122_33 : List Nat := []
theorem innerScan122_33 : scan edge (search edge 3) 4 [691] 691 innerTail122_33 best = best := rfl

def innerTail122_32 : List Nat := 7177 :: innerTail122_33
def innerCandidates122_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_32_checked : candidates edge 4 691 7177 innerTail122_33 best.bound = innerCandidates122_32 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_32 : scan edge (search edge 3) 4 [691] 691 innerTail122_32 best = best := by
  rw [innerTail122_32, scan, if_neg (by decide), innerCandidates122_32_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_33
    (if 3 ≤ innerCandidates122_32.length then search edge 3 [691, 7177] innerCandidates122_32 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_33

def innerTail122_31 : List Nat := 6673 :: innerTail122_32
def innerCandidates122_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_31_checked : candidates edge 4 691 6673 innerTail122_32 best.bound = innerCandidates122_31 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_31 : scan edge (search edge 3) 4 [691] 691 innerTail122_31 best = best := by
  rw [innerTail122_31, scan, if_neg (by decide), innerCandidates122_31_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_32
    (if 3 ≤ innerCandidates122_31.length then search edge 3 [691, 6673] innerCandidates122_31 7364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_32

def innerTail122_30 : List Nat := 6451 :: innerTail122_31
def innerCandidates122_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_30_checked : candidates edge 4 691 6451 innerTail122_31 best.bound = innerCandidates122_30 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_30 : scan edge (search edge 3) 4 [691] 691 innerTail122_30 best = best := by
  rw [innerTail122_30, scan, if_neg (by decide), innerCandidates122_30_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_31
    (if 3 ≤ innerCandidates122_30.length then search edge 3 [691, 6451] innerCandidates122_30 7142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_31

def innerTail122_29 : List Nat := 6397 :: innerTail122_30
def innerCandidates122_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_29_checked : candidates edge 4 691 6397 innerTail122_30 best.bound = innerCandidates122_29 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_29 : scan edge (search edge 3) 4 [691] 691 innerTail122_29 best = best := by
  rw [innerTail122_29, scan, if_neg (by decide), innerCandidates122_29_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_30
    (if 3 ≤ innerCandidates122_29.length then search edge 3 [691, 6397] innerCandidates122_29 7088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_30

def innerTail122_28 : List Nat := 6199 :: innerTail122_29
def innerCandidates122_28 : List Nat := [6673]
set_option maxRecDepth 4096 in
theorem innerCandidates122_28_checked : candidates edge 4 691 6199 innerTail122_29 best.bound = innerCandidates122_28 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_28 : scan edge (search edge 3) 4 [691] 691 innerTail122_28 best = best := by
  rw [innerTail122_28, scan, if_neg (by decide), innerCandidates122_28_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_29
    (if 3 ≤ innerCandidates122_28.length then search edge 3 [691, 6199] innerCandidates122_28 6890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_29

def innerTail122_27 : List Nat := 5869 :: innerTail122_28
def innerCandidates122_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_27_checked : candidates edge 4 691 5869 innerTail122_28 best.bound = innerCandidates122_27 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_27 : scan edge (search edge 3) 4 [691] 691 innerTail122_27 best = best := by
  rw [innerTail122_27, scan, if_neg (by decide), innerCandidates122_27_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_28
    (if 3 ≤ innerCandidates122_27.length then search edge 3 [691, 5869] innerCandidates122_27 6560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_28

def innerTail122_26 : List Nat := 5851 :: innerTail122_27
def innerCandidates122_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_26_checked : candidates edge 4 691 5851 innerTail122_27 best.bound = innerCandidates122_26 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_26 : scan edge (search edge 3) 4 [691] 691 innerTail122_26 best = best := by
  rw [innerTail122_26, scan, if_neg (by decide), innerCandidates122_26_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_27
    (if 3 ≤ innerCandidates122_26.length then search edge 3 [691, 5851] innerCandidates122_26 6542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_27

def innerTail122_25 : List Nat := 5827 :: innerTail122_26
def innerCandidates122_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_25_checked : candidates edge 4 691 5827 innerTail122_26 best.bound = innerCandidates122_25 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_25 : scan edge (search edge 3) 4 [691] 691 innerTail122_25 best = best := by
  rw [innerTail122_25, scan, if_neg (by decide), innerCandidates122_25_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_26
    (if 3 ≤ innerCandidates122_25.length then search edge 3 [691, 5827] innerCandidates122_25 6518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_26

def innerTail122_24 : List Nat := 5743 :: innerTail122_25
def innerCandidates122_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_24_checked : candidates edge 4 691 5743 innerTail122_25 best.bound = innerCandidates122_24 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_24 : scan edge (search edge 3) 4 [691] 691 innerTail122_24 best = best := by
  rw [innerTail122_24, scan, if_neg (by decide), innerCandidates122_24_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_25
    (if 3 ≤ innerCandidates122_24.length then search edge 3 [691, 5743] innerCandidates122_24 6434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_25

def innerTail122_23 : List Nat := 5431 :: innerTail122_24
def innerCandidates122_23 : List Nat := [5743]
set_option maxRecDepth 4096 in
theorem innerCandidates122_23_checked : candidates edge 4 691 5431 innerTail122_24 best.bound = innerCandidates122_23 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_23 : scan edge (search edge 3) 4 [691] 691 innerTail122_23 best = best := by
  rw [innerTail122_23, scan, if_neg (by decide), innerCandidates122_23_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_24
    (if 3 ≤ innerCandidates122_23.length then search edge 3 [691, 5431] innerCandidates122_23 6122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_24

def innerTail122_22 : List Nat := 5197 :: innerTail122_23
def innerCandidates122_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_22_checked : candidates edge 4 691 5197 innerTail122_23 best.bound = innerCandidates122_22 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_22 : scan edge (search edge 3) 4 [691] 691 innerTail122_22 best = best := by
  rw [innerTail122_22, scan, if_neg (by decide), innerCandidates122_22_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_23
    (if 3 ≤ innerCandidates122_22.length then search edge 3 [691, 5197] innerCandidates122_22 5888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_23

def innerTail122_21 : List Nat := 4951 :: innerTail122_22
def innerCandidates122_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_21_checked : candidates edge 4 691 4951 innerTail122_22 best.bound = innerCandidates122_21 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_21 : scan edge (search edge 3) 4 [691] 691 innerTail122_21 best = best := by
  rw [innerTail122_21, scan, if_neg (by decide), innerCandidates122_21_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_22
    (if 3 ≤ innerCandidates122_21.length then search edge 3 [691, 4951] innerCandidates122_21 5642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_22

def innerTail122_20 : List Nat := 4861 :: innerTail122_21
def innerCandidates122_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_20_checked : candidates edge 4 691 4861 innerTail122_21 best.bound = innerCandidates122_20 := by
  unfold candidates
  have row : edge 4861 = SearchData.row4861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_20 : scan edge (search edge 3) 4 [691] 691 innerTail122_20 best = best := by
  rw [innerTail122_20, scan, if_neg (by decide), innerCandidates122_20_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_21
    (if 3 ≤ innerCandidates122_20.length then search edge 3 [691, 4861] innerCandidates122_20 5552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_21

def innerTail122_19 : List Nat := 4729 :: innerTail122_20
def innerCandidates122_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_19_checked : candidates edge 4 691 4729 innerTail122_20 best.bound = innerCandidates122_19 := by
  unfold candidates
  have row : edge 4729 = SearchData.row4729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_19 : scan edge (search edge 3) 4 [691] 691 innerTail122_19 best = best := by
  rw [innerTail122_19, scan, if_neg (by decide), innerCandidates122_19_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_20
    (if 3 ≤ innerCandidates122_19.length then search edge 3 [691, 4729] innerCandidates122_19 5420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_20

def innerTail122_18 : List Nat := 4657 :: innerTail122_19
def innerCandidates122_18 : List Nat := [7177]
set_option maxRecDepth 4096 in
theorem innerCandidates122_18_checked : candidates edge 4 691 4657 innerTail122_19 best.bound = innerCandidates122_18 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_18 : scan edge (search edge 3) 4 [691] 691 innerTail122_18 best = best := by
  rw [innerTail122_18, scan, if_neg (by decide), innerCandidates122_18_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_19
    (if 3 ≤ innerCandidates122_18.length then search edge 3 [691, 4657] innerCandidates122_18 5348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_19

def innerTail122_17 : List Nat := 4603 :: innerTail122_18
def innerCandidates122_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_17_checked : candidates edge 4 691 4603 innerTail122_18 best.bound = innerCandidates122_17 := by
  unfold candidates
  have row : edge 4603 = SearchData.row4603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_17 : scan edge (search edge 3) 4 [691] 691 innerTail122_17 best = best := by
  rw [innerTail122_17, scan, if_neg (by decide), innerCandidates122_17_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_18
    (if 3 ≤ innerCandidates122_17.length then search edge 3 [691, 4603] innerCandidates122_17 5294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_18

def innerTail122_16 : List Nat := 3943 :: innerTail122_17
def innerCandidates122_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_16_checked : candidates edge 4 691 3943 innerTail122_17 best.bound = innerCandidates122_16 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_16 : scan edge (search edge 3) 4 [691] 691 innerTail122_16 best = best := by
  rw [innerTail122_16, scan, if_neg (by decide), innerCandidates122_16_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_17
    (if 3 ≤ innerCandidates122_16.length then search edge 3 [691, 3943] innerCandidates122_16 4634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_17

def innerTail122_15 : List Nat := 3607 :: innerTail122_16
def innerCandidates122_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_15_checked : candidates edge 4 691 3607 innerTail122_16 best.bound = innerCandidates122_15 := by
  unfold candidates
  have row : edge 3607 = SearchData.row3607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_15 : scan edge (search edge 3) 4 [691] 691 innerTail122_15 best = best := by
  rw [innerTail122_15, scan, if_neg (by decide), innerCandidates122_15_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_16
    (if 3 ≤ innerCandidates122_15.length then search edge 3 [691, 3607] innerCandidates122_15 4298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_16

def innerTail122_14 : List Nat := 3541 :: innerTail122_15
def innerCandidates122_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_14_checked : candidates edge 4 691 3541 innerTail122_15 best.bound = innerCandidates122_14 := by
  unfold candidates
  have row : edge 3541 = SearchData.row3541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_14 : scan edge (search edge 3) 4 [691] 691 innerTail122_14 best = best := by
  rw [innerTail122_14, scan, if_neg (by decide), innerCandidates122_14_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_15
    (if 3 ≤ innerCandidates122_14.length then search edge 3 [691, 3541] innerCandidates122_14 4232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_15

def innerTail122_13 : List Nat := 3469 :: innerTail122_14
def innerCandidates122_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_13_checked : candidates edge 4 691 3469 innerTail122_14 best.bound = innerCandidates122_13 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_13 : scan edge (search edge 3) 4 [691] 691 innerTail122_13 best = best := by
  rw [innerTail122_13, scan, if_neg (by decide), innerCandidates122_13_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_14
    (if 3 ≤ innerCandidates122_13.length then search edge 3 [691, 3469] innerCandidates122_13 4160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_14

def innerTail122_12 : List Nat := 3433 :: innerTail122_13
def innerCandidates122_12 : List Nat := [6451]
set_option maxRecDepth 4096 in
theorem innerCandidates122_12_checked : candidates edge 4 691 3433 innerTail122_13 best.bound = innerCandidates122_12 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_12 : scan edge (search edge 3) 4 [691] 691 innerTail122_12 best = best := by
  rw [innerTail122_12, scan, if_neg (by decide), innerCandidates122_12_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_13
    (if 3 ≤ innerCandidates122_12.length then search edge 3 [691, 3433] innerCandidates122_12 4124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_13

def innerTail122_11 : List Nat := 3229 :: innerTail122_12
def innerCandidates122_11 : List Nat := [6673]
set_option maxRecDepth 4096 in
theorem innerCandidates122_11_checked : candidates edge 4 691 3229 innerTail122_12 best.bound = innerCandidates122_11 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_11 : scan edge (search edge 3) 4 [691] 691 innerTail122_11 best = best := by
  rw [innerTail122_11, scan, if_neg (by decide), innerCandidates122_11_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_12
    (if 3 ≤ innerCandidates122_11.length then search edge 3 [691, 3229] innerCandidates122_11 3920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_12

def innerTail122_10 : List Nat := 3121 :: innerTail122_11
def innerCandidates122_10 : List Nat := [3433, 5851]
set_option maxRecDepth 4096 in
theorem innerCandidates122_10_checked : candidates edge 4 691 3121 innerTail122_11 best.bound = innerCandidates122_10 := by
  unfold candidates
  have row : edge 3121 = SearchData.row3121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_10 : scan edge (search edge 3) 4 [691] 691 innerTail122_10 best = best := by
  rw [innerTail122_10, scan, if_neg (by decide), innerCandidates122_10_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_11
    (if 3 ≤ innerCandidates122_10.length then search edge 3 [691, 3121] innerCandidates122_10 3812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_11

def innerTail122_9 : List Nat := 2593 :: innerTail122_10
def innerCandidates122_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_9_checked : candidates edge 4 691 2593 innerTail122_10 best.bound = innerCandidates122_9 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_9 : scan edge (search edge 3) 4 [691] 691 innerTail122_9 best = best := by
  rw [innerTail122_9, scan, if_neg (by decide), innerCandidates122_9_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_10
    (if 3 ≤ innerCandidates122_9.length then search edge 3 [691, 2593] innerCandidates122_9 3284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_10

def innerTail122_8 : List Nat := 2131 :: innerTail122_9
def innerCandidates122_8 : List Nat := [3541]
set_option maxRecDepth 4096 in
theorem innerCandidates122_8_checked : candidates edge 4 691 2131 innerTail122_9 best.bound = innerCandidates122_8 := by
  unfold candidates
  have row : edge 2131 = PrimePairCache.smallRow2131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_8 : scan edge (search edge 3) 4 [691] 691 innerTail122_8 best = best := by
  rw [innerTail122_8, scan, if_neg (by decide), innerCandidates122_8_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_9
    (if 3 ≤ innerCandidates122_8.length then search edge 3 [691, 2131] innerCandidates122_8 2822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_9

def innerTail122_7 : List Nat := 1693 :: innerTail122_8
def innerCandidates122_7 : List Nat := [2593]
set_option maxRecDepth 4096 in
theorem innerCandidates122_7_checked : candidates edge 4 691 1693 innerTail122_8 best.bound = innerCandidates122_7 := by
  unfold candidates
  have row : edge 1693 = PrimePairCache.smallRow1693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_7 : scan edge (search edge 3) 4 [691] 691 innerTail122_7 best = best := by
  rw [innerTail122_7, scan, if_neg (by decide), innerCandidates122_7_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_8
    (if 3 ≤ innerCandidates122_7.length then search edge 3 [691, 1693] innerCandidates122_7 2384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_8

def innerTail122_6 : List Nat := 1423 :: innerTail122_7
def innerCandidates122_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_6_checked : candidates edge 4 691 1423 innerTail122_7 best.bound = innerCandidates122_6 := by
  unfold candidates
  have row : edge 1423 = PrimePairCache.smallRow1423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_6 : scan edge (search edge 3) 4 [691] 691 innerTail122_6 best = best := by
  rw [innerTail122_6, scan, if_neg (by decide), innerCandidates122_6_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_7
    (if 3 ≤ innerCandidates122_6.length then search edge 3 [691, 1423] innerCandidates122_6 2114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_7

def innerTail122_5 : List Nat := 1291 :: innerTail122_6
def innerCandidates122_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_5_checked : candidates edge 4 691 1291 innerTail122_6 best.bound = innerCandidates122_5 := by
  unfold candidates
  have row : edge 1291 = PrimePairCache.smallRow1291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_5 : scan edge (search edge 3) 4 [691] 691 innerTail122_5 best = best := by
  rw [innerTail122_5, scan, if_neg (by decide), innerCandidates122_5_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_6
    (if 3 ≤ innerCandidates122_5.length then search edge 3 [691, 1291] innerCandidates122_5 1982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_6

def innerTail122_4 : List Nat := 1249 :: innerTail122_5
def innerCandidates122_4 : List Nat := [2131, 5851]
set_option maxRecDepth 4096 in
theorem innerCandidates122_4_checked : candidates edge 4 691 1249 innerTail122_5 best.bound = innerCandidates122_4 := by
  unfold candidates
  have row : edge 1249 = PrimePairCache.smallRow1249.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_4 : scan edge (search edge 3) 4 [691] 691 innerTail122_4 best = best := by
  rw [innerTail122_4, scan, if_neg (by decide), innerCandidates122_4_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_5
    (if 3 ≤ innerCandidates122_4.length then search edge 3 [691, 1249] innerCandidates122_4 1940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_5

def innerTail122_3 : List Nat := 919 :: innerTail122_4
def innerCandidates122_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates122_3_checked : candidates edge 4 691 919 innerTail122_4 best.bound = innerCandidates122_3 := by
  unfold candidates
  have row : edge 919 = PrimePairCache.smallRow919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_3 : scan edge (search edge 3) 4 [691] 691 innerTail122_3 best = best := by
  rw [innerTail122_3, scan, if_neg (by decide), innerCandidates122_3_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_4
    (if 3 ≤ innerCandidates122_3.length then search edge 3 [691, 919] innerCandidates122_3 1610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_4

def innerTail122_2 : List Nat := 907 :: innerTail122_3
def innerCandidates122_2 : List Nat := [4861, 5851]
set_option maxRecDepth 4096 in
theorem innerCandidates122_2_checked : candidates edge 4 691 907 innerTail122_3 best.bound = innerCandidates122_2 := by
  unfold candidates
  have row : edge 907 = PrimePairCache.smallRow907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_2 : scan edge (search edge 3) 4 [691] 691 innerTail122_2 best = best := by
  rw [innerTail122_2, scan, if_neg (by decide), innerCandidates122_2_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_3
    (if 3 ≤ innerCandidates122_2.length then search edge 3 [691, 907] innerCandidates122_2 1598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_3

def innerTail122_1 : List Nat := 727 :: innerTail122_2
def innerCandidates122_1 : List Nat := [5197]
set_option maxRecDepth 4096 in
theorem innerCandidates122_1_checked : candidates edge 4 691 727 innerTail122_2 best.bound = innerCandidates122_1 := by
  unfold candidates
  have row : edge 727 = PrimePairCache.smallRow727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_1 : scan edge (search edge 3) 4 [691] 691 innerTail122_1 best = best := by
  rw [innerTail122_1, scan, if_neg (by decide), innerCandidates122_1_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_2
    (if 3 ≤ innerCandidates122_1.length then search edge 3 [691, 727] innerCandidates122_1 1418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_2

def innerTail122_0 : List Nat := 709 :: innerTail122_1
def innerCandidates122_0 : List Nat := [3469]
set_option maxRecDepth 4096 in
theorem innerCandidates122_0_checked : candidates edge 4 691 709 innerTail122_1 best.bound = innerCandidates122_0 := by
  unfold candidates
  have row : edge 709 = PrimePairCache.smallRow709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan122_0 : scan edge (search edge 3) 4 [691] 691 innerTail122_0 best = best := by
  rw [innerTail122_0, scan, if_neg (by decide), innerCandidates122_0_checked]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_1
    (if 3 ≤ innerCandidates122_0.length then search edge 3 [691, 709] innerCandidates122_0 1400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan122_1

theorem rootBranch122 : search edge 4 [691] rootCandidates122 691 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [691] 691 innerTail122_0 best = best
  exact innerScan122_0

def rootCandidates121 : List Nat := [719, 911, 1499, 2417, 2843, 2939, 2957, 3083, 3209, 3797, 3929, 3989, 4073, 4637, 4649, 4679, 4937, 5657, 6047, 6131, 6317, 6653, 6659, 7499, 7643]
set_option maxRecDepth 4096 in
theorem rootCandidates121_checked : candidates edge 5 0 683 tail122 best.bound = rootCandidates121 := by
  unfold candidates
  have row : edge 683 = PrimePairCache.smallRow683.contains := by funext q; rfl
  rw [row]
  decide

def innerTail121_25 : List Nat := []
theorem innerScan121_25 : scan edge (search edge 3) 4 [683] 683 innerTail121_25 best = best := rfl

def innerTail121_24 : List Nat := 7643 :: innerTail121_25
def innerCandidates121_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_24_checked : candidates edge 4 683 7643 innerTail121_25 best.bound = innerCandidates121_24 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_24 : scan edge (search edge 3) 4 [683] 683 innerTail121_24 best = best := by
  rw [innerTail121_24, scan, if_neg (by decide), innerCandidates121_24_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_25
    (if 3 ≤ innerCandidates121_24.length then search edge 3 [683, 7643] innerCandidates121_24 8326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_25

def innerTail121_23 : List Nat := 7499 :: innerTail121_24
def innerCandidates121_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_23_checked : candidates edge 4 683 7499 innerTail121_24 best.bound = innerCandidates121_23 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_23 : scan edge (search edge 3) 4 [683] 683 innerTail121_23 best = best := by
  rw [innerTail121_23, scan, if_neg (by decide), innerCandidates121_23_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_24
    (if 3 ≤ innerCandidates121_23.length then search edge 3 [683, 7499] innerCandidates121_23 8182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_24

def innerTail121_22 : List Nat := 6659 :: innerTail121_23
def innerCandidates121_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_22_checked : candidates edge 4 683 6659 innerTail121_23 best.bound = innerCandidates121_22 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_22 : scan edge (search edge 3) 4 [683] 683 innerTail121_22 best = best := by
  rw [innerTail121_22, scan, if_neg (by decide), innerCandidates121_22_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_23
    (if 3 ≤ innerCandidates121_22.length then search edge 3 [683, 6659] innerCandidates121_22 7342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_23

def innerTail121_21 : List Nat := 6653 :: innerTail121_22
def innerCandidates121_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_21_checked : candidates edge 4 683 6653 innerTail121_22 best.bound = innerCandidates121_21 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_21 : scan edge (search edge 3) 4 [683] 683 innerTail121_21 best = best := by
  rw [innerTail121_21, scan, if_neg (by decide), innerCandidates121_21_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_22
    (if 3 ≤ innerCandidates121_21.length then search edge 3 [683, 6653] innerCandidates121_21 7336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_22

def innerTail121_20 : List Nat := 6317 :: innerTail121_21
def innerCandidates121_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_20_checked : candidates edge 4 683 6317 innerTail121_21 best.bound = innerCandidates121_20 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_20 : scan edge (search edge 3) 4 [683] 683 innerTail121_20 best = best := by
  rw [innerTail121_20, scan, if_neg (by decide), innerCandidates121_20_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_21
    (if 3 ≤ innerCandidates121_20.length then search edge 3 [683, 6317] innerCandidates121_20 7000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_21

def innerTail121_19 : List Nat := 6131 :: innerTail121_20
def innerCandidates121_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_19_checked : candidates edge 4 683 6131 innerTail121_20 best.bound = innerCandidates121_19 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_19 : scan edge (search edge 3) 4 [683] 683 innerTail121_19 best = best := by
  rw [innerTail121_19, scan, if_neg (by decide), innerCandidates121_19_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_20
    (if 3 ≤ innerCandidates121_19.length then search edge 3 [683, 6131] innerCandidates121_19 6814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_20

def innerTail121_18 : List Nat := 6047 :: innerTail121_19
def innerCandidates121_18 : List Nat := [7643]
set_option maxRecDepth 4096 in
theorem innerCandidates121_18_checked : candidates edge 4 683 6047 innerTail121_19 best.bound = innerCandidates121_18 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_18 : scan edge (search edge 3) 4 [683] 683 innerTail121_18 best = best := by
  rw [innerTail121_18, scan, if_neg (by decide), innerCandidates121_18_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_19
    (if 3 ≤ innerCandidates121_18.length then search edge 3 [683, 6047] innerCandidates121_18 6730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_19

def innerTail121_17 : List Nat := 5657 :: innerTail121_18
def innerCandidates121_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_17_checked : candidates edge 4 683 5657 innerTail121_18 best.bound = innerCandidates121_17 := by
  unfold candidates
  have row : edge 5657 = SearchData.row5657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_17 : scan edge (search edge 3) 4 [683] 683 innerTail121_17 best = best := by
  rw [innerTail121_17, scan, if_neg (by decide), innerCandidates121_17_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_18
    (if 3 ≤ innerCandidates121_17.length then search edge 3 [683, 5657] innerCandidates121_17 6340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_18

def innerTail121_16 : List Nat := 4937 :: innerTail121_17
def innerCandidates121_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_16_checked : candidates edge 4 683 4937 innerTail121_17 best.bound = innerCandidates121_16 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_16 : scan edge (search edge 3) 4 [683] 683 innerTail121_16 best = best := by
  rw [innerTail121_16, scan, if_neg (by decide), innerCandidates121_16_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_17
    (if 3 ≤ innerCandidates121_16.length then search edge 3 [683, 4937] innerCandidates121_16 5620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_17

def innerTail121_15 : List Nat := 4679 :: innerTail121_16
def innerCandidates121_15 : List Nat := [6653]
set_option maxRecDepth 4096 in
theorem innerCandidates121_15_checked : candidates edge 4 683 4679 innerTail121_16 best.bound = innerCandidates121_15 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_15 : scan edge (search edge 3) 4 [683] 683 innerTail121_15 best = best := by
  rw [innerTail121_15, scan, if_neg (by decide), innerCandidates121_15_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_16
    (if 3 ≤ innerCandidates121_15.length then search edge 3 [683, 4679] innerCandidates121_15 5362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_16

def innerTail121_14 : List Nat := 4649 :: innerTail121_15
def innerCandidates121_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_14_checked : candidates edge 4 683 4649 innerTail121_15 best.bound = innerCandidates121_14 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_14 : scan edge (search edge 3) 4 [683] 683 innerTail121_14 best = best := by
  rw [innerTail121_14, scan, if_neg (by decide), innerCandidates121_14_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_15
    (if 3 ≤ innerCandidates121_14.length then search edge 3 [683, 4649] innerCandidates121_14 5332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_15

def innerTail121_13 : List Nat := 4637 :: innerTail121_14
def innerCandidates121_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_13_checked : candidates edge 4 683 4637 innerTail121_14 best.bound = innerCandidates121_13 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_13 : scan edge (search edge 3) 4 [683] 683 innerTail121_13 best = best := by
  rw [innerTail121_13, scan, if_neg (by decide), innerCandidates121_13_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_14
    (if 3 ≤ innerCandidates121_13.length then search edge 3 [683, 4637] innerCandidates121_13 5320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_14

def innerTail121_12 : List Nat := 4073 :: innerTail121_13
def innerCandidates121_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_12_checked : candidates edge 4 683 4073 innerTail121_13 best.bound = innerCandidates121_12 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_12 : scan edge (search edge 3) 4 [683] 683 innerTail121_12 best = best := by
  rw [innerTail121_12, scan, if_neg (by decide), innerCandidates121_12_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_13
    (if 3 ≤ innerCandidates121_12.length then search edge 3 [683, 4073] innerCandidates121_12 4756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_13

def innerTail121_11 : List Nat := 3989 :: innerTail121_12
def innerCandidates121_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_11_checked : candidates edge 4 683 3989 innerTail121_12 best.bound = innerCandidates121_11 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_11 : scan edge (search edge 3) 4 [683] 683 innerTail121_11 best = best := by
  rw [innerTail121_11, scan, if_neg (by decide), innerCandidates121_11_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_12
    (if 3 ≤ innerCandidates121_11.length then search edge 3 [683, 3989] innerCandidates121_11 4672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_12

def innerTail121_10 : List Nat := 3929 :: innerTail121_11
def innerCandidates121_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_10_checked : candidates edge 4 683 3929 innerTail121_11 best.bound = innerCandidates121_10 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_10 : scan edge (search edge 3) 4 [683] 683 innerTail121_10 best = best := by
  rw [innerTail121_10, scan, if_neg (by decide), innerCandidates121_10_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_11
    (if 3 ≤ innerCandidates121_10.length then search edge 3 [683, 3929] innerCandidates121_10 4612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_11

def innerTail121_9 : List Nat := 3797 :: innerTail121_10
def innerCandidates121_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_9_checked : candidates edge 4 683 3797 innerTail121_10 best.bound = innerCandidates121_9 := by
  unfold candidates
  have row : edge 3797 = SearchData.row3797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_9 : scan edge (search edge 3) 4 [683] 683 innerTail121_9 best = best := by
  rw [innerTail121_9, scan, if_neg (by decide), innerCandidates121_9_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_10
    (if 3 ≤ innerCandidates121_9.length then search edge 3 [683, 3797] innerCandidates121_9 4480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_10

def innerTail121_8 : List Nat := 3209 :: innerTail121_9
def innerCandidates121_8 : List Nat := [6653]
set_option maxRecDepth 4096 in
theorem innerCandidates121_8_checked : candidates edge 4 683 3209 innerTail121_9 best.bound = innerCandidates121_8 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_8 : scan edge (search edge 3) 4 [683] 683 innerTail121_8 best = best := by
  rw [innerTail121_8, scan, if_neg (by decide), innerCandidates121_8_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_9
    (if 3 ≤ innerCandidates121_8.length then search edge 3 [683, 3209] innerCandidates121_8 3892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_9

def innerTail121_7 : List Nat := 3083 :: innerTail121_8
def innerCandidates121_7 : List Nat := [3797]
set_option maxRecDepth 4096 in
theorem innerCandidates121_7_checked : candidates edge 4 683 3083 innerTail121_8 best.bound = innerCandidates121_7 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_7 : scan edge (search edge 3) 4 [683] 683 innerTail121_7 best = best := by
  rw [innerTail121_7, scan, if_neg (by decide), innerCandidates121_7_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_8
    (if 3 ≤ innerCandidates121_7.length then search edge 3 [683, 3083] innerCandidates121_7 3766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_8

def innerTail121_6 : List Nat := 2957 :: innerTail121_7
def innerCandidates121_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_6_checked : candidates edge 4 683 2957 innerTail121_7 best.bound = innerCandidates121_6 := by
  unfold candidates
  have row : edge 2957 = SearchData.row2957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_6 : scan edge (search edge 3) 4 [683] 683 innerTail121_6 best = best := by
  rw [innerTail121_6, scan, if_neg (by decide), innerCandidates121_6_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_7
    (if 3 ≤ innerCandidates121_6.length then search edge 3 [683, 2957] innerCandidates121_6 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_7

def innerTail121_5 : List Nat := 2939 :: innerTail121_6
def innerCandidates121_5 : List Nat := [4679]
set_option maxRecDepth 4096 in
theorem innerCandidates121_5_checked : candidates edge 4 683 2939 innerTail121_6 best.bound = innerCandidates121_5 := by
  unfold candidates
  have row : edge 2939 = SearchData.row2939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_5 : scan edge (search edge 3) 4 [683] 683 innerTail121_5 best = best := by
  rw [innerTail121_5, scan, if_neg (by decide), innerCandidates121_5_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_6
    (if 3 ≤ innerCandidates121_5.length then search edge 3 [683, 2939] innerCandidates121_5 3622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_6

def innerTail121_4 : List Nat := 2843 :: innerTail121_5
def innerCandidates121_4 : List Nat := [3989, 5657, 6047]
set_option maxRecDepth 4096 in
theorem innerCandidates121_4_checked : candidates edge 4 683 2843 innerTail121_5 best.bound = innerCandidates121_4 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild121_4 : search edge 3 [683, 2843] innerCandidates121_4 3526 best = best := by decide

theorem innerScan121_4 : scan edge (search edge 3) 4 [683] 683 innerTail121_4 best = best := by
  rw [innerTail121_4, scan, if_neg (by decide), innerCandidates121_4_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_5
    (if 3 ≤ innerCandidates121_4.length then search edge 3 [683, 2843] innerCandidates121_4 3526 best else best) = best
  rw [if_pos (by decide), innerChild121_4]
  exact innerScan121_5

def innerTail121_3 : List Nat := 2417 :: innerTail121_4
def innerCandidates121_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_3_checked : candidates edge 4 683 2417 innerTail121_4 best.bound = innerCandidates121_3 := by
  unfold candidates
  have row : edge 2417 = PrimePairCache.smallRow2417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_3 : scan edge (search edge 3) 4 [683] 683 innerTail121_3 best = best := by
  rw [innerTail121_3, scan, if_neg (by decide), innerCandidates121_3_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_4
    (if 3 ≤ innerCandidates121_3.length then search edge 3 [683, 2417] innerCandidates121_3 3100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_4

def innerTail121_2 : List Nat := 1499 :: innerTail121_3
def innerCandidates121_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_2_checked : candidates edge 4 683 1499 innerTail121_3 best.bound = innerCandidates121_2 := by
  unfold candidates
  have row : edge 1499 = PrimePairCache.smallRow1499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_2 : scan edge (search edge 3) 4 [683] 683 innerTail121_2 best = best := by
  rw [innerTail121_2, scan, if_neg (by decide), innerCandidates121_2_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_3
    (if 3 ≤ innerCandidates121_2.length then search edge 3 [683, 1499] innerCandidates121_2 2182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_3

def innerTail121_1 : List Nat := 911 :: innerTail121_2
def innerCandidates121_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates121_1_checked : candidates edge 4 683 911 innerTail121_2 best.bound = innerCandidates121_1 := by
  unfold candidates
  have row : edge 911 = PrimePairCache.smallRow911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_1 : scan edge (search edge 3) 4 [683] 683 innerTail121_1 best = best := by
  rw [innerTail121_1, scan, if_neg (by decide), innerCandidates121_1_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_2
    (if 3 ≤ innerCandidates121_1.length then search edge 3 [683, 911] innerCandidates121_1 1594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_2

def innerTail121_0 : List Nat := 719 :: innerTail121_1
def innerCandidates121_0 : List Nat := [911, 6317]
set_option maxRecDepth 4096 in
theorem innerCandidates121_0_checked : candidates edge 4 683 719 innerTail121_1 best.bound = innerCandidates121_0 := by
  unfold candidates
  have row : edge 719 = PrimePairCache.smallRow719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan121_0 : scan edge (search edge 3) 4 [683] 683 innerTail121_0 best = best := by
  rw [innerTail121_0, scan, if_neg (by decide), innerCandidates121_0_checked]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_1
    (if 3 ≤ innerCandidates121_0.length then search edge 3 [683, 719] innerCandidates121_0 1402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan121_1

theorem rootBranch121 : search edge 4 [683] rootCandidates121 683 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [683] 683 innerTail121_0 best = best
  exact innerScan121_0

end ProjectEulerSolutions.PrimePairUpper.Search
