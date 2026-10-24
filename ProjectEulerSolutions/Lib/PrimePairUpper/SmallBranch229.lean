import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates48 : List Nat := [239, 251, 347, 617, 881, 911, 983, 1151, 1163, 1187, 1583, 2969, 3083, 3167, 3329, 3389, 4337, 4793, 5009, 5297, 5591, 5807, 5813, 6011, 6263, 6359, 6947, 7001, 7043, 7187, 7547, 7607, 7949, 8123, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates48_checked : candidates edge 5 0 233 tail49 best.bound = rootCandidates48 := by
  unfold candidates
  have row : edge 233 = PrimePairCache.smallRow233.contains := by funext q; rfl
  rw [row]
  decide

def innerTail48_35 : List Nat := []
theorem innerScan48_35 : scan edge (search edge 3) 4 [233] 233 innerTail48_35 best = best := rfl

def innerTail48_34 : List Nat := 8291 :: innerTail48_35
def innerCandidates48_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_34_checked : candidates edge 4 233 8291 innerTail48_35 best.bound = innerCandidates48_34 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_34 : scan edge (search edge 3) 4 [233] 233 innerTail48_34 best = best := by
  rw [innerTail48_34, scan, if_neg (by decide), innerCandidates48_34_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_35
    (if 3 ≤ innerCandidates48_34.length then search edge 3 [233, 8291] innerCandidates48_34 8524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_35

def innerTail48_33 : List Nat := 8123 :: innerTail48_34
def innerCandidates48_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_33_checked : candidates edge 4 233 8123 innerTail48_34 best.bound = innerCandidates48_33 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_33 : scan edge (search edge 3) 4 [233] 233 innerTail48_33 best = best := by
  rw [innerTail48_33, scan, if_neg (by decide), innerCandidates48_33_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_34
    (if 3 ≤ innerCandidates48_33.length then search edge 3 [233, 8123] innerCandidates48_33 8356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_34

def innerTail48_32 : List Nat := 7949 :: innerTail48_33
def innerCandidates48_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_32_checked : candidates edge 4 233 7949 innerTail48_33 best.bound = innerCandidates48_32 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_32 : scan edge (search edge 3) 4 [233] 233 innerTail48_32 best = best := by
  rw [innerTail48_32, scan, if_neg (by decide), innerCandidates48_32_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_33
    (if 3 ≤ innerCandidates48_32.length then search edge 3 [233, 7949] innerCandidates48_32 8182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_33

def innerTail48_31 : List Nat := 7607 :: innerTail48_32
def innerCandidates48_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_31_checked : candidates edge 4 233 7607 innerTail48_32 best.bound = innerCandidates48_31 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_31 : scan edge (search edge 3) 4 [233] 233 innerTail48_31 best = best := by
  rw [innerTail48_31, scan, if_neg (by decide), innerCandidates48_31_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_32
    (if 3 ≤ innerCandidates48_31.length then search edge 3 [233, 7607] innerCandidates48_31 7840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_32

def innerTail48_30 : List Nat := 7547 :: innerTail48_31
def innerCandidates48_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_30_checked : candidates edge 4 233 7547 innerTail48_31 best.bound = innerCandidates48_30 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_30 : scan edge (search edge 3) 4 [233] 233 innerTail48_30 best = best := by
  rw [innerTail48_30, scan, if_neg (by decide), innerCandidates48_30_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_31
    (if 3 ≤ innerCandidates48_30.length then search edge 3 [233, 7547] innerCandidates48_30 7780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_31

def innerTail48_29 : List Nat := 7187 :: innerTail48_30
def innerCandidates48_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_29_checked : candidates edge 4 233 7187 innerTail48_30 best.bound = innerCandidates48_29 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_29 : scan edge (search edge 3) 4 [233] 233 innerTail48_29 best = best := by
  rw [innerTail48_29, scan, if_neg (by decide), innerCandidates48_29_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_30
    (if 3 ≤ innerCandidates48_29.length then search edge 3 [233, 7187] innerCandidates48_29 7420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_30

def innerTail48_28 : List Nat := 7043 :: innerTail48_29
def innerCandidates48_28 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates48_28_checked : candidates edge 4 233 7043 innerTail48_29 best.bound = innerCandidates48_28 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_28 : scan edge (search edge 3) 4 [233] 233 innerTail48_28 best = best := by
  rw [innerTail48_28, scan, if_neg (by decide), innerCandidates48_28_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_29
    (if 3 ≤ innerCandidates48_28.length then search edge 3 [233, 7043] innerCandidates48_28 7276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_29

def innerTail48_27 : List Nat := 7001 :: innerTail48_28
def innerCandidates48_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_27_checked : candidates edge 4 233 7001 innerTail48_28 best.bound = innerCandidates48_27 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_27 : scan edge (search edge 3) 4 [233] 233 innerTail48_27 best = best := by
  rw [innerTail48_27, scan, if_neg (by decide), innerCandidates48_27_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_28
    (if 3 ≤ innerCandidates48_27.length then search edge 3 [233, 7001] innerCandidates48_27 7234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_28

def innerTail48_26 : List Nat := 6947 :: innerTail48_27
def innerCandidates48_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_26_checked : candidates edge 4 233 6947 innerTail48_27 best.bound = innerCandidates48_26 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_26 : scan edge (search edge 3) 4 [233] 233 innerTail48_26 best = best := by
  rw [innerTail48_26, scan, if_neg (by decide), innerCandidates48_26_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_27
    (if 3 ≤ innerCandidates48_26.length then search edge 3 [233, 6947] innerCandidates48_26 7180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_27

def innerTail48_25 : List Nat := 6359 :: innerTail48_26
def innerCandidates48_25 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates48_25_checked : candidates edge 4 233 6359 innerTail48_26 best.bound = innerCandidates48_25 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_25 : scan edge (search edge 3) 4 [233] 233 innerTail48_25 best = best := by
  rw [innerTail48_25, scan, if_neg (by decide), innerCandidates48_25_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_26
    (if 3 ≤ innerCandidates48_25.length then search edge 3 [233, 6359] innerCandidates48_25 6592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_26

def innerTail48_24 : List Nat := 6263 :: innerTail48_25
def innerCandidates48_24 : List Nat := [6359]
set_option maxRecDepth 4096 in
theorem innerCandidates48_24_checked : candidates edge 4 233 6263 innerTail48_25 best.bound = innerCandidates48_24 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_24 : scan edge (search edge 3) 4 [233] 233 innerTail48_24 best = best := by
  rw [innerTail48_24, scan, if_neg (by decide), innerCandidates48_24_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_25
    (if 3 ≤ innerCandidates48_24.length then search edge 3 [233, 6263] innerCandidates48_24 6496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_25

def innerTail48_23 : List Nat := 6011 :: innerTail48_24
def innerCandidates48_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_23_checked : candidates edge 4 233 6011 innerTail48_24 best.bound = innerCandidates48_23 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_23 : scan edge (search edge 3) 4 [233] 233 innerTail48_23 best = best := by
  rw [innerTail48_23, scan, if_neg (by decide), innerCandidates48_23_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_24
    (if 3 ≤ innerCandidates48_23.length then search edge 3 [233, 6011] innerCandidates48_23 6244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_24

def innerTail48_22 : List Nat := 5813 :: innerTail48_23
def innerCandidates48_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_22_checked : candidates edge 4 233 5813 innerTail48_23 best.bound = innerCandidates48_22 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_22 : scan edge (search edge 3) 4 [233] 233 innerTail48_22 best = best := by
  rw [innerTail48_22, scan, if_neg (by decide), innerCandidates48_22_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_23
    (if 3 ≤ innerCandidates48_22.length then search edge 3 [233, 5813] innerCandidates48_22 6046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_23

def innerTail48_21 : List Nat := 5807 :: innerTail48_22
def innerCandidates48_21 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates48_21_checked : candidates edge 4 233 5807 innerTail48_22 best.bound = innerCandidates48_21 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_21 : scan edge (search edge 3) 4 [233] 233 innerTail48_21 best = best := by
  rw [innerTail48_21, scan, if_neg (by decide), innerCandidates48_21_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_22
    (if 3 ≤ innerCandidates48_21.length then search edge 3 [233, 5807] innerCandidates48_21 6040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_22

def innerTail48_20 : List Nat := 5591 :: innerTail48_21
def innerCandidates48_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_20_checked : candidates edge 4 233 5591 innerTail48_21 best.bound = innerCandidates48_20 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_20 : scan edge (search edge 3) 4 [233] 233 innerTail48_20 best = best := by
  rw [innerTail48_20, scan, if_neg (by decide), innerCandidates48_20_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_21
    (if 3 ≤ innerCandidates48_20.length then search edge 3 [233, 5591] innerCandidates48_20 5824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_21

def innerTail48_19 : List Nat := 5297 :: innerTail48_20
def innerCandidates48_19 : List Nat := [6263]
set_option maxRecDepth 4096 in
theorem innerCandidates48_19_checked : candidates edge 4 233 5297 innerTail48_20 best.bound = innerCandidates48_19 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_19 : scan edge (search edge 3) 4 [233] 233 innerTail48_19 best = best := by
  rw [innerTail48_19, scan, if_neg (by decide), innerCandidates48_19_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_20
    (if 3 ≤ innerCandidates48_19.length then search edge 3 [233, 5297] innerCandidates48_19 5530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_20

def innerTail48_18 : List Nat := 5009 :: innerTail48_19
def innerCandidates48_18 : List Nat := [6947]
set_option maxRecDepth 4096 in
theorem innerCandidates48_18_checked : candidates edge 4 233 5009 innerTail48_19 best.bound = innerCandidates48_18 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_18 : scan edge (search edge 3) 4 [233] 233 innerTail48_18 best = best := by
  rw [innerTail48_18, scan, if_neg (by decide), innerCandidates48_18_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_19
    (if 3 ≤ innerCandidates48_18.length then search edge 3 [233, 5009] innerCandidates48_18 5242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_19

def innerTail48_17 : List Nat := 4793 :: innerTail48_18
def innerCandidates48_17 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates48_17_checked : candidates edge 4 233 4793 innerTail48_18 best.bound = innerCandidates48_17 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_17 : scan edge (search edge 3) 4 [233] 233 innerTail48_17 best = best := by
  rw [innerTail48_17, scan, if_neg (by decide), innerCandidates48_17_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_18
    (if 3 ≤ innerCandidates48_17.length then search edge 3 [233, 4793] innerCandidates48_17 5026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_18

def innerTail48_16 : List Nat := 4337 :: innerTail48_17
def innerCandidates48_16 : List Nat := [7043, 7547]
set_option maxRecDepth 4096 in
theorem innerCandidates48_16_checked : candidates edge 4 233 4337 innerTail48_17 best.bound = innerCandidates48_16 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_16 : scan edge (search edge 3) 4 [233] 233 innerTail48_16 best = best := by
  rw [innerTail48_16, scan, if_neg (by decide), innerCandidates48_16_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_17
    (if 3 ≤ innerCandidates48_16.length then search edge 3 [233, 4337] innerCandidates48_16 4570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_17

def innerTail48_15 : List Nat := 3389 :: innerTail48_16
def innerCandidates48_15 : List Nat := [7547]
set_option maxRecDepth 4096 in
theorem innerCandidates48_15_checked : candidates edge 4 233 3389 innerTail48_16 best.bound = innerCandidates48_15 := by
  unfold candidates
  have row : edge 3389 = SearchData.row3389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_15 : scan edge (search edge 3) 4 [233] 233 innerTail48_15 best = best := by
  rw [innerTail48_15, scan, if_neg (by decide), innerCandidates48_15_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_16
    (if 3 ≤ innerCandidates48_15.length then search edge 3 [233, 3389] innerCandidates48_15 3622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_16

def innerTail48_14 : List Nat := 3329 :: innerTail48_15
def innerCandidates48_14 : List Nat := [6359, 7001]
set_option maxRecDepth 4096 in
theorem innerCandidates48_14_checked : candidates edge 4 233 3329 innerTail48_15 best.bound = innerCandidates48_14 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_14 : scan edge (search edge 3) 4 [233] 233 innerTail48_14 best = best := by
  rw [innerTail48_14, scan, if_neg (by decide), innerCandidates48_14_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_15
    (if 3 ≤ innerCandidates48_14.length then search edge 3 [233, 3329] innerCandidates48_14 3562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_15

def innerTail48_13 : List Nat := 3167 :: innerTail48_14
def innerCandidates48_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_13_checked : candidates edge 4 233 3167 innerTail48_14 best.bound = innerCandidates48_13 := by
  unfold candidates
  have row : edge 3167 = SearchData.row3167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_13 : scan edge (search edge 3) 4 [233] 233 innerTail48_13 best = best := by
  rw [innerTail48_13, scan, if_neg (by decide), innerCandidates48_13_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_14
    (if 3 ≤ innerCandidates48_13.length then search edge 3 [233, 3167] innerCandidates48_13 3400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_14

def innerTail48_12 : List Nat := 3083 :: innerTail48_13
def innerCandidates48_12 : List Nat := [3329, 7187]
set_option maxRecDepth 4096 in
theorem innerCandidates48_12_checked : candidates edge 4 233 3083 innerTail48_13 best.bound = innerCandidates48_12 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_12 : scan edge (search edge 3) 4 [233] 233 innerTail48_12 best = best := by
  rw [innerTail48_12, scan, if_neg (by decide), innerCandidates48_12_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_13
    (if 3 ≤ innerCandidates48_12.length then search edge 3 [233, 3083] innerCandidates48_12 3316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_13

def innerTail48_11 : List Nat := 2969 :: innerTail48_12
def innerCandidates48_11 : List Nat := [5009]
set_option maxRecDepth 4096 in
theorem innerCandidates48_11_checked : candidates edge 4 233 2969 innerTail48_12 best.bound = innerCandidates48_11 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_11 : scan edge (search edge 3) 4 [233] 233 innerTail48_11 best = best := by
  rw [innerTail48_11, scan, if_neg (by decide), innerCandidates48_11_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_12
    (if 3 ≤ innerCandidates48_11.length then search edge 3 [233, 2969] innerCandidates48_11 3202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_12

def innerTail48_10 : List Nat := 1583 :: innerTail48_11
def innerCandidates48_10 : List Nat := [5009, 7043, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates48_10_checked : candidates edge 4 233 1583 innerTail48_11 best.bound = innerCandidates48_10 := by
  unfold candidates
  have row : edge 1583 = PrimePairCache.smallRow1583.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild48_10 : search edge 3 [233, 1583] innerCandidates48_10 1816 best = best := by decide

theorem innerScan48_10 : scan edge (search edge 3) 4 [233] 233 innerTail48_10 best = best := by
  rw [innerTail48_10, scan, if_neg (by decide), innerCandidates48_10_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_11
    (if 3 ≤ innerCandidates48_10.length then search edge 3 [233, 1583] innerCandidates48_10 1816 best else best) = best
  rw [if_pos (by decide), innerChild48_10]
  exact innerScan48_11

def innerTail48_9 : List Nat := 1187 :: innerTail48_10
def innerCandidates48_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_9_checked : candidates edge 4 233 1187 innerTail48_10 best.bound = innerCandidates48_9 := by
  unfold candidates
  have row : edge 1187 = PrimePairCache.smallRow1187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_9 : scan edge (search edge 3) 4 [233] 233 innerTail48_9 best = best := by
  rw [innerTail48_9, scan, if_neg (by decide), innerCandidates48_9_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_10
    (if 3 ≤ innerCandidates48_9.length then search edge 3 [233, 1187] innerCandidates48_9 1420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_10

def innerTail48_8 : List Nat := 1163 :: innerTail48_9
def innerCandidates48_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_8_checked : candidates edge 4 233 1163 innerTail48_9 best.bound = innerCandidates48_8 := by
  unfold candidates
  have row : edge 1163 = PrimePairCache.smallRow1163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_8 : scan edge (search edge 3) 4 [233] 233 innerTail48_8 best = best := by
  rw [innerTail48_8, scan, if_neg (by decide), innerCandidates48_8_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_9
    (if 3 ≤ innerCandidates48_8.length then search edge 3 [233, 1163] innerCandidates48_8 1396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_9

def innerTail48_7 : List Nat := 1151 :: innerTail48_8
def innerCandidates48_7 : List Nat := [4337]
set_option maxRecDepth 4096 in
theorem innerCandidates48_7_checked : candidates edge 4 233 1151 innerTail48_8 best.bound = innerCandidates48_7 := by
  unfold candidates
  have row : edge 1151 = PrimePairCache.smallRow1151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_7 : scan edge (search edge 3) 4 [233] 233 innerTail48_7 best = best := by
  rw [innerTail48_7, scan, if_neg (by decide), innerCandidates48_7_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_8
    (if 3 ≤ innerCandidates48_7.length then search edge 3 [233, 1151] innerCandidates48_7 1384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_8

def innerTail48_6 : List Nat := 983 :: innerTail48_7
def innerCandidates48_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates48_6_checked : candidates edge 4 233 983 innerTail48_7 best.bound = innerCandidates48_6 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_6 : scan edge (search edge 3) 4 [233] 233 innerTail48_6 best = best := by
  rw [innerTail48_6, scan, if_neg (by decide), innerCandidates48_6_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_7
    (if 3 ≤ innerCandidates48_6.length then search edge 3 [233, 983] innerCandidates48_6 1216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_7

def innerTail48_5 : List Nat := 911 :: innerTail48_6
def innerCandidates48_5 : List Nat := [4337]
set_option maxRecDepth 4096 in
theorem innerCandidates48_5_checked : candidates edge 4 233 911 innerTail48_6 best.bound = innerCandidates48_5 := by
  unfold candidates
  have row : edge 911 = PrimePairCache.smallRow911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_5 : scan edge (search edge 3) 4 [233] 233 innerTail48_5 best = best := by
  rw [innerTail48_5, scan, if_neg (by decide), innerCandidates48_5_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_6
    (if 3 ≤ innerCandidates48_5.length then search edge 3 [233, 911] innerCandidates48_5 1144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_6

def innerTail48_4 : List Nat := 881 :: innerTail48_5
def innerCandidates48_4 : List Nat := [983, 5591, 7607]
set_option maxRecDepth 4096 in
theorem innerCandidates48_4_checked : candidates edge 4 233 881 innerTail48_5 best.bound = innerCandidates48_4 := by
  unfold candidates
  have row : edge 881 = PrimePairCache.smallRow881.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild48_4 : search edge 3 [233, 881] innerCandidates48_4 1114 best = best := by decide

theorem innerScan48_4 : scan edge (search edge 3) 4 [233] 233 innerTail48_4 best = best := by
  rw [innerTail48_4, scan, if_neg (by decide), innerCandidates48_4_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_5
    (if 3 ≤ innerCandidates48_4.length then search edge 3 [233, 881] innerCandidates48_4 1114 best else best) = best
  rw [if_pos (by decide), innerChild48_4]
  exact innerScan48_5

def innerTail48_3 : List Nat := 617 :: innerTail48_4
def innerCandidates48_3 : List Nat := [983, 6011]
set_option maxRecDepth 4096 in
theorem innerCandidates48_3_checked : candidates edge 4 233 617 innerTail48_4 best.bound = innerCandidates48_3 := by
  unfold candidates
  have row : edge 617 = PrimePairCache.smallRow617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_3 : scan edge (search edge 3) 4 [233] 233 innerTail48_3 best = best := by
  rw [innerTail48_3, scan, if_neg (by decide), innerCandidates48_3_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_4
    (if 3 ≤ innerCandidates48_3.length then search edge 3 [233, 617] innerCandidates48_3 850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_4

def innerTail48_2 : List Nat := 347 :: innerTail48_3
def innerCandidates48_2 : List Nat := [983]
set_option maxRecDepth 4096 in
theorem innerCandidates48_2_checked : candidates edge 4 233 347 innerTail48_3 best.bound = innerCandidates48_2 := by
  unfold candidates
  have row : edge 347 = PrimePairCache.smallRow347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_2 : scan edge (search edge 3) 4 [233] 233 innerTail48_2 best = best := by
  rw [innerTail48_2, scan, if_neg (by decide), innerCandidates48_2_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_3
    (if 3 ≤ innerCandidates48_2.length then search edge 3 [233, 347] innerCandidates48_2 580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_3

def innerTail48_1 : List Nat := 251 :: innerTail48_2
def innerCandidates48_1 : List Nat := [347, 1151]
set_option maxRecDepth 4096 in
theorem innerCandidates48_1_checked : candidates edge 4 233 251 innerTail48_2 best.bound = innerCandidates48_1 := by
  unfold candidates
  have row : edge 251 = PrimePairCache.smallRow251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_1 : scan edge (search edge 3) 4 [233] 233 innerTail48_1 best = best := by
  rw [innerTail48_1, scan, if_neg (by decide), innerCandidates48_1_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_2
    (if 3 ≤ innerCandidates48_1.length then search edge 3 [233, 251] innerCandidates48_1 484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_2

def innerTail48_0 : List Nat := 239 :: innerTail48_1
def innerCandidates48_0 : List Nat := [347, 5807]
set_option maxRecDepth 4096 in
theorem innerCandidates48_0_checked : candidates edge 4 233 239 innerTail48_1 best.bound = innerCandidates48_0 := by
  unfold candidates
  have row : edge 239 = PrimePairCache.smallRow239.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan48_0 : scan edge (search edge 3) 4 [233] 233 innerTail48_0 best = best := by
  rw [innerTail48_0, scan, if_neg (by decide), innerCandidates48_0_checked]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_1
    (if 3 ≤ innerCandidates48_0.length then search edge 3 [233, 239] innerCandidates48_0 472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan48_1

theorem rootBranch48 : search edge 4 [233] rootCandidates48 233 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [233] 233 innerTail48_0 best = best
  exact innerScan48_0

end ProjectEulerSolutions.PrimePairUpper.Search
