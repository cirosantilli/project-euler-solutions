import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates41 : List Nat := [283, 373, 433, 541, 577, 601, 751, 811, 883, 1009, 1291, 1297, 1399, 1627, 1669, 1723, 2011, 2131, 2719, 2803, 3037, 3457, 3499, 3727, 4657, 4969, 5827, 6427, 6733, 7237, 8161, 8191]
set_option maxRecDepth 4096 in
theorem rootCandidates41_checked : candidates edge 5 0 193 tail42 best.bound = rootCandidates41 := by
  unfold candidates
  have row : edge 193 = PrimePairCache.smallRow193.contains := by funext q; rfl
  rw [row]
  decide

def innerTail41_32 : List Nat := []
theorem innerScan41_32 : scan edge (search edge 3) 4 [193] 193 innerTail41_32 best = best := rfl

def innerTail41_31 : List Nat := 8191 :: innerTail41_32
def innerCandidates41_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_31_checked : candidates edge 4 193 8191 innerTail41_32 best.bound = innerCandidates41_31 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_31 : scan edge (search edge 3) 4 [193] 193 innerTail41_31 best = best := by
  rw [innerTail41_31, scan, if_neg (by decide), innerCandidates41_31_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_32
    (if 3 ≤ innerCandidates41_31.length then search edge 3 [193, 8191] innerCandidates41_31 8384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_32

def innerTail41_30 : List Nat := 8161 :: innerTail41_31
def innerCandidates41_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_30_checked : candidates edge 4 193 8161 innerTail41_31 best.bound = innerCandidates41_30 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_30 : scan edge (search edge 3) 4 [193] 193 innerTail41_30 best = best := by
  rw [innerTail41_30, scan, if_neg (by decide), innerCandidates41_30_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_31
    (if 3 ≤ innerCandidates41_30.length then search edge 3 [193, 8161] innerCandidates41_30 8354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_31

def innerTail41_29 : List Nat := 7237 :: innerTail41_30
def innerCandidates41_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_29_checked : candidates edge 4 193 7237 innerTail41_30 best.bound = innerCandidates41_29 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_29 : scan edge (search edge 3) 4 [193] 193 innerTail41_29 best = best := by
  rw [innerTail41_29, scan, if_neg (by decide), innerCandidates41_29_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_30
    (if 3 ≤ innerCandidates41_29.length then search edge 3 [193, 7237] innerCandidates41_29 7430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_30

def innerTail41_28 : List Nat := 6733 :: innerTail41_29
def innerCandidates41_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_28_checked : candidates edge 4 193 6733 innerTail41_29 best.bound = innerCandidates41_28 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_28 : scan edge (search edge 3) 4 [193] 193 innerTail41_28 best = best := by
  rw [innerTail41_28, scan, if_neg (by decide), innerCandidates41_28_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_29
    (if 3 ≤ innerCandidates41_28.length then search edge 3 [193, 6733] innerCandidates41_28 6926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_29

def innerTail41_27 : List Nat := 6427 :: innerTail41_28
def innerCandidates41_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_27_checked : candidates edge 4 193 6427 innerTail41_28 best.bound = innerCandidates41_27 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_27 : scan edge (search edge 3) 4 [193] 193 innerTail41_27 best = best := by
  rw [innerTail41_27, scan, if_neg (by decide), innerCandidates41_27_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_28
    (if 3 ≤ innerCandidates41_27.length then search edge 3 [193, 6427] innerCandidates41_27 6620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_28

def innerTail41_26 : List Nat := 5827 :: innerTail41_27
def innerCandidates41_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_26_checked : candidates edge 4 193 5827 innerTail41_27 best.bound = innerCandidates41_26 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_26 : scan edge (search edge 3) 4 [193] 193 innerTail41_26 best = best := by
  rw [innerTail41_26, scan, if_neg (by decide), innerCandidates41_26_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_27
    (if 3 ≤ innerCandidates41_26.length then search edge 3 [193, 5827] innerCandidates41_26 6020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_27

def innerTail41_25 : List Nat := 4969 :: innerTail41_26
def innerCandidates41_25 : List Nat := [5827]
set_option maxRecDepth 4096 in
theorem innerCandidates41_25_checked : candidates edge 4 193 4969 innerTail41_26 best.bound = innerCandidates41_25 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_25 : scan edge (search edge 3) 4 [193] 193 innerTail41_25 best = best := by
  rw [innerTail41_25, scan, if_neg (by decide), innerCandidates41_25_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_26
    (if 3 ≤ innerCandidates41_25.length then search edge 3 [193, 4969] innerCandidates41_25 5162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_26

def innerTail41_24 : List Nat := 4657 :: innerTail41_25
def innerCandidates41_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_24_checked : candidates edge 4 193 4657 innerTail41_25 best.bound = innerCandidates41_24 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_24 : scan edge (search edge 3) 4 [193] 193 innerTail41_24 best = best := by
  rw [innerTail41_24, scan, if_neg (by decide), innerCandidates41_24_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_25
    (if 3 ≤ innerCandidates41_24.length then search edge 3 [193, 4657] innerCandidates41_24 4850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_25

def innerTail41_23 : List Nat := 3727 :: innerTail41_24
def innerCandidates41_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_23_checked : candidates edge 4 193 3727 innerTail41_24 best.bound = innerCandidates41_23 := by
  unfold candidates
  have row : edge 3727 = SearchData.row3727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_23 : scan edge (search edge 3) 4 [193] 193 innerTail41_23 best = best := by
  rw [innerTail41_23, scan, if_neg (by decide), innerCandidates41_23_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_24
    (if 3 ≤ innerCandidates41_23.length then search edge 3 [193, 3727] innerCandidates41_23 3920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_24

def innerTail41_22 : List Nat := 3499 :: innerTail41_23
def innerCandidates41_22 : List Nat := [8191]
set_option maxRecDepth 4096 in
theorem innerCandidates41_22_checked : candidates edge 4 193 3499 innerTail41_23 best.bound = innerCandidates41_22 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_22 : scan edge (search edge 3) 4 [193] 193 innerTail41_22 best = best := by
  rw [innerTail41_22, scan, if_neg (by decide), innerCandidates41_22_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_23
    (if 3 ≤ innerCandidates41_22.length then search edge 3 [193, 3499] innerCandidates41_22 3692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_23

def innerTail41_21 : List Nat := 3457 :: innerTail41_22
def innerCandidates41_21 : List Nat := [3499]
set_option maxRecDepth 4096 in
theorem innerCandidates41_21_checked : candidates edge 4 193 3457 innerTail41_22 best.bound = innerCandidates41_21 := by
  unfold candidates
  have row : edge 3457 = SearchData.row3457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_21 : scan edge (search edge 3) 4 [193] 193 innerTail41_21 best = best := by
  rw [innerTail41_21, scan, if_neg (by decide), innerCandidates41_21_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_22
    (if 3 ≤ innerCandidates41_21.length then search edge 3 [193, 3457] innerCandidates41_21 3650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_22

def innerTail41_20 : List Nat := 3037 :: innerTail41_21
def innerCandidates41_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_20_checked : candidates edge 4 193 3037 innerTail41_21 best.bound = innerCandidates41_20 := by
  unfold candidates
  have row : edge 3037 = SearchData.row3037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_20 : scan edge (search edge 3) 4 [193] 193 innerTail41_20 best = best := by
  rw [innerTail41_20, scan, if_neg (by decide), innerCandidates41_20_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_21
    (if 3 ≤ innerCandidates41_20.length then search edge 3 [193, 3037] innerCandidates41_20 3230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_21

def innerTail41_19 : List Nat := 2803 :: innerTail41_20
def innerCandidates41_19 : List Nat := [4969, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates41_19_checked : candidates edge 4 193 2803 innerTail41_20 best.bound = innerCandidates41_19 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_19 : scan edge (search edge 3) 4 [193] 193 innerTail41_19 best = best := by
  rw [innerTail41_19, scan, if_neg (by decide), innerCandidates41_19_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_20
    (if 3 ≤ innerCandidates41_19.length then search edge 3 [193, 2803] innerCandidates41_19 2996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_20

def innerTail41_18 : List Nat := 2719 :: innerTail41_19
def innerCandidates41_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_18_checked : candidates edge 4 193 2719 innerTail41_19 best.bound = innerCandidates41_18 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_18 : scan edge (search edge 3) 4 [193] 193 innerTail41_18 best = best := by
  rw [innerTail41_18, scan, if_neg (by decide), innerCandidates41_18_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_19
    (if 3 ≤ innerCandidates41_18.length then search edge 3 [193, 2719] innerCandidates41_18 2912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_19

def innerTail41_17 : List Nat := 2131 :: innerTail41_18
def innerCandidates41_17 : List Nat := [3037]
set_option maxRecDepth 4096 in
theorem innerCandidates41_17_checked : candidates edge 4 193 2131 innerTail41_18 best.bound = innerCandidates41_17 := by
  unfold candidates
  have row : edge 2131 = PrimePairCache.smallRow2131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_17 : scan edge (search edge 3) 4 [193] 193 innerTail41_17 best = best := by
  rw [innerTail41_17, scan, if_neg (by decide), innerCandidates41_17_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_18
    (if 3 ≤ innerCandidates41_17.length then search edge 3 [193, 2131] innerCandidates41_17 2324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_18

def innerTail41_16 : List Nat := 2011 :: innerTail41_17
def innerCandidates41_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_16_checked : candidates edge 4 193 2011 innerTail41_17 best.bound = innerCandidates41_16 := by
  unfold candidates
  have row : edge 2011 = PrimePairCache.smallRow2011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_16 : scan edge (search edge 3) 4 [193] 193 innerTail41_16 best = best := by
  rw [innerTail41_16, scan, if_neg (by decide), innerCandidates41_16_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_17
    (if 3 ≤ innerCandidates41_16.length then search edge 3 [193, 2011] innerCandidates41_16 2204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_17

def innerTail41_15 : List Nat := 1723 :: innerTail41_16
def innerCandidates41_15 : List Nat := [3457]
set_option maxRecDepth 4096 in
theorem innerCandidates41_15_checked : candidates edge 4 193 1723 innerTail41_16 best.bound = innerCandidates41_15 := by
  unfold candidates
  have row : edge 1723 = PrimePairCache.smallRow1723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_15 : scan edge (search edge 3) 4 [193] 193 innerTail41_15 best = best := by
  rw [innerTail41_15, scan, if_neg (by decide), innerCandidates41_15_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_16
    (if 3 ≤ innerCandidates41_15.length then search edge 3 [193, 1723] innerCandidates41_15 1916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_16

def innerTail41_14 : List Nat := 1669 :: innerTail41_15
def innerCandidates41_14 : List Nat := [3499, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates41_14_checked : candidates edge 4 193 1669 innerTail41_15 best.bound = innerCandidates41_14 := by
  unfold candidates
  have row : edge 1669 = PrimePairCache.smallRow1669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_14 : scan edge (search edge 3) 4 [193] 193 innerTail41_14 best = best := by
  rw [innerTail41_14, scan, if_neg (by decide), innerCandidates41_14_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_15
    (if 3 ≤ innerCandidates41_14.length then search edge 3 [193, 1669] innerCandidates41_14 1862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_15

def innerTail41_13 : List Nat := 1627 :: innerTail41_14
def innerCandidates41_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_13_checked : candidates edge 4 193 1627 innerTail41_14 best.bound = innerCandidates41_13 := by
  unfold candidates
  have row : edge 1627 = PrimePairCache.smallRow1627.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_13 : scan edge (search edge 3) 4 [193] 193 innerTail41_13 best = best := by
  rw [innerTail41_13, scan, if_neg (by decide), innerCandidates41_13_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_14
    (if 3 ≤ innerCandidates41_13.length then search edge 3 [193, 1627] innerCandidates41_13 1820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_14

def innerTail41_12 : List Nat := 1399 :: innerTail41_13
def innerCandidates41_12 : List Nat := [2131]
set_option maxRecDepth 4096 in
theorem innerCandidates41_12_checked : candidates edge 4 193 1399 innerTail41_13 best.bound = innerCandidates41_12 := by
  unfold candidates
  have row : edge 1399 = PrimePairCache.smallRow1399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_12 : scan edge (search edge 3) 4 [193] 193 innerTail41_12 best = best := by
  rw [innerTail41_12, scan, if_neg (by decide), innerCandidates41_12_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_13
    (if 3 ≤ innerCandidates41_12.length then search edge 3 [193, 1399] innerCandidates41_12 1592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_13

def innerTail41_11 : List Nat := 1297 :: innerTail41_12
def innerCandidates41_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_11_checked : candidates edge 4 193 1297 innerTail41_12 best.bound = innerCandidates41_11 := by
  unfold candidates
  have row : edge 1297 = PrimePairCache.smallRow1297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_11 : scan edge (search edge 3) 4 [193] 193 innerTail41_11 best = best := by
  rw [innerTail41_11, scan, if_neg (by decide), innerCandidates41_11_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_12
    (if 3 ≤ innerCandidates41_11.length then search edge 3 [193, 1297] innerCandidates41_11 1490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_12

def innerTail41_10 : List Nat := 1291 :: innerTail41_11
def innerCandidates41_10 : List Nat := [3457, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates41_10_checked : candidates edge 4 193 1291 innerTail41_11 best.bound = innerCandidates41_10 := by
  unfold candidates
  have row : edge 1291 = PrimePairCache.smallRow1291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_10 : scan edge (search edge 3) 4 [193] 193 innerTail41_10 best = best := by
  rw [innerTail41_10, scan, if_neg (by decide), innerCandidates41_10_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_11
    (if 3 ≤ innerCandidates41_10.length then search edge 3 [193, 1291] innerCandidates41_10 1484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_11

def innerTail41_9 : List Nat := 1009 :: innerTail41_10
def innerCandidates41_9 : List Nat := [4657]
set_option maxRecDepth 4096 in
theorem innerCandidates41_9_checked : candidates edge 4 193 1009 innerTail41_10 best.bound = innerCandidates41_9 := by
  unfold candidates
  have row : edge 1009 = PrimePairCache.smallRow1009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_9 : scan edge (search edge 3) 4 [193] 193 innerTail41_9 best = best := by
  rw [innerTail41_9, scan, if_neg (by decide), innerCandidates41_9_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_10
    (if 3 ≤ innerCandidates41_9.length then search edge 3 [193, 1009] innerCandidates41_9 1202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_10

def innerTail41_8 : List Nat := 883 :: innerTail41_9
def innerCandidates41_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_8_checked : candidates edge 4 193 883 innerTail41_9 best.bound = innerCandidates41_8 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_8 : scan edge (search edge 3) 4 [193] 193 innerTail41_8 best = best := by
  rw [innerTail41_8, scan, if_neg (by decide), innerCandidates41_8_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_9
    (if 3 ≤ innerCandidates41_8.length then search edge 3 [193, 883] innerCandidates41_8 1076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_9

def innerTail41_7 : List Nat := 811 :: innerTail41_8
def innerCandidates41_7 : List Nat := [1723, 2011, 8191]
set_option maxRecDepth 4096 in
theorem innerCandidates41_7_checked : candidates edge 4 193 811 innerTail41_8 best.bound = innerCandidates41_7 := by
  unfold candidates
  have row : edge 811 = PrimePairCache.smallRow811.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild41_7 : search edge 3 [193, 811] innerCandidates41_7 1004 best = best := by decide

theorem innerScan41_7 : scan edge (search edge 3) 4 [193] 193 innerTail41_7 best = best := by
  rw [innerTail41_7, scan, if_neg (by decide), innerCandidates41_7_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_8
    (if 3 ≤ innerCandidates41_7.length then search edge 3 [193, 811] innerCandidates41_7 1004 best else best) = best
  rw [if_pos (by decide), innerChild41_7]
  exact innerScan41_8

def innerTail41_6 : List Nat := 751 :: innerTail41_7
def innerCandidates41_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_6_checked : candidates edge 4 193 751 innerTail41_7 best.bound = innerCandidates41_6 := by
  unfold candidates
  have row : edge 751 = PrimePairCache.smallRow751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_6 : scan edge (search edge 3) 4 [193] 193 innerTail41_6 best = best := by
  rw [innerTail41_6, scan, if_neg (by decide), innerCandidates41_6_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_7
    (if 3 ≤ innerCandidates41_6.length then search edge 3 [193, 751] innerCandidates41_6 944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_7

def innerTail41_5 : List Nat := 601 :: innerTail41_6
def innerCandidates41_5 : List Nat := [1627, 2011, 2719]
set_option maxRecDepth 4096 in
theorem innerCandidates41_5_checked : candidates edge 4 193 601 innerTail41_6 best.bound = innerCandidates41_5 := by
  unfold candidates
  have row : edge 601 = PrimePairCache.smallRow601.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild41_5 : search edge 3 [193, 601] innerCandidates41_5 794 best = best := by decide

theorem innerScan41_5 : scan edge (search edge 3) 4 [193] 193 innerTail41_5 best = best := by
  rw [innerTail41_5, scan, if_neg (by decide), innerCandidates41_5_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_6
    (if 3 ≤ innerCandidates41_5.length then search edge 3 [193, 601] innerCandidates41_5 794 best else best) = best
  rw [if_pos (by decide), innerChild41_5]
  exact innerScan41_6

def innerTail41_4 : List Nat := 577 :: innerTail41_5
def innerCandidates41_4 : List Nat := [1723]
set_option maxRecDepth 4096 in
theorem innerCandidates41_4_checked : candidates edge 4 193 577 innerTail41_5 best.bound = innerCandidates41_4 := by
  unfold candidates
  have row : edge 577 = PrimePairCache.smallRow577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_4 : scan edge (search edge 3) 4 [193] 193 innerTail41_4 best = best := by
  rw [innerTail41_4, scan, if_neg (by decide), innerCandidates41_4_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_5
    (if 3 ≤ innerCandidates41_4.length then search edge 3 [193, 577] innerCandidates41_4 770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_5

def innerTail41_3 : List Nat := 541 :: innerTail41_4
def innerCandidates41_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates41_3_checked : candidates edge 4 193 541 innerTail41_4 best.bound = innerCandidates41_3 := by
  unfold candidates
  have row : edge 541 = PrimePairCache.smallRow541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_3 : scan edge (search edge 3) 4 [193] 193 innerTail41_3 best = best := by
  rw [innerTail41_3, scan, if_neg (by decide), innerCandidates41_3_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_4
    (if 3 ≤ innerCandidates41_3.length then search edge 3 [193, 541] innerCandidates41_3 734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_4

def innerTail41_2 : List Nat := 433 :: innerTail41_3
def innerCandidates41_2 : List Nat := [883, 1009, 2719]
set_option maxRecDepth 4096 in
theorem innerCandidates41_2_checked : candidates edge 4 193 433 innerTail41_3 best.bound = innerCandidates41_2 := by
  unfold candidates
  have row : edge 433 = PrimePairCache.smallRow433.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild41_2 : search edge 3 [193, 433] innerCandidates41_2 626 best = best := by decide

theorem innerScan41_2 : scan edge (search edge 3) 4 [193] 193 innerTail41_2 best = best := by
  rw [innerTail41_2, scan, if_neg (by decide), innerCandidates41_2_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_3
    (if 3 ≤ innerCandidates41_2.length then search edge 3 [193, 433] innerCandidates41_2 626 best else best) = best
  rw [if_pos (by decide), innerChild41_2]
  exact innerScan41_3

def innerTail41_1 : List Nat := 373 :: innerTail41_2
def innerCandidates41_1 : List Nat := [1009]
set_option maxRecDepth 4096 in
theorem innerCandidates41_1_checked : candidates edge 4 193 373 innerTail41_2 best.bound = innerCandidates41_1 := by
  unfold candidates
  have row : edge 373 = PrimePairCache.smallRow373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan41_1 : scan edge (search edge 3) 4 [193] 193 innerTail41_1 best = best := by
  rw [innerTail41_1, scan, if_neg (by decide), innerCandidates41_1_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_2
    (if 3 ≤ innerCandidates41_1.length then search edge 3 [193, 373] innerCandidates41_1 566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan41_2

def innerTail41_0 : List Nat := 283 :: innerTail41_1
def innerCandidates41_0 : List Nat := [541, 601, 6733, 8191]
set_option maxRecDepth 4096 in
theorem innerCandidates41_0_checked : candidates edge 4 193 283 innerTail41_1 best.bound = innerCandidates41_0 := by
  unfold candidates
  have row : edge 283 = PrimePairCache.smallRow283.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild41_0 : search edge 3 [193, 283] innerCandidates41_0 476 best = best := by decide

theorem innerScan41_0 : scan edge (search edge 3) 4 [193] 193 innerTail41_0 best = best := by
  rw [innerTail41_0, scan, if_neg (by decide), innerCandidates41_0_checked]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_1
    (if 3 ≤ innerCandidates41_0.length then search edge 3 [193, 283] innerCandidates41_0 476 best else best) = best
  rw [if_pos (by decide), innerChild41_0]
  exact innerScan41_1

theorem rootBranch41 : search edge 4 [193] rootCandidates41 193 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [193] 193 innerTail41_0 best = best
  exact innerScan41_0

def rootCandidates40 : List Nat := [227, 251, 281, 461, 599, 953, 977, 1103, 1319, 1373, 1583, 1697, 2069, 2213, 2423, 2903, 3251, 3581, 4127, 4259, 4673, 4691, 5099, 5153, 5813, 5843, 5981, 6287, 6311, 6599, 6917, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates40_checked : candidates edge 5 0 191 tail41 best.bound = rootCandidates40 := by
  unfold candidates
  have row : edge 191 = PrimePairCache.smallRow191.contains := by funext q; rfl
  rw [row]
  decide

def innerTail40_32 : List Nat := []
theorem innerScan40_32 : scan edge (search edge 3) 4 [191] 191 innerTail40_32 best = best := rfl

def innerTail40_31 : List Nat := 8237 :: innerTail40_32
def innerCandidates40_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_31_checked : candidates edge 4 191 8237 innerTail40_32 best.bound = innerCandidates40_31 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_31 : scan edge (search edge 3) 4 [191] 191 innerTail40_31 best = best := by
  rw [innerTail40_31, scan, if_neg (by decide), innerCandidates40_31_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_32
    (if 3 ≤ innerCandidates40_31.length then search edge 3 [191, 8237] innerCandidates40_31 8428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_32

def innerTail40_30 : List Nat := 6917 :: innerTail40_31
def innerCandidates40_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_30_checked : candidates edge 4 191 6917 innerTail40_31 best.bound = innerCandidates40_30 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_30 : scan edge (search edge 3) 4 [191] 191 innerTail40_30 best = best := by
  rw [innerTail40_30, scan, if_neg (by decide), innerCandidates40_30_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_31
    (if 3 ≤ innerCandidates40_30.length then search edge 3 [191, 6917] innerCandidates40_30 7108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_31

def innerTail40_29 : List Nat := 6599 :: innerTail40_30
def innerCandidates40_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_29_checked : candidates edge 4 191 6599 innerTail40_30 best.bound = innerCandidates40_29 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_29 : scan edge (search edge 3) 4 [191] 191 innerTail40_29 best = best := by
  rw [innerTail40_29, scan, if_neg (by decide), innerCandidates40_29_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_30
    (if 3 ≤ innerCandidates40_29.length then search edge 3 [191, 6599] innerCandidates40_29 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_30

def innerTail40_28 : List Nat := 6311 :: innerTail40_29
def innerCandidates40_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_28_checked : candidates edge 4 191 6311 innerTail40_29 best.bound = innerCandidates40_28 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_28 : scan edge (search edge 3) 4 [191] 191 innerTail40_28 best = best := by
  rw [innerTail40_28, scan, if_neg (by decide), innerCandidates40_28_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_29
    (if 3 ≤ innerCandidates40_28.length then search edge 3 [191, 6311] innerCandidates40_28 6502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_29

def innerTail40_27 : List Nat := 6287 :: innerTail40_28
def innerCandidates40_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_27_checked : candidates edge 4 191 6287 innerTail40_28 best.bound = innerCandidates40_27 := by
  unfold candidates
  have row : edge 6287 = SearchData.row6287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_27 : scan edge (search edge 3) 4 [191] 191 innerTail40_27 best = best := by
  rw [innerTail40_27, scan, if_neg (by decide), innerCandidates40_27_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_28
    (if 3 ≤ innerCandidates40_27.length then search edge 3 [191, 6287] innerCandidates40_27 6478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_28

def innerTail40_26 : List Nat := 5981 :: innerTail40_27
def innerCandidates40_26 : List Nat := [8237]
set_option maxRecDepth 4096 in
theorem innerCandidates40_26_checked : candidates edge 4 191 5981 innerTail40_27 best.bound = innerCandidates40_26 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_26 : scan edge (search edge 3) 4 [191] 191 innerTail40_26 best = best := by
  rw [innerTail40_26, scan, if_neg (by decide), innerCandidates40_26_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_27
    (if 3 ≤ innerCandidates40_26.length then search edge 3 [191, 5981] innerCandidates40_26 6172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_27

def innerTail40_25 : List Nat := 5843 :: innerTail40_26
def innerCandidates40_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_25_checked : candidates edge 4 191 5843 innerTail40_26 best.bound = innerCandidates40_25 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_25 : scan edge (search edge 3) 4 [191] 191 innerTail40_25 best = best := by
  rw [innerTail40_25, scan, if_neg (by decide), innerCandidates40_25_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_26
    (if 3 ≤ innerCandidates40_25.length then search edge 3 [191, 5843] innerCandidates40_25 6034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_26

def innerTail40_24 : List Nat := 5813 :: innerTail40_25
def innerCandidates40_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_24_checked : candidates edge 4 191 5813 innerTail40_25 best.bound = innerCandidates40_24 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_24 : scan edge (search edge 3) 4 [191] 191 innerTail40_24 best = best := by
  rw [innerTail40_24, scan, if_neg (by decide), innerCandidates40_24_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_25
    (if 3 ≤ innerCandidates40_24.length then search edge 3 [191, 5813] innerCandidates40_24 6004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_25

def innerTail40_23 : List Nat := 5153 :: innerTail40_24
def innerCandidates40_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_23_checked : candidates edge 4 191 5153 innerTail40_24 best.bound = innerCandidates40_23 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_23 : scan edge (search edge 3) 4 [191] 191 innerTail40_23 best = best := by
  rw [innerTail40_23, scan, if_neg (by decide), innerCandidates40_23_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_24
    (if 3 ≤ innerCandidates40_23.length then search edge 3 [191, 5153] innerCandidates40_23 5344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_24

def innerTail40_22 : List Nat := 5099 :: innerTail40_23
def innerCandidates40_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_22_checked : candidates edge 4 191 5099 innerTail40_23 best.bound = innerCandidates40_22 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_22 : scan edge (search edge 3) 4 [191] 191 innerTail40_22 best = best := by
  rw [innerTail40_22, scan, if_neg (by decide), innerCandidates40_22_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_23
    (if 3 ≤ innerCandidates40_22.length then search edge 3 [191, 5099] innerCandidates40_22 5290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_23

def innerTail40_21 : List Nat := 4691 :: innerTail40_22
def innerCandidates40_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_21_checked : candidates edge 4 191 4691 innerTail40_22 best.bound = innerCandidates40_21 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_21 : scan edge (search edge 3) 4 [191] 191 innerTail40_21 best = best := by
  rw [innerTail40_21, scan, if_neg (by decide), innerCandidates40_21_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_22
    (if 3 ≤ innerCandidates40_21.length then search edge 3 [191, 4691] innerCandidates40_21 4882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_22

def innerTail40_20 : List Nat := 4673 :: innerTail40_21
def innerCandidates40_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_20_checked : candidates edge 4 191 4673 innerTail40_21 best.bound = innerCandidates40_20 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_20 : scan edge (search edge 3) 4 [191] 191 innerTail40_20 best = best := by
  rw [innerTail40_20, scan, if_neg (by decide), innerCandidates40_20_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_21
    (if 3 ≤ innerCandidates40_20.length then search edge 3 [191, 4673] innerCandidates40_20 4864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_21

def innerTail40_19 : List Nat := 4259 :: innerTail40_20
def innerCandidates40_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_19_checked : candidates edge 4 191 4259 innerTail40_20 best.bound = innerCandidates40_19 := by
  unfold candidates
  have row : edge 4259 = SearchData.row4259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_19 : scan edge (search edge 3) 4 [191] 191 innerTail40_19 best = best := by
  rw [innerTail40_19, scan, if_neg (by decide), innerCandidates40_19_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_20
    (if 3 ≤ innerCandidates40_19.length then search edge 3 [191, 4259] innerCandidates40_19 4450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_20

def innerTail40_18 : List Nat := 4127 :: innerTail40_19
def innerCandidates40_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_18_checked : candidates edge 4 191 4127 innerTail40_19 best.bound = innerCandidates40_18 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_18 : scan edge (search edge 3) 4 [191] 191 innerTail40_18 best = best := by
  rw [innerTail40_18, scan, if_neg (by decide), innerCandidates40_18_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_19
    (if 3 ≤ innerCandidates40_18.length then search edge 3 [191, 4127] innerCandidates40_18 4318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_19

def innerTail40_17 : List Nat := 3581 :: innerTail40_18
def innerCandidates40_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_17_checked : candidates edge 4 191 3581 innerTail40_18 best.bound = innerCandidates40_17 := by
  unfold candidates
  have row : edge 3581 = SearchData.row3581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_17 : scan edge (search edge 3) 4 [191] 191 innerTail40_17 best = best := by
  rw [innerTail40_17, scan, if_neg (by decide), innerCandidates40_17_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_18
    (if 3 ≤ innerCandidates40_17.length then search edge 3 [191, 3581] innerCandidates40_17 3772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_18

def innerTail40_16 : List Nat := 3251 :: innerTail40_17
def innerCandidates40_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_16_checked : candidates edge 4 191 3251 innerTail40_17 best.bound = innerCandidates40_16 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_16 : scan edge (search edge 3) 4 [191] 191 innerTail40_16 best = best := by
  rw [innerTail40_16, scan, if_neg (by decide), innerCandidates40_16_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_17
    (if 3 ≤ innerCandidates40_16.length then search edge 3 [191, 3251] innerCandidates40_16 3442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_17

def innerTail40_15 : List Nat := 2903 :: innerTail40_16
def innerCandidates40_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_15_checked : candidates edge 4 191 2903 innerTail40_16 best.bound = innerCandidates40_15 := by
  unfold candidates
  have row : edge 2903 = SearchData.row2903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_15 : scan edge (search edge 3) 4 [191] 191 innerTail40_15 best = best := by
  rw [innerTail40_15, scan, if_neg (by decide), innerCandidates40_15_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_16
    (if 3 ≤ innerCandidates40_15.length then search edge 3 [191, 2903] innerCandidates40_15 3094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_16

def innerTail40_14 : List Nat := 2423 :: innerTail40_15
def innerCandidates40_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_14_checked : candidates edge 4 191 2423 innerTail40_15 best.bound = innerCandidates40_14 := by
  unfold candidates
  have row : edge 2423 = PrimePairCache.smallRow2423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_14 : scan edge (search edge 3) 4 [191] 191 innerTail40_14 best = best := by
  rw [innerTail40_14, scan, if_neg (by decide), innerCandidates40_14_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_15
    (if 3 ≤ innerCandidates40_14.length then search edge 3 [191, 2423] innerCandidates40_14 2614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_15

def innerTail40_13 : List Nat := 2213 :: innerTail40_14
def innerCandidates40_13 : List Nat := [3581, 4691]
set_option maxRecDepth 4096 in
theorem innerCandidates40_13_checked : candidates edge 4 191 2213 innerTail40_14 best.bound = innerCandidates40_13 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_13 : scan edge (search edge 3) 4 [191] 191 innerTail40_13 best = best := by
  rw [innerTail40_13, scan, if_neg (by decide), innerCandidates40_13_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_14
    (if 3 ≤ innerCandidates40_13.length then search edge 3 [191, 2213] innerCandidates40_13 2404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_14

def innerTail40_12 : List Nat := 2069 :: innerTail40_13
def innerCandidates40_12 : List Nat := [4691]
set_option maxRecDepth 4096 in
theorem innerCandidates40_12_checked : candidates edge 4 191 2069 innerTail40_13 best.bound = innerCandidates40_12 := by
  unfold candidates
  have row : edge 2069 = PrimePairCache.smallRow2069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_12 : scan edge (search edge 3) 4 [191] 191 innerTail40_12 best = best := by
  rw [innerTail40_12, scan, if_neg (by decide), innerCandidates40_12_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_13
    (if 3 ≤ innerCandidates40_12.length then search edge 3 [191, 2069] innerCandidates40_12 2260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_13

def innerTail40_11 : List Nat := 1697 :: innerTail40_12
def innerCandidates40_11 : List Nat := [5099, 8237]
set_option maxRecDepth 4096 in
theorem innerCandidates40_11_checked : candidates edge 4 191 1697 innerTail40_12 best.bound = innerCandidates40_11 := by
  unfold candidates
  have row : edge 1697 = PrimePairCache.smallRow1697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_11 : scan edge (search edge 3) 4 [191] 191 innerTail40_11 best = best := by
  rw [innerTail40_11, scan, if_neg (by decide), innerCandidates40_11_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_12
    (if 3 ≤ innerCandidates40_11.length then search edge 3 [191, 1697] innerCandidates40_11 1888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_12

def innerTail40_10 : List Nat := 1583 :: innerTail40_11
def innerCandidates40_10 : List Nat := [6287]
set_option maxRecDepth 4096 in
theorem innerCandidates40_10_checked : candidates edge 4 191 1583 innerTail40_11 best.bound = innerCandidates40_10 := by
  unfold candidates
  have row : edge 1583 = PrimePairCache.smallRow1583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_10 : scan edge (search edge 3) 4 [191] 191 innerTail40_10 best = best := by
  rw [innerTail40_10, scan, if_neg (by decide), innerCandidates40_10_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_11
    (if 3 ≤ innerCandidates40_10.length then search edge 3 [191, 1583] innerCandidates40_10 1774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_11

def innerTail40_9 : List Nat := 1373 :: innerTail40_10
def innerCandidates40_9 : List Nat := [2423]
set_option maxRecDepth 4096 in
theorem innerCandidates40_9_checked : candidates edge 4 191 1373 innerTail40_10 best.bound = innerCandidates40_9 := by
  unfold candidates
  have row : edge 1373 = PrimePairCache.smallRow1373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_9 : scan edge (search edge 3) 4 [191] 191 innerTail40_9 best = best := by
  rw [innerTail40_9, scan, if_neg (by decide), innerCandidates40_9_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_10
    (if 3 ≤ innerCandidates40_9.length then search edge 3 [191, 1373] innerCandidates40_9 1564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_10

def innerTail40_8 : List Nat := 1319 :: innerTail40_9
def innerCandidates40_8 : List Nat := [5843, 5981]
set_option maxRecDepth 4096 in
theorem innerCandidates40_8_checked : candidates edge 4 191 1319 innerTail40_9 best.bound = innerCandidates40_8 := by
  unfold candidates
  have row : edge 1319 = PrimePairCache.smallRow1319.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_8 : scan edge (search edge 3) 4 [191] 191 innerTail40_8 best = best := by
  rw [innerTail40_8, scan, if_neg (by decide), innerCandidates40_8_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_9
    (if 3 ≤ innerCandidates40_8.length then search edge 3 [191, 1319] innerCandidates40_8 1510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_9

def innerTail40_7 : List Nat := 1103 :: innerTail40_8
def innerCandidates40_7 : List Nat := [1373, 5813]
set_option maxRecDepth 4096 in
theorem innerCandidates40_7_checked : candidates edge 4 191 1103 innerTail40_8 best.bound = innerCandidates40_7 := by
  unfold candidates
  have row : edge 1103 = PrimePairCache.smallRow1103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_7 : scan edge (search edge 3) 4 [191] 191 innerTail40_7 best = best := by
  rw [innerTail40_7, scan, if_neg (by decide), innerCandidates40_7_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_8
    (if 3 ≤ innerCandidates40_7.length then search edge 3 [191, 1103] innerCandidates40_7 1294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_8

def innerTail40_6 : List Nat := 977 :: innerTail40_7
def innerCandidates40_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_6_checked : candidates edge 4 191 977 innerTail40_7 best.bound = innerCandidates40_6 := by
  unfold candidates
  have row : edge 977 = PrimePairCache.smallRow977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_6 : scan edge (search edge 3) 4 [191] 191 innerTail40_6 best = best := by
  rw [innerTail40_6, scan, if_neg (by decide), innerCandidates40_6_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_7
    (if 3 ≤ innerCandidates40_6.length then search edge 3 [191, 977] innerCandidates40_6 1168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_7

def innerTail40_5 : List Nat := 953 :: innerTail40_6
def innerCandidates40_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_5_checked : candidates edge 4 191 953 innerTail40_6 best.bound = innerCandidates40_5 := by
  unfold candidates
  have row : edge 953 = PrimePairCache.smallRow953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_5 : scan edge (search edge 3) 4 [191] 191 innerTail40_5 best = best := by
  rw [innerTail40_5, scan, if_neg (by decide), innerCandidates40_5_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_6
    (if 3 ≤ innerCandidates40_5.length then search edge 3 [191, 953] innerCandidates40_5 1144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_6

def innerTail40_4 : List Nat := 599 :: innerTail40_5
def innerCandidates40_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_4_checked : candidates edge 4 191 599 innerTail40_5 best.bound = innerCandidates40_4 := by
  unfold candidates
  have row : edge 599 = PrimePairCache.smallRow599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_4 : scan edge (search edge 3) 4 [191] 191 innerTail40_4 best = best := by
  rw [innerTail40_4, scan, if_neg (by decide), innerCandidates40_4_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_5
    (if 3 ≤ innerCandidates40_4.length then search edge 3 [191, 599] innerCandidates40_4 790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_5

def innerTail40_3 : List Nat := 461 :: innerTail40_4
def innerCandidates40_3 : List Nat := [1697]
set_option maxRecDepth 4096 in
theorem innerCandidates40_3_checked : candidates edge 4 191 461 innerTail40_4 best.bound = innerCandidates40_3 := by
  unfold candidates
  have row : edge 461 = PrimePairCache.smallRow461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_3 : scan edge (search edge 3) 4 [191] 191 innerTail40_3 best = best := by
  rw [innerTail40_3, scan, if_neg (by decide), innerCandidates40_3_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_4
    (if 3 ≤ innerCandidates40_3.length then search edge 3 [191, 461] innerCandidates40_3 652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_4

def innerTail40_2 : List Nat := 281 :: innerTail40_3
def innerCandidates40_2 : List Nat := [5843]
set_option maxRecDepth 4096 in
theorem innerCandidates40_2_checked : candidates edge 4 191 281 innerTail40_3 best.bound = innerCandidates40_2 := by
  unfold candidates
  have row : edge 281 = PrimePairCache.smallRow281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_2 : scan edge (search edge 3) 4 [191] 191 innerTail40_2 best = best := by
  rw [innerTail40_2, scan, if_neg (by decide), innerCandidates40_2_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_3
    (if 3 ≤ innerCandidates40_2.length then search edge 3 [191, 281] innerCandidates40_2 472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_3

def innerTail40_1 : List Nat := 251 :: innerTail40_2
def innerCandidates40_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates40_1_checked : candidates edge 4 191 251 innerTail40_2 best.bound = innerCandidates40_1 := by
  unfold candidates
  have row : edge 251 = PrimePairCache.smallRow251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan40_1 : scan edge (search edge 3) 4 [191] 191 innerTail40_1 best = best := by
  rw [innerTail40_1, scan, if_neg (by decide), innerCandidates40_1_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_2
    (if 3 ≤ innerCandidates40_1.length then search edge 3 [191, 251] innerCandidates40_1 442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan40_2

def innerTail40_0 : List Nat := 227 :: innerTail40_1
def innerCandidates40_0 : List Nat := [281, 2903, 4691]
set_option maxRecDepth 4096 in
theorem innerCandidates40_0_checked : candidates edge 4 191 227 innerTail40_1 best.bound = innerCandidates40_0 := by
  unfold candidates
  have row : edge 227 = PrimePairCache.smallRow227.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild40_0 : search edge 3 [191, 227] innerCandidates40_0 418 best = best := by decide

theorem innerScan40_0 : scan edge (search edge 3) 4 [191] 191 innerTail40_0 best = best := by
  rw [innerTail40_0, scan, if_neg (by decide), innerCandidates40_0_checked]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_1
    (if 3 ≤ innerCandidates40_0.length then search edge 3 [191, 227] innerCandidates40_0 418 best else best) = best
  rw [if_pos (by decide), innerChild40_0]
  exact innerScan40_1

theorem rootBranch40 : search edge 4 [191] rootCandidates40 191 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [191] 191 innerTail40_0 best = best
  exact innerScan40_0

end ProjectEulerSolutions.PrimePairUpper.Search
