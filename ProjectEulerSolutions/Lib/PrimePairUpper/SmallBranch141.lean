import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates252 : List Nat := [1637, 1949, 1979, 2477, 2693, 3023, 3779, 4133, 4673, 4967, 5297, 5813, 6311, 6899, 6971, 7229, 7481, 7487, 7853]
set_option maxRecDepth 4096 in
theorem rootCandidates252_checked : candidates edge 5 0 1613 tail253 best.bound = rootCandidates252 := by
  unfold candidates
  have row : edge 1613 = PrimePairCache.smallRow1613.contains := by funext q; rfl
  rw [row]
  decide

def innerTail252_19 : List Nat := []
theorem innerScan252_19 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_19 best = best := rfl

def innerTail252_18 : List Nat := 7853 :: innerTail252_19
def innerCandidates252_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_18_checked : candidates edge 4 1613 7853 innerTail252_19 best.bound = innerCandidates252_18 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_18 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_18 best = best := by
  rw [innerTail252_18, scan, if_neg (by decide), innerCandidates252_18_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_19
    (if 3 ≤ innerCandidates252_18.length then search edge 3 [1613, 7853] innerCandidates252_18 9466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_19

def innerTail252_17 : List Nat := 7487 :: innerTail252_18
def innerCandidates252_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_17_checked : candidates edge 4 1613 7487 innerTail252_18 best.bound = innerCandidates252_17 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_17 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_17 best = best := by
  rw [innerTail252_17, scan, if_neg (by decide), innerCandidates252_17_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_18
    (if 3 ≤ innerCandidates252_17.length then search edge 3 [1613, 7487] innerCandidates252_17 9100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_18

def innerTail252_16 : List Nat := 7481 :: innerTail252_17
def innerCandidates252_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_16_checked : candidates edge 4 1613 7481 innerTail252_17 best.bound = innerCandidates252_16 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_16 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_16 best = best := by
  rw [innerTail252_16, scan, if_neg (by decide), innerCandidates252_16_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_17
    (if 3 ≤ innerCandidates252_16.length then search edge 3 [1613, 7481] innerCandidates252_16 9094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_17

def innerTail252_15 : List Nat := 7229 :: innerTail252_16
def innerCandidates252_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_15_checked : candidates edge 4 1613 7229 innerTail252_16 best.bound = innerCandidates252_15 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_15 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_15 best = best := by
  rw [innerTail252_15, scan, if_neg (by decide), innerCandidates252_15_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_16
    (if 3 ≤ innerCandidates252_15.length then search edge 3 [1613, 7229] innerCandidates252_15 8842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_16

def innerTail252_14 : List Nat := 6971 :: innerTail252_15
def innerCandidates252_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_14_checked : candidates edge 4 1613 6971 innerTail252_15 best.bound = innerCandidates252_14 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_14 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_14 best = best := by
  rw [innerTail252_14, scan, if_neg (by decide), innerCandidates252_14_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_15
    (if 3 ≤ innerCandidates252_14.length then search edge 3 [1613, 6971] innerCandidates252_14 8584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_15

def innerTail252_13 : List Nat := 6899 :: innerTail252_14
def innerCandidates252_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_13_checked : candidates edge 4 1613 6899 innerTail252_14 best.bound = innerCandidates252_13 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_13 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_13 best = best := by
  rw [innerTail252_13, scan, if_neg (by decide), innerCandidates252_13_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_14
    (if 3 ≤ innerCandidates252_13.length then search edge 3 [1613, 6899] innerCandidates252_13 8512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_14

def innerTail252_12 : List Nat := 6311 :: innerTail252_13
def innerCandidates252_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_12_checked : candidates edge 4 1613 6311 innerTail252_13 best.bound = innerCandidates252_12 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_12 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_12 best = best := by
  rw [innerTail252_12, scan, if_neg (by decide), innerCandidates252_12_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_13
    (if 3 ≤ innerCandidates252_12.length then search edge 3 [1613, 6311] innerCandidates252_12 7924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_13

def innerTail252_11 : List Nat := 5813 :: innerTail252_12
def innerCandidates252_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_11_checked : candidates edge 4 1613 5813 innerTail252_12 best.bound = innerCandidates252_11 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_11 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_11 best = best := by
  rw [innerTail252_11, scan, if_neg (by decide), innerCandidates252_11_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_12
    (if 3 ≤ innerCandidates252_11.length then search edge 3 [1613, 5813] innerCandidates252_11 7426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_12

def innerTail252_10 : List Nat := 5297 :: innerTail252_11
def innerCandidates252_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_10_checked : candidates edge 4 1613 5297 innerTail252_11 best.bound = innerCandidates252_10 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_10 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_10 best = best := by
  rw [innerTail252_10, scan, if_neg (by decide), innerCandidates252_10_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_11
    (if 3 ≤ innerCandidates252_10.length then search edge 3 [1613, 5297] innerCandidates252_10 6910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_11

def innerTail252_9 : List Nat := 4967 :: innerTail252_10
def innerCandidates252_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_9_checked : candidates edge 4 1613 4967 innerTail252_10 best.bound = innerCandidates252_9 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_9 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_9 best = best := by
  rw [innerTail252_9, scan, if_neg (by decide), innerCandidates252_9_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_10
    (if 3 ≤ innerCandidates252_9.length then search edge 3 [1613, 4967] innerCandidates252_9 6580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_10

def innerTail252_8 : List Nat := 4673 :: innerTail252_9
def innerCandidates252_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_8_checked : candidates edge 4 1613 4673 innerTail252_9 best.bound = innerCandidates252_8 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_8 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_8 best = best := by
  rw [innerTail252_8, scan, if_neg (by decide), innerCandidates252_8_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_9
    (if 3 ≤ innerCandidates252_8.length then search edge 3 [1613, 4673] innerCandidates252_8 6286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_9

def innerTail252_7 : List Nat := 4133 :: innerTail252_8
def innerCandidates252_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_7_checked : candidates edge 4 1613 4133 innerTail252_8 best.bound = innerCandidates252_7 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_7 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_7 best = best := by
  rw [innerTail252_7, scan, if_neg (by decide), innerCandidates252_7_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_8
    (if 3 ≤ innerCandidates252_7.length then search edge 3 [1613, 4133] innerCandidates252_7 5746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_8

def innerTail252_6 : List Nat := 3779 :: innerTail252_7
def innerCandidates252_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_6_checked : candidates edge 4 1613 3779 innerTail252_7 best.bound = innerCandidates252_6 := by
  unfold candidates
  have row : edge 3779 = SearchData.row3779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_6 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_6 best = best := by
  rw [innerTail252_6, scan, if_neg (by decide), innerCandidates252_6_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_7
    (if 3 ≤ innerCandidates252_6.length then search edge 3 [1613, 3779] innerCandidates252_6 5392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_7

def innerTail252_5 : List Nat := 3023 :: innerTail252_6
def innerCandidates252_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_5_checked : candidates edge 4 1613 3023 innerTail252_6 best.bound = innerCandidates252_5 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_5 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_5 best = best := by
  rw [innerTail252_5, scan, if_neg (by decide), innerCandidates252_5_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_6
    (if 3 ≤ innerCandidates252_5.length then search edge 3 [1613, 3023] innerCandidates252_5 4636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_6

def innerTail252_4 : List Nat := 2693 :: innerTail252_5
def innerCandidates252_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_4_checked : candidates edge 4 1613 2693 innerTail252_5 best.bound = innerCandidates252_4 := by
  unfold candidates
  have row : edge 2693 = SearchData.row2693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_4 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_4 best = best := by
  rw [innerTail252_4, scan, if_neg (by decide), innerCandidates252_4_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_5
    (if 3 ≤ innerCandidates252_4.length then search edge 3 [1613, 2693] innerCandidates252_4 4306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_5

def innerTail252_3 : List Nat := 2477 :: innerTail252_4
def innerCandidates252_3 : List Nat := [4133, 7853]
set_option maxRecDepth 4096 in
theorem innerCandidates252_3_checked : candidates edge 4 1613 2477 innerTail252_4 best.bound = innerCandidates252_3 := by
  unfold candidates
  have row : edge 2477 = PrimePairCache.smallRow2477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_3 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_3 best = best := by
  rw [innerTail252_3, scan, if_neg (by decide), innerCandidates252_3_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_4
    (if 3 ≤ innerCandidates252_3.length then search edge 3 [1613, 2477] innerCandidates252_3 4090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_4

def innerTail252_2 : List Nat := 1979 :: innerTail252_3
def innerCandidates252_2 : List Nat := [2693]
set_option maxRecDepth 4096 in
theorem innerCandidates252_2_checked : candidates edge 4 1613 1979 innerTail252_3 best.bound = innerCandidates252_2 := by
  unfold candidates
  have row : edge 1979 = PrimePairCache.smallRow1979.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_2 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_2 best = best := by
  rw [innerTail252_2, scan, if_neg (by decide), innerCandidates252_2_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_3
    (if 3 ≤ innerCandidates252_2.length then search edge 3 [1613, 1979] innerCandidates252_2 3592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_3

def innerTail252_1 : List Nat := 1949 :: innerTail252_2
def innerCandidates252_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_1_checked : candidates edge 4 1613 1949 innerTail252_2 best.bound = innerCandidates252_1 := by
  unfold candidates
  have row : edge 1949 = PrimePairCache.smallRow1949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_1 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_1 best = best := by
  rw [innerTail252_1, scan, if_neg (by decide), innerCandidates252_1_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_2
    (if 3 ≤ innerCandidates252_1.length then search edge 3 [1613, 1949] innerCandidates252_1 3562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_2

def innerTail252_0 : List Nat := 1637 :: innerTail252_1
def innerCandidates252_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates252_0_checked : candidates edge 4 1613 1637 innerTail252_1 best.bound = innerCandidates252_0 := by
  unfold candidates
  have row : edge 1637 = PrimePairCache.smallRow1637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan252_0 : scan edge (search edge 3) 4 [1613] 1613 innerTail252_0 best = best := by
  rw [innerTail252_0, scan, if_neg (by decide), innerCandidates252_0_checked]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_1
    (if 3 ≤ innerCandidates252_0.length then search edge 3 [1613, 1637] innerCandidates252_0 3250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan252_1

theorem rootBranch252 : search edge 4 [1613] rootCandidates252 1613 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1613] 1613 innerTail252_0 best = best
  exact innerScan252_0

def rootCandidates251 : List Nat := [1783, 2017, 2161, 2467, 2803, 2833, 3613, 3673, 3697, 3769, 3793, 3967, 4129, 4201, 4621, 4723, 5011, 5179, 5647, 6991, 7243, 7603, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates251_checked : candidates edge 5 0 1609 tail252 best.bound = rootCandidates251 := by
  unfold candidates
  have row : edge 1609 = PrimePairCache.smallRow1609.contains := by funext q; rfl
  rw [row]
  decide

def innerTail251_23 : List Nat := []
theorem innerScan251_23 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_23 best = best := rfl

def innerTail251_22 : List Nat := 8287 :: innerTail251_23
def innerCandidates251_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_22_checked : candidates edge 4 1609 8287 innerTail251_23 best.bound = innerCandidates251_22 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_22 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_22 best = best := by
  rw [innerTail251_22, scan, if_neg (by decide), innerCandidates251_22_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_23
    (if 3 ≤ innerCandidates251_22.length then search edge 3 [1609, 8287] innerCandidates251_22 9896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_23

def innerTail251_21 : List Nat := 7603 :: innerTail251_22
def innerCandidates251_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_21_checked : candidates edge 4 1609 7603 innerTail251_22 best.bound = innerCandidates251_21 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_21 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_21 best = best := by
  rw [innerTail251_21, scan, if_neg (by decide), innerCandidates251_21_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_22
    (if 3 ≤ innerCandidates251_21.length then search edge 3 [1609, 7603] innerCandidates251_21 9212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_22

def innerTail251_20 : List Nat := 7243 :: innerTail251_21
def innerCandidates251_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_20_checked : candidates edge 4 1609 7243 innerTail251_21 best.bound = innerCandidates251_20 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_20 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_20 best = best := by
  rw [innerTail251_20, scan, if_neg (by decide), innerCandidates251_20_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_21
    (if 3 ≤ innerCandidates251_20.length then search edge 3 [1609, 7243] innerCandidates251_20 8852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_21

def innerTail251_19 : List Nat := 6991 :: innerTail251_20
def innerCandidates251_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_19_checked : candidates edge 4 1609 6991 innerTail251_20 best.bound = innerCandidates251_19 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_19 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_19 best = best := by
  rw [innerTail251_19, scan, if_neg (by decide), innerCandidates251_19_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_20
    (if 3 ≤ innerCandidates251_19.length then search edge 3 [1609, 6991] innerCandidates251_19 8600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_20

def innerTail251_18 : List Nat := 5647 :: innerTail251_19
def innerCandidates251_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_18_checked : candidates edge 4 1609 5647 innerTail251_19 best.bound = innerCandidates251_18 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_18 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_18 best = best := by
  rw [innerTail251_18, scan, if_neg (by decide), innerCandidates251_18_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_19
    (if 3 ≤ innerCandidates251_18.length then search edge 3 [1609, 5647] innerCandidates251_18 7256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_19

def innerTail251_17 : List Nat := 5179 :: innerTail251_18
def innerCandidates251_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_17_checked : candidates edge 4 1609 5179 innerTail251_18 best.bound = innerCandidates251_17 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_17 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_17 best = best := by
  rw [innerTail251_17, scan, if_neg (by decide), innerCandidates251_17_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_18
    (if 3 ≤ innerCandidates251_17.length then search edge 3 [1609, 5179] innerCandidates251_17 6788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_18

def innerTail251_16 : List Nat := 5011 :: innerTail251_17
def innerCandidates251_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_16_checked : candidates edge 4 1609 5011 innerTail251_17 best.bound = innerCandidates251_16 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_16 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_16 best = best := by
  rw [innerTail251_16, scan, if_neg (by decide), innerCandidates251_16_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_17
    (if 3 ≤ innerCandidates251_16.length then search edge 3 [1609, 5011] innerCandidates251_16 6620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_17

def innerTail251_15 : List Nat := 4723 :: innerTail251_16
def innerCandidates251_15 : List Nat := [7243]
set_option maxRecDepth 4096 in
theorem innerCandidates251_15_checked : candidates edge 4 1609 4723 innerTail251_16 best.bound = innerCandidates251_15 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_15 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_15 best = best := by
  rw [innerTail251_15, scan, if_neg (by decide), innerCandidates251_15_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_16
    (if 3 ≤ innerCandidates251_15.length then search edge 3 [1609, 4723] innerCandidates251_15 6332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_16

def innerTail251_14 : List Nat := 4621 :: innerTail251_15
def innerCandidates251_14 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates251_14_checked : candidates edge 4 1609 4621 innerTail251_15 best.bound = innerCandidates251_14 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_14 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_14 best = best := by
  rw [innerTail251_14, scan, if_neg (by decide), innerCandidates251_14_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_15
    (if 3 ≤ innerCandidates251_14.length then search edge 3 [1609, 4621] innerCandidates251_14 6230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_15

def innerTail251_13 : List Nat := 4201 :: innerTail251_14
def innerCandidates251_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_13_checked : candidates edge 4 1609 4201 innerTail251_14 best.bound = innerCandidates251_13 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_13 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_13 best = best := by
  rw [innerTail251_13, scan, if_neg (by decide), innerCandidates251_13_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_14
    (if 3 ≤ innerCandidates251_13.length then search edge 3 [1609, 4201] innerCandidates251_13 5810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_14

def innerTail251_12 : List Nat := 4129 :: innerTail251_13
def innerCandidates251_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_12_checked : candidates edge 4 1609 4129 innerTail251_13 best.bound = innerCandidates251_12 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_12 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_12 best = best := by
  rw [innerTail251_12, scan, if_neg (by decide), innerCandidates251_12_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_13
    (if 3 ≤ innerCandidates251_12.length then search edge 3 [1609, 4129] innerCandidates251_12 5738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_13

def innerTail251_11 : List Nat := 3967 :: innerTail251_12
def innerCandidates251_11 : List Nat := [5011]
set_option maxRecDepth 4096 in
theorem innerCandidates251_11_checked : candidates edge 4 1609 3967 innerTail251_12 best.bound = innerCandidates251_11 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_11 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_11 best = best := by
  rw [innerTail251_11, scan, if_neg (by decide), innerCandidates251_11_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_12
    (if 3 ≤ innerCandidates251_11.length then search edge 3 [1609, 3967] innerCandidates251_11 5576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_12

def innerTail251_10 : List Nat := 3793 :: innerTail251_11
def innerCandidates251_10 : List Nat := [5179]
set_option maxRecDepth 4096 in
theorem innerCandidates251_10_checked : candidates edge 4 1609 3793 innerTail251_11 best.bound = innerCandidates251_10 := by
  unfold candidates
  have row : edge 3793 = SearchData.row3793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_10 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_10 best = best := by
  rw [innerTail251_10, scan, if_neg (by decide), innerCandidates251_10_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_11
    (if 3 ≤ innerCandidates251_10.length then search edge 3 [1609, 3793] innerCandidates251_10 5402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_11

def innerTail251_9 : List Nat := 3769 :: innerTail251_10
def innerCandidates251_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_9_checked : candidates edge 4 1609 3769 innerTail251_10 best.bound = innerCandidates251_9 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_9 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_9 best = best := by
  rw [innerTail251_9, scan, if_neg (by decide), innerCandidates251_9_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_10
    (if 3 ≤ innerCandidates251_9.length then search edge 3 [1609, 3769] innerCandidates251_9 5378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_10

def innerTail251_8 : List Nat := 3697 :: innerTail251_9
def innerCandidates251_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_8_checked : candidates edge 4 1609 3697 innerTail251_9 best.bound = innerCandidates251_8 := by
  unfold candidates
  have row : edge 3697 = SearchData.row3697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_8 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_8 best = best := by
  rw [innerTail251_8, scan, if_neg (by decide), innerCandidates251_8_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_9
    (if 3 ≤ innerCandidates251_8.length then search edge 3 [1609, 3697] innerCandidates251_8 5306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_9

def innerTail251_7 : List Nat := 3673 :: innerTail251_8
def innerCandidates251_7 : List Nat := [4129]
set_option maxRecDepth 4096 in
theorem innerCandidates251_7_checked : candidates edge 4 1609 3673 innerTail251_8 best.bound = innerCandidates251_7 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_7 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_7 best = best := by
  rw [innerTail251_7, scan, if_neg (by decide), innerCandidates251_7_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_8
    (if 3 ≤ innerCandidates251_7.length then search edge 3 [1609, 3673] innerCandidates251_7 5282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_8

def innerTail251_6 : List Nat := 3613 :: innerTail251_7
def innerCandidates251_6 : List Nat := [3697]
set_option maxRecDepth 4096 in
theorem innerCandidates251_6_checked : candidates edge 4 1609 3613 innerTail251_7 best.bound = innerCandidates251_6 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_6 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_6 best = best := by
  rw [innerTail251_6, scan, if_neg (by decide), innerCandidates251_6_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_7
    (if 3 ≤ innerCandidates251_6.length then search edge 3 [1609, 3613] innerCandidates251_6 5222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_7

def innerTail251_5 : List Nat := 2833 :: innerTail251_6
def innerCandidates251_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_5_checked : candidates edge 4 1609 2833 innerTail251_6 best.bound = innerCandidates251_5 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_5 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_5 best = best := by
  rw [innerTail251_5, scan, if_neg (by decide), innerCandidates251_5_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_6
    (if 3 ≤ innerCandidates251_5.length then search edge 3 [1609, 2833] innerCandidates251_5 4442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_6

def innerTail251_4 : List Nat := 2803 :: innerTail251_5
def innerCandidates251_4 : List Nat := [3769]
set_option maxRecDepth 4096 in
theorem innerCandidates251_4_checked : candidates edge 4 1609 2803 innerTail251_5 best.bound = innerCandidates251_4 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_4 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_4 best = best := by
  rw [innerTail251_4, scan, if_neg (by decide), innerCandidates251_4_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_5
    (if 3 ≤ innerCandidates251_4.length then search edge 3 [1609, 2803] innerCandidates251_4 4412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_5

def innerTail251_3 : List Nat := 2467 :: innerTail251_4
def innerCandidates251_3 : List Nat := [4723]
set_option maxRecDepth 4096 in
theorem innerCandidates251_3_checked : candidates edge 4 1609 2467 innerTail251_4 best.bound = innerCandidates251_3 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_3 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_3 best = best := by
  rw [innerTail251_3, scan, if_neg (by decide), innerCandidates251_3_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_4
    (if 3 ≤ innerCandidates251_3.length then search edge 3 [1609, 2467] innerCandidates251_3 4076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_4

def innerTail251_2 : List Nat := 2161 :: innerTail251_3
def innerCandidates251_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates251_2_checked : candidates edge 4 1609 2161 innerTail251_3 best.bound = innerCandidates251_2 := by
  unfold candidates
  have row : edge 2161 = PrimePairCache.smallRow2161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_2 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_2 best = best := by
  rw [innerTail251_2, scan, if_neg (by decide), innerCandidates251_2_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_3
    (if 3 ≤ innerCandidates251_2.length then search edge 3 [1609, 2161] innerCandidates251_2 3770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_3

def innerTail251_1 : List Nat := 2017 :: innerTail251_2
def innerCandidates251_1 : List Nat := [4621, 7603]
set_option maxRecDepth 4096 in
theorem innerCandidates251_1_checked : candidates edge 4 1609 2017 innerTail251_2 best.bound = innerCandidates251_1 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_1 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_1 best = best := by
  rw [innerTail251_1, scan, if_neg (by decide), innerCandidates251_1_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_2
    (if 3 ≤ innerCandidates251_1.length then search edge 3 [1609, 2017] innerCandidates251_1 3626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_2

def innerTail251_0 : List Nat := 1783 :: innerTail251_1
def innerCandidates251_0 : List Nat := [3769, 5011]
set_option maxRecDepth 4096 in
theorem innerCandidates251_0_checked : candidates edge 4 1609 1783 innerTail251_1 best.bound = innerCandidates251_0 := by
  unfold candidates
  have row : edge 1783 = PrimePairCache.smallRow1783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan251_0 : scan edge (search edge 3) 4 [1609] 1609 innerTail251_0 best = best := by
  rw [innerTail251_0, scan, if_neg (by decide), innerCandidates251_0_checked]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_1
    (if 3 ≤ innerCandidates251_0.length then search edge 3 [1609, 1783] innerCandidates251_0 3392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan251_1

theorem rootBranch251 : search edge 4 [1609] rootCandidates251 1609 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1609] 1609 innerTail251_0 best = best
  exact innerScan251_0

def rootCandidates250 : List Nat := [1901, 2549, 3203, 3767, 4271, 4481, 4637, 5009, 5099, 5351, 5471, 5849, 5903, 7109, 7541, 7673, 7877, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates250_checked : candidates edge 5 0 1607 tail251 best.bound = rootCandidates250 := by
  unfold candidates
  have row : edge 1607 = PrimePairCache.smallRow1607.contains := by funext q; rfl
  rw [row]
  decide

def innerTail250_18 : List Nat := []
theorem innerScan250_18 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_18 best = best := rfl

def innerTail250_17 : List Nat := 8171 :: innerTail250_18
def innerCandidates250_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates250_17_checked : candidates edge 4 1607 8171 innerTail250_18 best.bound = innerCandidates250_17 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_17 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_17 best = best := by
  rw [innerTail250_17, scan, if_neg (by decide), innerCandidates250_17_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_18
    (if 3 ≤ innerCandidates250_17.length then search edge 3 [1607, 8171] innerCandidates250_17 9778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_18

def innerTail250_16 : List Nat := 7877 :: innerTail250_17
def innerCandidates250_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates250_16_checked : candidates edge 4 1607 7877 innerTail250_17 best.bound = innerCandidates250_16 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_16 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_16 best = best := by
  rw [innerTail250_16, scan, if_neg (by decide), innerCandidates250_16_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_17
    (if 3 ≤ innerCandidates250_16.length then search edge 3 [1607, 7877] innerCandidates250_16 9484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_17

def innerTail250_15 : List Nat := 7673 :: innerTail250_16
def innerCandidates250_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates250_15_checked : candidates edge 4 1607 7673 innerTail250_16 best.bound = innerCandidates250_15 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_15 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_15 best = best := by
  rw [innerTail250_15, scan, if_neg (by decide), innerCandidates250_15_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_16
    (if 3 ≤ innerCandidates250_15.length then search edge 3 [1607, 7673] innerCandidates250_15 9280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_16

def innerTail250_14 : List Nat := 7541 :: innerTail250_15
def innerCandidates250_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates250_14_checked : candidates edge 4 1607 7541 innerTail250_15 best.bound = innerCandidates250_14 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_14 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_14 best = best := by
  rw [innerTail250_14, scan, if_neg (by decide), innerCandidates250_14_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_15
    (if 3 ≤ innerCandidates250_14.length then search edge 3 [1607, 7541] innerCandidates250_14 9148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_15

def innerTail250_13 : List Nat := 7109 :: innerTail250_14
def innerCandidates250_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates250_13_checked : candidates edge 4 1607 7109 innerTail250_14 best.bound = innerCandidates250_13 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_13 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_13 best = best := by
  rw [innerTail250_13, scan, if_neg (by decide), innerCandidates250_13_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_14
    (if 3 ≤ innerCandidates250_13.length then search edge 3 [1607, 7109] innerCandidates250_13 8716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_14

def innerTail250_12 : List Nat := 5903 :: innerTail250_13
def innerCandidates250_12 : List Nat := [7109, 7877]
set_option maxRecDepth 4096 in
theorem innerCandidates250_12_checked : candidates edge 4 1607 5903 innerTail250_13 best.bound = innerCandidates250_12 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_12 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_12 best = best := by
  rw [innerTail250_12, scan, if_neg (by decide), innerCandidates250_12_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_13
    (if 3 ≤ innerCandidates250_12.length then search edge 3 [1607, 5903] innerCandidates250_12 7510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_13

def innerTail250_11 : List Nat := 5849 :: innerTail250_12
def innerCandidates250_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates250_11_checked : candidates edge 4 1607 5849 innerTail250_12 best.bound = innerCandidates250_11 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_11 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_11 best = best := by
  rw [innerTail250_11, scan, if_neg (by decide), innerCandidates250_11_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_12
    (if 3 ≤ innerCandidates250_11.length then search edge 3 [1607, 5849] innerCandidates250_11 7456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_12

def innerTail250_10 : List Nat := 5471 :: innerTail250_11
def innerCandidates250_10 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates250_10_checked : candidates edge 4 1607 5471 innerTail250_11 best.bound = innerCandidates250_10 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_10 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_10 best = best := by
  rw [innerTail250_10, scan, if_neg (by decide), innerCandidates250_10_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_11
    (if 3 ≤ innerCandidates250_10.length then search edge 3 [1607, 5471] innerCandidates250_10 7078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_11

def innerTail250_9 : List Nat := 5351 :: innerTail250_10
def innerCandidates250_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates250_9_checked : candidates edge 4 1607 5351 innerTail250_10 best.bound = innerCandidates250_9 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_9 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_9 best = best := by
  rw [innerTail250_9, scan, if_neg (by decide), innerCandidates250_9_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_10
    (if 3 ≤ innerCandidates250_9.length then search edge 3 [1607, 5351] innerCandidates250_9 6958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_10

def innerTail250_8 : List Nat := 5099 :: innerTail250_9
def innerCandidates250_8 : List Nat := [5471]
set_option maxRecDepth 4096 in
theorem innerCandidates250_8_checked : candidates edge 4 1607 5099 innerTail250_9 best.bound = innerCandidates250_8 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_8 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_8 best = best := by
  rw [innerTail250_8, scan, if_neg (by decide), innerCandidates250_8_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_9
    (if 3 ≤ innerCandidates250_8.length then search edge 3 [1607, 5099] innerCandidates250_8 6706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_9

def innerTail250_7 : List Nat := 5009 :: innerTail250_8
def innerCandidates250_7 : List Nat := [5351, 5849, 7109]
set_option maxRecDepth 4096 in
theorem innerCandidates250_7_checked : candidates edge 4 1607 5009 innerTail250_8 best.bound = innerCandidates250_7 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild250_7 : search edge 3 [1607, 5009] innerCandidates250_7 6616 best = best := by decide

theorem innerScan250_7 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_7 best = best := by
  rw [innerTail250_7, scan, if_neg (by decide), innerCandidates250_7_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_8
    (if 3 ≤ innerCandidates250_7.length then search edge 3 [1607, 5009] innerCandidates250_7 6616 best else best) = best
  rw [if_pos (by decide), innerChild250_7]
  exact innerScan250_8

def innerTail250_6 : List Nat := 4637 :: innerTail250_7
def innerCandidates250_6 : List Nat := [5009, 5471]
set_option maxRecDepth 4096 in
theorem innerCandidates250_6_checked : candidates edge 4 1607 4637 innerTail250_7 best.bound = innerCandidates250_6 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_6 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_6 best = best := by
  rw [innerTail250_6, scan, if_neg (by decide), innerCandidates250_6_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_7
    (if 3 ≤ innerCandidates250_6.length then search edge 3 [1607, 4637] innerCandidates250_6 6244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_7

def innerTail250_5 : List Nat := 4481 :: innerTail250_6
def innerCandidates250_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates250_5_checked : candidates edge 4 1607 4481 innerTail250_6 best.bound = innerCandidates250_5 := by
  unfold candidates
  have row : edge 4481 = SearchData.row4481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_5 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_5 best = best := by
  rw [innerTail250_5, scan, if_neg (by decide), innerCandidates250_5_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_6
    (if 3 ≤ innerCandidates250_5.length then search edge 3 [1607, 4481] innerCandidates250_5 6088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_6

def innerTail250_4 : List Nat := 4271 :: innerTail250_5
def innerCandidates250_4 : List Nat := [5009]
set_option maxRecDepth 4096 in
theorem innerCandidates250_4_checked : candidates edge 4 1607 4271 innerTail250_5 best.bound = innerCandidates250_4 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_4 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_4 best = best := by
  rw [innerTail250_4, scan, if_neg (by decide), innerCandidates250_4_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_5
    (if 3 ≤ innerCandidates250_4.length then search edge 3 [1607, 4271] innerCandidates250_4 5878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_5

def innerTail250_3 : List Nat := 3767 :: innerTail250_4
def innerCandidates250_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates250_3_checked : candidates edge 4 1607 3767 innerTail250_4 best.bound = innerCandidates250_3 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_3 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_3 best = best := by
  rw [innerTail250_3, scan, if_neg (by decide), innerCandidates250_3_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_4
    (if 3 ≤ innerCandidates250_3.length then search edge 3 [1607, 3767] innerCandidates250_3 5374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_4

def innerTail250_2 : List Nat := 3203 :: innerTail250_3
def innerCandidates250_2 : List Nat := [4481, 7541]
set_option maxRecDepth 4096 in
theorem innerCandidates250_2_checked : candidates edge 4 1607 3203 innerTail250_3 best.bound = innerCandidates250_2 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_2 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_2 best = best := by
  rw [innerTail250_2, scan, if_neg (by decide), innerCandidates250_2_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_3
    (if 3 ≤ innerCandidates250_2.length then search edge 3 [1607, 3203] innerCandidates250_2 4810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_3

def innerTail250_1 : List Nat := 2549 :: innerTail250_2
def innerCandidates250_1 : List Nat := [5471, 7673]
set_option maxRecDepth 4096 in
theorem innerCandidates250_1_checked : candidates edge 4 1607 2549 innerTail250_2 best.bound = innerCandidates250_1 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_1 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_1 best = best := by
  rw [innerTail250_1, scan, if_neg (by decide), innerCandidates250_1_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_2
    (if 3 ≤ innerCandidates250_1.length then search edge 3 [1607, 2549] innerCandidates250_1 4156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_2

def innerTail250_0 : List Nat := 1901 :: innerTail250_1
def innerCandidates250_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates250_0_checked : candidates edge 4 1607 1901 innerTail250_1 best.bound = innerCandidates250_0 := by
  unfold candidates
  have row : edge 1901 = PrimePairCache.smallRow1901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan250_0 : scan edge (search edge 3) 4 [1607] 1607 innerTail250_0 best = best := by
  rw [innerTail250_0, scan, if_neg (by decide), innerCandidates250_0_checked]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_1
    (if 3 ≤ innerCandidates250_0.length then search edge 3 [1607, 1901] innerCandidates250_0 3508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan250_1

theorem rootBranch250 : search edge 4 [1607] rootCandidates250 1607 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1607] 1607 innerTail250_0 best = best
  exact innerScan250_0

end ProjectEulerSolutions.PrimePairUpper.Search
