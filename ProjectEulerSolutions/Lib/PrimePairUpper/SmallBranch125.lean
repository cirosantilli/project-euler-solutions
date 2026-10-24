import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates302 : List Nat := [3253, 3673, 4093, 4099, 4177, 5413, 5521, 6379, 6637, 6781, 7963]
set_option maxRecDepth 4096 in
theorem rootCandidates302_checked : candidates edge 5 0 2011 tail303 best.bound = rootCandidates302 := by
  unfold candidates
  have row : edge 2011 = PrimePairCache.smallRow2011.contains := by funext q; rfl
  rw [row]
  decide

def innerTail302_11 : List Nat := []
theorem innerScan302_11 : scan edge (search edge 3) 4 [2011] 2011 innerTail302_11 best = best := rfl

def innerTail302_10 : List Nat := 7963 :: innerTail302_11
def innerCandidates302_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates302_10_checked : candidates edge 4 2011 7963 innerTail302_11 best.bound = innerCandidates302_10 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan302_10 : scan edge (search edge 3) 4 [2011] 2011 innerTail302_10 best = best := by
  rw [innerTail302_10, scan, if_neg (by decide), innerCandidates302_10_checked]
  change scan edge (search edge 3) 4 [2011] 2011 innerTail302_11
    (if 3 ≤ innerCandidates302_10.length then search edge 3 [2011, 7963] innerCandidates302_10 9974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan302_11

def innerTail302_9 : List Nat := 6781 :: innerTail302_10
def innerCandidates302_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates302_9_checked : candidates edge 4 2011 6781 innerTail302_10 best.bound = innerCandidates302_9 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan302_9 : scan edge (search edge 3) 4 [2011] 2011 innerTail302_9 best = best := by
  rw [innerTail302_9, scan, if_neg (by decide), innerCandidates302_9_checked]
  change scan edge (search edge 3) 4 [2011] 2011 innerTail302_10
    (if 3 ≤ innerCandidates302_9.length then search edge 3 [2011, 6781] innerCandidates302_9 8792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan302_10

def innerTail302_8 : List Nat := 6637 :: innerTail302_9
def innerCandidates302_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates302_8_checked : candidates edge 4 2011 6637 innerTail302_9 best.bound = innerCandidates302_8 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan302_8 : scan edge (search edge 3) 4 [2011] 2011 innerTail302_8 best = best := by
  rw [innerTail302_8, scan, if_neg (by decide), innerCandidates302_8_checked]
  change scan edge (search edge 3) 4 [2011] 2011 innerTail302_9
    (if 3 ≤ innerCandidates302_8.length then search edge 3 [2011, 6637] innerCandidates302_8 8648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan302_9

def innerTail302_7 : List Nat := 6379 :: innerTail302_8
def innerCandidates302_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates302_7_checked : candidates edge 4 2011 6379 innerTail302_8 best.bound = innerCandidates302_7 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan302_7 : scan edge (search edge 3) 4 [2011] 2011 innerTail302_7 best = best := by
  rw [innerTail302_7, scan, if_neg (by decide), innerCandidates302_7_checked]
  change scan edge (search edge 3) 4 [2011] 2011 innerTail302_8
    (if 3 ≤ innerCandidates302_7.length then search edge 3 [2011, 6379] innerCandidates302_7 8390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan302_8

def innerTail302_6 : List Nat := 5521 :: innerTail302_7
def innerCandidates302_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates302_6_checked : candidates edge 4 2011 5521 innerTail302_7 best.bound = innerCandidates302_6 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan302_6 : scan edge (search edge 3) 4 [2011] 2011 innerTail302_6 best = best := by
  rw [innerTail302_6, scan, if_neg (by decide), innerCandidates302_6_checked]
  change scan edge (search edge 3) 4 [2011] 2011 innerTail302_7
    (if 3 ≤ innerCandidates302_6.length then search edge 3 [2011, 5521] innerCandidates302_6 7532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan302_7

def innerTail302_5 : List Nat := 5413 :: innerTail302_6
def innerCandidates302_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates302_5_checked : candidates edge 4 2011 5413 innerTail302_6 best.bound = innerCandidates302_5 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan302_5 : scan edge (search edge 3) 4 [2011] 2011 innerTail302_5 best = best := by
  rw [innerTail302_5, scan, if_neg (by decide), innerCandidates302_5_checked]
  change scan edge (search edge 3) 4 [2011] 2011 innerTail302_6
    (if 3 ≤ innerCandidates302_5.length then search edge 3 [2011, 5413] innerCandidates302_5 7424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan302_6

def innerTail302_4 : List Nat := 4177 :: innerTail302_5
def innerCandidates302_4 : List Nat := [6379, 6781]
set_option maxRecDepth 4096 in
theorem innerCandidates302_4_checked : candidates edge 4 2011 4177 innerTail302_5 best.bound = innerCandidates302_4 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan302_4 : scan edge (search edge 3) 4 [2011] 2011 innerTail302_4 best = best := by
  rw [innerTail302_4, scan, if_neg (by decide), innerCandidates302_4_checked]
  change scan edge (search edge 3) 4 [2011] 2011 innerTail302_5
    (if 3 ≤ innerCandidates302_4.length then search edge 3 [2011, 4177] innerCandidates302_4 6188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan302_5

def innerTail302_3 : List Nat := 4099 :: innerTail302_4
def innerCandidates302_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates302_3_checked : candidates edge 4 2011 4099 innerTail302_4 best.bound = innerCandidates302_3 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan302_3 : scan edge (search edge 3) 4 [2011] 2011 innerTail302_3 best = best := by
  rw [innerTail302_3, scan, if_neg (by decide), innerCandidates302_3_checked]
  change scan edge (search edge 3) 4 [2011] 2011 innerTail302_4
    (if 3 ≤ innerCandidates302_3.length then search edge 3 [2011, 4099] innerCandidates302_3 6110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan302_4

def innerTail302_2 : List Nat := 4093 :: innerTail302_3
def innerCandidates302_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates302_2_checked : candidates edge 4 2011 4093 innerTail302_3 best.bound = innerCandidates302_2 := by
  unfold candidates
  have row : edge 4093 = SearchData.row4093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan302_2 : scan edge (search edge 3) 4 [2011] 2011 innerTail302_2 best = best := by
  rw [innerTail302_2, scan, if_neg (by decide), innerCandidates302_2_checked]
  change scan edge (search edge 3) 4 [2011] 2011 innerTail302_3
    (if 3 ≤ innerCandidates302_2.length then search edge 3 [2011, 4093] innerCandidates302_2 6104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan302_3

def innerTail302_1 : List Nat := 3673 :: innerTail302_2
def innerCandidates302_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates302_1_checked : candidates edge 4 2011 3673 innerTail302_2 best.bound = innerCandidates302_1 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan302_1 : scan edge (search edge 3) 4 [2011] 2011 innerTail302_1 best = best := by
  rw [innerTail302_1, scan, if_neg (by decide), innerCandidates302_1_checked]
  change scan edge (search edge 3) 4 [2011] 2011 innerTail302_2
    (if 3 ≤ innerCandidates302_1.length then search edge 3 [2011, 3673] innerCandidates302_1 5684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan302_2

def innerTail302_0 : List Nat := 3253 :: innerTail302_1
def innerCandidates302_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates302_0_checked : candidates edge 4 2011 3253 innerTail302_1 best.bound = innerCandidates302_0 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan302_0 : scan edge (search edge 3) 4 [2011] 2011 innerTail302_0 best = best := by
  rw [innerTail302_0, scan, if_neg (by decide), innerCandidates302_0_checked]
  change scan edge (search edge 3) 4 [2011] 2011 innerTail302_1
    (if 3 ≤ innerCandidates302_0.length then search edge 3 [2011, 3253] innerCandidates302_0 5264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan302_1

theorem rootBranch302 : search edge 4 [2011] rootCandidates302 2011 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2011] 2011 innerTail302_0 best = best
  exact innerScan302_0

def rootCandidates301 : List Nat := [2087, 2237, 2543, 3413, 3557, 4271, 4673, 5039, 5051, 5153, 5189, 5651, 5657, 5693, 6563, 6689, 6791, 6833, 6983, 7013, 7211, 8171, 8273, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates301_checked : candidates edge 5 0 2003 tail302 best.bound = rootCandidates301 := by
  unfold candidates
  have row : edge 2003 = PrimePairCache.smallRow2003.contains := by funext q; rfl
  rw [row]
  decide

def innerTail301_24 : List Nat := []
theorem innerScan301_24 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_24 best = best := rfl

def innerTail301_23 : List Nat := 8297 :: innerTail301_24
def innerCandidates301_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates301_23_checked : candidates edge 4 2003 8297 innerTail301_24 best.bound = innerCandidates301_23 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_23 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_23 best = best := by
  rw [innerTail301_23, scan, if_neg (by decide), innerCandidates301_23_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_24
    (if 3 ≤ innerCandidates301_23.length then search edge 3 [2003, 8297] innerCandidates301_23 10300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_24

def innerTail301_22 : List Nat := 8273 :: innerTail301_23
def innerCandidates301_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates301_22_checked : candidates edge 4 2003 8273 innerTail301_23 best.bound = innerCandidates301_22 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_22 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_22 best = best := by
  rw [innerTail301_22, scan, if_neg (by decide), innerCandidates301_22_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_23
    (if 3 ≤ innerCandidates301_22.length then search edge 3 [2003, 8273] innerCandidates301_22 10276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_23

def innerTail301_21 : List Nat := 8171 :: innerTail301_22
def innerCandidates301_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates301_21_checked : candidates edge 4 2003 8171 innerTail301_22 best.bound = innerCandidates301_21 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_21 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_21 best = best := by
  rw [innerTail301_21, scan, if_neg (by decide), innerCandidates301_21_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_22
    (if 3 ≤ innerCandidates301_21.length then search edge 3 [2003, 8171] innerCandidates301_21 10174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_22

def innerTail301_20 : List Nat := 7211 :: innerTail301_21
def innerCandidates301_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates301_20_checked : candidates edge 4 2003 7211 innerTail301_21 best.bound = innerCandidates301_20 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_20 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_20 best = best := by
  rw [innerTail301_20, scan, if_neg (by decide), innerCandidates301_20_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_21
    (if 3 ≤ innerCandidates301_20.length then search edge 3 [2003, 7211] innerCandidates301_20 9214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_21

def innerTail301_19 : List Nat := 7013 :: innerTail301_20
def innerCandidates301_19 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates301_19_checked : candidates edge 4 2003 7013 innerTail301_20 best.bound = innerCandidates301_19 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_19 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_19 best = best := by
  rw [innerTail301_19, scan, if_neg (by decide), innerCandidates301_19_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_20
    (if 3 ≤ innerCandidates301_19.length then search edge 3 [2003, 7013] innerCandidates301_19 9016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_20

def innerTail301_18 : List Nat := 6983 :: innerTail301_19
def innerCandidates301_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates301_18_checked : candidates edge 4 2003 6983 innerTail301_19 best.bound = innerCandidates301_18 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_18 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_18 best = best := by
  rw [innerTail301_18, scan, if_neg (by decide), innerCandidates301_18_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_19
    (if 3 ≤ innerCandidates301_18.length then search edge 3 [2003, 6983] innerCandidates301_18 8986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_19

def innerTail301_17 : List Nat := 6833 :: innerTail301_18
def innerCandidates301_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates301_17_checked : candidates edge 4 2003 6833 innerTail301_18 best.bound = innerCandidates301_17 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_17 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_17 best = best := by
  rw [innerTail301_17, scan, if_neg (by decide), innerCandidates301_17_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_18
    (if 3 ≤ innerCandidates301_17.length then search edge 3 [2003, 6833] innerCandidates301_17 8836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_18

def innerTail301_16 : List Nat := 6791 :: innerTail301_17
def innerCandidates301_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates301_16_checked : candidates edge 4 2003 6791 innerTail301_17 best.bound = innerCandidates301_16 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_16 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_16 best = best := by
  rw [innerTail301_16, scan, if_neg (by decide), innerCandidates301_16_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_17
    (if 3 ≤ innerCandidates301_16.length then search edge 3 [2003, 6791] innerCandidates301_16 8794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_17

def innerTail301_15 : List Nat := 6689 :: innerTail301_16
def innerCandidates301_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates301_15_checked : candidates edge 4 2003 6689 innerTail301_16 best.bound = innerCandidates301_15 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_15 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_15 best = best := by
  rw [innerTail301_15, scan, if_neg (by decide), innerCandidates301_15_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_16
    (if 3 ≤ innerCandidates301_15.length then search edge 3 [2003, 6689] innerCandidates301_15 8692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_16

def innerTail301_14 : List Nat := 6563 :: innerTail301_15
def innerCandidates301_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates301_14_checked : candidates edge 4 2003 6563 innerTail301_15 best.bound = innerCandidates301_14 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_14 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_14 best = best := by
  rw [innerTail301_14, scan, if_neg (by decide), innerCandidates301_14_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_15
    (if 3 ≤ innerCandidates301_14.length then search edge 3 [2003, 6563] innerCandidates301_14 8566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_15

def innerTail301_13 : List Nat := 5693 :: innerTail301_14
def innerCandidates301_13 : List Nat := [6563, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates301_13_checked : candidates edge 4 2003 5693 innerTail301_14 best.bound = innerCandidates301_13 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_13 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_13 best = best := by
  rw [innerTail301_13, scan, if_neg (by decide), innerCandidates301_13_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_14
    (if 3 ≤ innerCandidates301_13.length then search edge 3 [2003, 5693] innerCandidates301_13 7696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_14

def innerTail301_12 : List Nat := 5657 :: innerTail301_13
def innerCandidates301_12 : List Nat := [6791, 6983]
set_option maxRecDepth 4096 in
theorem innerCandidates301_12_checked : candidates edge 4 2003 5657 innerTail301_13 best.bound = innerCandidates301_12 := by
  unfold candidates
  have row : edge 5657 = SearchData.row5657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_12 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_12 best = best := by
  rw [innerTail301_12, scan, if_neg (by decide), innerCandidates301_12_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_13
    (if 3 ≤ innerCandidates301_12.length then search edge 3 [2003, 5657] innerCandidates301_12 7660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_13

def innerTail301_11 : List Nat := 5651 :: innerTail301_12
def innerCandidates301_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates301_11_checked : candidates edge 4 2003 5651 innerTail301_12 best.bound = innerCandidates301_11 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_11 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_11 best = best := by
  rw [innerTail301_11, scan, if_neg (by decide), innerCandidates301_11_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_12
    (if 3 ≤ innerCandidates301_11.length then search edge 3 [2003, 5651] innerCandidates301_11 7654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_12

def innerTail301_10 : List Nat := 5189 :: innerTail301_11
def innerCandidates301_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates301_10_checked : candidates edge 4 2003 5189 innerTail301_11 best.bound = innerCandidates301_10 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_10 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_10 best = best := by
  rw [innerTail301_10, scan, if_neg (by decide), innerCandidates301_10_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_11
    (if 3 ≤ innerCandidates301_10.length then search edge 3 [2003, 5189] innerCandidates301_10 7192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_11

def innerTail301_9 : List Nat := 5153 :: innerTail301_10
def innerCandidates301_9 : List Nat := [8297]
set_option maxRecDepth 4096 in
theorem innerCandidates301_9_checked : candidates edge 4 2003 5153 innerTail301_10 best.bound = innerCandidates301_9 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_9 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_9 best = best := by
  rw [innerTail301_9, scan, if_neg (by decide), innerCandidates301_9_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_10
    (if 3 ≤ innerCandidates301_9.length then search edge 3 [2003, 5153] innerCandidates301_9 7156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_10

def innerTail301_8 : List Nat := 5051 :: innerTail301_9
def innerCandidates301_8 : List Nat := [6833]
set_option maxRecDepth 4096 in
theorem innerCandidates301_8_checked : candidates edge 4 2003 5051 innerTail301_9 best.bound = innerCandidates301_8 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_8 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_8 best = best := by
  rw [innerTail301_8, scan, if_neg (by decide), innerCandidates301_8_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_9
    (if 3 ≤ innerCandidates301_8.length then search edge 3 [2003, 5051] innerCandidates301_8 7054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_9

def innerTail301_7 : List Nat := 5039 :: innerTail301_8
def innerCandidates301_7 : List Nat := [5153]
set_option maxRecDepth 4096 in
theorem innerCandidates301_7_checked : candidates edge 4 2003 5039 innerTail301_8 best.bound = innerCandidates301_7 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_7 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_7 best = best := by
  rw [innerTail301_7, scan, if_neg (by decide), innerCandidates301_7_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_8
    (if 3 ≤ innerCandidates301_7.length then search edge 3 [2003, 5039] innerCandidates301_7 7042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_8

def innerTail301_6 : List Nat := 4673 :: innerTail301_7
def innerCandidates301_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates301_6_checked : candidates edge 4 2003 4673 innerTail301_7 best.bound = innerCandidates301_6 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_6 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_6 best = best := by
  rw [innerTail301_6, scan, if_neg (by decide), innerCandidates301_6_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_7
    (if 3 ≤ innerCandidates301_6.length then search edge 3 [2003, 4673] innerCandidates301_6 6676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_7

def innerTail301_5 : List Nat := 4271 :: innerTail301_6
def innerCandidates301_5 : List Nat := [4673, 5693]
set_option maxRecDepth 4096 in
theorem innerCandidates301_5_checked : candidates edge 4 2003 4271 innerTail301_6 best.bound = innerCandidates301_5 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_5 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_5 best = best := by
  rw [innerTail301_5, scan, if_neg (by decide), innerCandidates301_5_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_6
    (if 3 ≤ innerCandidates301_5.length then search edge 3 [2003, 4271] innerCandidates301_5 6274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_6

def innerTail301_4 : List Nat := 3557 :: innerTail301_5
def innerCandidates301_4 : List Nat := [5051]
set_option maxRecDepth 4096 in
theorem innerCandidates301_4_checked : candidates edge 4 2003 3557 innerTail301_5 best.bound = innerCandidates301_4 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_4 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_4 best = best := by
  rw [innerTail301_4, scan, if_neg (by decide), innerCandidates301_4_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_5
    (if 3 ≤ innerCandidates301_4.length then search edge 3 [2003, 3557] innerCandidates301_4 5560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_5

def innerTail301_3 : List Nat := 3413 :: innerTail301_4
def innerCandidates301_3 : List Nat := [6689, 7211]
set_option maxRecDepth 4096 in
theorem innerCandidates301_3_checked : candidates edge 4 2003 3413 innerTail301_4 best.bound = innerCandidates301_3 := by
  unfold candidates
  have row : edge 3413 = SearchData.row3413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_3 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_3 best = best := by
  rw [innerTail301_3, scan, if_neg (by decide), innerCandidates301_3_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_4
    (if 3 ≤ innerCandidates301_3.length then search edge 3 [2003, 3413] innerCandidates301_3 5416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_4

def innerTail301_2 : List Nat := 2543 :: innerTail301_3
def innerCandidates301_2 : List Nat := [5051, 5153, 6791]
set_option maxRecDepth 4096 in
theorem innerCandidates301_2_checked : candidates edge 4 2003 2543 innerTail301_3 best.bound = innerCandidates301_2 := by
  unfold candidates
  have row : edge 2543 = PrimePairCache.smallRow2543.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild301_2 : search edge 3 [2003, 2543] innerCandidates301_2 4546 best = best := by decide

theorem innerScan301_2 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_2 best = best := by
  rw [innerTail301_2, scan, if_neg (by decide), innerCandidates301_2_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_3
    (if 3 ≤ innerCandidates301_2.length then search edge 3 [2003, 2543] innerCandidates301_2 4546 best else best) = best
  rw [if_pos (by decide), innerChild301_2]
  exact innerScan301_3

def innerTail301_1 : List Nat := 2237 :: innerTail301_2
def innerCandidates301_1 : List Nat := [5693, 6689, 6791]
set_option maxRecDepth 4096 in
theorem innerCandidates301_1_checked : candidates edge 4 2003 2237 innerTail301_2 best.bound = innerCandidates301_1 := by
  unfold candidates
  have row : edge 2237 = PrimePairCache.smallRow2237.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild301_1 : search edge 3 [2003, 2237] innerCandidates301_1 4240 best = best := by decide

theorem innerScan301_1 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_1 best = best := by
  rw [innerTail301_1, scan, if_neg (by decide), innerCandidates301_1_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_2
    (if 3 ≤ innerCandidates301_1.length then search edge 3 [2003, 2237] innerCandidates301_1 4240 best else best) = best
  rw [if_pos (by decide), innerChild301_1]
  exact innerScan301_2

def innerTail301_0 : List Nat := 2087 :: innerTail301_1
def innerCandidates301_0 : List Nat := [6689]
set_option maxRecDepth 4096 in
theorem innerCandidates301_0_checked : candidates edge 4 2003 2087 innerTail301_1 best.bound = innerCandidates301_0 := by
  unfold candidates
  have row : edge 2087 = PrimePairCache.smallRow2087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan301_0 : scan edge (search edge 3) 4 [2003] 2003 innerTail301_0 best = best := by
  rw [innerTail301_0, scan, if_neg (by decide), innerCandidates301_0_checked]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_1
    (if 3 ≤ innerCandidates301_0.length then search edge 3 [2003, 2087] innerCandidates301_0 4090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan301_1

theorem rootBranch301 : search edge 4 [2003] rootCandidates301 2003 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2003] 2003 innerTail301_0 best = best
  exact innerScan301_0

def rootCandidates300 : List Nat := [2347, 2521, 3361, 3469, 3607, 3733, 5869, 6121, 6469, 6961, 7717, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates300_checked : candidates edge 5 0 1999 tail301 best.bound = rootCandidates300 := by
  unfold candidates
  have row : edge 1999 = PrimePairCache.smallRow1999.contains := by funext q; rfl
  rw [row]
  decide

def innerTail300_12 : List Nat := []
theorem innerScan300_12 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_12 best = best := rfl

def innerTail300_11 : List Nat := 8263 :: innerTail300_12
def innerCandidates300_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates300_11_checked : candidates edge 4 1999 8263 innerTail300_12 best.bound = innerCandidates300_11 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan300_11 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_11 best = best := by
  rw [innerTail300_11, scan, if_neg (by decide), innerCandidates300_11_checked]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_12
    (if 3 ≤ innerCandidates300_11.length then search edge 3 [1999, 8263] innerCandidates300_11 10262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan300_12

def innerTail300_10 : List Nat := 7717 :: innerTail300_11
def innerCandidates300_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates300_10_checked : candidates edge 4 1999 7717 innerTail300_11 best.bound = innerCandidates300_10 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan300_10 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_10 best = best := by
  rw [innerTail300_10, scan, if_neg (by decide), innerCandidates300_10_checked]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_11
    (if 3 ≤ innerCandidates300_10.length then search edge 3 [1999, 7717] innerCandidates300_10 9716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan300_11

def innerTail300_9 : List Nat := 6961 :: innerTail300_10
def innerCandidates300_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates300_9_checked : candidates edge 4 1999 6961 innerTail300_10 best.bound = innerCandidates300_9 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan300_9 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_9 best = best := by
  rw [innerTail300_9, scan, if_neg (by decide), innerCandidates300_9_checked]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_10
    (if 3 ≤ innerCandidates300_9.length then search edge 3 [1999, 6961] innerCandidates300_9 8960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan300_10

def innerTail300_8 : List Nat := 6469 :: innerTail300_9
def innerCandidates300_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates300_8_checked : candidates edge 4 1999 6469 innerTail300_9 best.bound = innerCandidates300_8 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan300_8 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_8 best = best := by
  rw [innerTail300_8, scan, if_neg (by decide), innerCandidates300_8_checked]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_9
    (if 3 ≤ innerCandidates300_8.length then search edge 3 [1999, 6469] innerCandidates300_8 8468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan300_9

def innerTail300_7 : List Nat := 6121 :: innerTail300_8
def innerCandidates300_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates300_7_checked : candidates edge 4 1999 6121 innerTail300_8 best.bound = innerCandidates300_7 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan300_7 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_7 best = best := by
  rw [innerTail300_7, scan, if_neg (by decide), innerCandidates300_7_checked]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_8
    (if 3 ≤ innerCandidates300_7.length then search edge 3 [1999, 6121] innerCandidates300_7 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan300_8

def innerTail300_6 : List Nat := 5869 :: innerTail300_7
def innerCandidates300_6 : List Nat := [6961]
set_option maxRecDepth 4096 in
theorem innerCandidates300_6_checked : candidates edge 4 1999 5869 innerTail300_7 best.bound = innerCandidates300_6 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan300_6 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_6 best = best := by
  rw [innerTail300_6, scan, if_neg (by decide), innerCandidates300_6_checked]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_7
    (if 3 ≤ innerCandidates300_6.length then search edge 3 [1999, 5869] innerCandidates300_6 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan300_7

def innerTail300_5 : List Nat := 3733 :: innerTail300_6
def innerCandidates300_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates300_5_checked : candidates edge 4 1999 3733 innerTail300_6 best.bound = innerCandidates300_5 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan300_5 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_5 best = best := by
  rw [innerTail300_5, scan, if_neg (by decide), innerCandidates300_5_checked]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_6
    (if 3 ≤ innerCandidates300_5.length then search edge 3 [1999, 3733] innerCandidates300_5 5732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan300_6

def innerTail300_4 : List Nat := 3607 :: innerTail300_5
def innerCandidates300_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates300_4_checked : candidates edge 4 1999 3607 innerTail300_5 best.bound = innerCandidates300_4 := by
  unfold candidates
  have row : edge 3607 = SearchData.row3607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan300_4 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_4 best = best := by
  rw [innerTail300_4, scan, if_neg (by decide), innerCandidates300_4_checked]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_5
    (if 3 ≤ innerCandidates300_4.length then search edge 3 [1999, 3607] innerCandidates300_4 5606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan300_5

def innerTail300_3 : List Nat := 3469 :: innerTail300_4
def innerCandidates300_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates300_3_checked : candidates edge 4 1999 3469 innerTail300_4 best.bound = innerCandidates300_3 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan300_3 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_3 best = best := by
  rw [innerTail300_3, scan, if_neg (by decide), innerCandidates300_3_checked]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_4
    (if 3 ≤ innerCandidates300_3.length then search edge 3 [1999, 3469] innerCandidates300_3 5468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan300_4

def innerTail300_2 : List Nat := 3361 :: innerTail300_3
def innerCandidates300_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates300_2_checked : candidates edge 4 1999 3361 innerTail300_3 best.bound = innerCandidates300_2 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan300_2 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_2 best = best := by
  rw [innerTail300_2, scan, if_neg (by decide), innerCandidates300_2_checked]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_3
    (if 3 ≤ innerCandidates300_2.length then search edge 3 [1999, 3361] innerCandidates300_2 5360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan300_3

def innerTail300_1 : List Nat := 2521 :: innerTail300_2
def innerCandidates300_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates300_1_checked : candidates edge 4 1999 2521 innerTail300_2 best.bound = innerCandidates300_1 := by
  unfold candidates
  have row : edge 2521 = PrimePairCache.smallRow2521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan300_1 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_1 best = best := by
  rw [innerTail300_1, scan, if_neg (by decide), innerCandidates300_1_checked]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_2
    (if 3 ≤ innerCandidates300_1.length then search edge 3 [1999, 2521] innerCandidates300_1 4520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan300_2

def innerTail300_0 : List Nat := 2347 :: innerTail300_1
def innerCandidates300_0 : List Nat := [6469]
set_option maxRecDepth 4096 in
theorem innerCandidates300_0_checked : candidates edge 4 1999 2347 innerTail300_1 best.bound = innerCandidates300_0 := by
  unfold candidates
  have row : edge 2347 = PrimePairCache.smallRow2347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan300_0 : scan edge (search edge 3) 4 [1999] 1999 innerTail300_0 best = best := by
  rw [innerTail300_0, scan, if_neg (by decide), innerCandidates300_0_checked]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_1
    (if 3 ≤ innerCandidates300_0.length then search edge 3 [1999, 2347] innerCandidates300_0 4346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan300_1

theorem rootBranch300 : search edge 4 [1999] rootCandidates300 1999 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1999] 1999 innerTail300_0 best = best
  exact innerScan300_0

def rootCandidates299 : List Nat := [2003, 2423, 2903, 3119, 3323, 3677, 4397, 4583, 4793, 6053, 7043, 7583]
set_option maxRecDepth 4096 in
theorem rootCandidates299_checked : candidates edge 5 0 1997 tail300 best.bound = rootCandidates299 := by
  unfold candidates
  have row : edge 1997 = PrimePairCache.smallRow1997.contains := by funext q; rfl
  rw [row]
  decide

def innerTail299_12 : List Nat := []
theorem innerScan299_12 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_12 best = best := rfl

def innerTail299_11 : List Nat := 7583 :: innerTail299_12
def innerCandidates299_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates299_11_checked : candidates edge 4 1997 7583 innerTail299_12 best.bound = innerCandidates299_11 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan299_11 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_11 best = best := by
  rw [innerTail299_11, scan, if_neg (by decide), innerCandidates299_11_checked]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_12
    (if 3 ≤ innerCandidates299_11.length then search edge 3 [1997, 7583] innerCandidates299_11 9580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan299_12

def innerTail299_10 : List Nat := 7043 :: innerTail299_11
def innerCandidates299_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates299_10_checked : candidates edge 4 1997 7043 innerTail299_11 best.bound = innerCandidates299_10 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan299_10 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_10 best = best := by
  rw [innerTail299_10, scan, if_neg (by decide), innerCandidates299_10_checked]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_11
    (if 3 ≤ innerCandidates299_10.length then search edge 3 [1997, 7043] innerCandidates299_10 9040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan299_11

def innerTail299_9 : List Nat := 6053 :: innerTail299_10
def innerCandidates299_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates299_9_checked : candidates edge 4 1997 6053 innerTail299_10 best.bound = innerCandidates299_9 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan299_9 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_9 best = best := by
  rw [innerTail299_9, scan, if_neg (by decide), innerCandidates299_9_checked]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_10
    (if 3 ≤ innerCandidates299_9.length then search edge 3 [1997, 6053] innerCandidates299_9 8050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan299_10

def innerTail299_8 : List Nat := 4793 :: innerTail299_9
def innerCandidates299_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates299_8_checked : candidates edge 4 1997 4793 innerTail299_9 best.bound = innerCandidates299_8 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan299_8 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_8 best = best := by
  rw [innerTail299_8, scan, if_neg (by decide), innerCandidates299_8_checked]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_9
    (if 3 ≤ innerCandidates299_8.length then search edge 3 [1997, 4793] innerCandidates299_8 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan299_9

def innerTail299_7 : List Nat := 4583 :: innerTail299_8
def innerCandidates299_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates299_7_checked : candidates edge 4 1997 4583 innerTail299_8 best.bound = innerCandidates299_7 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan299_7 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_7 best = best := by
  rw [innerTail299_7, scan, if_neg (by decide), innerCandidates299_7_checked]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_8
    (if 3 ≤ innerCandidates299_7.length then search edge 3 [1997, 4583] innerCandidates299_7 6580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan299_8

def innerTail299_6 : List Nat := 4397 :: innerTail299_7
def innerCandidates299_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates299_6_checked : candidates edge 4 1997 4397 innerTail299_7 best.bound = innerCandidates299_6 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan299_6 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_6 best = best := by
  rw [innerTail299_6, scan, if_neg (by decide), innerCandidates299_6_checked]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_7
    (if 3 ≤ innerCandidates299_6.length then search edge 3 [1997, 4397] innerCandidates299_6 6394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan299_7

def innerTail299_5 : List Nat := 3677 :: innerTail299_6
def innerCandidates299_5 : List Nat := [7583]
set_option maxRecDepth 4096 in
theorem innerCandidates299_5_checked : candidates edge 4 1997 3677 innerTail299_6 best.bound = innerCandidates299_5 := by
  unfold candidates
  have row : edge 3677 = SearchData.row3677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan299_5 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_5 best = best := by
  rw [innerTail299_5, scan, if_neg (by decide), innerCandidates299_5_checked]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_6
    (if 3 ≤ innerCandidates299_5.length then search edge 3 [1997, 3677] innerCandidates299_5 5674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan299_6

def innerTail299_4 : List Nat := 3323 :: innerTail299_5
def innerCandidates299_4 : List Nat := [4583]
set_option maxRecDepth 4096 in
theorem innerCandidates299_4_checked : candidates edge 4 1997 3323 innerTail299_5 best.bound = innerCandidates299_4 := by
  unfold candidates
  have row : edge 3323 = SearchData.row3323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan299_4 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_4 best = best := by
  rw [innerTail299_4, scan, if_neg (by decide), innerCandidates299_4_checked]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_5
    (if 3 ≤ innerCandidates299_4.length then search edge 3 [1997, 3323] innerCandidates299_4 5320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan299_5

def innerTail299_3 : List Nat := 3119 :: innerTail299_4
def innerCandidates299_3 : List Nat := [3677]
set_option maxRecDepth 4096 in
theorem innerCandidates299_3_checked : candidates edge 4 1997 3119 innerTail299_4 best.bound = innerCandidates299_3 := by
  unfold candidates
  have row : edge 3119 = SearchData.row3119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan299_3 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_3 best = best := by
  rw [innerTail299_3, scan, if_neg (by decide), innerCandidates299_3_checked]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_4
    (if 3 ≤ innerCandidates299_3.length then search edge 3 [1997, 3119] innerCandidates299_3 5116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan299_4

def innerTail299_2 : List Nat := 2903 :: innerTail299_3
def innerCandidates299_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates299_2_checked : candidates edge 4 1997 2903 innerTail299_3 best.bound = innerCandidates299_2 := by
  unfold candidates
  have row : edge 2903 = SearchData.row2903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan299_2 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_2 best = best := by
  rw [innerTail299_2, scan, if_neg (by decide), innerCandidates299_2_checked]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_3
    (if 3 ≤ innerCandidates299_2.length then search edge 3 [1997, 2903] innerCandidates299_2 4900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan299_3

def innerTail299_1 : List Nat := 2423 :: innerTail299_2
def innerCandidates299_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates299_1_checked : candidates edge 4 1997 2423 innerTail299_2 best.bound = innerCandidates299_1 := by
  unfold candidates
  have row : edge 2423 = PrimePairCache.smallRow2423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan299_1 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_1 best = best := by
  rw [innerTail299_1, scan, if_neg (by decide), innerCandidates299_1_checked]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_2
    (if 3 ≤ innerCandidates299_1.length then search edge 3 [1997, 2423] innerCandidates299_1 4420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan299_2

def innerTail299_0 : List Nat := 2003 :: innerTail299_1
def innerCandidates299_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates299_0_checked : candidates edge 4 1997 2003 innerTail299_1 best.bound = innerCandidates299_0 := by
  unfold candidates
  have row : edge 2003 = PrimePairCache.smallRow2003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan299_0 : scan edge (search edge 3) 4 [1997] 1997 innerTail299_0 best = best := by
  rw [innerTail299_0, scan, if_neg (by decide), innerCandidates299_0_checked]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_1
    (if 3 ≤ innerCandidates299_0.length then search edge 3 [1997, 2003] innerCandidates299_0 4000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan299_1

theorem rootBranch299 : search edge 4 [1997] rootCandidates299 1997 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1997] 1997 innerTail299_0 best = best
  exact innerScan299_0

end ProjectEulerSolutions.PrimePairUpper.Search
